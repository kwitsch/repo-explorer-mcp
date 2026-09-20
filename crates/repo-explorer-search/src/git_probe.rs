//! git2-backed `RepoStateProbe`: repository fingerprinting for the result
//! caches. Uses the in-process `git2` (libgit2) bindings — no external `git`
//! binary at runtime or in tests. Lives in this crate alongside the other
//! backend probes; every failure degrades to `None`.

use git2::{ObjectType, Oid, Repository, Status, StatusOptions};
use repo_explorer_core::fingerprint::{RepoFingerprint, RepoStateProbe};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Single-slot cache for the most-recently-opened top-level `Repository`,
/// keyed by the exact `repo_root` it was opened for.
type RepoSlot = Mutex<Option<(PathBuf, Repository)>>;
/// Per-file `(len, mtime) -> content oid` cache, guarded by the
/// [`RACY_MARGIN`] check in `cached_hash_or_compute`.
type HashCache = Mutex<HashMap<PathBuf, (u64, Option<SystemTime>, Oid)>>;

/// Probes repository state via the in-process `git2` (libgit2) library. Every
/// failure (not a repository, bare repo, unborn HEAD, per-file stat/hash
/// error) degrades to `None` — "no fingerprint" simply disables caching for
/// that call. Holds a single-slot repo handle (shared between `fingerprint`
/// and `changed_paths`) and a per-file content-hash cache, both behind
/// `Arc<Mutex<_>>` so they can be moved into the `spawn_blocking` closures.
pub struct GitStateProbe {
    timeout: Duration,
    repo_slot: Arc<RepoSlot>,
    hash_cache: Arc<HashCache>,
}

impl GitStateProbe {
    /// `timeout_seconds = 0` means "no timeout", matching `SearchConfig`.
    pub fn new(timeout_seconds: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_seconds),
            repo_slot: Arc::new(Mutex::new(None)),
            hash_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// The owner-executable bit of a regular tracked file, folded into its
/// dirty-hash entry alongside its content oid. `Oid::hash_file` only covers
/// content, so a `chmod +x`/`-x` with no byte change would otherwise leave
/// `dirty_hash` unchanged even though `git status`/`git diff` report it.
/// Windows has no such bit in git's own tracking model here, so this is
/// always `0` there.
#[cfg(unix)]
fn exec_bit(full: &Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(full)
        .map(|m| m.permissions().mode() & 0o100)
        .unwrap_or(0)
}

#[cfg(not(unix))]
fn exec_bit(_full: &Path) -> u32 {
    0
}

/// `(len, mtime)` stat tag shared by the untracked-entry fingerprint and
/// every tracked-changed hash-failure fallback below, so a future tweak to
/// the stat heuristic (e.g. adding another field) only needs to land here
/// instead of drifting across separately-typed-out copies.
fn stat_tag(path: &str, tag: impl std::fmt::Display, full: &Path) -> String {
    match std::fs::symlink_metadata(full) {
        Ok(meta) => format!("{path}:{tag}:{}:{:?}", meta.len(), meta.modified().ok()),
        Err(_) => format!("{path}:{tag}:missing"),
    }
}

fn sha256_hex(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0u8]);
    }
    hex::encode(hasher.finalize())
}

/// Racy-git guard: a cache entry is only trusted (read or written) when the
/// file's mtime is already safely older than this margin relative to "now".
/// Two writes to the same path within one filesystem mtime tick would
/// otherwise be indistinguishable by `(len, mtime)` alone, silently hiding a
/// real content change (exactly the risk the design explicitly rejected for
/// "path-set-only" fingerprinting). Requiring the file to have been quiet for
/// at least `RACY_MARGIN` before trusting its stat means a just-written file
/// is always rehashed — a large *stable* dirty file (the actual perf target)
/// gets the fast path; a file mid-edit never does.
const RACY_MARGIN: Duration = Duration::from_secs(2);

/// Content oid for a tracked-changed file, with a racy-git-safe stat cache to
/// avoid re-reading and re-hashing a large file that hasn't changed since the
/// last call. No eviction: an upgrade path if long-running processes with
/// many distinct dirty files ever make this cache's memory use a concern.
fn cached_hash_or_compute(full: &Path, cache: &HashCache) -> Option<Oid> {
    let meta = std::fs::metadata(full).ok()?;
    let len = meta.len();
    let mtime = meta.modified().ok();
    let stable = mtime
        .and_then(|m| SystemTime::now().duration_since(m).ok())
        .is_some_and(|age| age > RACY_MARGIN);

    if stable
        && let Some((cached_len, cached_mtime, oid)) = cache.lock().unwrap().get(full)
        && *cached_len == len
        && *cached_mtime == mtime
    {
        return Some(*oid);
    }

    let oid = Oid::hash_file(ObjectType::Blob, full).ok()?;
    if stable {
        cache
            .lock()
            .unwrap()
            .insert(full.to_path_buf(), (len, mtime, oid));
    }
    Some(oid)
}

/// Runs `f` against the cached top-level `Repository` for `repo_root`,
/// (re)opening it only when the slot is empty or holds a different
/// `repo_root` — shared between `fingerprint` and `changed_paths` so a
/// same-repo call pair (the common `AgentLoop` pattern) discovers `.git`
/// once, not twice. Single slot, so concurrent calls for different repos
/// serialize on this lock; upgrade to a keyed table if that ever matters.
fn with_repo<T>(
    repo_root: &Path,
    repo_slot: &RepoSlot,
    f: impl FnOnce(&Repository) -> Option<T>,
) -> Option<T> {
    let mut guard = repo_slot.lock().unwrap();
    let reuse = matches!(guard.as_ref(), Some((cached, _)) if cached == repo_root);
    if !reuse {
        let repo = Repository::discover(repo_root).ok()?;
        *guard = Some((repo_root.to_path_buf(), repo));
    }
    f(&guard.as_ref()?.1)
}

fn fingerprint_blocking(
    repo_root: &Path,
    repo_slot: &RepoSlot,
    hash_cache: &HashCache,
) -> Option<RepoFingerprint> {
    with_repo(repo_root, repo_slot, |repo| {
        fingerprint_from_repo(repo, hash_cache)
    })
}

/// Single blocking status walk over an already-opened `repo`. Every relative
/// path is joined against `workdir` (never the caller's `repo_root`, which
/// may be a strict subdirectory of the git top level). Tracked-changed
/// entries fold in a cheap content-sensitive blob oid (`Oid::hash_file`, no
/// odb write, works for text AND binary, cached — see
/// `cached_hash_or_compute`); untracked entries fold in a `(len, mtime)` stat
/// via `symlink_metadata` (F-07) — deliberately NOT a content hash,
/// preserving the large-untracked-tree perf guard, and stating the link
/// itself so a dangling untracked symlink's own target string is what
/// varies. Ignored files stay excluded (default `StatusOptions`), a
/// deliberate carried-forward gap. `parts` is sorted because git2's status
/// entry order is not guaranteed.
fn fingerprint_from_repo(repo: &Repository, hash_cache: &HashCache) -> Option<RepoFingerprint> {
    let workdir = repo.workdir()?.to_path_buf();

    // Unborn branch (commit-less repo) → head() fails → None, matching the
    // old CLI's empty-`rev-parse`→None behavior.
    let head_sha = repo.head().ok()?.peel_to_commit().ok()?.id().to_string();

    let mut opts = StatusOptions::new();
    opts.include_untracked(true).recurse_untracked_dirs(true);
    let statuses = repo.statuses(Some(&mut opts)).ok()?;

    let mut parts: Vec<String> = Vec::new();
    for entry in statuses.iter() {
        // Skip non-UTF-8 paths (matching the existing exotic-name blind spot).
        let Ok(path) = entry.path() else {
            continue;
        };
        let s = entry.status();
        let bits = s.bits();
        let full = workdir.join(path);
        if s.contains(Status::WT_NEW) {
            parts.push(stat_tag(path, "U", &full));
        } else if (s.contains(Status::WT_DELETED) || s.contains(Status::INDEX_DELETED))
            && !full.exists()
        {
            parts.push(format!("{path}:{bits}:gone"));
        } else if std::fs::symlink_metadata(&full)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
        {
            // A tracked symlink's own blob content is its target-path text,
            // not the pointee's bytes — Oid::hash_file opens (and so
            // follows) the link and would hash the wrong content. Hash the
            // link text itself, exactly what git stores for a symlink blob.
            // Checked before the directory branch below: a symlink to a
            // directory must still be hashed as a symlink, not recursed
            // into as if it were a submodule.
            let part = match std::fs::read_link(&full).ok().and_then(|target| {
                Oid::hash_object(ObjectType::Blob, target.as_os_str().as_encoded_bytes()).ok()
            }) {
                Some(oid) => format!("{path}:{bits}:{oid}"),
                None => stat_tag(path, bits, &full),
            };
            parts.push(part);
        } else if full.is_dir() {
            // A submodule (gitlink) or a typechange to a plain directory:
            // Oid::hash_file cannot read a directory, and the directory's
            // own mtime does not change either when a file nested inside it
            // is edited or when it is checked out to a different commit.
            // Recurse as its own repo so both transitions are visible; a
            // fresh (uncached) discover, since a submodule isn't the
            // top-level repo_root the shared slot tracks, and reusing that
            // slot here would recursively lock it. If it isn't actually a
            // git repo (or has no commits yet), fall back to the stat
            // string like every other hash failure.
            let part = match Repository::discover(&full)
                .ok()
                .and_then(|sub| fingerprint_from_repo(&sub, hash_cache))
            {
                Some(fp) => format!("{path}:{bits}:sub:{}:{}", fp.head_sha, fp.dirty_hash),
                None => stat_tag(path, bits, &full),
            };
            parts.push(part);
        } else {
            // Tracked-changed (modified/added/typechange/renamed, staged or
            // unstaged). Content-sensitive blob oid plus the owner-exec bit
            // (so a mode-only `chmod +x`/`-x` with no content change still
            // varies the fingerprint, matching git's own mode tracking); on
            // error fall back to a stat string so the entry still varies.
            let part = match cached_hash_or_compute(&full, hash_cache) {
                Some(oid) => format!("{path}:{bits}:{}:{oid}", exec_bit(&full)),
                None => stat_tag(path, bits, &full),
            };
            parts.push(part);
        }
    }
    parts.sort();
    let joined = parts.join("\n");
    Some(RepoFingerprint {
        head_sha,
        dirty_hash: sha256_hex(&[&joined]),
    })
}

/// Committed leg of `changed_paths`: diff the two resolved commit trees.
/// Deleted deltas leave `new_file().path()` empty in libgit2 — only
/// `old_file()` carries the path — so fall back to it, otherwise every deleted
/// file is silently dropped (regresses vs `git diff --name-only`). Any
/// resolution failure → `None`.
fn changed_paths_blocking(
    repo_root: &Path,
    from_sha: &str,
    to_sha: &str,
    repo_slot: &RepoSlot,
) -> Option<Vec<PathBuf>> {
    with_repo(repo_root, repo_slot, |repo| {
        let from_tree = repo
            .find_commit(Oid::from_str(from_sha).ok()?)
            .ok()?
            .tree()
            .ok()?;
        let to_tree = repo
            .find_commit(Oid::from_str(to_sha).ok()?)
            .ok()?
            .tree()
            .ok()?;
        let diff = repo
            .diff_tree_to_tree(Some(&from_tree), Some(&to_tree), None)
            .ok()?;
        let paths = diff
            .deltas()
            .filter_map(|d| {
                d.new_file()
                    .path()
                    .or_else(|| d.old_file().path())
                    .map(PathBuf::from)
            })
            .collect();
        Some(paths)
    })
}

/// Runs blocking git2 work off the async runtime, bounded by `timeout`
/// (`Duration::ZERO` = no timeout, matching `SearchConfig`). A `spawn_blocking`
/// task cannot be cancelled, so on timeout the blocking call keeps running in
/// the background while this returns `None` — acceptable for the fast local
/// reads git2 does here; a stuck network-mounted `.git` or pathologically
/// large repo degrades to "no fingerprint" instead of hanging the caller.
async fn run_blocking<T: Send + 'static>(
    timeout: Duration,
    f: impl FnOnce() -> Option<T> + Send + 'static,
) -> Option<T> {
    let handle = tokio::task::spawn_blocking(f);
    if timeout.is_zero() {
        handle.await.ok().flatten()
    } else {
        tokio::time::timeout(timeout, handle)
            .await
            .ok()
            .and_then(Result::ok)
            .flatten()
    }
}

impl RepoStateProbe for GitStateProbe {
    async fn fingerprint(&self, repo_root: &Path) -> Option<RepoFingerprint> {
        let repo_root = repo_root.to_path_buf();
        let repo_slot = Arc::clone(&self.repo_slot);
        let hash_cache = Arc::clone(&self.hash_cache);
        run_blocking(self.timeout, move || {
            fingerprint_blocking(&repo_root, &repo_slot, &hash_cache)
        })
        .await
    }

    async fn changed_paths(
        &self,
        repo_root: &Path,
        from: &RepoFingerprint,
        to: &RepoFingerprint,
    ) -> Option<Vec<PathBuf>> {
        // A differing dirty state cannot be enumerated after the fact — report
        // "unknown" and let the caller invalidate. Same head → no committed
        // change. Both short-circuits are pure Rust (no git2).
        if from.dirty_hash != to.dirty_hash {
            return None;
        }
        if from.head_sha == to.head_sha {
            return Some(Vec::new());
        }
        let repo_root = repo_root.to_path_buf();
        let from_sha = from.head_sha.clone();
        let to_sha = to.head_sha.clone();
        let repo_slot = Arc::clone(&self.repo_slot);
        run_blocking(self.timeout, move || {
            changed_paths_blocking(&repo_root, &from_sha, &to_sha, &repo_slot)
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use git2::{Commit, IndexAddOption, Signature};

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("git_probe_{tag}_{}", std::process::id()))
    }

    fn init_repo(dir: &Path) -> Repository {
        std::fs::create_dir_all(dir).unwrap();
        let repo = Repository::init(dir).unwrap();
        {
            let mut cfg = repo.config().unwrap();
            cfg.set_str("user.email", "t@example.com").unwrap();
            cfg.set_str("user.name", "t").unwrap();
        }
        repo
    }

    /// Commit whatever is currently staged in the index (parent = current
    /// HEAD if any, else a root commit).
    fn commit_index(repo: &Repository, msg: &str) {
        let mut index = repo.index().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = Signature::now("t", "t@example.com").unwrap();
        let parent = repo.head().ok().and_then(|h| h.peel_to_commit().ok());
        let parents: Vec<&Commit> = parent.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, msg, &tree, &parents)
            .unwrap();
    }

    /// Stage all working-tree files and commit.
    fn write_commit(repo: &Repository, msg: &str) {
        {
            let mut index = repo.index().unwrap();
            index
                .add_all(["*"].iter(), IndexAddOption::DEFAULT, None)
                .unwrap();
            index.write().unwrap();
        }
        commit_index(repo, msg);
    }

    #[tokio::test]
    async fn non_repo_yields_no_fingerprint() {
        let dir = temp_dir("nonrepo");
        std::fs::create_dir_all(&dir).unwrap();
        let probe = GitStateProbe::new(30);
        // The temp dir might live under a parent repository; only assert the
        // None contract when git2 itself reports "not a repository".
        if Repository::discover(&dir).is_err() {
            assert_eq!(probe.fingerprint(&dir).await, None);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn dirty_change_and_commit_change_the_fingerprint() {
        let dir = temp_dir("fp");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        let clean = probe.fingerprint(&dir).await.expect("fingerprint");

        std::fs::write(dir.join("a.txt"), "two\n").unwrap();
        let dirty = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_eq!(clean.head_sha, dirty.head_sha);
        assert_ne!(clean.dirty_hash, dirty.dirty_hash);

        write_commit(&repo, "c2");
        let committed = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(clean.head_sha, committed.head_sha);
        assert_eq!(clean.dirty_hash, committed.dirty_hash, "clean == clean");

        // Committed diff between the two clean states names the file.
        let changed = probe
            .changed_paths(&dir, &clean, &committed)
            .await
            .expect("changed paths");
        assert_eq!(changed, vec![PathBuf::from("a.txt")]);

        // Same fingerprint → empty change set; differing dirty state → unknown.
        assert_eq!(
            probe.changed_paths(&dir, &clean, &clean).await,
            Some(Vec::new())
        );
        assert_eq!(probe.changed_paths(&dir, &clean, &dirty).await, None);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// F-07: editing bytes inside an already-untracked file must change the
    /// fingerprint even though neither status nor a HEAD diff notices the
    /// content change (only the file's continued presence).
    #[tokio::test]
    async fn untracked_file_content_edit_changes_the_fingerprint() {
        let dir = temp_dir("untracked");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        std::fs::write(dir.join("scratch.txt"), "one\n").unwrap();
        let before = probe.fingerprint(&dir).await.expect("fingerprint");

        // A size change, so it is visible regardless of mtime resolution.
        std::fs::write(dir.join("scratch.txt"), "one\ntwo\n").unwrap();
        let after = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            before.dirty_hash, after.dirty_hash,
            "content edit inside an already-untracked file must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// F-07 follow-up: a brand-new wholly-untracked directory must be listed
    /// file-by-file (recurse_untracked_dirs), so an edit inside it is visible
    /// rather than collapsed to the directory's own unchanged mtime.
    #[tokio::test]
    async fn untracked_file_inside_a_new_directory_content_edit_changes_the_fingerprint() {
        let dir = temp_dir("untracked_dir");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        std::fs::create_dir_all(dir.join("newdir")).unwrap();
        std::fs::write(dir.join("newdir/scratch.txt"), "one\n").unwrap();
        let before = probe.fingerprint(&dir).await.expect("fingerprint");

        std::fs::write(dir.join("newdir/scratch.txt"), "one\ntwo\n").unwrap();
        let after = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            before.dirty_hash, after.dirty_hash,
            "content edit inside a file in a brand-new untracked directory must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An untracked dangling symlink retargeted to a different, still-missing
    /// path must change the fingerprint: symlink_metadata (not metadata)
    /// stats the link itself, and recreating it yields a fresh mtime.
    #[cfg(unix)]
    #[tokio::test]
    async fn untracked_dangling_symlink_retarget_changes_the_fingerprint() {
        let dir = temp_dir("dangling_symlink");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        std::os::unix::fs::symlink("missing-a", dir.join("link")).unwrap();
        let before = probe.fingerprint(&dir).await.expect("fingerprint");

        std::fs::remove_file(dir.join("link")).unwrap();
        std::os::unix::fs::symlink("missing-b", dir.join("link")).unwrap();
        let after = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            before.dirty_hash, after.dirty_hash,
            "retargeting a dangling untracked symlink must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tracked symlink retargeted to a different existing file must change
    /// the fingerprint: the blob content is the link's target-path text, not
    /// the pointee's bytes, so `Oid::hash_file` (which follows the link) must
    /// not be used for it.
    #[cfg(unix)]
    #[tokio::test]
    async fn tracked_symlink_retarget_changes_the_fingerprint() {
        let dir = temp_dir("tracked_symlink");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("target-a"), "same length\n").unwrap();
        std::fs::write(dir.join("target-b"), "same length\n").unwrap();
        std::os::unix::fs::symlink("target-a", dir.join("link")).unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        let clean = probe.fingerprint(&dir).await.expect("fingerprint");

        std::fs::remove_file(dir.join("link")).unwrap();
        std::os::unix::fs::symlink("target-b", dir.join("link")).unwrap();
        let retargeted = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            clean.dirty_hash, retargeted.dirty_hash,
            "retargeting a tracked symlink to a different (same-content) file must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Toggling a tracked file's executable bit with no content change must
    /// still change the fingerprint (git tracks the mode as part of the
    /// tree entry, independent of blob content).
    #[cfg(unix)]
    #[tokio::test]
    async fn tracked_file_chmod_exec_changes_the_fingerprint() {
        use std::os::unix::fs::PermissionsExt;

        let dir = temp_dir("chmod_exec");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("script.sh"), "echo hi\n").unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        let clean = probe.fingerprint(&dir).await.expect("fingerprint");

        let path = dir.join("script.sh");
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(perms.mode() | 0o100);
        std::fs::set_permissions(&path, perms).unwrap();
        let chmodded = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            clean.dirty_hash, chmodded.dirty_hash,
            "chmod +x with no content change must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// New regression (F-07 + workdir-vs-repo_root): all status paths must be
    /// joined against workdir(), never repo_root. Probe a SUBDIRECTORY of the
    /// repo while an untracked file lives at the git TOP LEVEL; its content
    /// edit must still be visible.
    #[tokio::test]
    async fn untracked_content_edit_visible_when_repo_root_is_a_subdirectory() {
        let dir = temp_dir("subdir_workdir");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        write_commit(&repo, "c1");

        std::fs::create_dir_all(dir.join("subdir")).unwrap();
        // Untracked file at the repo TOP LEVEL, deliberately outside subdir.
        std::fs::write(dir.join("scratch.txt"), "one\n").unwrap();

        let probe = GitStateProbe::new(30);
        let sub = dir.join("subdir");
        let before = probe.fingerprint(&sub).await.expect("fingerprint");

        std::fs::write(dir.join("scratch.txt"), "one\ntwo\n").unwrap();
        let after = probe.fingerprint(&sub).await.expect("fingerprint");
        assert_ne!(
            before.dirty_hash, after.dirty_hash,
            "an untracked-content edit at the git top level must be visible even when repo_root is a subdirectory"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// F-08: a submodule (gitlink) shows the same `WT_MODIFIED` status bits
    /// both when a file nested inside it is edited and when it is checked
    /// out to a different commit — the fingerprint must still change for
    /// both transitions, not just stat the submodule's own top-level
    /// directory (whose mtime is stable across either).
    #[tokio::test]
    async fn submodule_dirty_content_and_head_change_the_fingerprint() {
        let outer_dir = temp_dir("submodule_outer");
        let inner_dir = outer_dir.join("sub");

        let inner_repo = init_repo(&inner_dir);
        std::fs::write(inner_dir.join("f.txt"), "one\n").unwrap();
        write_commit(&inner_repo, "inner c1");
        let inner_c1 = inner_repo.head().unwrap().peel_to_commit().unwrap().id();
        std::fs::write(inner_dir.join("f.txt"), "two\n").unwrap();
        write_commit(&inner_repo, "inner c2");
        let inner_c2 = inner_repo.head().unwrap().peel_to_commit().unwrap().id();

        let outer_repo = init_repo(&outer_dir);
        std::fs::write(outer_dir.join("top.txt"), "top\n").unwrap();
        // Manually stage a gitlink entry for "sub" (mode 160000) pointing at
        // the inner repo's current HEAD — the same tree shape `git submodule
        // add` produces, without needing a network/file transport.
        {
            let mut index = outer_repo.index().unwrap();
            index
                .add_all(["top.txt"].iter(), IndexAddOption::DEFAULT, None)
                .unwrap();
            index
                .add(&git2::IndexEntry {
                    ctime: git2::IndexTime::new(0, 0),
                    mtime: git2::IndexTime::new(0, 0),
                    dev: 0,
                    ino: 0,
                    mode: 0o160000,
                    uid: 0,
                    gid: 0,
                    file_size: 0,
                    id: inner_c2,
                    flags: 0,
                    flags_extended: 0,
                    path: b"sub".to_vec(),
                })
                .unwrap();
            index.write().unwrap();
        }
        commit_index(&outer_repo, "add submodule");

        let probe = GitStateProbe::new(30);
        let clean = probe.fingerprint(&outer_dir).await.expect("fingerprint");

        // Edit a file INSIDE the submodule's own working tree — no outer
        // commit, so the outer gitlink entry shows the same WT_MODIFIED bits
        // before and after.
        std::fs::write(inner_dir.join("f.txt"), "two\nthree\n").unwrap();
        let dirty_content = probe.fingerprint(&outer_dir).await.expect("fingerprint");
        assert_ne!(
            clean.dirty_hash, dirty_content.dirty_hash,
            "editing a file nested inside a submodule must be visible"
        );

        // Revert the in-submodule edit, then check the submodule's own repo
        // out to a different (still fully clean) commit.
        std::fs::write(inner_dir.join("f.txt"), "two\n").unwrap();
        inner_repo.set_head_detached(inner_c1).unwrap();
        inner_repo
            .checkout_head(Some(git2::build::CheckoutBuilder::new().force()))
            .unwrap();
        let checked_out_c1 = probe.fingerprint(&outer_dir).await.expect("fingerprint");
        assert_ne!(
            clean.dirty_hash, checked_out_c1.dirty_hash,
            "checking a submodule out to a different commit must be visible"
        );

        std::fs::remove_dir_all(&outer_dir).ok();
    }

    /// New regression: a committed deletion must appear in changed_paths.
    /// Deleted deltas carry the path only on old_file() in libgit2, so a
    /// new_file()-only collector would silently drop it.
    #[tokio::test]
    async fn changed_paths_includes_deleted_files() {
        let dir = temp_dir("deleted_paths");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("a.txt"), "a\n").unwrap();
        std::fs::write(dir.join("b.txt"), "b\n").unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        let from = probe.fingerprint(&dir).await.expect("fingerprint");

        std::fs::remove_file(dir.join("b.txt")).unwrap();
        {
            let mut index = repo.index().unwrap();
            index.remove_path(Path::new("b.txt")).unwrap();
            index.write().unwrap();
        }
        commit_index(&repo, "c2");
        let to = probe.fingerprint(&dir).await.expect("fingerprint");

        let changed = probe
            .changed_paths(&dir, &from, &to)
            .await
            .expect("changed paths");
        assert!(
            changed.contains(&PathBuf::from("b.txt")),
            "a committed deletion must appear in changed_paths, got {changed:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// New regression: two same-length, same-tick tracked edits must never
    /// collapse via the hash cache's racy-git guard — the second edit lands
    /// well within `RACY_MARGIN` of the first, so it must always be rehashed.
    #[tokio::test]
    async fn rapid_same_length_tracked_edits_are_never_hidden_by_the_hash_cache() {
        let dir = temp_dir("racy");
        let repo = init_repo(&dir);
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        write_commit(&repo, "c1");

        let probe = GitStateProbe::new(30);
        std::fs::write(dir.join("a.txt"), "AAA\n").unwrap();
        let first = probe.fingerprint(&dir).await.expect("fingerprint");
        // Same length as "AAA\n", written immediately after — well inside
        // RACY_MARGIN, so a naive (len, mtime) cache could wrongly hit.
        std::fs::write(dir.join("a.txt"), "BBB\n").unwrap();
        let second = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            first.dirty_hash, second.dirty_hash,
            "a same-length rapid re-edit must never be hidden by the stat cache"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
