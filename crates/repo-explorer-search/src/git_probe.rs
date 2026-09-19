//! git2-backed `RepoStateProbe`: repository fingerprinting for the result
//! caches. Uses the in-process `git2` (libgit2) bindings — no external `git`
//! binary at runtime or in tests. Lives in this crate alongside the other
//! backend probes; every failure degrades to `None`.

use git2::{ObjectType, Oid, Repository, Status, StatusOptions};
use repo_explorer_core::fingerprint::{RepoFingerprint, RepoStateProbe};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Probes repository state via the in-process `git2` (libgit2) library. Every
/// failure (not a repository, bare repo, unborn HEAD, per-file stat/hash
/// error) degrades to `None` — "no fingerprint" simply disables caching for
/// that call.
pub struct GitStateProbe;

impl GitStateProbe {
    /// `timeout_seconds` is accepted for call-site compatibility but unused:
    /// git2 does local-only filesystem ops with no cancellation point.
    pub fn new(_timeout_seconds: u64) -> Self {
        Self
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

/// Single blocking status walk. Every relative path is joined against
/// `workdir` (never the caller's `repo_root`, which may be a strict
/// subdirectory of the git top level). Tracked-changed entries fold in a
/// cheap content-sensitive blob oid (`Oid::hash_file`, no odb write, works for
/// text AND binary); untracked entries fold in a `(len, mtime)` stat via
/// `symlink_metadata` (F-07) — deliberately NOT a content hash, preserving the
/// large-untracked-tree perf guard, and stating the link itself so a dangling
/// untracked symlink's own target string is what varies. Ignored files stay
/// excluded (default `StatusOptions`), a deliberate carried-forward gap.
/// `parts` is sorted because git2's status entry order is not guaranteed.
fn fingerprint_blocking(repo_root: &Path) -> Option<RepoFingerprint> {
    let repo = Repository::discover(repo_root).ok()?;
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
            let part = match std::fs::symlink_metadata(&full) {
                Ok(meta) => format!("{path}:U:{}:{:?}", meta.len(), meta.modified().ok()),
                Err(_) => format!("{path}:U:missing"),
            };
            parts.push(part);
        } else if (s.contains(Status::WT_DELETED) || s.contains(Status::INDEX_DELETED))
            && !full.exists()
        {
            parts.push(format!("{path}:{bits}:gone"));
        } else {
            // Tracked-changed (modified/added/typechange/renamed, staged or
            // unstaged). Content-sensitive blob oid; on error (e.g. a
            // typechange to a directory) fall back to a stat string so the
            // entry still varies.
            let part = match Oid::hash_file(ObjectType::Blob, &full) {
                Ok(oid) => format!("{path}:{bits}:{oid}"),
                Err(_) => match std::fs::symlink_metadata(&full) {
                    Ok(meta) => {
                        format!("{path}:{bits}:{}:{:?}", meta.len(), meta.modified().ok())
                    }
                    Err(_) => format!("{path}:{bits}:missing"),
                },
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
fn changed_paths_blocking(repo_root: &Path, from_sha: &str, to_sha: &str) -> Option<Vec<PathBuf>> {
    let repo = Repository::discover(repo_root).ok()?;
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
}

impl RepoStateProbe for GitStateProbe {
    async fn fingerprint(&self, repo_root: &Path) -> Option<RepoFingerprint> {
        let repo_root = repo_root.to_path_buf();
        tokio::task::spawn_blocking(move || fingerprint_blocking(&repo_root))
            .await
            .ok()
            .flatten()
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
        tokio::task::spawn_blocking(move || changed_paths_blocking(&repo_root, &from_sha, &to_sha))
            .await
            .ok()
            .flatten()
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
}
