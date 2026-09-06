//! Git-backed `RepoStateProbe`: repository fingerprinting for the result
//! caches. Lives in this crate because subprocess concerns belong here (the
//! same `process::run` used for rtk/rg drives `git`).

use crate::process::{SpawnSpec, run};
use repo_explorer_core::fingerprint::{RepoFingerprint, RepoStateProbe};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Probes repository state via the `git` CLI. Every failure (no git binary,
/// not a repository, timeout) degrades to `None` — "no fingerprint" simply
/// disables caching for that call.
pub struct GitStateProbe {
    timeout: Duration,
}

impl GitStateProbe {
    /// `timeout_seconds = 0` means "no timeout", matching `SearchConfig`.
    pub fn new(timeout_seconds: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_seconds),
        }
    }

    async fn git(&self, repo_root: &Path, args: &[&str]) -> Option<String> {
        let spec = SpawnSpec {
            backend: "git",
            program: PathBuf::from("git"),
            args: args.iter().map(|s| s.to_string()).collect(),
            cwd: repo_root.to_path_buf(),
            timeout: self.timeout,
        };
        run(&spec).await.ok()
    }
}

/// Fold each untracked file's `(size, mtime)` into a stable string, so an
/// edit to an already-untracked file's bytes changes the dirty fingerprint
/// even though neither `git status --porcelain` nor `git diff HEAD` would
/// notice (F-07). Cheap stat, not a full read — avoids the perf risk of
/// hashing content across a large untracked directory that isn't gitignored.
/// Untracked paths are read straight out of the already-fetched `status`
/// text (lines prefixed `"?? "`), so no extra `git` subprocess call is
/// needed. `parts` is sorted so this is stable regardless of `status`'s own
/// line order.
///
/// ponytail: git's C-style quoting of exotic filenames in `--porcelain`
/// output isn't unescaped here, so such a path won't resolve via
/// `repo_root.join(rel)` and falls into the `Err` "missing" arm — no worse
/// than the total blind spot this replaces; unquote it if it ever bites.
///
/// Stats the path itself (`symlink_metadata`), not through a symlink: a
/// symlink's *target* content is already covered separately, by the
/// target's own `?? `/tracked status entry, so following the link here would
/// only mean a dangling symlink (a real, common untracked-scratch case)
/// always reports "missing" regardless of what it points at or how that's
/// repointed — collapsing every such retarget into the same fingerprint.
///
/// ponytail: `(size, mtime)` is a heuristic, not a content hash — a same-byte-
/// length edit landing inside one mtime tick (coarse on some filesystems) is
/// invisible; hash real content instead if that ever bites. Also: a
/// gitignored file never appears in `status` at all (git omits it unless
/// `--ignored` is passed), so its content stays outside this fingerprint
/// entirely even though it's fully readable via `read_file` — a deliberately
/// separate, unaddressed gap (whether an ignored file's content should count
/// as "dirty" for caching is a design question, not a one-line fix).
fn untracked_fingerprint(repo_root: &Path, status: &str) -> String {
    let mut parts: Vec<String> = status
        .lines()
        .filter_map(|l| l.strip_prefix("?? "))
        .map(|rel| match std::fs::symlink_metadata(repo_root.join(rel)) {
            Ok(meta) => format!("{rel}:{}:{:?}", meta.len(), meta.modified().ok()),
            Err(_) => format!("{rel}:missing"),
        })
        .collect();
    parts.sort();
    parts.join("\n")
}

fn sha256_hex(parts: &[&str]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part.as_bytes());
        hasher.update([0u8]);
    }
    hex::encode(hasher.finalize())
}

impl RepoStateProbe for GitStateProbe {
    async fn fingerprint(&self, repo_root: &Path) -> Option<RepoFingerprint> {
        // The dirty digest covers the porcelain status (path set incl.
        // untracked files) plus the full `git diff HEAD` patch text, whose
        // `index` lines pin the base blobs — so an equal digest means equal
        // dirty *content* for tracked files, not merely an equal path set.
        // `untracked_fingerprint` covers the remaining blind spot: content
        // edits inside an already-untracked file (F-07) — `diff HEAD` never
        // covers untracked content, and `status` only ever shows the same
        // one-line "??" entry regardless of what changed inside.
        // `--untracked-files=all` (rather than git's directory-collapsing
        // default) so a brand-new, not-yet-`git add`ed directory is listed
        // file-by-file — otherwise git reports it as one `?? dir/` line, and
        // `untracked_fingerprint` below would stat the directory itself
        // (whose own mtime doesn't change when a file inside it is edited),
        // silently missing every content edit inside a new scratch/feature
        // subdirectory. ponytail: this makes `git status` walk into large
        // untracked-and-not-gitignored trees instead of stopping at the
        // directory boundary; acceptable since each call is already
        // timeout-bounded (`self.timeout`), same tradeoff already accepted
        // for `--sort path` in `repo-explorer-search::backend`.
        let (head, status, diff) = tokio::join!(
            self.git(repo_root, &["rev-parse", "HEAD"]),
            self.git(
                repo_root,
                &["status", "--porcelain", "--untracked-files=all"]
            ),
            self.git(repo_root, &["diff", "HEAD"])
        );
        let head_sha = head?.trim().to_string();
        if head_sha.is_empty() {
            return None;
        }
        let status = status?;
        let diff = diff?;
        // Off the async runtime thread: `untracked_fingerprint` does one
        // blocking `std::fs::metadata` per untracked path, and this runs on
        // every cache-consulting `explore_repository` call (`AgentLoop::run`
        // Stage 0), same reasoning as `dispatch::read_file_canonical`'s own
        // `spawn_blocking` wrap.
        let untracked = {
            let repo_root = repo_root.to_path_buf();
            let status = status.clone();
            tokio::task::spawn_blocking(move || untracked_fingerprint(&repo_root, &status))
                .await
                .unwrap_or_default()
        };
        Some(RepoFingerprint {
            head_sha,
            dirty_hash: sha256_hex(&[&status, &diff, &untracked]),
        })
    }

    async fn changed_paths(
        &self,
        repo_root: &Path,
        from: &RepoFingerprint,
        to: &RepoFingerprint,
    ) -> Option<Vec<PathBuf>> {
        // A differing dirty state cannot be enumerated after the fact (the
        // `from` side's dirty paths are gone) — report "unknown" and let the
        // caller invalidate.
        if from.dirty_hash != to.dirty_hash {
            return None;
        }
        if from.head_sha == to.head_sha {
            return Some(Vec::new());
        }
        let out = self
            .git(
                repo_root,
                &["diff", "--name-only", &from.head_sha, &to.head_sha],
            )
            .await?;
        Some(
            out.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(PathBuf::from)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git_available() -> bool {
        which::which("git").is_ok()
    }

    async fn sh_git(dir: &Path, args: &[&str]) {
        let probe = GitStateProbe::new(30);
        probe
            .git(dir, args)
            .await
            .unwrap_or_else(|| panic!("git {args:?} failed in {}", dir.display()));
    }

    async fn init_repo(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        sh_git(dir, &["init", "-q"]).await;
        sh_git(dir, &["config", "user.email", "t@example.com"]).await;
        sh_git(dir, &["config", "user.name", "t"]).await;
    }

    async fn commit_all(dir: &Path, msg: &str) {
        sh_git(dir, &["add", "-A"]).await;
        sh_git(dir, &["commit", "-q", "-m", msg]).await;
    }

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("git_probe_{tag}_{}", std::process::id()))
    }

    #[tokio::test]
    async fn non_repo_yields_no_fingerprint() {
        if !git_available() {
            eprintln!("skipping: git not on PATH");
            return;
        }
        let dir = temp_dir("nonrepo");
        std::fs::create_dir_all(&dir).unwrap();
        // Guard against the temp dir living under some parent repository:
        // `rev-parse` succeeding there would still be a real answer, so only
        // assert when git itself reports failure.
        let probe = GitStateProbe::new(30);
        if probe.git(&dir, &["rev-parse", "HEAD"]).await.is_none() {
            assert_eq!(probe.fingerprint(&dir).await, None);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn dirty_change_and_commit_change_the_fingerprint() {
        if !git_available() {
            eprintln!("skipping: git not on PATH");
            return;
        }
        let dir = temp_dir("fp");
        init_repo(&dir).await;
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        commit_all(&dir, "c1").await;

        let probe = GitStateProbe::new(30);
        let clean = probe.fingerprint(&dir).await.expect("fingerprint");

        std::fs::write(dir.join("a.txt"), "two\n").unwrap();
        let dirty = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_eq!(clean.head_sha, dirty.head_sha);
        assert_ne!(clean.dirty_hash, dirty.dirty_hash);

        commit_all(&dir, "c2").await;
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
    /// fingerprint even though neither `status --porcelain` nor `diff HEAD`
    /// notices the content change (only the file's continued presence).
    #[tokio::test]
    async fn untracked_file_content_edit_changes_the_fingerprint() {
        if !git_available() {
            eprintln!("skipping: git not on PATH");
            return;
        }
        let dir = temp_dir("untracked");
        init_repo(&dir).await;
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        commit_all(&dir, "c1").await;

        let probe = GitStateProbe::new(30);
        std::fs::write(dir.join("scratch.txt"), "one\n").unwrap();
        let before = probe.fingerprint(&dir).await.expect("fingerprint");

        // Editing the untracked file's bytes (no `git add`) must change it —
        // this is a *size* change, so it's visible regardless of mtime
        // resolution on the test filesystem.
        std::fs::write(dir.join("scratch.txt"), "one\ntwo\n").unwrap();
        let after = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            before.dirty_hash, after.dirty_hash,
            "content edit inside an already-untracked file must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// F-07 follow-up: git's default `status --porcelain` collapses a
    /// brand-new, wholly-untracked directory into one `?? dir/` line, which
    /// would make `untracked_fingerprint` stat the directory (whose own mtime
    /// doesn't change) instead of the file inside it — `--untracked-files=all`
    /// must keep this case visible too, not just a loose untracked file.
    #[tokio::test]
    async fn untracked_file_inside_a_new_directory_content_edit_changes_the_fingerprint() {
        if !git_available() {
            eprintln!("skipping: git not on PATH");
            return;
        }
        let dir = temp_dir("untracked_dir");
        init_repo(&dir).await;
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        commit_all(&dir, "c1").await;

        let probe = GitStateProbe::new(30);
        std::fs::create_dir_all(dir.join("newdir")).unwrap();
        std::fs::write(dir.join("newdir/scratch.txt"), "one\n").unwrap();
        let before = probe.fingerprint(&dir).await.expect("fingerprint");

        // Same length as the F-07 test above (a size change), so this isn't
        // relying on mtime resolution either — only on `newdir/` being
        // listed file-by-file instead of collapsed to one line.
        std::fs::write(dir.join("newdir/scratch.txt"), "one\ntwo\n").unwrap();
        let after = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            before.dirty_hash, after.dirty_hash,
            "content edit inside a file in a brand-new untracked directory must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// An untracked dangling symlink retargeted to a *different*, still-
    /// missing path must change the fingerprint: following the link
    /// (`std::fs::metadata`) would make both targets resolve to the same
    /// `Err(NotFound)` and collapse to an identical `"<name>:missing"`
    /// string regardless of what the link actually points at.
    #[cfg(unix)]
    #[tokio::test]
    async fn untracked_dangling_symlink_retarget_changes_the_fingerprint() {
        if !git_available() {
            eprintln!("skipping: git not on PATH");
            return;
        }
        let dir = temp_dir("dangling_symlink");
        init_repo(&dir).await;
        std::fs::write(dir.join("tracked.txt"), "one\n").unwrap();
        commit_all(&dir, "c1").await;

        let probe = GitStateProbe::new(30);
        std::os::unix::fs::symlink("missing-a", dir.join("link")).unwrap();
        let before = probe.fingerprint(&dir).await.expect("fingerprint");

        // Both targets are still missing — only the link's own target string
        // changed.
        std::fs::remove_file(dir.join("link")).unwrap();
        std::os::unix::fs::symlink("missing-b", dir.join("link")).unwrap();
        let after = probe.fingerprint(&dir).await.expect("fingerprint");
        assert_ne!(
            before.dirty_hash, after.dirty_hash,
            "retargeting a dangling untracked symlink must be visible"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
