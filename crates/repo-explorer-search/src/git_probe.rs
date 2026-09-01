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
            extra_path_dir: None,
        };
        run(&spec).await.ok()
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

impl RepoStateProbe for GitStateProbe {
    async fn fingerprint(&self, repo_root: &Path) -> Option<RepoFingerprint> {
        // The dirty digest covers the porcelain status (path set incl.
        // untracked files) plus the full `git diff HEAD` patch text, whose
        // `index` lines pin the base blobs — so an equal digest means equal
        // dirty *content* for tracked files, not merely an equal path set.
        // Blind spot: content edits inside an already-untracked file.
        let (head, status, diff) = tokio::join!(
            self.git(repo_root, &["rev-parse", "HEAD"]),
            self.git(repo_root, &["status", "--porcelain"]),
            self.git(repo_root, &["diff", "HEAD"])
        );
        let head_sha = head?.trim().to_string();
        if head_sha.is_empty() {
            return None;
        }
        let status = status?;
        let diff = diff?;
        Some(RepoFingerprint {
            head_sha,
            dirty_hash: sha256_hex(&[&status, &diff]),
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
}
