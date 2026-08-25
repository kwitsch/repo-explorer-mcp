//! Repository-state fingerprinting for cache keying and invalidation.
//!
//! Core owns only the value type and the trait; the production impl
//! (`GitStateProbe`, subprocess-driven) lives in `repo-explorer-search`, the
//! crate that owns subprocess concerns.

use std::path::{Path, PathBuf};

/// A snapshot of the repository's content state: the checked-out commit plus a
/// digest of the dirty working-tree paths. Two equal fingerprints mean "same
/// content as far as caching is concerned".
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RepoFingerprint {
    pub head_sha: String,
    /// Digest over `git status --porcelain` output (empty tree state included),
    /// so any working-tree change produces a different fingerprint.
    pub dirty_hash: String,
}

/// Probes the repository's version-control state. Every method degrades to
/// `None` instead of failing: "no fingerprint" simply disables caching.
#[allow(async_fn_in_trait)]
pub trait RepoStateProbe {
    /// Current fingerprint, or `None` when the repo state cannot be determined
    /// (not a git repository, git missing, subprocess failure).
    async fn fingerprint(&self, repo_root: &Path) -> Option<RepoFingerprint>;

    /// Paths that differ between two fingerprints, or `None` when unknown.
    /// Must include paths that are dirty in either state.
    async fn changed_paths(
        &self,
        repo_root: &Path,
        from: &RepoFingerprint,
        to: &RepoFingerprint,
    ) -> Option<Vec<PathBuf>>;
}

/// Test double: scripted fingerprint and changed-path answers.
#[cfg(any(test, feature = "test-support"))]
pub mod mock {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Default)]
    pub struct MockRepoStateProbe {
        fingerprint: Arc<Mutex<Option<RepoFingerprint>>>,
        changed_paths: Arc<Mutex<Option<Vec<PathBuf>>>>,
    }

    impl MockRepoStateProbe {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn with_fingerprint(self, f: Option<RepoFingerprint>) -> Self {
            *self.fingerprint.lock().expect("mock fingerprint poisoned") = f;
            self
        }

        pub fn with_changed_paths(self, p: Option<Vec<PathBuf>>) -> Self {
            *self
                .changed_paths
                .lock()
                .expect("mock changed_paths poisoned") = p;
            self
        }

        /// Replace the scripted fingerprint after construction (simulates the
        /// repository changing between two queries).
        pub fn set_fingerprint(&self, f: Option<RepoFingerprint>) {
            *self.fingerprint.lock().expect("mock fingerprint poisoned") = f;
        }
    }

    impl RepoStateProbe for MockRepoStateProbe {
        async fn fingerprint(&self, _repo_root: &Path) -> Option<RepoFingerprint> {
            self.fingerprint
                .lock()
                .expect("mock fingerprint poisoned")
                .clone()
        }

        async fn changed_paths(
            &self,
            _repo_root: &Path,
            _from: &RepoFingerprint,
            _to: &RepoFingerprint,
        ) -> Option<Vec<PathBuf>> {
            self.changed_paths
                .lock()
                .expect("mock changed_paths poisoned")
                .clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_eq_and_hash_usable() {
        let a = RepoFingerprint {
            head_sha: "abc".to_string(),
            dirty_hash: "d0".to_string(),
        };
        let b = RepoFingerprint {
            head_sha: "abc".to_string(),
            dirty_hash: "d1".to_string(),
        };
        assert_eq!(a, a.clone());
        assert_ne!(a, b, "a dirty-state change must change the fingerprint");
        let mut set = std::collections::HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&a));
        assert!(!set.contains(&b));
    }

    #[tokio::test]
    async fn mock_probe_scripts_answers() {
        use mock::MockRepoStateProbe;
        let f = RepoFingerprint {
            head_sha: "abc".to_string(),
            dirty_hash: "d0".to_string(),
        };
        let probe = MockRepoStateProbe::new()
            .with_fingerprint(Some(f.clone()))
            .with_changed_paths(Some(vec![PathBuf::from("src/lib.rs")]));
        assert_eq!(probe.fingerprint(Path::new("/r")).await, Some(f.clone()));
        assert_eq!(
            probe.changed_paths(Path::new("/r"), &f, &f).await,
            Some(vec![PathBuf::from("src/lib.rs")])
        );
        probe.set_fingerprint(None);
        assert_eq!(probe.fingerprint(Path::new("/r")).await, None);
    }
}
