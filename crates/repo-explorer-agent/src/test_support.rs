//! Shared filesystem fixture for this crate's tests: build a temp repo dir
//! containing whatever files a test needs. Gated behind `test` and the
//! `test-support` feature — the latter is what lets `tests/integration.rs` (a
//! separate test binary) reach it, the same way this crate reaches
//! `repo_explorer_core::*::mock` via that crate's own `test-support` feature
//! (see this crate's `Cargo.toml` dev-dependencies).

use std::path::PathBuf;

/// Create a unique temp dir (named `{prefix}_{test}_{pid}`, so parallel tests
/// never collide) and write each `(relative_path, contents)` pair into it,
/// creating parent directories as needed. Caller removes the dir when done.
pub fn temp_repo_with(prefix: &str, test: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("{prefix}_{test}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for &(rel, contents) in files {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, contents).unwrap();
    }
    dir
}
