//! In-process text/filename search backend (`NativeSearchBackend`) built on
//! ripgrep's own `ignore` + `grep` crates, implementing
//! `repo_explorer_core::search::SearchBackend` with no external binary; plus
//! the git-backed `RepoStateProbe` (`GitStateProbe`), whose subprocess concern
//! is this crate's remaining dependency domain.
mod backend;
mod git_probe;
mod process;

pub use backend::NativeSearchBackend;
pub use git_probe::GitStateProbe;
