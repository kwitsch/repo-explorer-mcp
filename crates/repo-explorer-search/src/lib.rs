//! CLI-driven text search backend (ripgrep) implementing
//! `repo_explorer_core::search::SearchBackend`, plus the git2-backed
//! `RepoStateProbe` (`GitStateProbe`). The ripgrep path is subprocess-driven
//! (this crate's dependency domain); `GitStateProbe` uses the in-process
//! `git2` (libgit2) bindings — no external `git` binary.
mod backend;
mod git_probe;
mod process;

pub use backend::NativeSearchBackend;
pub use git_probe::GitStateProbe;
