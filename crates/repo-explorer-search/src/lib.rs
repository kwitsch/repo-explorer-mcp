//! In-process text/filename search backend (`NativeSearchBackend`) built on
//! ripgrep's own `ignore` + `grep` crates, plus the git2-backed
//! `RepoStateProbe` (`GitStateProbe`) — neither spawns an external binary.
mod backend;
mod git_probe;

pub use backend::NativeSearchBackend;
pub use git_probe::GitStateProbe;
