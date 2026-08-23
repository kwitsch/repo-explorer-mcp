//! CLI-driven text search backend (rtk / ripgrep) implementing
//! `repo_explorer_core::search::SearchBackend`, plus the git-backed
//! `RepoStateProbe` (`GitStateProbe`) — both subprocess-driven, which is this
//! crate's dependency domain.
mod backend;
mod git_probe;
mod parser;
mod process;
mod resolver;

pub use backend::CliSearchBackend;
pub use git_probe::GitStateProbe;
