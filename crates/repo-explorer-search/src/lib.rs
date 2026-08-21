//! CLI-driven text search backend (rtk / ripgrep) implementing
//! `repo_explorer_core::search::SearchBackend`.
mod backend;
mod parser;
mod process;
mod resolver;

pub use backend::CliSearchBackend;
