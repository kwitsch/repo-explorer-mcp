//! `CliSearchBackend`: resolves the `rg` binary once at construction, then
//! wires resolver -> process -> parser on each `search` call. `rg` is invoked
//! directly as `rg -H -n ...` (one fixed line grammar); `-H` guarantees
//! filenames so the parser grammar is fixed. `rg` is the sole, mandatory
//! search backend: an unresolved backend is still constructible (`new` is
//! infallible) and `search` returns `BackendNotFound`, while the serve-time
//! gate in `main.rs` (see `rg_available`) refuses to start.

use crate::parser::parse_rg;
use crate::process::{SpawnSpec, run};
use crate::resolver::resolve_rg;
use repo_explorer_core::config::SearchConfig;
use repo_explorer_core::domain::ExplorationFinding;
use repo_explorer_core::search::{SearchBackend, SearchError, SearchOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct CliSearchBackend {
    rg: Option<PathBuf>,
    timeout_seconds: u64,
}

impl CliSearchBackend {
    /// Resolve the `rg` binary once from config + PATH + the managed fallback.
    /// Precedence: explicit `[search] rg_path` > system `rg` on PATH (`which`) >
    /// the repo-explorer-managed `rg` copy (`managed_rg_path`) when it exists on
    /// disk. Still infallible: an unresolved backend is constructible; `search`
    /// fails fast via `BackendNotFound` and the serve-time gate in `main.rs`
    /// (see `rg_available`) refuses to start.
    ///
    /// Async: only when no explicit path is configured does resolution fall
    /// through to `which::which`, whose PATH-wide stat calls run via
    /// `spawn_blocking` rather than directly on a tokio worker thread (an
    /// unusually long/slow-to-stat PATH must not stall it) — mirroring
    /// `update.rs`'s `which_rg_blocking`.
    pub async fn new(config: &SearchConfig, managed_rg_path: Option<PathBuf>) -> Self {
        let system_rg = if config.rg_path.is_some() {
            None
        } else {
            tokio::task::spawn_blocking(|| which::which("rg").ok())
                .await
                .unwrap_or(None)
        };
        let rg = resolve_rg(config.rg_path.as_deref(), || {
            system_rg
                .clone()
                .or_else(|| managed_rg_path.clone().filter(|p| p.exists()))
        });
        Self {
            rg,
            timeout_seconds: config.timeout_seconds,
        }
    }

    /// True when the `rg` binary resolved. Consumed by the serve-time
    /// fail-fast in `main.rs`, since a resolvable `rg` is required for search.
    pub fn rg_available(&self) -> bool {
        self.rg.is_some()
    }
}

/// Append the flags common to the `rg` invocation.
fn push_flags(args: &mut Vec<String>, options: &SearchOptions) {
    if options.case_sensitive {
        args.push("-s".to_string());
    } else {
        args.push("-S".to_string());
    }
    if let Some(n) = options.context_lines {
        args.push("-C".to_string());
        args.push(n.to_string());
    }
    if let Some(g) = &options.file_glob {
        args.push("-g".to_string());
        args.push(g.clone());
    }
}

impl SearchBackend for CliSearchBackend {
    async fn search(
        &self,
        repo_root: &Path,
        pattern: &str,
        scope: Option<&Path>,
        options: &SearchOptions,
    ) -> Result<Vec<ExplorationFinding>, SearchError> {
        if pattern.is_empty() {
            return Err(SearchError::InvalidInput(
                "empty search pattern".to_string(),
            ));
        }
        let program = self
            .rg
            .as_ref()
            .ok_or_else(|| SearchError::BackendNotFound("rg could not be resolved".to_string()))?;

        let target = scope
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());

        // Direct rg invocation: `rg -H -n <flags> -- <pattern> <target>`.
        let mut args: Vec<String> = ["-H", "-n"].iter().map(|s| s.to_string()).collect();
        push_flags(&mut args, options);
        // `--` ends option parsing so a pattern/target starting with `-` (e.g.
        // `-\d+` or a leading-dash file name) is never misread as a flag.
        args.push("--".to_string());
        args.push(pattern.to_string());
        args.push(target);

        let spec = SpawnSpec {
            backend: "rg",
            program: program.clone(),
            args,
            cwd: repo_root.to_path_buf(),
            timeout: Duration::from_secs(self.timeout_seconds),
        };

        let stdout = run(&spec).await?;

        let mut findings = parse_rg(&stdout);
        if let Some(max) = options.max_results {
            findings.truncate(max as usize);
        }
        Ok(findings)
    }
}
