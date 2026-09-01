//! `CliSearchBackend`: resolves the `rtk` binary once at construction, then
//! wires resolver -> process -> parser on each `search` call. rtk is invoked as
//! `rtk rg -H -n ...` (raw rg passthrough, one fixed line grammar); `-H`
//! guarantees filenames so the parser grammar is fixed. rtk is mandatory: an
//! unresolved backend is still constructible (`new` is infallible) and `search`
//! returns `BackendNotFound`, while the serve-time gate in `main.rs` (see
//! `rtk_available`) refuses to start.

use crate::parser::parse_rtk;
use crate::process::{SpawnSpec, run};
use crate::resolver::resolve_rtk;
use repo_explorer_core::config::SearchConfig;
use repo_explorer_core::domain::ExplorationFinding;
use repo_explorer_core::search::{SearchBackend, SearchError, SearchOptions};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub struct CliSearchBackend {
    rtk: Option<PathBuf>,
    timeout_seconds: u64,
}

impl CliSearchBackend {
    /// Resolve the `rtk` binary from config + PATH once. Still infallible: an
    /// unresolved backend is constructible, and `search` fails fast via
    /// `BackendNotFound` while the serve-time gate in `main.rs` refuses to
    /// start (see `rtk_available`).
    pub fn new(config: &SearchConfig) -> Self {
        Self {
            rtk: resolve_rtk(config.rtk_path.as_deref(), || which::which("rtk").ok()),
            timeout_seconds: config.timeout_seconds,
        }
    }

    /// True when the `rtk` binary resolved. Consumed by the serve-time
    /// fail-fast in `main.rs`, since rtk is a hard requirement of search.
    pub fn rtk_available(&self) -> bool {
        self.rtk.is_some()
    }
}

/// Append the flags common to rtk's native rg surface.
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
            .rtk
            .as_ref()
            .ok_or_else(|| SearchError::BackendNotFound("rtk could not be resolved".to_string()))?;

        let target = scope
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());

        // rtk's raw rg passthrough: `rtk rg -H -n <flags> -- <pattern> <target>`.
        let mut args: Vec<String> = ["rg", "-H", "-n"].iter().map(|s| s.to_string()).collect();
        push_flags(&mut args, options);
        // `--` ends option parsing so a pattern/target starting with `-` (e.g.
        // `-\d+` or a leading-dash file name) is never misread as a flag.
        args.push("--".to_string());
        args.push(pattern.to_string());
        args.push(target);

        let spec = SpawnSpec {
            backend: "rtk",
            program: program.clone(),
            args,
            cwd: repo_root.to_path_buf(),
            timeout: Duration::from_secs(self.timeout_seconds),
        };

        let stdout = run(&spec).await?;

        let mut findings = parse_rtk(&stdout);
        if let Some(max) = options.max_results {
            findings.truncate(max as usize);
        }
        Ok(findings)
    }
}
