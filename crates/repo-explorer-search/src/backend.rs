//! `CliSearchBackend`: resolves the search binary once at construction, then
//! wires resolver -> process -> parser on each `search` call. rtk is invoked as
//! `rtk rg -H -n ...` (raw rg passthrough, one fixed line grammar); rg as
//! `rg --json -H ...`. `-H` guarantees filenames so the parser grammar is fixed.

use crate::parser::{parse_rg_json, parse_rtk};
use crate::process::{SpawnSpec, run};
use crate::resolver::{ResolvedBackend, Tool, resolve_backend};
use repo_explorer_core::config::SearchConfig;
use repo_explorer_core::domain::ExplorationFinding;
use repo_explorer_core::search::{SearchBackend, SearchError, SearchOptions};
use std::path::Path;
use std::time::Duration;

pub struct CliSearchBackend {
    backend: Option<ResolvedBackend>,
    timeout_seconds: u64,
}

impl CliSearchBackend {
    /// Resolve the search binary from config + PATH once. A backend that
    /// resolves to nothing is still constructible; `search` returns
    /// `BackendNotFound` when invoked.
    pub fn new(config: &SearchConfig) -> Self {
        let backend = resolve_backend(
            config.rtk_path.as_deref(),
            config.ripgrep_path.as_deref(),
            config.prefer_rtk,
            || which::which("rtk").ok(),
            || which::which("rg").ok(),
        );
        Self {
            backend,
            timeout_seconds: config.timeout_seconds,
        }
    }
}

/// Append the flags common to both tools' native rg surface.
fn push_flags(args: &mut Vec<String>, options: &SearchOptions) {
    if options.case_sensitive {
        args.push("-s".to_string());
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
        let resolved = self.backend.as_ref().ok_or_else(|| {
            SearchError::BackendNotFound("neither rtk nor ripgrep could be resolved".to_string())
        })?;

        let target = scope
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());

        // Only the leading flags and the reported backend name differ per tool;
        // everything after them is identical, so it is built exactly once.
        let (backend_name, leading) = match resolved.tool {
            Tool::Rtk => ("rtk", ["rg", "-H", "-n"].as_slice()),
            Tool::Ripgrep => ("ripgrep", ["--json", "-H"].as_slice()),
        };
        let mut args: Vec<String> = leading.iter().map(|s| s.to_string()).collect();
        push_flags(&mut args, options);
        // `--` ends option parsing so a pattern/target starting with `-` (e.g.
        // `-\d+` or a leading-dash file name) is never misread as a flag.
        args.push("--".to_string());
        args.push(pattern.to_string());
        args.push(target);

        let spec = SpawnSpec {
            backend: backend_name,
            program: resolved.path.clone(),
            args,
            cwd: repo_root.to_path_buf(),
            timeout: Duration::from_secs(self.timeout_seconds),
        };

        let stdout = run(&spec).await?;

        let mut findings = match resolved.tool {
            Tool::Rtk => parse_rtk(&stdout),
            Tool::Ripgrep => parse_rg_json(&stdout)?,
        };
        if let Some(max) = options.max_results {
            findings.truncate(max as usize);
        }
        Ok(findings)
    }
}
