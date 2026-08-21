//! `MemoryBackend` implementation: orchestrates `ensure_fresh_index` and maps
//! each of the six query methods onto an upstream `codebase-memory-mcp` tool,
//! decoding every response into the uniform `ExplorationResult`.

use crate::client::{MemoryClient, decode_result, project_name};
use crate::freshness::{FreshnessDecision, IndexProbe, decide_freshness};
use repo_explorer_core::config::CodebaseMemoryConfig;
use repo_explorer_core::domain::{
    ExplorationFinding, ExplorationQuery, ExplorationResult, FileLocation,
};
use repo_explorer_core::memory::{
    GraphQuery, IndexStatus, MemoryBackend, MemoryError, SnippetTarget,
};
use serde_json::{Map, Value};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A `MemoryBackend` backed by a live `rmcp` client. Holds the staleness
/// threshold from config so `ensure_fresh_index` needs no extra arguments.
#[derive(Debug)]
pub struct MemoryClientBackend {
    client: MemoryClient,
    staleness: Duration,
}

impl MemoryClientBackend {
    /// Connect to the configured `codebase-memory-mcp` (stdio only).
    pub async fn connect(config: &CodebaseMemoryConfig) -> Result<Self, MemoryError> {
        let client = MemoryClient::connect(config).await?;
        Ok(Self {
            client,
            staleness: Duration::from_secs(config.staleness_seconds),
        })
    }

    /// Best-effort graceful shutdown.
    pub async fn close(&mut self) {
        self.client.close().await;
    }

    /// Probe `index_status` for the project; a tool error meaning "not indexed"
    /// is reported as `exists = false` rather than an `Err`.
    async fn probe_status(&self, project: &str) -> Result<IndexProbe, MemoryError> {
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project.to_string()));
        match self.client.call("index_status", args).await {
            Ok(result) => {
                let json = decode_result("index_status", &result)?;
                // An unrecognized/empty response must NOT be optimistically
                // treated as "already indexed" — default to `false` so an
                // unknown shape forces a (safe) reindex instead of skipping one.
                let exists = json
                    .get("indexed")
                    .and_then(Value::as_bool)
                    .or_else(|| json.get("exists").and_then(Value::as_bool))
                    .unwrap_or(false);
                let last_indexed_at = json
                    .get("last_indexed_at")
                    .and_then(Value::as_i64)
                    .and_then(|secs| u64::try_from(secs).ok())
                    .map(|secs| UNIX_EPOCH + Duration::from_secs(secs));
                Ok(IndexProbe {
                    exists,
                    last_indexed_at,
                    changed_files: 0,
                })
            }
            // Only a tool error that explicitly indicates the project is
            // unknown/not-yet-indexed is downgraded to "not indexed"; any other
            // tool failure (permission error, malformed input, internal fault)
            // is surfaced to the caller instead of being silently reinterpreted.
            Err(MemoryError::ToolFailed { message, .. }) if is_not_indexed_error(&message) => {
                Ok(IndexProbe {
                    exists: false,
                    last_indexed_at: None,
                    changed_files: 0,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Fill `changed_files` from `detect_changes` for an existing project.
    async fn probe_changes(&self, project: &str) -> Result<usize, MemoryError> {
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project.to_string()));
        match self.client.call("detect_changes", args).await {
            Ok(result) => {
                let json = decode_result("detect_changes", &result)?;
                let changed = json
                    .get("changed_files")
                    .and_then(Value::as_array)
                    .map(|a| a.len())
                    .or_else(|| {
                        json.get("changed_count")
                            .and_then(Value::as_u64)
                            .map(|n| n as usize)
                    })
                    .unwrap_or(0);
                Ok(changed)
            }
            // Per the `MemoryBackend::ensure_fresh_index` contract, a soft tool
            // failure must not abort exploration outright. We cannot confirm
            // freshness, so report an (arbitrary, nonzero) unknown change count
            // that forces `decide_freshness` toward a reindex attempt rather
            // than returning `Err` for a recoverable hiccup.
            Err(MemoryError::ToolFailed { .. }) => Ok(1),
            Err(e) => Err(e),
        }
    }

    /// Run `index_repository` against the absolute repo root. Soft tool failure
    /// -> `IndexingFailed`; transport failure -> `Err`.
    async fn run_index(&self, repo_root: &Path) -> Result<IndexStatus, MemoryError> {
        let abs = std::fs::canonicalize(repo_root).unwrap_or_else(|_| repo_root.to_path_buf());
        let mut args = Map::new();
        args.insert(
            "path".to_string(),
            Value::String(abs.to_string_lossy().into_owned()),
        );
        match self.client.call("index_repository", args).await {
            Ok(_) => Ok(IndexStatus::Reindexed),
            Err(MemoryError::ToolFailed { message, .. }) => {
                Ok(IndexStatus::IndexingFailed { reason: message })
            }
            Err(e) => Err(e),
        }
    }
}

/// Does a tool-failure message indicate "this project is not indexed yet"
/// (as opposed to some other recoverable-or-not failure)? Matches the
/// observed `codebase-memory-mcp` phrasing ("project not found or not
/// indexed") case-insensitively so close variants still match.
fn is_not_indexed_error(message: &str) -> bool {
    let lower = message.to_lowercase();
    lower.contains("not indexed") || lower.contains("not found")
}

/// Build a `FileLocation` from a JSON row's `file`/`line_start`/`line_end`,
/// tolerating missing line fields (defaulting to 0).
fn location_from(json: &Value) -> Option<FileLocation> {
    let file = json
        .get("file")
        .or_else(|| json.get("path"))
        .and_then(Value::as_str)?;
    // Saturate rather than `as`-truncate: a line number beyond `u32::MAX` (or a
    // malformed huge value) must not silently wrap around to a small, wrong one.
    let line_start = json
        .get("line_start")
        .and_then(Value::as_u64)
        .map(|n| n.min(u32::MAX as u64) as u32)
        .unwrap_or(0);
    let line_end = json
        .get("line_end")
        .and_then(Value::as_u64)
        .map(|n| n.min(u32::MAX as u64) as u32)
        .unwrap_or(line_start);
    Some(FileLocation {
        path: std::path::PathBuf::from(file),
        line_start,
        line_end,
    })
}

/// Turn a JSON array of hit rows into findings, plus a compact summary string.
fn findings_and_summary(tool: &'static str, json: &Value) -> ExplorationResult {
    let rows = json
        .get("results")
        .or_else(|| json.get("rows"))
        .or_else(|| json.get("hits"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut findings = Vec::new();
    for row in &rows {
        if let Some(location) = location_from(row) {
            let snippet = row
                .get("snippet")
                .or_else(|| row.get("text"))
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            findings.push(ExplorationFinding {
                location,
                snippet,
                note: None,
            });
        }
    }
    let summary = format!(
        "{tool}: {} row(s), {} locatable finding(s)",
        rows.len(),
        findings.len()
    );
    ExplorationResult { findings, summary }
}

impl MemoryBackend for MemoryClientBackend {
    async fn ensure_fresh_index(&self, repo_root: &Path) -> Result<IndexStatus, MemoryError> {
        let project = project_name(repo_root)?;
        let mut probe = self.probe_status(&project).await?;
        if probe.exists {
            probe.changed_files = self.probe_changes(&project).await?;
        }
        match decide_freshness(&probe, self.staleness, SystemTime::now()) {
            FreshnessDecision::UpToDate => Ok(IndexStatus::UpToDate),
            FreshnessDecision::Reindex => self.run_index(repo_root).await,
        }
    }

    async fn search_code(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
    ) -> Result<ExplorationResult, MemoryError> {
        let project = project_name(repo_root)?;
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project));
        args.insert("pattern".to_string(), Value::String(query.text.clone()));
        if let Some(scope) = &query.scope_hint {
            args.insert(
                "file_pattern".to_string(),
                Value::String(scope.to_string_lossy().into_owned()),
            );
        }
        if let Some(limit) = query.max_results {
            args.insert("limit".to_string(), Value::Number(limit.into()));
        }
        let result = self.client.call("search_code", args).await?;
        let json = decode_result("search_code", &result)?;
        Ok(findings_and_summary("search_code", &json))
    }

    async fn search_graph(
        &self,
        repo_root: &Path,
        query: &GraphQuery,
    ) -> Result<ExplorationResult, MemoryError> {
        let project = project_name(repo_root)?;
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project));
        args.insert("format".to_string(), Value::String("json".to_string()));
        if let Some(v) = &query.name_pattern {
            args.insert("name_pattern".to_string(), Value::String(v.clone()));
        }
        if let Some(v) = &query.file_pattern {
            args.insert("file_pattern".to_string(), Value::String(v.clone()));
        }
        if let Some(v) = &query.label {
            args.insert("label".to_string(), Value::String(v.clone()));
        }
        if let Some(limit) = query.max_results {
            args.insert("limit".to_string(), Value::Number(limit.into()));
        }
        let result = self.client.call("search_graph", args).await?;
        let json = decode_result("search_graph", &result)?;
        Ok(findings_and_summary("search_graph", &json))
    }

    async fn query_graph(
        &self,
        repo_root: &Path,
        query: &str,
        max_results: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError> {
        let project = project_name(repo_root)?;
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project));
        args.insert("query".to_string(), Value::String(query.to_string()));
        if let Some(limit) = max_results {
            args.insert("limit".to_string(), Value::Number(limit.into()));
        }
        let result = self.client.call("query_graph", args).await?;
        let json = decode_result("query_graph", &result)?;
        Ok(findings_and_summary("query_graph", &json))
    }

    async fn trace_path(
        &self,
        repo_root: &Path,
        from: &str,
        to: &str,
        max_depth: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError> {
        let project = project_name(repo_root)?;
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project));
        args.insert("from".to_string(), Value::String(from.to_string()));
        args.insert("to".to_string(), Value::String(to.to_string()));
        if let Some(depth) = max_depth {
            args.insert("max_depth".to_string(), Value::Number(depth.into()));
        }
        let result = self.client.call("trace_path", args).await?;
        let json = decode_result("trace_path", &result)?;
        Ok(findings_and_summary("trace_path", &json))
    }

    async fn get_architecture(
        &self,
        repo_root: &Path,
        depth: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError> {
        let project = project_name(repo_root)?;
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project));
        if let Some(d) = depth {
            args.insert("depth".to_string(), Value::Number(d.into()));
        }
        let result = self.client.call("get_architecture", args).await?;
        let json = decode_result("get_architecture", &result)?;
        Ok(findings_and_summary("get_architecture", &json))
    }

    async fn get_code_snippet(
        &self,
        repo_root: &Path,
        target: &SnippetTarget,
    ) -> Result<ExplorationResult, MemoryError> {
        let project = project_name(repo_root)?;
        let mut args = Map::new();
        args.insert("project".to_string(), Value::String(project));
        match target {
            SnippetTarget::QualifiedName(name) => {
                args.insert("qualified_name".to_string(), Value::String(name.clone()));
            }
            SnippetTarget::FileRange {
                file,
                start_line,
                end_line,
            } => {
                args.insert(
                    "file".to_string(),
                    Value::String(file.to_string_lossy().into_owned()),
                );
                if let Some(s) = start_line {
                    args.insert("start_line".to_string(), Value::Number((*s).into()));
                }
                if let Some(e) = end_line {
                    args.insert("end_line".to_string(), Value::Number((*e).into()));
                }
            }
        }
        let result = self.client.call("get_code_snippet", args).await?;
        let json = decode_result("get_code_snippet", &result)?;
        // 0 or 1 finding: reuse the row shape if a location resolves.
        let mut findings = Vec::new();
        if let Some(location) = location_from(&json) {
            let snippet = json
                .get("snippet")
                .or_else(|| json.get("code"))
                .or_else(|| json.get("text"))
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            findings.push(ExplorationFinding {
                location,
                snippet,
                note: None,
            });
        }
        let summary = if findings.is_empty() {
            "get_code_snippet: no snippet resolved".to_string()
        } else {
            "get_code_snippet: 1 snippet".to_string()
        };
        Ok(ExplorationResult { findings, summary })
    }
}
