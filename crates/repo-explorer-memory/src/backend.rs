//! `MemoryBackend` implementation: orchestrates `ensure_fresh_index` and maps
//! each of the six query methods onto an upstream `codebase-memory-mcp` tool,
//! decoding every response into the uniform `ExplorationResult`.

use crate::client::{
    MemoryClient, canonicalize_repo_root, decode_result, project_name, project_name_from_abs,
};
use crate::freshness::{ChangeCount, FreshnessDecision, IndexProbe, decide_freshness};
use repo_explorer_core::config::CodebaseMemoryConfig;
use repo_explorer_core::domain::{
    ExplorationFinding, ExplorationQuery, ExplorationResult, FileLocation, saturate_u32,
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

    /// Shared `base_args -> client.call -> decode_result` sequence used by
    /// both `probe_status` and `probe_changes`; each caller keeps its own
    /// error-downgrading on top of this.
    async fn call_and_decode(
        &self,
        tool: &'static str,
        project: &str,
    ) -> Result<Value, MemoryError> {
        let args = base_args(project.to_string());
        let result = self.client.call(tool, args).await?;
        decode_result(result)
    }

    /// Probe `index_status` for the project, returning `(exists,
    /// last_indexed_at)`; a tool error meaning "not indexed" is reported as
    /// `exists = false` rather than an `Err`. The changed-file count is not
    /// this call's to know, so the full `IndexProbe` is assembled by
    /// `ensure_fresh_index` instead of being returned half-filled here.
    async fn probe_status(&self, project: &str) -> Result<(bool, Option<SystemTime>), MemoryError> {
        match self.call_and_decode("index_status", project).await {
            Ok(json) => {
                // An unrecognized/empty response must NOT be optimistically
                // treated as "already indexed" — default to `false` so an
                // unknown shape forces a (safe) reindex instead of skipping one.
                let exists = first_field(&json, &["indexed", "exists"])
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let last_indexed_at = json
                    .get("last_indexed_at")
                    .and_then(Value::as_i64)
                    .and_then(|secs| u64::try_from(secs).ok())
                    .map(|secs| UNIX_EPOCH + Duration::from_secs(secs));
                Ok((exists, last_indexed_at))
            }
            // Only a tool error that explicitly indicates the project is
            // unknown/not-yet-indexed is downgraded to "not indexed"; any other
            // tool failure (permission error, malformed input, internal fault)
            // is surfaced to the caller instead of being silently reinterpreted.
            Err(MemoryError::ToolFailed { message, .. }) if is_not_indexed_error(&message) => {
                Ok((false, None))
            }
            Err(e) => Err(e),
        }
    }

    /// Fill `changed_files` from `detect_changes` for an existing project.
    async fn probe_changes(&self, project: &str) -> Result<ChangeCount, MemoryError> {
        match self.call_and_decode("detect_changes", project).await {
            Ok(json) => {
                // Only a shape we actually understand yields a `Known` count.
                // An absent field, an unexpected type, or a number that is not
                // a plain non-negative integer is `Unknown` — never `Known(0)`,
                // which `decide_freshness` reads as "confirmed no changes" and
                // would optimistically skip a needed reindex (same rule as
                // `probe_status`: an unrecognized response must not be treated
                // as "already indexed").
                let changed = match first_field(&json, &["changed_files", "changed_count"]) {
                    Some(Value::Array(a)) => ChangeCount::Known(a.len()),
                    // Saturate rather than `as`-truncate: on a platform where
                    // `usize` is narrower than `u64`, a huge count must clamp,
                    // not wrap to a small, wrong value (matches the
                    // `line_start`/`line_end` saturating casts in this module).
                    Some(Value::Number(n)) => match n.as_u64() {
                        Some(n) => ChangeCount::Known(n.min(usize::MAX as u64) as usize),
                        None => ChangeCount::Unknown,
                    },
                    _ => ChangeCount::Unknown,
                };
                Ok(changed)
            }
            // Per the `MemoryBackend::ensure_fresh_index` contract, a soft tool
            // failure must not abort exploration outright. We cannot confirm
            // freshness, so report `Unknown` explicitly (rather than an
            // arbitrary nonzero count) — `decide_freshness` treats it as
            // forcing a reindex attempt, same as a real change, without
            // pretending to know how many files changed.
            Err(MemoryError::ToolFailed { .. }) => Ok(ChangeCount::Unknown),
            Err(e) => Err(e),
        }
    }

    /// Run `index_repository` against the given, already-canonicalized repo
    /// root. Soft tool failure -> `IndexingFailed`; transport failure -> `Err`.
    ///
    /// Takes the canonicalized path directly (rather than re-canonicalizing a
    /// raw `repo_root`) because `ensure_fresh_index` already resolved it once
    /// for `project_name_from_abs`; canonicalizing twice per reindex would
    /// duplicate a blocking filesystem syscall for no benefit.
    async fn run_index(&self, abs_repo_root: &Path) -> Result<IndexStatus, MemoryError> {
        let mut args = Map::new();
        args.insert(
            "path".to_string(),
            Value::String(abs_repo_root.to_string_lossy().into_owned()),
        );
        match self.client.call("index_repository", args).await {
            Ok(_) => Ok(IndexStatus::Reindexed),
            Err(MemoryError::ToolFailed { message, .. }) => {
                Ok(IndexStatus::IndexingFailed { reason: message })
            }
            Err(e) => Err(e),
        }
    }

    /// Shared tail of every read-only memory-query method: resolve
    /// `repo_root` to its project name, build `{"project": ...}` plus
    /// whatever `build_args` inserts, call `tool`, decode, and turn the
    /// response into an `ExplorationResult` — the one place that
    /// resolve/call/decode sequence lives. `map` is the only per-tool
    /// difference (row-array responses use [`findings_and_summary`];
    /// `get_code_snippet` decodes a single row).
    async fn call_memory_tool_with(
        &self,
        tool: &'static str,
        repo_root: &Path,
        build_args: impl FnOnce(&mut Map<String, Value>),
        map: impl FnOnce(&'static str, &Value) -> ExplorationResult,
    ) -> Result<ExplorationResult, MemoryError> {
        let project = project_name(repo_root).await?;
        let mut args = base_args(project);
        build_args(&mut args);
        let result = self.client.call(tool, args).await?;
        let json = decode_result(result)?;
        Ok(map(tool, &json))
    }

    /// [`call_memory_tool_with`] for the common case: a response holding an
    /// array of hit rows.
    async fn call_memory_tool(
        &self,
        tool: &'static str,
        repo_root: &Path,
        build_args: impl FnOnce(&mut Map<String, Value>),
    ) -> Result<ExplorationResult, MemoryError> {
        self.call_memory_tool_with(tool, repo_root, build_args, findings_and_summary)
            .await
    }
}

/// Build the `{"project": ...}` argument map every tool call site starts
/// from — the one place the upstream `project` key name/encoding lives.
fn base_args(project: String) -> Map<String, Value> {
    let mut args = Map::new();
    args.insert("project".to_string(), Value::String(project));
    args
}

/// Read the first present field among `keys`, in order — the single place
/// every "which key name did the upstream tool use this time" guess goes
/// through, instead of a separate `.or_else()` chain per call site.
fn first_field<'a>(json: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().find_map(|k| json.get(*k))
}

/// Does a tool-failure message indicate "this project is not indexed yet"
/// (as opposed to some other recoverable-or-not failure)? Matches the
/// observed `codebase-memory-mcp` phrasing ("project not found or not
/// indexed") case-insensitively so close variants still match, while
/// requiring "project" alongside a bare "not found" so unrelated failures
/// (e.g. "config file not found") are not misclassified as "not indexed".
fn is_not_indexed_error(message: &str) -> bool {
    let lower = message.to_lowercase();
    lower.contains("not indexed") || (lower.contains("project") && lower.contains("not found"))
}

/// Build a `FileLocation` from a JSON row's `file`/`path`/`file_path` plus
/// `line_start`/`start_line` (and end variants), tolerating missing line
/// fields (defaulting to 0).
fn location_from(json: &Value) -> Option<FileLocation> {
    let file = first_field(json, &["file", "path", "file_path"]).and_then(Value::as_str)?;
    // Saturate rather than `as`-truncate: a line number beyond `u32::MAX` (or a
    // malformed huge value) must not silently wrap around to a small, wrong one.
    let line_start = first_field(json, &["line_start", "start_line"])
        .and_then(Value::as_u64)
        .map(saturate_u32)
        .unwrap_or(0);
    let line_end = first_field(json, &["line_end", "end_line"])
        .and_then(Value::as_u64)
        .map(saturate_u32)
        .unwrap_or(line_start);
    Some(FileLocation {
        path: std::path::PathBuf::from(file),
        line_start,
        line_end,
    })
}

/// Decode a `get_code_snippet` response: 0 or 1 finding, reusing the same row
/// shape as [`findings_and_summary`] if a location resolves.
fn single_snippet(tool: &'static str, json: &Value) -> ExplorationResult {
    let mut findings = Vec::new();
    if let Some(location) = location_from(json) {
        let snippet = first_field(json, &["snippet", "code", "source", "text"])
            .and_then(Value::as_str)
            .map(|s| s.to_string());
        findings.push(ExplorationFinding {
            location,
            snippet,
            note: symbol_note(json),
        });
    }
    let summary = if findings.is_empty() {
        format!("{tool}: no snippet resolved")
    } else {
        format!("{tool}: 1 snippet")
    };
    ExplorationResult { findings, summary }
}

/// The row's symbol identity, when the upstream tool reported one. Carried on
/// `ExplorationFinding.note` (the domain type has no symbol field) so
/// downstream consumers — the retrieval pre-stage's exact/fuzzy symbol
/// classification and the skeleton renderer — can use it.
fn symbol_note(row: &Value) -> Option<String> {
    first_field(row, &["qualified_name", "name", "symbol"])
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string())
}

/// Parse a `"start-end"` / `"start"` line-range cell (tolerating quotes) into
/// a `(line_start, line_end)` pair; anything unparseable defaults to 0.
fn parse_line_range(cell: &str) -> (u32, u32) {
    let cell = cell.trim().trim_matches('"');
    let mut parts = cell.splitn(2, '-');
    let start = parts
        .next()
        .and_then(|p| p.parse::<u64>().ok())
        .map(saturate_u32)
        .unwrap_or(0);
    let end = parts
        .next()
        .and_then(|p| p.parse::<u64>().ok())
        .map(saturate_u32)
        .unwrap_or(start);
    (start, end)
}

/// Build the `ExplorationFinding` shape both table-row parsers
/// (`columnar_findings`, `text_table_findings`) return: no snippet (neither
/// tool shape carries one) plus whatever symbol/name `note` was resolved.
fn finding(file: &str, line_start: u32, line_end: u32, note: Option<String>) -> ExplorationFinding {
    ExplorationFinding {
        location: FileLocation {
            path: std::path::PathBuf::from(file),
            line_start,
            line_end,
        },
        snippet: None,
        note,
    }
}

/// Decode `codebase-memory-mcp`'s columnar graph payload
/// `{cols, groups: [{qn_prefix?, file, rows: [[cell, ...], ...]}, ...]}`
/// (what `search_graph` with `format: "json"` actually returns). `None` when
/// the value has no such shape.
fn columnar_findings(json: &Value) -> Option<Vec<ExplorationFinding>> {
    let cols = json.get("cols")?.as_array()?;
    let col = |name: &str| cols.iter().position(|c| c.as_str() == Some(name));
    let name_col = col("name");
    let lines_col = col("lines");
    let groups = json.get("groups")?.as_array()?;
    let mut findings = Vec::new();
    for group in groups {
        let Some(file) = group.get("file").and_then(Value::as_str) else {
            continue;
        };
        let qn_prefix = group.get("qn_prefix").and_then(Value::as_str).unwrap_or("");
        let rows = group
            .get("rows")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for row in rows {
            let Some(cells) = row.as_array() else {
                continue;
            };
            let cell = |i: Option<usize>| i.and_then(|i| cells.get(i)).and_then(Value::as_str);
            let (line_start, line_end) = cell(lines_col).map(parse_line_range).unwrap_or((0, 0));
            let note = cell(name_col).map(|name| {
                if qn_prefix.is_empty() {
                    name.to_string()
                } else {
                    format!("{qn_prefix}.{name}")
                }
            });
            findings.push(finding(file, line_start, line_end, note));
        }
    }
    Some(findings)
}

/// A text-table section's `(cols: ...)` header, resolved once into the fixed
/// positions `text_table_findings` needs per row, instead of re-running a
/// linear `position()` scan over the column names for every row.
struct TableCols {
    len: usize,
    file: Option<usize>,
    lines: Option<usize>,
    qn: Option<usize>,
    name: Option<usize>,
}

/// Parse a header line's tail (after the section name's `:`) for a
/// `(cols: ...)` list; `None` when the line carries no such list (e.g. a
/// trailing summary line like `total_grep_matches: 44`).
fn parse_header_cols(rest: &str) -> Option<TableCols> {
    let (_, tail) = rest.split_once("(cols:")?;
    let names: Vec<&str> = tail.trim_end_matches(')').split_whitespace().collect();
    let pos = |name: &str| names.iter().position(|c| *c == name);
    Some(TableCols {
        len: names.len(),
        file: pos("file"),
        lines: pos("lines"),
        qn: pos("qn"),
        name: pos("name"),
    })
}

/// Parse the plain-text tables `codebase-memory-mcp` answers with — one or
/// more sections shaped like:
///
/// ```text
/// results: 3  (cols: qn label file lines matches in out)
///   crates.x.src.a.foo Method crates/x/src/a.rs 78-113 86;107 1 4
/// dirs: 1  (cols: dir hits)
/// ```
///
/// `search_code` sends a single `results:` section; `get_architecture`
/// (which, unlike `search_graph`, never requests `format: "json"`) sends
/// several — `node_labels:`, `edge_types:`, `packages:`, `entry_points:`,
/// etc. Every unindented line is treated as a new section header and parsed
/// for its `(cols: …)` list; only the indented rows under a header whose
/// columns include `file` become findings, so a column-less section (a
/// summary line) or a file-less one (e.g. `packages:`) is walked but simply
/// contributes nothing.
fn text_table_findings(text: &str) -> Vec<ExplorationFinding> {
    let mut findings = Vec::new();
    let mut cols: Option<TableCols> = None;
    for line in text.lines() {
        if !line.starts_with(' ') {
            cols = line
                .split_once(':')
                .and_then(|(_, rest)| parse_header_cols(rest));
            continue;
        }
        let Some(t) = &cols else { continue };
        let cells: Vec<&str> = line.split_whitespace().collect();
        if cells.len() != t.len {
            continue;
        }
        let Some(file) = t.file.and_then(|i| cells.get(i).copied()) else {
            continue;
        };
        let (line_start, line_end) = t
            .lines
            .and_then(|i| cells.get(i).copied())
            .map(parse_line_range)
            .unwrap_or((0, 0));
        let note =
            t.qn.or(t.name)
                .and_then(|i| cells.get(i).copied())
                .map(str::to_string);
        findings.push(finding(file, line_start, line_end, note));
    }
    findings
}

/// Turn a tool response into findings plus a compact summary string. Handles
/// the three shapes `codebase-memory-mcp` actually produces: an array of
/// object rows (`results`/`rows`/`hits`), the columnar `{cols, groups}` JSON,
/// and the plain-text table (reaching here as `Value::String`).
fn findings_and_summary(tool: &'static str, json: &Value) -> ExplorationResult {
    if let Value::String(text) = json {
        let findings = text_table_findings(text);
        let summary = format!("{tool}: {} locatable finding(s)", findings.len());
        return ExplorationResult { findings, summary };
    }
    let mut findings = Vec::new();
    if let Some(rows) = first_field(json, &["results", "rows", "hits"]).and_then(Value::as_array) {
        for row in rows {
            if let Some(location) = location_from(row) {
                let snippet = first_field(row, &["snippet", "text"])
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());
                findings.push(ExplorationFinding {
                    location,
                    snippet,
                    note: symbol_note(row),
                });
            }
        }
        let summary = format!(
            "{tool}: {} row(s), {} locatable finding(s)",
            rows.len(),
            findings.len()
        );
        return ExplorationResult { findings, summary };
    }
    let findings = columnar_findings(json).unwrap_or_default();
    let summary = format!("{tool}: {} locatable finding(s)", findings.len());
    ExplorationResult { findings, summary }
}

impl MemoryBackend for MemoryClientBackend {
    async fn ensure_fresh_index(&self, repo_root: &Path) -> Result<IndexStatus, MemoryError> {
        // Canonicalize once and derive the project name from that same
        // resolved path, instead of calling `project_name` (which
        // canonicalizes again internally) and then re-canonicalizing a
        // second time inside `run_index`.
        let abs = canonicalize_repo_root(repo_root).await;
        let project = project_name_from_abs(repo_root, &abs)?;
        let (exists, last_indexed_at) = self.probe_status(&project).await?;
        // `detect_changes` is only meaningful for a project that exists.
        let changed_files = if exists {
            self.probe_changes(&project).await?
        } else {
            ChangeCount::Known(0)
        };
        let probe = IndexProbe {
            exists,
            last_indexed_at,
            changed_files,
        };
        match decide_freshness(&probe, self.staleness, SystemTime::now()) {
            FreshnessDecision::UpToDate => Ok(IndexStatus::UpToDate),
            FreshnessDecision::Reindex => self.run_index(&abs).await,
        }
    }

    async fn search_code(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
    ) -> Result<ExplorationResult, MemoryError> {
        self.call_memory_tool("search_code", repo_root, |args| {
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
        })
        .await
    }

    async fn search_graph(
        &self,
        repo_root: &Path,
        query: &GraphQuery,
    ) -> Result<ExplorationResult, MemoryError> {
        self.call_memory_tool("search_graph", repo_root, |args| {
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
        })
        .await
    }

    async fn query_graph(
        &self,
        repo_root: &Path,
        query: &str,
        max_results: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError> {
        self.call_memory_tool("query_graph", repo_root, |args| {
            args.insert("query".to_string(), Value::String(query.to_string()));
            if let Some(limit) = max_results {
                args.insert("limit".to_string(), Value::Number(limit.into()));
            }
        })
        .await
    }

    async fn trace_path(
        &self,
        repo_root: &Path,
        from: &str,
        to: &str,
        max_depth: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError> {
        self.call_memory_tool("trace_path", repo_root, |args| {
            args.insert("from".to_string(), Value::String(from.to_string()));
            args.insert("to".to_string(), Value::String(to.to_string()));
            if let Some(depth) = max_depth {
                args.insert("max_depth".to_string(), Value::Number(depth.into()));
            }
        })
        .await
    }

    async fn get_architecture(
        &self,
        repo_root: &Path,
        depth: Option<u32>,
    ) -> Result<ExplorationResult, MemoryError> {
        self.call_memory_tool("get_architecture", repo_root, |args| {
            if let Some(d) = depth {
                args.insert("depth".to_string(), Value::Number(d.into()));
            }
        })
        .await
    }

    async fn get_code_snippet(
        &self,
        repo_root: &Path,
        target: &SnippetTarget,
    ) -> Result<ExplorationResult, MemoryError> {
        self.call_memory_tool_with(
            "get_code_snippet",
            repo_root,
            |args| match target {
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
            },
            single_snippet,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Real `search_graph format:"json"` payload shape (columnar).
    #[test]
    fn columnar_search_graph_payload_decodes() {
        let payload = json!({
            "total": 1,
            "count": 1,
            "cols": ["name", "label", "lines", "in", "out"],
            "groups": [{
                "qn_prefix": "repo.crates.repo-explorer-memory.src.freshness",
                "file": "crates/repo-explorer-memory/src/freshness.rs",
                "rows": [["decide_freshness", "Function", "40-60", 9, 6]]
            }],
            "has_more": false
        });
        let res = findings_and_summary("search_graph", &payload);
        assert_eq!(res.findings.len(), 1);
        let f = &res.findings[0];
        assert_eq!(
            f.location.path,
            std::path::PathBuf::from("crates/repo-explorer-memory/src/freshness.rs")
        );
        assert_eq!((f.location.line_start, f.location.line_end), (40, 60));
        assert_eq!(
            f.note.as_deref(),
            Some("repo.crates.repo-explorer-memory.src.freshness.decide_freshness")
        );
        assert!(res.summary.contains("1 locatable finding"));
    }

    /// Real `search_code` payload shape (plain-text table via `Value::String`).
    #[test]
    fn text_table_search_code_payload_decodes() {
        let text = "results: 3  (cols: qn label file lines matches in out)\n  \
repo.crates.a.src.b.MemoryClientBackend.probe_changes Method crates/a/src/b.rs 78-113 86;107 1 4\n  \
repo.crates.a.src.b.MemoryClientBackend.ensure_fresh_index Method crates/a/src/b.rs 280-303 \"299\" 1 7\n  \
repo.crates.a.src.f.decide_freshness Function crates/a/src/f.rs 40-60 \"40\" 1 0\n\
dirs: 1  (cols: dir hits)\n  crates/ 28\ntotal_grep_matches: 44\n";
        let res = findings_and_summary("search_code", &Value::String(text.to_string()));
        assert_eq!(res.findings.len(), 3);
        assert_eq!(
            res.findings[2].location.path,
            std::path::PathBuf::from("crates/a/src/f.rs")
        );
        assert_eq!(
            (
                res.findings[2].location.line_start,
                res.findings[2].location.line_end
            ),
            (40, 60)
        );
        assert_eq!(
            res.findings[0].note.as_deref(),
            Some("repo.crates.a.src.b.MemoryClientBackend.probe_changes")
        );
        // The dirs section must not leak into findings.
        assert!(
            res.findings
                .iter()
                .all(|f| f.location.path != std::path::Path::new("crates/"))
        );
    }

    /// Real `get_architecture` payload shape: multiple plain-text sections,
    /// none named `results:`, only some (`entry_points:`) carrying a `file`
    /// column.
    #[test]
    fn text_table_get_architecture_payload_decodes() {
        let text = "\
node_labels: 2  (cols: label count)\n  Function 120\n  Method 136\n\
packages: 1  (cols: name nodes fan_in fan_out)\n  repo-explorer-core 256 0 0\n\
entry_points: 2  (cols: qn file)\n  \
repo.crates.repo-explorer-mcp.src.main.main crates/repo-explorer-mcp/src/main.rs\n  \
repo.crates.repo-explorer-core.src.lib.run crates/repo-explorer-core/src/lib.rs\n";
        let res = findings_and_summary("get_architecture", &Value::String(text.to_string()));
        assert_eq!(res.findings.len(), 2);
        assert_eq!(
            res.findings[0].location.path,
            std::path::PathBuf::from("crates/repo-explorer-mcp/src/main.rs")
        );
        assert_eq!(
            res.findings[0].note.as_deref(),
            Some("repo.crates.repo-explorer-mcp.src.main.main")
        );
        // No `file` column on node_labels/packages must not produce findings.
        assert!(
            res.findings
                .iter()
                .all(|f| f.location.path != std::path::Path::new("Function"))
        );
    }

    /// Real `get_code_snippet` payload shape (file_path/start_line/source).
    #[test]
    fn get_code_snippet_payload_decodes() {
        let payload = json!({
            "name": "decide_freshness",
            "qualified_name": "repo.crates.a.src.f.decide_freshness",
            "label": "Function",
            "file_path": "/repo/crates/a/src/f.rs",
            "start_line": 40,
            "end_line": 60,
            "source": "pub(crate) fn decide_freshness() {}",
            "callers": 1,
            "callees": 0
        });
        let res = single_snippet("get_code_snippet", &payload);
        assert_eq!(res.findings.len(), 1);
        let f = &res.findings[0];
        assert_eq!((f.location.line_start, f.location.line_end), (40, 60));
        assert_eq!(
            f.snippet.as_deref(),
            Some("pub(crate) fn decide_freshness() {}")
        );
        assert_eq!(
            f.note.as_deref(),
            Some("repo.crates.a.src.f.decide_freshness")
        );
    }

    /// Object-row arrays (the previously supported shape) still decode.
    #[test]
    fn object_rows_still_decode() {
        let payload = json!({
            "results": [
                {"file": "src/a.rs", "line_start": 1, "line_end": 2, "snippet": "x", "name": "foo"},
                {"no_file": true}
            ]
        });
        let res = findings_and_summary("search_graph", &payload);
        assert_eq!(res.findings.len(), 1);
        assert_eq!(res.findings[0].note.as_deref(), Some("foo"));
        assert!(res.summary.contains("2 row(s), 1 locatable finding(s)"));
    }

    #[test]
    fn parse_line_range_boundaries() {
        assert_eq!(parse_line_range("40-60"), (40, 60));
        assert_eq!(parse_line_range("\"40\""), (40, 40));
        assert_eq!(parse_line_range("7"), (7, 7));
        assert_eq!(parse_line_range("garbage"), (0, 0));
        assert_eq!(parse_line_range("5-x"), (5, 5));
    }
}
