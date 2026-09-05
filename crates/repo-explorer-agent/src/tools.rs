//! The 10-tool catalog (JSON-Schema literals), per-tool argument DTOs, and
//! `finish` validation. `repo_root` is never a tool parameter — the dispatcher
//! supplies it from loop state.

use crate::dispatch::{canonical_repo_root, read_file_canonical};
use repo_explorer_core::domain::{ExplorationFinding, ExplorationResult, FileLocation};
use repo_explorer_core::llm::{Message, Tool, ToolCall};
use repo_explorer_core::retrieval::normalize_location;
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Appended to every `tool_catalog()` tool other than `finish`: the fallback
/// loop rejects single-call turns (2-strike), so the demand must also live
/// where the model reads tool contracts. `expand` never gets this — it lives
/// only in `verify_catalog()`, whose loop has no such rejection and whose
/// system prompt asks for exactly one `expand` call per turn.
const BATCH_SUFFIX: &str = " Batch all independent tool calls of a turn into ONE response with multiple tool calls; single-call turns are rejected.";

fn tool(name: &str, description: &str, schema: serde_json::Value) -> Tool {
    let description = if name == "finish" || name == "expand" {
        description.to_string()
    } else {
        format!("{description}{BATCH_SUFFIX}")
    };
    Tool {
        name: name.to_string(),
        description,
        parameters_schema_json: schema.to_string(),
    }
}

/// The 10-tool catalog, built once per process. `finish` is always included;
/// the loop re-offers the whole catalog on every turn, which is what makes
/// `finish` "forced".
pub(crate) fn tool_catalog() -> &'static [Tool] {
    static CATALOG: LazyLock<Vec<Tool>> = LazyLock::new(build_catalog);
    &CATALOG
}

/// The verification-stage catalog: `expand` plus `finish`. Built once per
/// process, like `tool_catalog`.
pub(crate) fn verify_catalog() -> &'static [Tool] {
    static CATALOG: LazyLock<Vec<Tool>> = LazyLock::new(|| vec![expand_tool(), finish_tool()]);
    &CATALOG
}

/// Only `finish` — offered together with a forced tool choice for the final
/// budget-exhausted turn.
pub(crate) fn finish_only_catalog() -> &'static [Tool] {
    static CATALOG: LazyLock<Vec<Tool>> = LazyLock::new(|| vec![finish_tool()]);
    &CATALOG
}

fn expand_tool() -> Tool {
    tool(
        "expand",
        "Fetch the full bodies of the numbered candidates you cannot judge from their skeletons alone. Args: candidate_ids (required, 1-based ids from the candidate list).",
        json!({
            "type": "object",
            "properties": {
                "candidate_ids": {
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["candidate_ids"],
            "additionalProperties": false
        }),
    )
}

/// Shared JSON-Schema for `grep` and `find`, mirroring `PatternArgs` — one
/// shape, so one schema literal.
fn pattern_args_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "scope": {"type": "string"},
            "max_results": {"type": "integer"}
        },
        "required": ["pattern"],
        "additionalProperties": false
    })
}

fn build_catalog() -> Vec<Tool> {
    vec![
        tool(
            "search_code",
            "PRIMARY/authoritative: literal/regex text search (grep-style, not natural language) over the indexed memory graph. Args: query (required, a literal string or regex pattern — not a natural-language question), optional scope_hint, max_results.",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope_hint": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        ),
        tool(
            "search_graph",
            "PRIMARY: structural search of the memory graph by name/file/label pattern. All args optional.",
            json!({
                "type": "object",
                "properties": {
                    "name_pattern": {"type": "string"},
                    "file_pattern": {"type": "string"},
                    "label": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": [],
                "additionalProperties": false
            }),
        ),
        tool(
            "query_graph",
            "PRIMARY: run a graph query against the memory graph. Args: query (required), optional max_results.",
            json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        ),
        tool(
            "trace_path",
            "PRIMARY: report a single function's callers and callees (both directions) in the memory graph — not a path between two symbols; the connected backend has no two-endpoint concept and takes no `to` argument. Args: from (required, the function name), optional max_depth.",
            json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "max_depth": {"type": "integer"}
                },
                "required": ["from"],
                "additionalProperties": false
            }),
        ),
        tool(
            "get_architecture",
            "PRIMARY: retrieve a high-level architecture overview from memory. Args: optional depth.",
            json!({
                "type": "object",
                "properties": {
                    "depth": {"type": "integer"}
                },
                "required": [],
                "additionalProperties": false
            }),
        ),
        tool(
            "get_code_snippet",
            "PRIMARY: fetch a code snippet from memory by qualified_name. The connected backend requires qualified_name unconditionally — a file plus start_line/end_line alone is not supported and always fails; use read_file for a path/line-range read instead.",
            json!({
                "type": "object",
                "properties": {
                    "qualified_name": {"type": "string"}
                },
                "required": ["qualified_name"],
                "additionalProperties": false
            }),
        ),
        tool(
            "grep",
            "SUPPLEMENT/fallback: raw content search of files under the repo. Use only when the memory tools are insufficient. Args: pattern (required), optional scope, max_results.",
            pattern_args_schema(),
        ),
        tool(
            "find",
            "SUPPLEMENT/fallback: search for files by name pattern. Use only when the memory tools are insufficient. Args: pattern (required), optional scope, max_results.",
            pattern_args_schema(),
        ),
        tool(
            "read_file",
            "SUPPLEMENT/fallback: read a file (or a line range of it) relative to the repository root. Use only when the memory tools are insufficient. Args: path (required), optional start_line, end_line.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        ),
        finish_tool(),
    ]
}

/// The `finish` tool, shared by the exploration, verification, and
/// finish-only catalogs.
fn finish_tool() -> Tool {
    tool(
        "finish",
        "REQUIRED to conclude: report the located findings and a summary. Call this once you have gathered enough information.",
        json!({
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "line_start": {"type": "integer"},
                                    "line_end": {"type": "integer"}
                                },
                                "required": ["path", "line_start", "line_end"],
                                "additionalProperties": false
                            },
                            "snippet": {"type": "string"},
                            "note": {"type": "string"}
                        },
                        "required": ["location"],
                        "additionalProperties": false
                    }
                },
                "summary": {"type": "string"}
            },
            "required": ["summary"],
            "additionalProperties": false
        }),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SearchCodeArgs {
    pub query: String,
    #[serde(default)]
    pub scope_hint: Option<String>,
    #[serde(default)]
    pub max_results: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SearchGraphArgs {
    #[serde(default)]
    pub name_pattern: Option<String>,
    #[serde(default)]
    pub file_pattern: Option<String>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub max_results: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct QueryGraphArgs {
    pub query: String,
    #[serde(default)]
    pub max_results: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TracePathArgs {
    pub from: String,
    #[serde(default)]
    pub max_depth: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GetArchitectureArgs {
    #[serde(default)]
    pub depth: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GetCodeSnippetArgs {
    pub qualified_name: String,
}

/// Arguments of both `grep` (content pattern) and `find` (file-name glob) —
/// one shape, so one type.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PatternArgs {
    pub pattern: String,
    #[serde(default)]
    pub scope: Option<String>,
    #[serde(default)]
    pub max_results: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReadFileArgs {
    pub path: String,
    #[serde(default)]
    pub start_line: Option<u32>,
    #[serde(default)]
    pub end_line: Option<u32>,
}

/// Arguments of the verification stage's `expand` tool: 1-based ids into the
/// numbered candidate list.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExpandArgs {
    pub candidate_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FinishArgs {
    #[serde(default)]
    pub findings: Vec<FinishFinding>,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FinishFinding {
    pub location: FinishLocation,
    #[serde(default)]
    pub snippet: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FinishLocation {
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
}

/// Validate one finding's location against the filesystem under `repo_root`:
/// non-empty path, `normalize_location`, then existence/escape/inside-repo
/// check by reusing `read_file_canonical`. A `line_end` past the file's real
/// length is clamped to the line count (never below `line_start`), the same
/// defect class as F-13. Returns a human-readable reason string on rejection.
async fn validate_finding(
    f: FinishFinding,
    repo_root: &Path,
    canonical_root: &Path,
) -> Result<ExplorationFinding, String> {
    if f.location.path.trim().is_empty() {
        return Err("finding location.path must be non-empty".to_string());
    }
    let location = normalize_location(FileLocation {
        path: PathBuf::from(&f.location.path),
        line_start: f.location.line_start,
        line_end: f.location.line_end,
    });
    let path_str = location.path.to_string_lossy();
    // Existence + escape + inside-repo check, reusing the read_file safe-read.
    // ponytail: reads the whole file just to count lines; swap for a
    // stat-only helper if finish-time I/O ever profiles hot.
    let content = read_file_canonical(repo_root, canonical_root, &path_str, None, None)
        .await
        .map_err(|_| {
            format!("finding location.path `{path_str}` does not exist in the repository")
        })?;
    // Clamp a line_end past EOF to the file length, never below line_start.
    let line_count = content.lines().count() as u32;
    let line_end = location.line_end.min(line_count.max(location.line_start));
    Ok(ExplorationFinding {
        location: FileLocation {
            line_end,
            ..location
        },
        snippet: f.snippet,
        note: f.note,
    })
}

/// Deserialize and hand-validate `finish` arguments into an `ExplorationResult`.
/// `deny_unknown_fields` plus the required `summary`/`line_start`/`line_end`
/// keys reject malformed payloads at parse time; every finding is validated by
/// [`validate_finding`], and the first rejection fails the whole call — the
/// loop feeds the reason back to the model as a retry rejection message via
/// [`resolve_finish`]. See [`parse_finish_lenient`] for the one-shot forced-finish
/// path, which has no retry loop to feed a rejection back to.
pub(crate) async fn parse_finish(
    arguments_json: &str,
    repo_root: &Path,
) -> Result<ExplorationResult, String> {
    let args: FinishArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("could not parse finish arguments: {e}"))?;
    // Resolve the canonical repo root once, and only when there is at least
    // one path to validate (see canonical_repo_root's reuse guidance).
    let canonical_root = if args.findings.is_empty() {
        None
    } else {
        Some(canonical_repo_root(repo_root).await?)
    };
    let mut findings = Vec::with_capacity(args.findings.len());
    for f in args.findings {
        let canonical_root = canonical_root
            .as_ref()
            .expect("canonical_root is Some when findings are non-empty");
        findings.push(validate_finding(f, repo_root, canonical_root).await?);
    }
    Ok(ExplorationResult {
        findings,
        summary: args.summary,
    })
}

/// Lenient counterpart to [`parse_finish`] for `forced_finish`, whose one-shot
/// call has no retry path to feed a rejection back to the model: rather than
/// discarding the whole payload over a single bad finding, this keeps every
/// finding that validates and silently drops only the invalid ones. An empty
/// `findings` array is still a legitimate "nothing found" completion; only a
/// non-empty array left with zero survivors is an error, matching
/// [`parse_finish`]'s all-or-nothing behavior for that case. Still propagates
/// a JSON parse error, since there is nothing salvageable then.
pub(crate) async fn parse_finish_lenient(
    arguments_json: &str,
    repo_root: &Path,
) -> Result<ExplorationResult, String> {
    let args: FinishArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("could not parse finish arguments: {e}"))?;
    let had_findings = !args.findings.is_empty();
    let canonical_root = if args.findings.is_empty() {
        None
    } else {
        Some(canonical_repo_root(repo_root).await?)
    };
    let mut findings = Vec::with_capacity(args.findings.len());
    for f in args.findings {
        let canonical_root = canonical_root
            .as_ref()
            .expect("canonical_root is Some when findings are non-empty");
        match validate_finding(f, repo_root, canonical_root).await {
            Ok(finding) => findings.push(finding),
            Err(reason) => {
                tracing::debug!(reason = %reason, "forced finish dropped an invalid finding")
            }
        }
    }
    if had_findings && findings.is_empty() {
        return Err("no finding in the finish call had a valid path".to_string());
    }
    Ok(ExplorationResult {
        findings,
        summary: args.summary,
    })
}

/// Resolve the `finish` calls of one model turn: the first one that parses
/// successfully ends the turn immediately; otherwise a rejection
/// `Role::Tool` message is built for every one that failed to parse, so the
/// caller can feed it back to the model and let it retry. Shared by the
/// fallback loop and the verification stage, which otherwise duplicated this
/// exact resolution logic.
pub(crate) async fn resolve_finish(
    calls: &[ToolCall],
    repo_root: &Path,
) -> Result<ExplorationResult, Vec<Message>> {
    let mut rejections = Vec::new();
    for c in calls.iter().filter(|c| c.name == "finish") {
        match parse_finish(&c.arguments_json, repo_root).await {
            Ok(result) => return Ok(result),
            Err(reason) => rejections.push(Message::tool(
                &c.id,
                format!("finish rejected: {reason}; fix the arguments and call finish again"),
            )),
        }
    }
    Err(rejections)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Temp repo containing `src/main.rs` with `n_lines` numbered lines, via
    /// the crate's shared `test_support::temp_repo_with` fixture. Caller
    /// removes the dir at the end.
    fn temp_repo_main(test: &str, n_lines: usize) -> std::path::PathBuf {
        let body = (1..=n_lines)
            .map(|i| format!("l{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        crate::test_support::temp_repo_with("agent_tools_finish", test, &[("src/main.rs", &body)])
    }

    const NAMES: [&str; 10] = [
        "search_code",
        "search_graph",
        "query_graph",
        "trace_path",
        "get_architecture",
        "get_code_snippet",
        "grep",
        "find",
        "read_file",
        "finish",
    ];

    #[test]
    fn catalog_has_ten_tools_with_verbatim_names_in_order() {
        let catalog = tool_catalog();
        assert_eq!(catalog.len(), 10);
        let got: Vec<&str> = catalog.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(got, NAMES.to_vec());
    }

    #[test]
    fn every_schema_is_valid_json_object() {
        for t in tool_catalog() {
            let v: serde_json::Value = serde_json::from_str(&t.parameters_schema_json)
                .unwrap_or_else(|e| panic!("tool {} schema is not valid JSON: {e}", t.name));
            assert_eq!(v["type"], "object", "tool {} schema type", t.name);
            assert_eq!(
                v["additionalProperties"], false,
                "tool {} must set additionalProperties false",
                t.name
            );
        }
    }

    #[test]
    fn finish_is_present() {
        assert!(tool_catalog().iter().any(|t| t.name == "finish"));
    }

    #[test]
    fn search_code_args_parse() {
        let a: SearchCodeArgs =
            serde_json::from_str(r#"{"query":"main","max_results":5}"#).unwrap();
        assert_eq!(a.query, "main");
        assert_eq!(a.scope_hint, None);
        assert_eq!(a.max_results, Some(5));
    }

    #[test]
    fn get_code_snippet_args_parse_qualified_name() {
        let a: GetCodeSnippetArgs =
            serde_json::from_str(r#"{"qualified_name":"foo::bar"}"#).unwrap();
        assert_eq!(a.qualified_name, "foo::bar");
    }

    #[test]
    fn unknown_field_is_rejected() {
        let r: Result<SearchCodeArgs, _> = serde_json::from_str(r#"{"query":"x","bogus":1}"#);
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn parse_finish_round_trips() {
        let dir = temp_repo_main("round_trips", 5);
        let json = r#"{"findings":[{"location":{"path":"src/main.rs","line_start":1,"line_end":3},"note":"here"}],"summary":"done"}"#;
        let result = parse_finish(json, &dir).await.unwrap();
        assert_eq!(result.summary, "done");
        assert_eq!(result.findings.len(), 1);
        assert_eq!(
            result.findings[0].location,
            repo_explorer_core::domain::FileLocation {
                path: std::path::PathBuf::from("src/main.rs"),
                line_start: 1,
                line_end: 3,
            }
        );
        assert_eq!(result.findings[0].note, Some("here".to_string()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_allows_empty_findings() {
        let result = parse_finish(r#"{"summary":"nothing found"}"#, &std::env::temp_dir())
            .await
            .unwrap();
        assert!(result.findings.is_empty());
        assert_eq!(result.summary, "nothing found");
    }

    #[tokio::test]
    async fn parse_finish_rejects_missing_summary() {
        assert!(
            parse_finish(r#"{"findings":[]}"#, &std::env::temp_dir())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn parse_finish_rejects_empty_path() {
        let json =
            r#"{"findings":[{"location":{"path":"","line_start":1,"line_end":2}}],"summary":"s"}"#;
        assert!(parse_finish(json, &std::env::temp_dir()).await.is_err());
    }

    #[tokio::test]
    async fn parse_finish_rejects_missing_line_numbers() {
        let json = r#"{"findings":[{"location":{"path":"a.rs"}}],"summary":"s"}"#;
        assert!(parse_finish(json, &std::env::temp_dir()).await.is_err());
    }

    #[tokio::test]
    async fn parse_finish_normalizes_inverted_range() {
        let dir = temp_repo_main("inverted", 60);
        let json = r#"{"findings":[{"location":{"path":"src/main.rs","line_start":50,"line_end":10}}],"summary":"done"}"#;
        let result = parse_finish(json, &dir).await.unwrap();
        assert_eq!(result.findings[0].location.line_start, 10);
        assert_eq!(result.findings[0].location.line_end, 50);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_widens_zero_end_to_start() {
        let dir = temp_repo_main("zero_end", 10);
        let json = r#"{"findings":[{"location":{"path":"src/main.rs","line_start":7,"line_end":0}}],"summary":"done"}"#;
        let result = parse_finish(json, &dir).await.unwrap();
        assert_eq!(result.findings[0].location.line_start, 7);
        assert_eq!(result.findings[0].location.line_end, 7);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_accepts_existing_path() {
        let dir = temp_repo_main("accepts", 5);
        let json = r#"{"findings":[{"location":{"path":"src/main.rs","line_start":2,"line_end":4}}],"summary":"ok"}"#;
        let result = parse_finish(json, &dir).await.unwrap();
        assert_eq!(
            result.findings[0].location,
            repo_explorer_core::domain::FileLocation {
                path: std::path::PathBuf::from("src/main.rs"),
                line_start: 2,
                line_end: 4,
            }
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_rejects_nonexistent_path() {
        let dir = temp_repo_main("nonexistent", 5);
        let json = r#"{"findings":[{"location":{"path":"src/gone.rs","line_start":1,"line_end":2}}],"summary":"s"}"#;
        let err = parse_finish(json, &dir).await.unwrap_err();
        assert!(err.contains("does not exist"), "reason was: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_clamps_line_end_past_eof() {
        let dir = temp_repo_main("clamp", 3);
        let json = r#"{"findings":[{"location":{"path":"src/main.rs","line_start":2,"line_end":99}}],"summary":"s"}"#;
        let result = parse_finish(json, &dir).await.unwrap();
        assert_eq!(result.findings[0].location.line_start, 2);
        assert_eq!(result.findings[0].location.line_end, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn resolve_finish_rejects_nonexistent_path() {
        let dir = temp_repo_main("resolve_nonexistent", 5);
        let json = r#"{"findings":[{"location":{"path":"src/gone.rs","line_start":1,"line_end":2}}],"summary":"s"}"#;
        let call = ToolCall {
            id: "call-1".to_string(),
            name: "finish".to_string(),
            arguments_json: json.to_string(),
            thought_signatures: None,
        };
        let rejections = resolve_finish(&[call], &dir).await.unwrap_err();
        assert_eq!(rejections.len(), 1);
        let msg = &rejections[0];
        assert_eq!(msg.role, repo_explorer_core::llm::Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call-1"));
        assert!(
            msg.content.contains("finish rejected"),
            "content: {}",
            msg.content
        );
        assert!(
            msg.content.contains("does not exist"),
            "content: {}",
            msg.content
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_lenient_drops_invalid_findings_keeps_valid() {
        let dir = temp_repo_main("lenient_partial", 5);
        let json = r#"{"findings":[
            {"location":{"path":"src/main.rs","line_start":1,"line_end":2}},
            {"location":{"path":"src/gone.rs","line_start":1,"line_end":2}}
        ],"summary":"partial"}"#;
        let result = parse_finish_lenient(json, &dir).await.unwrap();
        assert_eq!(result.summary, "partial");
        assert_eq!(result.findings.len(), 1);
        assert_eq!(
            result.findings[0].location.path,
            std::path::PathBuf::from("src/main.rs")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_lenient_rejects_when_all_findings_invalid() {
        let dir = temp_repo_main("lenient_all_invalid", 5);
        let json = r#"{"findings":[{"location":{"path":"src/gone.rs","line_start":1,"line_end":2}}],"summary":"s"}"#;
        assert!(parse_finish_lenient(json, &dir).await.is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn parse_finish_lenient_allows_empty_findings() {
        let result = parse_finish_lenient(r#"{"summary":"nothing found"}"#, &std::env::temp_dir())
            .await
            .unwrap();
        assert!(result.findings.is_empty());
        assert_eq!(result.summary, "nothing found");
    }
}
