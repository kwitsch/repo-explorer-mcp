//! The 10-tool catalog (JSON-Schema literals), per-tool argument DTOs, and
//! `finish` validation. `repo_root` is never a tool parameter — the dispatcher
//! supplies it from loop state.

use repo_explorer_core::domain::{ExplorationFinding, ExplorationResult, FileLocation};
use repo_explorer_core::llm::{Message, Tool, ToolCall};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;
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
            "PRIMARY/authoritative: semantic code search over the indexed memory graph. Prefer the memory tools first. Args: query (required), optional scope_hint, max_results.",
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
            "PRIMARY: trace a dependency/reference path between two symbols in the memory graph. Args: from (required), to (required), optional max_depth.",
            json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "max_depth": {"type": "integer"}
                },
                "required": ["from", "to"],
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
            "PRIMARY: fetch a code snippet from memory, either by qualified_name or by file plus optional start_line/end_line. Provide qualified_name OR file (qualified_name wins if both are present).",
            json!({
                "type": "object",
                "properties": {
                    "qualified_name": {"type": "string"},
                    "file": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                },
                "required": [],
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
    pub to: String,
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
    #[serde(default)]
    pub qualified_name: Option<String>,
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub start_line: Option<u32>,
    #[serde(default)]
    pub end_line: Option<u32>,
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

/// Deserialize and hand-validate `finish` arguments into an `ExplorationResult`.
/// `deny_unknown_fields` plus the required `summary`/`line_start`/`line_end`
/// keys reject malformed payloads at parse time; the explicit non-empty-path
/// check covers the one rule serde cannot express. Returns a human-readable
/// reason string on rejection so the loop can feed it back to the model.
pub(crate) fn parse_finish(arguments_json: &str) -> Result<ExplorationResult, String> {
    let args: FinishArgs = serde_json::from_str(arguments_json)
        .map_err(|e| format!("could not parse finish arguments: {e}"))?;
    let mut findings = Vec::with_capacity(args.findings.len());
    for f in args.findings {
        if f.location.path.trim().is_empty() {
            return Err("finding location.path must be non-empty".to_string());
        }
        findings.push(ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(f.location.path),
                line_start: f.location.line_start,
                line_end: f.location.line_end,
            },
            snippet: f.snippet,
            note: f.note,
        });
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
pub(crate) fn resolve_finish(calls: &[ToolCall]) -> Result<ExplorationResult, Vec<Message>> {
    let mut rejections = Vec::new();
    for c in calls.iter().filter(|c| c.name == "finish") {
        match parse_finish(&c.arguments_json) {
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
    fn get_code_snippet_args_parse_both_modes() {
        let a: GetCodeSnippetArgs =
            serde_json::from_str(r#"{"qualified_name":"foo::bar"}"#).unwrap();
        assert_eq!(a.qualified_name, Some("foo::bar".to_string()));
        let b: GetCodeSnippetArgs =
            serde_json::from_str(r#"{"file":"src/lib.rs","start_line":1,"end_line":9}"#).unwrap();
        assert_eq!(b.file, Some("src/lib.rs".to_string()));
        assert_eq!(b.start_line, Some(1));
        assert_eq!(b.end_line, Some(9));
    }

    #[test]
    fn unknown_field_is_rejected() {
        let r: Result<SearchCodeArgs, _> = serde_json::from_str(r#"{"query":"x","bogus":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn parse_finish_round_trips() {
        let json = r#"{"findings":[{"location":{"path":"src/main.rs","line_start":1,"line_end":3},"note":"here"}],"summary":"done"}"#;
        let result = parse_finish(json).unwrap();
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
    }

    #[test]
    fn parse_finish_allows_empty_findings() {
        let result = parse_finish(r#"{"summary":"nothing found"}"#).unwrap();
        assert!(result.findings.is_empty());
        assert_eq!(result.summary, "nothing found");
    }

    #[test]
    fn parse_finish_rejects_missing_summary() {
        assert!(parse_finish(r#"{"findings":[]}"#).is_err());
    }

    #[test]
    fn parse_finish_rejects_empty_path() {
        let json =
            r#"{"findings":[{"location":{"path":"","line_start":1,"line_end":2}}],"summary":"s"}"#;
        assert!(parse_finish(json).is_err());
    }

    #[test]
    fn parse_finish_rejects_missing_line_numbers() {
        let json = r#"{"findings":[{"location":{"path":"a.rs"}}],"summary":"s"}"#;
        assert!(parse_finish(json).is_err());
    }
}
