//! Parse one `ToolCall` → typed args → backend call (or inline `read_file`) →
//! render the outcome as a single `Role::Tool` message plus any findings to
//! accumulate. Backend errors, malformed arguments, unknown tools, and
//! `read_file` IO/path-escape failures all become `Role::Tool` messages — never
//! panics, never fatal. `finish` is handled by the loop, not here.

use repo_explorer_core::domain::{ExplorationFinding, ExplorationQuery, ExplorationResult};
use repo_explorer_core::llm::{Message, ToolCall};
use repo_explorer_core::memory::{GraphQuery, MemoryBackend, SnippetTarget};
use repo_explorer_core::search::{SearchBackend, SearchOptions};
use serde::Serialize;
use std::path::{Component, Path, PathBuf};

use crate::tools::{
    FindArgs, GetArchitectureArgs, GetCodeSnippetArgs, GrepArgs, QueryGraphArgs, ReadFileArgs,
    SearchCodeArgs, SearchGraphArgs, TracePathArgs,
};

/// Dispatch a single non-`finish` tool call, returning the `Role::Tool` message
/// to push and any findings to accumulate.
pub(crate) async fn dispatch_call<M: MemoryBackend, S: SearchBackend>(
    memory: &M,
    search: &S,
    repo_root: &Path,
    call: &ToolCall,
) -> (Message, Vec<ExplorationFinding>) {
    match dispatch_inner(memory, search, repo_root, call).await {
        Ok((content, findings)) => (tool_message(&call.id, content), findings),
        Err(msg) => (tool_message(&call.id, msg), Vec::new()),
    }
}

async fn dispatch_inner<M: MemoryBackend, S: SearchBackend>(
    memory: &M,
    search: &S,
    repo_root: &Path,
    call: &ToolCall,
) -> Result<(String, Vec<ExplorationFinding>), String> {
    match call.name.as_str() {
        "search_code" => {
            let args: SearchCodeArgs = parse_args(&call.arguments_json)?;
            let query = ExplorationQuery {
                text: args.query,
                scope_hint: args.scope_hint.map(PathBuf::from),
                max_results: args.max_results,
            };
            call_and_render("search_code", memory.search_code(repo_root, &query)).await
        }
        "search_graph" => {
            let args: SearchGraphArgs = parse_args(&call.arguments_json)?;
            let query = GraphQuery {
                name_pattern: args.name_pattern,
                file_pattern: args.file_pattern,
                label: args.label,
                max_results: args.max_results,
            };
            call_and_render("search_graph", memory.search_graph(repo_root, &query)).await
        }
        "query_graph" => {
            let args: QueryGraphArgs = parse_args(&call.arguments_json)?;
            call_and_render(
                "query_graph",
                memory.query_graph(repo_root, &args.query, args.max_results),
            )
            .await
        }
        "trace_path" => {
            let args: TracePathArgs = parse_args(&call.arguments_json)?;
            call_and_render(
                "trace_path",
                memory.trace_path(repo_root, &args.from, &args.to, args.max_depth),
            )
            .await
        }
        "get_architecture" => {
            let args: GetArchitectureArgs = parse_args(&call.arguments_json)?;
            call_and_render(
                "get_architecture",
                memory.get_architecture(repo_root, args.depth),
            )
            .await
        }
        "get_code_snippet" => {
            let args: GetCodeSnippetArgs = parse_args(&call.arguments_json)?;
            let target = snippet_target(args)?;
            call_and_render(
                "get_code_snippet",
                memory.get_code_snippet(repo_root, &target),
            )
            .await
        }
        "grep" => {
            let args: GrepArgs = parse_args(&call.arguments_json)?;
            let opts = SearchOptions {
                max_results: args.max_results,
                ..SearchOptions::default()
            };
            let scope = validate_scope(args.scope.as_deref())?;
            let findings = search
                .search(repo_root, &args.pattern, scope.as_deref(), &opts)
                .await
                .map_err(|e| format!("grep failed: {e}"))?;
            Ok(render_findings(findings))
        }
        "find" => {
            let args: FindArgs = parse_args(&call.arguments_json)?;
            // The real `SearchBackend` only exposes a content search; a
            // filename search is approximated by matching any non-empty line
            // (pattern `.`) restricted to files matching `pattern` as a glob.
            let opts = SearchOptions {
                max_results: args.max_results,
                file_glob: Some(args.pattern.clone()),
                ..SearchOptions::default()
            };
            let scope = validate_scope(args.scope.as_deref())?;
            let findings = search
                .search(repo_root, ".", scope.as_deref(), &opts)
                .await
                .map_err(|e| format!("find failed: {e}"))?;
            Ok(render_findings(findings))
        }
        "read_file" => {
            let args: ReadFileArgs = parse_args(&call.arguments_json)?;
            let content = read_file(repo_root, &args.path, args.start_line, args.end_line)?;
            Ok((content, Vec::new()))
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

/// Await a memory-backend call, mapping any error to the shared
/// `"{name} failed: {e}"` format, then render its `ExplorationResult` — the
/// one place the six memory-tool branches' identical
/// call/map_err/render shape lives.
async fn call_and_render<E: std::fmt::Display>(
    name: &str,
    fut: impl std::future::Future<Output = Result<ExplorationResult, E>>,
) -> Result<(String, Vec<ExplorationFinding>), String> {
    let res = fut.await.map_err(|e| format!("{name} failed: {e}"))?;
    Ok(render_result(res))
}

fn parse_args<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, String> {
    serde_json::from_str(json).map_err(|e| format!("invalid arguments: {e}"))
}

fn snippet_target(args: GetCodeSnippetArgs) -> Result<SnippetTarget, String> {
    if let Some(name) = args.qualified_name {
        Ok(SnippetTarget::QualifiedName(name))
    } else if let Some(file) = args.file {
        Ok(SnippetTarget::FileRange {
            file: PathBuf::from(file),
            start_line: args.start_line,
            end_line: args.end_line,
        })
    } else {
        Err("get_code_snippet requires either qualified_name or file".to_string())
    }
}

/// Reject a model-supplied `scope` that is absolute or escapes the repository
/// root via a `..` component, mirroring `read_file`'s guard below — a
/// filesystem-backed `SearchBackend` must not be handed a path that walks
/// outside `repo_root`.
fn validate_scope(scope: Option<&str>) -> Result<Option<PathBuf>, String> {
    let Some(scope) = scope else {
        return Ok(None);
    };
    let rel = Path::new(scope);
    if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("scope `{scope}` escapes the repository root"));
    }
    Ok(Some(rel.to_path_buf()))
}

fn read_file(
    repo_root: &Path,
    path: &str,
    start_line: Option<u32>,
    end_line: Option<u32>,
) -> Result<String, String> {
    let rel = Path::new(path);
    if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!(
            "read_file path `{path}` escapes the repository root"
        ));
    }
    let full = repo_root.join(rel);
    let canonical_full =
        std::fs::canonicalize(&full).map_err(|e| format!("read_file failed for `{path}`: {e}"))?;
    let canonical_root = std::fs::canonicalize(repo_root)
        .map_err(|e| format!("read_file failed for `{path}`: {e}"))?;
    if !canonical_full.starts_with(&canonical_root) {
        return Err(format!(
            "read_file path `{path}` escapes the repository root"
        ));
    }
    let contents = std::fs::read_to_string(&canonical_full)
        .map_err(|e| format!("read_file failed for `{path}`: {e}"))?;
    Ok(slice_lines(&contents, start_line, end_line))
}

/// Slice `contents` to the 1-based inclusive `[start_line, end_line]` window.
/// With neither bound, return the whole file unchanged.
fn slice_lines(contents: &str, start_line: Option<u32>, end_line: Option<u32>) -> String {
    if start_line.is_none() && end_line.is_none() {
        return contents.to_string();
    }
    let start = start_line.unwrap_or(1).max(1);
    let end = end_line.unwrap_or(u32::MAX);
    contents
        .lines()
        .enumerate()
        .filter(|(i, _)| {
            let ln = (*i as u32) + 1;
            ln >= start && ln <= end
        })
        .map(|(_, l)| l)
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Serialize)]
struct FindingDto {
    path: String,
    line_start: u32,
    line_end: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Serialize)]
struct ResultDto {
    findings: Vec<FindingDto>,
    summary: String,
}

fn finding_dto(f: &ExplorationFinding) -> FindingDto {
    FindingDto {
        path: f.location.path.display().to_string(),
        line_start: f.location.line_start,
        line_end: f.location.line_end,
        snippet: f.snippet.clone(),
        note: f.note.clone(),
    }
}

/// Serialize `value`, falling back to `fallback` (a literal empty-JSON shape,
/// e.g. `"{}"`/`"[]"`) on the practically-unreachable serialize failure — the
/// one place that serialize-or-empty-fallback pattern lives.
fn serialize_or_empty<T: Serialize>(value: &T, fallback: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| fallback.to_string())
}

fn render_result(res: ExplorationResult) -> (String, Vec<ExplorationFinding>) {
    let dto = ResultDto {
        findings: res.findings.iter().map(finding_dto).collect(),
        summary: res.summary,
    };
    let content = serialize_or_empty(&dto, "{}");
    (content, res.findings)
}

fn render_findings(findings: Vec<ExplorationFinding>) -> (String, Vec<ExplorationFinding>) {
    let dtos: Vec<FindingDto> = findings.iter().map(finding_dto).collect();
    let content = serialize_or_empty(&dtos, "[]");
    (content, findings)
}

fn tool_message(call_id: &str, content: String) -> Message {
    Message::tool(call_id, content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::FileLocation;
    use repo_explorer_core::llm::Role;
    use repo_explorer_core::memory::IndexStatus;
    use repo_explorer_core::memory::mock::{Call as MemCall, MockMemoryBackend};
    use repo_explorer_core::search::mock::{Call as SearchCall, MockSearchBackend};
    use std::path::PathBuf;

    fn call(id: &str, name: &str, args: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments_json: args.to_string(),
        }
    }

    fn finding(path: &str) -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: 1,
                line_end: 5,
            },
            snippet: None,
            note: None,
        }
    }

    #[tokio::test]
    async fn search_code_dispatches_to_memory_with_parsed_args() {
        let memory = MockMemoryBackend::new().with_search_code_result(Ok(ExplorationResult {
            findings: vec![finding("src/a.rs")],
            summary: "mem".to_string(),
        }));
        let search = MockSearchBackend::new();
        let root = PathBuf::from("/repo");
        let c = call(
            "c1",
            "search_code",
            r#"{"query":"main","scope_hint":"src","max_results":7}"#,
        );
        let (message, findings) = dispatch_call(&memory, &search, &root, &c).await;

        assert_eq!(message.role, Role::Tool);
        assert_eq!(message.tool_call_id.as_deref(), Some("c1"));
        assert!(message.content.contains("src/a.rs"));
        assert_eq!(findings, vec![finding("src/a.rs")]);

        let calls = memory.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            MemCall::SearchCode {
                repo_root: PathBuf::from("/repo"),
                query: ExplorationQuery {
                    text: "main".to_string(),
                    scope_hint: Some(PathBuf::from("src")),
                    max_results: Some(7),
                },
            }
        );
    }

    #[tokio::test]
    async fn grep_routes_to_search_with_content_mode_and_repo_root_from_loop() {
        let memory = MockMemoryBackend::new();
        let search = MockSearchBackend::new().with_search_result(Ok(vec![finding("src/b.rs")]));
        let root = PathBuf::from("/repo");
        let c = call("c2", "grep", r#"{"pattern":"fn main","scope":"src"}"#);
        let (message, findings) = dispatch_call(&memory, &search, &root, &c).await;

        assert_eq!(message.tool_call_id.as_deref(), Some("c2"));
        assert_eq!(findings, vec![finding("src/b.rs")]);
        let calls = search.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            SearchCall::Search {
                repo_root: PathBuf::from("/repo"),
                pattern: "fn main".to_string(),
                scope: Some(PathBuf::from("src")),
                options: SearchOptions {
                    max_results: None,
                    ..SearchOptions::default()
                },
            }
        );
    }

    #[tokio::test]
    async fn find_routes_to_search_with_file_glob() {
        let search = MockSearchBackend::new();
        let root = PathBuf::from("/repo");
        let c = call("c3", "find", r#"{"pattern":"*.rs"}"#);
        let _ = dispatch_call(&MockMemoryBackend::new(), &search, &root, &c).await;
        match &search.calls()[0] {
            SearchCall::Search {
                pattern,
                options,
                scope,
                ..
            } => {
                assert_eq!(pattern, ".");
                assert_eq!(options.file_glob.as_deref(), Some("*.rs"));
                assert_eq!(scope, &None);
            }
        }
    }

    #[tokio::test]
    async fn get_code_snippet_qualified_name_takes_precedence() {
        let memory = MockMemoryBackend::new();
        let root = PathBuf::from("/repo");
        let c = call(
            "c4",
            "get_code_snippet",
            r#"{"qualified_name":"a::b","file":"x.rs"}"#,
        );
        let _ = dispatch_call(&memory, &MockSearchBackend::new(), &root, &c).await;
        assert_eq!(
            memory.calls()[0],
            MemCall::GetCodeSnippet {
                repo_root: PathBuf::from("/repo"),
                target: SnippetTarget::QualifiedName("a::b".to_string()),
            }
        );
    }

    #[tokio::test]
    async fn get_code_snippet_without_either_is_error_message() {
        let memory = MockMemoryBackend::new();
        let root = PathBuf::from("/repo");
        let c = call("c5", "get_code_snippet", r#"{}"#);
        let (message, findings) =
            dispatch_call(&memory, &MockSearchBackend::new(), &root, &c).await;
        assert_eq!(message.role, Role::Tool);
        assert!(message.content.contains("qualified_name") || message.content.contains("file"));
        assert!(findings.is_empty());
        // No backend call was made.
        assert!(memory.calls().is_empty());
    }

    #[tokio::test]
    async fn malformed_arguments_produce_error_message_not_panic() {
        let memory = MockMemoryBackend::new();
        let root = PathBuf::from("/repo");
        let c = call("c6", "search_code", r#"{"not_query":1}"#);
        let (message, findings) =
            dispatch_call(&memory, &MockSearchBackend::new(), &root, &c).await;
        assert_eq!(message.tool_call_id.as_deref(), Some("c6"));
        assert!(message.content.contains("invalid arguments"));
        assert!(findings.is_empty());
        assert!(memory.calls().is_empty());
    }

    #[tokio::test]
    async fn unknown_tool_is_error_message() {
        let (message, _) = dispatch_call(
            &MockMemoryBackend::new(),
            &MockSearchBackend::new(),
            &PathBuf::from("/repo"),
            &call("c7", "nonesuch", r#"{}"#),
        )
        .await;
        assert!(message.content.contains("unknown tool: nonesuch"));
    }

    #[tokio::test]
    async fn backend_error_becomes_tool_message_not_fatal() {
        let memory = MockMemoryBackend::new().with_search_code_result(Err(
            repo_explorer_core::memory::MemoryError::ToolFailed {
                tool: "search_code",
                message: "boom".to_string(),
            },
        ));
        let root = PathBuf::from("/repo");
        let c = call("c8", "search_code", r#"{"query":"x"}"#);
        let (message, findings) =
            dispatch_call(&memory, &MockSearchBackend::new(), &root, &c).await;
        assert!(message.content.contains("search_code failed"));
        assert!(message.content.contains("boom"));
        assert!(findings.is_empty());
    }

    #[tokio::test]
    async fn read_file_reads_range_within_repo_root() {
        let dir = std::env::temp_dir().join(format!("agent_dispatch_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "l1\nl2\nl3\nl4\n").unwrap();
        let c = call(
            "c9",
            "read_file",
            r#"{"path":"f.txt","start_line":2,"end_line":3}"#,
        );
        let (message, findings) = dispatch_call(
            &MockMemoryBackend::new(),
            &MockSearchBackend::new(),
            &dir,
            &c,
        )
        .await;
        assert_eq!(message.content, "l2\nl3");
        assert!(findings.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_file_rejects_parent_traversal() {
        let c = call("c10", "read_file", r#"{"path":"../secret.txt"}"#);
        let (message, _) = dispatch_call(
            &MockMemoryBackend::new(),
            &MockSearchBackend::new(),
            &PathBuf::from("/repo"),
            &c,
        )
        .await;
        assert!(message.content.contains("escapes the repository root"));
    }

    // Silence unused-import warnings for items only used in some cfgs.
    #[allow(unused_imports)]
    use IndexStatus as _IndexStatus;
}
