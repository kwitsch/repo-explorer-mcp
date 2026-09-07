//! MCP server handler exposing the single `explore_repository` tool, plus the
//! serde/schemars request/response DTOs and their mapping to/from the pure,
//! serde-free `repo-explorer-core` domain types. Keeping the DTOs here (not on
//! core) preserves the one-impure-dependency-per-crate convention.

use repo_explorer_agent::AgentLoop;
use repo_explorer_core::domain::{ExplorationQuery, ExplorationResult};
use repo_explorer_core::llm::SystemClock;
use repo_explorer_core::retrieval::is_unknown_location;
use repo_explorer_llm::GenaiProvider;
use repo_explorer_memory::MemoryClientBackend;
use repo_explorer_search::{CliSearchBackend, GitStateProbe};
use rmcp::{
    Json, ServerHandler,
    handler::server::{
        router::{prompt::PromptRouter, tool::ToolRouter},
        wrapper::Parameters,
    },
    model::{PromptMessage, Role, ServerCapabilities, ServerInfo},
    prompt, prompt_handler, prompt_router, tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::Instrument;

/// The concrete agent type wired to the production backends. All backend trait
/// methods and `AgentLoop::run` take `&self`, so a shared `Arc<Agent>` (no
/// `Mutex`) supports concurrent tool calls.
pub type Agent =
    AgentLoop<MemoryClientBackend, CliSearchBackend, GenaiProvider, GitStateProbe, SystemClock>;

/// Input schema for `explore_repository`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct ExploreRepositoryRequest {
    /// Absolute path to the repository's base directory to explore. Required.
    /// `scope_hint` is interpreted relative to this path.
    repo_path: String,
    /// Free-text search request in English only. Include the exact
    /// identifier, symbol, or file path as it appears in the code (e.g. a
    /// snake_case or camelCase name) for the fastest, most precise results.
    query: String,
    /// Optional path prefix (relative to the repo root) to restrict the
    /// search.
    #[serde(default)]
    scope_hint: Option<String>,
    /// Optional cap on the number of findings.
    #[serde(default)]
    max_results: Option<u32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct FileLocationDto {
    path: String,
    /// Omitted (rather than a misleading `0`) when the underlying location is
    /// core's "unknown" sentinel — see `is_unknown_location`.
    #[serde(skip_serializing_if = "Option::is_none")]
    line_start: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line_end: Option<u32>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct ExplorationFindingDto {
    location: FileLocationDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
struct ExplorationResultDto {
    findings: Vec<ExplorationFindingDto>,
    summary: String,
}

impl From<ExplorationResult> for ExplorationResultDto {
    fn from(result: ExplorationResult) -> Self {
        Self {
            findings: result
                .findings
                .into_iter()
                .map(|f| {
                    let known = !is_unknown_location(&f.location);
                    ExplorationFindingDto {
                        location: FileLocationDto {
                            // Reuses the existing buffer for valid UTF-8 paths; only
                            // non-UTF-8 paths pay for the lossy allocation.
                            path: f
                                .location
                                .path
                                .into_os_string()
                                .into_string()
                                .unwrap_or_else(|s| s.to_string_lossy().into_owned()),
                            line_start: known.then_some(f.location.line_start),
                            line_end: known.then_some(f.location.line_end),
                        },
                        snippet: f.snippet,
                        note: f.note,
                    }
                })
                .collect(),
            summary: result.summary,
        }
    }
}

/// Build a short per-call correlation id — the first 8 hex chars of a
/// SHA-256 over the same normalized query cache key `AgentLoop` uses
/// internally, plus a per-process monotonically increasing counter — so
/// every log line for one call (including concurrent calls) can be
/// correlated by an external harness without fragile timestamp-slicing.
fn build_req_id(
    repo_path: &std::path::Path,
    query: &ExplorationQuery,
    counter: &AtomicU64,
) -> String {
    let key = Agent::query_cache_key(repo_path, query);
    let hash = hex::encode(Sha256::digest(key.as_bytes()));
    let n = counter.fetch_add(1, Ordering::Relaxed);
    format!("{}-{n}", &hash[..8])
}

/// Reject a query that is empty or whitespace-only before it reaches the
/// agent loop. Such a query derives zero retrieval patterns and would burn
/// the full LLM fallback budget on an effectively empty prompt with no
/// signal to act on (F-04). Boundary-level check: constructs no
/// `ExplorationQuery` and does no agent work.
fn reject_blank_query(query: &str) -> Result<(), String> {
    if query.trim().is_empty() {
        return Err("query must not be empty".to_string());
    }
    Ok(())
}

/// Reject `max_results: 0` before it reaches the agent loop. `0` truncates
/// `findings` to an empty list (F-09) while the model-authored `summary`
/// passes through unchanged, so the response can describe findings that
/// aren't there. `max_results` has no documented "return nothing" mode —
/// `None` already means unlimited — so `0` is an unintended edge case, not a
/// valid request, and is rejected the same way as a blank query.
fn reject_zero_max_results(max_results: Option<u32>) -> Result<(), String> {
    if max_results == Some(0) {
        return Err("max_results must be greater than 0; omit it for unlimited".to_string());
    }
    Ok(())
}

/// Reject a `repo_path` that is not an absolute path to an existing directory
/// before it reaches the agent loop (F-22). Absolute-only: a relative path is
/// rejected outright — the long-lived server has only an incidental cwd, so
/// resolving relative paths against it is fragile and surprising, and with the
/// server-level root removed there is no fence-root to resolve against. A
/// missing/typo'd/relative/file-not-dir path would otherwise burn a
/// git-fingerprint probe and a query-cache lookup against a bogus root before
/// failing deep inside `ensure_fresh_index`. Boundary-level check: constructs
/// no `ExplorationQuery` and does no agent work. Duplicates (does not replace)
/// the deeper async check as defense-in-depth against TOCTOU.
fn reject_invalid_repo_path(repo_path: &str) -> Result<(), String> {
    let path = std::path::Path::new(repo_path);
    if !path.is_absolute() {
        return Err(format!("repo_path must be an absolute path: {repo_path}"));
    }
    if !path.is_dir() {
        return Err(format!(
            "repo_path is not an existing directory: {repo_path}"
        ));
    }
    Ok(())
}

/// Every `explore_repository` boundary check, in order — the single place
/// new request-shape validation (one per discovered bug so far: F-04, F-09,
/// F-22) gets added, instead of each check's caller re-pasting its own
/// log-and-return wrapper.
fn validate_request(req: &ExploreRepositoryRequest) -> Result<(), String> {
    reject_blank_query(&req.query)?;
    reject_zero_max_results(req.max_results)?;
    reject_invalid_repo_path(&req.repo_path)?;
    Ok(())
}

/// The MCP server handler: a shared `Arc<Agent>`.
#[derive(Clone)]
pub struct RepoExplorerServer {
    tool_router: ToolRouter<Self>,
    prompt_router: PromptRouter<Self>,
    agent: Arc<Agent>,
    /// Per-process counter feeding [`build_req_id`]; shared (not per-clone)
    /// so every `RepoExplorerServer` clone contributes to one sequence.
    req_counter: Arc<AtomicU64>,
}

/// Arguments for the `find-definition` and `bare-symbol` example prompts.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct SymbolPromptArgs {
    /// The exact code identifier or symbol, written as it appears in the
    /// source (snake_case, camelCase, or PascalCase), in English.
    symbol: String,
}

/// Arguments for the `locate-at-line` example prompt.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct LocateAtLineArgs {
    /// Repo-relative file path, written exactly as it appears in the source.
    path: String,
    /// 1-based line number within `path`, as sent by the MCP client (prompt
    /// arguments are always strings on the wire per the MCP spec).
    line: String,
    /// The exact code identifier or symbol at that location, in English.
    symbol: String,
}

#[tool_router]
#[prompt_router]
impl RepoExplorerServer {
    pub fn new(agent: Arc<Agent>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            prompt_router: Self::prompt_router(),
            agent,
            req_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Explore the repository and return structured findings plus a summary.
    #[tool(
        name = "explore_repository",
        description = "This server handles English-language requests only; \
                       send the query in English. Explore the repository for \
                       the given request and return matching file locations \
                       (path always present; line numbers included when \
                       resolvable, omitted entirely for an \
                       unresolved/symbol-only match, plus optional \
                       snippet/context) plus a summary. Phrase the query with \
                       the exact code identifier, symbol, or file path you are \
                       looking for. Args: repo_path (required, absolute path to \
                       the repository base directory), query (required), optional \
                       scope_hint (path prefix, relative to repo_path), optional \
                       max_results."
    )]
    async fn explore_repository(
        &self,
        params: Parameters<ExploreRepositoryRequest>,
    ) -> Result<Json<ExplorationResultDto>, String> {
        let req = params.0;
        if let Err(e) = validate_request(&req) {
            tracing::warn!(error_class = "validation", message = %e, "exploration rejected");
            return Err(e);
        }
        let repo_path = PathBuf::from(&req.repo_path);
        let query = ExplorationQuery {
            text: req.query,
            scope_hint: req.scope_hint.map(PathBuf::from),
            max_results: req.max_results,
        };
        let req_id = build_req_id(&repo_path, &query, &self.req_counter);
        let span = tracing::info_span!("explore", req_id = %req_id);
        let result = self.agent.run(&repo_path, &query).instrument(span).await;
        match result {
            Ok(result) => Ok(Json(ExplorationResultDto::from(result))),
            Err(e) => {
                tracing::warn!(error_class = "provider", message = %e, "exploration failed");
                Err(e.to_string())
            }
        }
    }

    /// Example prompt: locate a symbol's definition. Phrase queries in
    /// English and use the exact code identifier as written in the source.
    #[prompt(
        name = "find-definition",
        description = "Locate where a symbol is defined. Supply the exact \
                       code identifier as written in the source; the query \
                       is phrased in English."
    )]
    async fn find_definition_prompt(
        &self,
        Parameters(args): Parameters<SymbolPromptArgs>,
    ) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            Role::User,
            format!("where is `{}` defined", args.symbol),
        )]
    }

    /// Example prompt: search for a bare exact symbol. Phrase queries in
    /// English and use the exact code identifier as written in the source.
    #[prompt(
        name = "bare-symbol",
        description = "Search for a bare exact symbol with no surrounding \
                       prose. Supply the exact code identifier as written in \
                       the source."
    )]
    async fn bare_symbol_prompt(
        &self,
        Parameters(args): Parameters<SymbolPromptArgs>,
    ) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(Role::User, args.symbol)]
    }

    /// Example prompt: ask about a symbol at a specific path and line.
    /// Phrase queries in English and use the exact code identifier and path
    /// as written in the source.
    #[prompt(
        name = "locate-at-line",
        description = "Ask what a symbol at a specific path and line does. \
                       Supply the exact path and code identifier as written \
                       in the source; the query is phrased in English."
    )]
    async fn locate_at_line_prompt(
        &self,
        Parameters(args): Parameters<LocateAtLineArgs>,
    ) -> Vec<PromptMessage> {
        vec![PromptMessage::new_text(
            Role::User,
            format!("{}:{} what does {} do", args.path, args.line, args.symbol),
        )]
    }
}

#[tool_handler(router = self.tool_router)]
#[prompt_handler(router = self.prompt_router)]
impl ServerHandler for RepoExplorerServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .build(),
        )
        .with_instructions(
            "Repository exploration server. Handles English-language requests \
             only — send queries in English. Call `explore_repository` with a \
             free-text query (best results when it names an exact identifier, \
             symbol, or file path) to receive structured findings and a \
             summary. See the listed prompts for example query phrasings.",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::{ExplorationFinding, FileLocation};

    #[test]
    fn maps_result_to_dto_json_shape() {
        let result = ExplorationResult {
            findings: vec![
                ExplorationFinding {
                    location: FileLocation {
                        path: PathBuf::from("src").join("lib.rs"),
                        line_start: 10,
                        line_end: 20,
                    },
                    snippet: Some("fn main() {}".to_string()),
                    note: Some("entry".to_string()),
                },
                ExplorationFinding {
                    location: FileLocation {
                        path: PathBuf::from("src/other.rs"),
                        line_start: 1,
                        line_end: 1,
                    },
                    snippet: None,
                    note: None,
                },
            ],
            summary: "two findings".to_string(),
        };

        let dto = ExplorationResultDto::from(result);
        let value = serde_json::to_value(&dto).expect("serialize dto");

        assert_eq!(value["summary"], "two findings");
        let findings = value["findings"].as_array().expect("findings array");
        assert_eq!(findings.len(), 2);

        assert_eq!(findings[0]["location"]["line_start"], 10);
        assert_eq!(findings[0]["location"]["line_end"], 20);
        assert_eq!(findings[0]["snippet"], "fn main() {}");
        assert_eq!(findings[0]["note"], "entry");
        assert!(
            findings[0]["location"]["path"]
                .as_str()
                .expect("path string")
                .contains("lib.rs")
        );

        // snippet/note omitted when None.
        assert!(findings[1].get("snippet").is_none());
        assert!(findings[1].get("note").is_none());
    }

    #[test]
    fn omits_line_numbers_for_unknown_location() {
        // Core's (0, 0) sentinel (e.g. a symbol row with no resolvable line)
        // must never be serialized as a literal `line_start: 0` — that reads
        // as real data instead of "unknown".
        let result = ExplorationResult {
            findings: vec![ExplorationFinding {
                location: FileLocation {
                    path: PathBuf::from("src/lib.rs"),
                    line_start: 0,
                    line_end: 0,
                },
                snippet: None,
                note: Some("exact symbol match: `Foo`".to_string()),
            }],
            summary: "one finding".to_string(),
        };

        let dto = ExplorationResultDto::from(result);
        let value = serde_json::to_value(&dto).expect("serialize dto");

        let location = &value["findings"][0]["location"];
        assert!(location.get("line_start").is_none());
        assert!(location.get("line_end").is_none());
        assert!(
            location["path"]
                .as_str()
                .expect("path string")
                .contains("lib.rs")
        );
    }

    #[test]
    fn deserializes_minimal_request() {
        let json = format!(
            r#"{{"repo_path":{:?},"query":"where is main"}}"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let req: ExploreRepositoryRequest = serde_json::from_str(&json).expect("minimal request");
        assert_eq!(req.query, "where is main");
        assert_eq!(req.repo_path, env!("CARGO_MANIFEST_DIR"));
        assert!(req.scope_hint.is_none());
        assert!(req.max_results.is_none());
    }

    #[test]
    fn deserializes_full_request() {
        let json = format!(
            r#"{{"repo_path":{:?},"query":"q","scope_hint":"src","max_results":5}}"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let req: ExploreRepositoryRequest = serde_json::from_str(&json).expect("full request");
        assert_eq!(req.query, "q");
        assert_eq!(req.scope_hint.as_deref(), Some("src"));
        assert_eq!(req.max_results, Some(5));
    }

    #[test]
    fn rejects_unknown_field() {
        let json = format!(
            r#"{{"repo_path":{:?},"query":"q","bogus":true}}"#,
            env!("CARGO_MANIFEST_DIR")
        );
        let err = serde_json::from_str::<ExploreRepositoryRequest>(&json);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_request_missing_repo_path() {
        // repo_path carries no #[serde(default)], so its absence is a
        // deserialize error (surfaced by rmcp as invalid-params).
        let err = serde_json::from_str::<ExploreRepositoryRequest>(r#"{"query":"q"}"#);
        assert!(err.is_err());
    }

    #[test]
    fn reject_invalid_repo_path_accepts_absolute_existing_dir() {
        assert!(reject_invalid_repo_path(env!("CARGO_MANIFEST_DIR")).is_ok());
    }

    #[test]
    fn reject_invalid_repo_path_rejects_relative_even_if_it_exists() {
        // "src" exists relative to the crate dir, but the absolute-only guard
        // must still reject it — the long-lived server has only an incidental
        // cwd to resolve against.
        let err = reject_invalid_repo_path("src").unwrap_err();
        assert!(
            err.contains("must be an absolute path"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn reject_invalid_repo_path_rejects_absolute_nonexistent() {
        let err = reject_invalid_repo_path("/nonexistent/repo/xyz").unwrap_err();
        assert!(
            err.contains("is not an existing directory"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn reject_invalid_repo_path_messages_are_pinned() {
        assert_eq!(
            reject_invalid_repo_path("src").unwrap_err(),
            "repo_path must be an absolute path: src"
        );
        assert_eq!(
            reject_invalid_repo_path("/nonexistent/repo/xyz").unwrap_err(),
            "repo_path is not an existing directory: /nonexistent/repo/xyz"
        );
    }

    #[test]
    fn rejects_empty_query() {
        assert!(reject_blank_query("").is_err());
    }

    #[test]
    fn rejects_whitespace_only_spaces() {
        assert!(reject_blank_query("   ").is_err());
    }

    #[test]
    fn rejects_whitespace_only_tabs_newlines() {
        assert!(reject_blank_query("\t\n ").is_err());
    }

    #[test]
    fn accepts_normal_query() {
        assert!(reject_blank_query("where is main").is_ok());
    }

    #[test]
    fn accepts_content_surrounded_by_whitespace() {
        // Real content with surrounding whitespace must pass; the guard trims
        // only for the emptiness test and never mutates the value.
        assert!(reject_blank_query("  q  ").is_ok());
    }

    #[test]
    fn blank_query_error_message_is_pinned() {
        assert_eq!(
            reject_blank_query("").unwrap_err(),
            "query must not be empty"
        );
    }

    #[test]
    fn rejects_zero_max_results() {
        assert!(reject_zero_max_results(Some(0)).is_err());
    }

    #[test]
    fn accepts_missing_or_positive_max_results() {
        assert!(reject_zero_max_results(None).is_ok());
        assert!(reject_zero_max_results(Some(1)).is_ok());
        assert!(reject_zero_max_results(Some(5)).is_ok());
    }

    #[test]
    fn zero_max_results_error_message_is_pinned() {
        assert_eq!(
            reject_zero_max_results(Some(0)).unwrap_err(),
            "max_results must be greater than 0; omit it for unlimited"
        );
    }

    #[test]
    fn prompt_router_lists_the_example_prompts() {
        // list_all() returns prompts sorted by name.
        let names: Vec<String> = RepoExplorerServer::prompt_router()
            .list_all()
            .into_iter()
            .map(|p| p.name)
            .collect();
        assert_eq!(names, ["bare-symbol", "find-definition", "locate-at-line"]);
    }

    #[test]
    fn validate_request_runs_all_boundary_checks() {
        let dir = env!("CARGO_MANIFEST_DIR");
        let ok: ExploreRepositoryRequest = serde_json::from_str(&format!(
            r#"{{"repo_path":{dir:?},"query":"q","max_results":5}}"#
        ))
        .unwrap();
        assert!(validate_request(&ok).is_ok());

        let blank: ExploreRepositoryRequest = serde_json::from_str(&format!(
            r#"{{"repo_path":{dir:?},"query":"  ","max_results":5}}"#
        ))
        .unwrap();
        assert!(validate_request(&blank).is_err());

        let zero: ExploreRepositoryRequest = serde_json::from_str(&format!(
            r#"{{"repo_path":{dir:?},"query":"q","max_results":0}}"#
        ))
        .unwrap();
        assert!(validate_request(&zero).is_err());

        let bad_path: ExploreRepositoryRequest =
            serde_json::from_str(r#"{"repo_path":"/nonexistent/xyz","query":"q"}"#).unwrap();
        assert!(validate_request(&bad_path).is_err());
    }
}
