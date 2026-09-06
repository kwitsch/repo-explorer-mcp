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
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
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
    /// Free-text exploration request.
    query: String,
    /// Optional path prefix to restrict the search to.
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
fn build_req_id(query: &ExplorationQuery, counter: &AtomicU64) -> String {
    let key = Agent::query_cache_key(query);
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

/// Every `explore_repository` boundary check, in order — the single place
/// new request-shape validation (one per discovered bug so far: F-04, F-09)
/// gets added, instead of each check's caller re-pasting its own
/// log-and-return wrapper.
fn validate_request(req: &ExploreRepositoryRequest) -> Result<(), String> {
    reject_blank_query(&req.query)?;
    reject_zero_max_results(req.max_results)?;
    Ok(())
}

/// The MCP server handler: a shared `Arc<Agent>` plus the repo root to explore.
#[derive(Clone)]
pub struct RepoExplorerServer {
    tool_router: ToolRouter<Self>,
    agent: Arc<Agent>,
    repo_root: Arc<PathBuf>,
    /// Per-process counter feeding [`build_req_id`]; shared (not per-clone)
    /// so every `RepoExplorerServer` clone contributes to one sequence.
    req_counter: Arc<AtomicU64>,
}

#[tool_router]
impl RepoExplorerServer {
    pub fn new(agent: Arc<Agent>, repo_root: PathBuf) -> Self {
        Self {
            tool_router: Self::tool_router(),
            agent,
            repo_root: Arc::new(repo_root),
            req_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Explore the repository and return structured findings plus a summary.
    #[tool(
        name = "explore_repository",
        description = "Explore the repository for the given request and return \
                       matching file locations (path always present; line \
                       numbers included when resolvable, omitted entirely for \
                       an unresolved/symbol-only match, plus optional \
                       snippet/context) plus a summary. Args: query (required), \
                       optional scope_hint (path prefix), optional max_results."
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
        let query = ExplorationQuery {
            text: req.query,
            scope_hint: req.scope_hint.map(PathBuf::from),
            max_results: req.max_results,
        };
        let req_id = build_req_id(&query, &self.req_counter);
        let span = tracing::info_span!("explore", req_id = %req_id);
        let result = self
            .agent
            .run(self.repo_root.as_ref(), &query)
            .instrument(span)
            .await;
        match result {
            Ok(result) => Ok(Json(ExplorationResultDto::from(result))),
            Err(e) => {
                tracing::warn!(error_class = "provider", message = %e, "exploration failed");
                Err(e.to_string())
            }
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for RepoExplorerServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Repository exploration server. Call `explore_repository` with a \
             free-text query to receive structured findings and a summary.",
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
        let req: ExploreRepositoryRequest =
            serde_json::from_str(r#"{"query":"where is main"}"#).expect("minimal request");
        assert_eq!(req.query, "where is main");
        assert!(req.scope_hint.is_none());
        assert!(req.max_results.is_none());
    }

    #[test]
    fn deserializes_full_request() {
        let req: ExploreRepositoryRequest =
            serde_json::from_str(r#"{"query":"q","scope_hint":"src","max_results":5}"#)
                .expect("full request");
        assert_eq!(req.query, "q");
        assert_eq!(req.scope_hint.as_deref(), Some("src"));
        assert_eq!(req.max_results, Some(5));
    }

    #[test]
    fn rejects_unknown_field() {
        let err = serde_json::from_str::<ExploreRepositoryRequest>(r#"{"query":"q","bogus":true}"#);
        assert!(err.is_err());
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
    fn validate_request_runs_both_boundary_checks() {
        let ok: ExploreRepositoryRequest =
            serde_json::from_str(r#"{"query":"q","max_results":5}"#).unwrap();
        assert!(validate_request(&ok).is_ok());

        let blank: ExploreRepositoryRequest =
            serde_json::from_str(r#"{"query":"  ","max_results":5}"#).unwrap();
        assert!(validate_request(&blank).is_err());

        let zero: ExploreRepositoryRequest =
            serde_json::from_str(r#"{"query":"q","max_results":0}"#).unwrap();
        assert!(validate_request(&zero).is_err());
    }
}
