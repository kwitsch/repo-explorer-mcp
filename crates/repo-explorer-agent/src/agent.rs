//! The `AgentLoop`: a generic, transport-free exploration loop. One turn = one
//! `router.complete_with_tools` call; `AgentConfig.max_iterations` bounds turns.
//! Tool/backend failures and malformed model output degrade into `Role::Tool`
//! messages fed back to the model; only a `RouterError` is a hard failure.

use repo_explorer_core::domain::{ExplorationFinding, ExplorationQuery, ExplorationResult};
use repo_explorer_core::llm::{
    Clock, LlmProvider, Message, ProviderResponse, ProviderRouter, SystemClock,
};
use repo_explorer_core::memory::{IndexStatus, MemoryBackend};
use repo_explorer_core::search::SearchBackend;
use std::path::Path;

use crate::dispatch::dispatch_call;
use crate::tools::{parse_finish, tool_catalog};

/// The only hard-failure mode: the provider router could not produce a response
/// (all providers exhausted, or a non-failover provider error). Flattened to a
/// `String` to stay comparable, matching the crate-boundary convention.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum AgentLoopError {
    #[error("llm provider error: {0}")]
    Provider(String),
}

/// Loop configuration. `max_iterations` bounds the number of provider turns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    pub max_iterations: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_iterations: 24 }
    }
}

/// The generic exploration loop. Owns `memory`, `search`, and the `router`
/// (which owns its providers) — static dispatch, mirroring `ProviderRouter`.
pub struct AgentLoop<M, S, P, C = SystemClock>
where
    M: MemoryBackend,
    S: SearchBackend,
    P: LlmProvider,
    C: Clock,
{
    memory: M,
    search: S,
    router: ProviderRouter<P, C>,
    config: AgentConfig,
}

impl<M, S, P, C> AgentLoop<M, S, P, C>
where
    M: MemoryBackend,
    S: SearchBackend,
    P: LlmProvider,
    C: Clock,
{
    pub fn new(memory: M, search: S, router: ProviderRouter<P, C>, config: AgentConfig) -> Self {
        Self {
            memory,
            search,
            router,
            config,
        }
    }

    pub async fn run(
        &self,
        repo_root: &Path,
        query: &ExplorationQuery,
    ) -> Result<ExplorationResult, AgentLoopError> {
        let tools = tool_catalog();

        // Step 1: ensure a fresh index (once). Failures are non-fatal notes.
        let index_note = match self.memory.ensure_fresh_index(repo_root).await {
            Ok(IndexStatus::Reindexed) | Ok(IndexStatus::UpToDate) => None,
            Ok(IndexStatus::IndexingFailed { reason }) => Some(format!(
                "Note: the memory index could not be refreshed ({reason}); memory results may be stale."
            )),
            Err(e) => Some(format!(
                "Note: the memory backend is unavailable ({e}); rely on the grep/find/read_file search tools."
            )),
        };

        // Step 2: seed messages.
        let mut messages: Vec<Message> = vec![
            Message::system(system_prompt(index_note.as_deref())),
            Message::user(user_prompt(query)),
        ];

        let mut findings: Vec<ExplorationFinding> = Vec::new();

        // Step 3: turn loop.
        for _turn in 0..self.config.max_iterations {
            match self.router.complete_with_tools(&messages, &tools).await {
                Ok(ProviderResponse::ToolCalls(calls)) if calls.is_empty() => {
                    messages.push(Message::assistant_tool_calls(Vec::new()));
                    messages.push(Message::user(
                        "You must respond with a tool call; call finish when done.",
                    ));
                }
                Ok(ProviderResponse::ToolCalls(calls)) => {
                    messages.push(Message::assistant_tool_calls(calls.clone()));
                    for call in &calls {
                        if call.name == "finish" {
                            match parse_finish(&call.arguments_json) {
                                Ok(result) => return Ok(result),
                                Err(reason) => {
                                    messages.push(Message::tool(
                                        call.id.clone(),
                                        format!(
                                            "finish rejected: {reason}; fix the arguments and call finish again"
                                        ),
                                    ));
                                }
                            }
                        } else {
                            let (message, new_findings) =
                                dispatch_call(&self.memory, &self.search, repo_root, call).await;
                            messages.push(message);
                            for f in new_findings {
                                accumulate(&mut findings, f);
                            }
                        }
                    }
                }
                Ok(ProviderResponse::Text(text)) => {
                    messages.push(Message::assistant_text(text));
                    messages.push(Message::user(
                        "You must respond with a tool call; call finish when done.",
                    ));
                }
                Err(router_err) => {
                    return Err(AgentLoopError::Provider(router_err.to_string()));
                }
            }
        }

        // Step 4: iteration limit reached without finish -> degraded result.
        Ok(ExplorationResult {
            findings,
            summary: format!(
                "Exploration stopped after reaching the iteration limit ({}) without an explicit finish; returning best-effort findings gathered so far.",
                self.config.max_iterations
            ),
        })
    }
}

/// Push `f` unless a finding with the same `FileLocation` is already present
/// (dedupe by location; first-seen snippet/note wins).
fn accumulate(findings: &mut Vec<ExplorationFinding>, f: ExplorationFinding) {
    if !findings
        .iter()
        .any(|existing| existing.location == f.location)
    {
        findings.push(f);
    }
}

fn system_prompt(index_note: Option<&str>) -> String {
    let mut s = String::from(
        "You are a repository exploration agent. Use the provided tools to locate the code relevant to the user's query. \
The memory tools (search_code, search_graph, query_graph, trace_path, get_architecture, get_code_snippet) are primary and authoritative — prefer them first. \
The grep, find, and read_file tools are a supplement/fallback, to be used only when the memory tools are insufficient. \
When you have gathered enough information, you MUST conclude by calling the finish tool with your findings and a summary.",
    );
    if let Some(note) = index_note {
        s.push(' ');
        s.push_str(note);
    }
    s
}

fn user_prompt(query: &ExplorationQuery) -> String {
    let mut s = format!("Exploration query: {}", query.text);
    if let Some(scope) = &query.scope_hint {
        s.push_str(&format!("\nScope hint: {}", scope.display()));
    }
    if let Some(max) = query.max_results {
        s.push_str(&format!("\nDesired maximum results: {max}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::llm::ToolCall;
    use repo_explorer_core::llm::mock::{FakeClock, MockLlmProvider};
    use repo_explorer_core::memory::mock::MockMemoryBackend;
    use repo_explorer_core::search::mock::MockSearchBackend;
    use std::path::PathBuf;

    fn finish_call() -> ToolCall {
        ToolCall {
            id: "c1".to_string(),
            name: "finish".to_string(),
            arguments_json:
                r#"{"findings":[{"location":{"path":"src/lib.rs","line_start":1,"line_end":2},"note":"here"}],"summary":"done"}"#
                    .to_string(),
        }
    }

    #[tokio::test]
    async fn immediate_finish_returns_its_payload() {
        let provider = MockLlmProvider::new()
            .with_responses(vec![Ok(ProviderResponse::ToolCalls(vec![finish_call()]))]);
        let router = ProviderRouter::with_clock(
            vec![("primary".to_string(), provider)],
            60,
            FakeClock::new(),
        );
        let agent = AgentLoop::new(
            MockMemoryBackend::new(),
            MockSearchBackend::new(),
            router,
            AgentConfig::default(),
        );
        let query = ExplorationQuery {
            text: "where is main".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let got = agent.run(&PathBuf::from("/repo"), &query).await.unwrap();
        assert_eq!(got.summary, "done");
        assert_eq!(got.findings.len(), 1);
        assert_eq!(got.findings[0].location.line_start, 1);
    }

    #[tokio::test]
    async fn router_error_is_hard_fail() {
        // Empty provider list -> RouterError::NoProviders on the first turn.
        let router: ProviderRouter<MockLlmProvider, FakeClock> =
            ProviderRouter::with_clock(vec![], 60, FakeClock::new());
        let agent = AgentLoop::new(
            MockMemoryBackend::new(),
            MockSearchBackend::new(),
            router,
            AgentConfig::default(),
        );
        let query = ExplorationQuery {
            text: "x".to_string(),
            scope_hint: None,
            max_results: None,
        };
        let got = agent.run(&PathBuf::from("/repo"), &query).await;
        assert!(matches!(got, Err(AgentLoopError::Provider(_))));
    }

    #[test]
    fn agent_loop_error_is_comparable() {
        let a = AgentLoopError::Provider("x".to_string());
        assert_eq!(a, AgentLoopError::Provider("x".to_string()));
        assert_eq!(a.to_string(), "llm provider error: x");
    }
}
