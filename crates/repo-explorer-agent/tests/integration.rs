//! Stage 5 acceptance tests: fake-provider dispatch/assembly, iteration-limit
//! degradation, and mid-exploration failover driven through a real
//! `ProviderRouter` with two `MockLlmProvider`s.

use repo_explorer_agent::{AgentConfig, AgentLoop};
use repo_explorer_core::domain::{
    ExplorationFinding, ExplorationQuery, ExplorationResult, FileLocation,
};
use repo_explorer_core::llm::mock::{FakeClock, MockLlmProvider};
use repo_explorer_core::llm::{ProviderError, ProviderResponse, ProviderRouter, Role, ToolCall};
use repo_explorer_core::memory::mock::{Call as MemCall, MockMemoryBackend};
use repo_explorer_core::search::mock::{Call as SearchCall, MockSearchBackend};
use std::path::PathBuf;

fn tc(id: &str, name: &str, args: &str) -> ToolCall {
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
async fn fake_provider_dispatch_and_assembly() {
    let provider = MockLlmProvider::new().with_responses(vec![
        Ok(ProviderResponse::ToolCalls(vec![tc(
            "c1",
            "search_code",
            r#"{"query":"main","max_results":5}"#,
        )])),
        Ok(ProviderResponse::ToolCalls(vec![tc(
            "c2",
            "grep",
            r#"{"pattern":"fn main","scope":"src"}"#,
        )])),
        Ok(ProviderResponse::ToolCalls(vec![tc(
            "c3",
            "finish",
            r#"{"findings":[{"location":{"path":"src/main.rs","line_start":1,"line_end":3}}],"summary":"found main"}"#,
        )])),
    ]);
    let provider_probe = provider.clone();
    let router = ProviderRouter::with_clock(
        vec![("primary".to_string(), vec![("m".to_string(), provider)])],
        60,
        FakeClock::new(),
    );

    let memory = MockMemoryBackend::new().with_search_code_result(Ok(ExplorationResult {
        findings: vec![finding("src/a.rs")],
        summary: "mem".to_string(),
    }));
    let mem_probe = memory.clone();
    let search = MockSearchBackend::new().with_search_result(Ok(vec![finding("src/b.rs")]));
    let search_probe = search.clone();

    let agent = AgentLoop::new(memory, search, router, AgentConfig::default());
    let query = ExplorationQuery {
        text: "where is main".to_string(),
        scope_hint: None,
        max_results: None,
    };
    let result = agent.run(&PathBuf::from("/repo"), &query).await.unwrap();

    // Returned result equals the finish payload.
    assert_eq!(result.summary, "found main");
    assert_eq!(
        result.findings,
        vec![ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from("src/main.rs"),
                line_start: 1,
                line_end: 3,
            },
            snippet: None,
            note: None,
        }]
    );

    // Memory dispatch: ensure_fresh_index + search_code with parsed args + loop repo_root.
    let mem_calls = mem_probe.calls();
    assert!(mem_calls.iter().any(|c| matches!(
        c,
        MemCall::EnsureFreshIndex { repo_root } if repo_root == &PathBuf::from("/repo")
    )));
    assert!(mem_calls.iter().any(|c| matches!(
        c,
        MemCall::SearchCode { repo_root, query }
            if repo_root == &PathBuf::from("/repo")
                && query.text == "main"
                && query.max_results == Some(5)
    )));

    // Search dispatch: repo_root supplied by the loop, content search.
    let search_calls = search_probe.calls();
    assert_eq!(search_calls.len(), 1);
    match &search_calls[0] {
        SearchCall::Search {
            repo_root,
            pattern,
            scope,
            ..
        } => {
            assert_eq!(repo_root, &PathBuf::from("/repo"));
            assert_eq!(pattern, "fn main");
            assert_eq!(scope, &Some(PathBuf::from("src")));
        }
    }

    // Every turn's tools slice includes finish; Tool messages appended per dispatch.
    let llm_calls = provider_probe.calls();
    assert_eq!(llm_calls.len(), 3);
    for c in &llm_calls {
        assert!(c.tools.iter().any(|t| t.name == "finish"));
    }
    assert!(
        llm_calls[1]
            .messages
            .iter()
            .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c1"))
    );
    assert!(
        llm_calls[2]
            .messages
            .iter()
            .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c2"))
    );
}

#[tokio::test]
async fn iteration_limit_degrades_gracefully() {
    let provider = MockLlmProvider::new().with_fallback(Ok(ProviderResponse::ToolCalls(vec![tc(
        "c",
        "grep",
        r#"{"pattern":"x"}"#,
    )])));
    let provider_probe = provider.clone();
    let router = ProviderRouter::with_clock(
        vec![("primary".to_string(), vec![("m".to_string(), provider)])],
        60,
        FakeClock::new(),
    );
    let search = MockSearchBackend::new().with_search_result(Ok(vec![finding("src/x.rs")]));
    let agent = AgentLoop::new(
        MockMemoryBackend::new(),
        search,
        router,
        AgentConfig { max_iterations: 3 },
    );
    let query = ExplorationQuery {
        text: "q".to_string(),
        scope_hint: None,
        max_results: None,
    };
    let result = agent.run(&PathBuf::from("/repo"), &query).await.unwrap();

    assert!(result.summary.contains("iteration limit"));
    // Same finding returned every turn -> deduped by location to one entry.
    assert_eq!(result.findings, vec![finding("src/x.rs")]);
    assert_eq!(provider_probe.calls().len(), 3);
}

#[tokio::test]
async fn mid_exploration_failover_across_providers() {
    let primary = MockLlmProvider::new().with_responses(vec![
        Ok(ProviderResponse::ToolCalls(vec![tc(
            "c1",
            "search_code",
            r#"{"query":"widget"}"#,
        )])),
        Err(ProviderError::RateLimited {
            provider: "primary".to_string(),
            message: "429".to_string(),
        }),
    ]);
    let secondary = MockLlmProvider::new().with_responses(vec![
        Ok(ProviderResponse::ToolCalls(vec![tc(
            "c2",
            "get_architecture",
            r#"{"depth":2}"#,
        )])),
        Ok(ProviderResponse::ToolCalls(vec![tc(
            "c3",
            "finish",
            r#"{"findings":[],"summary":"done via secondary"}"#,
        )])),
    ]);
    let primary_probe = primary.clone();
    let secondary_probe = secondary.clone();
    let router = ProviderRouter::with_clock(
        vec![
            ("primary".to_string(), vec![("m1".to_string(), primary)]),
            ("secondary".to_string(), vec![("m2".to_string(), secondary)]),
        ],
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
        text: "widget".to_string(),
        scope_hint: None,
        max_results: None,
    };
    let result = agent.run(&PathBuf::from("/repo"), &query).await.unwrap();

    assert_eq!(result.summary, "done via secondary");
    // Turn 1: primary succeeds. Turn 2: primary rate-limited -> secondary arch.
    // Turn 3: primary cooling (clock not advanced) -> secondary finish.
    assert_eq!(primary_probe.calls().len(), 2);
    assert_eq!(secondary_probe.calls().len(), 2);
}
