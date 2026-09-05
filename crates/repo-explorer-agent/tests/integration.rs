//! Acceptance tests for the exploration orchestrator: deterministic early
//! exit, LLM verification, fallback-loop dispatch/assembly, batch enforcement,
//! budgets, mid-exploration failover, and the query cache — all driven through
//! a real `ProviderRouter` with `MockLlmProvider`s.

use repo_explorer_agent::AgentLoop;
use repo_explorer_core::config::{AgentSettings, CacheSettings};
use repo_explorer_core::domain::{
    ExplorationFinding, ExplorationQuery, ExplorationResult, FileLocation,
};
use repo_explorer_core::fingerprint::RepoFingerprint;
use repo_explorer_core::fingerprint::mock::MockRepoStateProbe;
use repo_explorer_core::llm::mock::{FakeClock, MockLlmProvider};
use repo_explorer_core::llm::{
    Completion, ProviderError, ProviderResponse, ProviderRouter, Role, TokenUsage, ToolCall,
};
use repo_explorer_core::memory::mock::{Call as MemCall, MockMemoryBackend};
use repo_explorer_core::search::mock::{Call as SearchCall, MockSearchBackend};
use std::path::PathBuf;

fn tc(id: &str, name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments_json: args.to_string(),
        thought_signatures: None,
    }
}

fn ok_calls(calls: Vec<ToolCall>) -> Result<Completion, ProviderError> {
    Ok(Completion::from(ProviderResponse::ToolCalls(calls)))
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

fn symbol_finding(path: &str, symbol: &str) -> ExplorationFinding {
    ExplorationFinding {
        location: FileLocation {
            path: PathBuf::from(path),
            line_start: 10,
            line_end: 20,
        },
        snippet: Some("fn decide_freshness() {}".to_string()),
        note: Some(symbol.to_string()),
    }
}

fn query(text: &str) -> ExplorationQuery {
    ExplorationQuery {
        text: text.to_string(),
        scope_hint: None,
        max_results: None,
    }
}

/// Temp repo whose `src/` holds the files the finish-payload tests reference,
/// each 40 lines long, so `finish` path validation accepts the model's
/// findings and clamps nothing. Named per test + pid so parallel tests never
/// collide. Caller removes the dir.
fn temp_repo(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("agent_integ_{test}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let body = (1..=40)
        .map(|i| format!("l{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    for name in ["main.rs", "fresh_a.rs", "fresh_b.rs"] {
        std::fs::write(dir.join("src").join(name), &body).unwrap();
    }
    dir
}

fn router(
    providers: Vec<(String, Vec<(String, MockLlmProvider)>)>,
) -> ProviderRouter<MockLlmProvider, FakeClock> {
    ProviderRouter::with_clock(providers, 60, FakeClock::new())
}

fn single_router(provider: MockLlmProvider) -> ProviderRouter<MockLlmProvider, FakeClock> {
    router(vec![(
        "primary".to_string(),
        vec![("m".to_string(), provider)],
    )])
}

/// Settings that force the explorative fallback loop (confidence can never
/// reach 101), keeping the LLM-loop acceptance tests on a deterministic path.
fn fallback_only() -> AgentSettings {
    AgentSettings {
        early_exit_confidence: 101,
        fallback_confidence: 101,
        ..AgentSettings::default()
    }
}

#[tokio::test]
async fn fake_provider_dispatch_and_assembly() {
    let provider = MockLlmProvider::new().with_responses(vec![
        // One batched turn: memory + grep concurrently.
        ok_calls(vec![
            tc("c1", "search_code", r#"{"query":"main","max_results":5}"#),
            tc("c2", "grep", r#"{"pattern":"fn main","scope":"src"}"#),
        ]),
        ok_calls(vec![tc(
            "c3",
            "finish",
            r#"{"findings":[{"location":{"path":"src/main.rs","line_start":1,"line_end":3}}],"summary":"found main"}"#,
        )]),
    ]);
    let provider_probe = provider.clone();

    let memory = MockMemoryBackend::new().with_search_code_result(Ok(ExplorationResult {
        findings: vec![finding("src/a.rs")],
        summary: "mem".to_string(),
    }));
    let mem_probe = memory.clone();
    let search = MockSearchBackend::new().with_search_result(Ok(vec![finding("src/b.rs")]));
    let search_probe = search.clone();

    let agent = AgentLoop::new(
        memory,
        search,
        single_router(provider),
        MockRepoStateProbe::new(),
        fallback_only(),
        CacheSettings::default(),
    );
    let dir = temp_repo("dispatch");
    let result = agent.run(&dir, &query("where is main")).await.unwrap();

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
        MemCall::EnsureFreshIndex { repo_root } if repo_root == &dir
    )));
    assert!(mem_calls.iter().any(|c| matches!(
        c,
        MemCall::SearchCode { repo_root, query }
            if repo_root == &dir
                && query.text == "main"
                && query.max_results == Some(5)
    )));

    // The dispatched grep (the retrieval pre-stage also greps; find the one
    // with the model's pattern).
    let search_calls = search_probe.calls();
    assert!(search_calls.iter().any(|c| matches!(
        c,
        SearchCall::Search { repo_root, pattern, scope, .. }
            if repo_root == &dir
                && pattern == "fn main"
                && scope == &Some(PathBuf::from("src"))
    )));

    // Two turns; every turn's tools slice includes finish; both batched tool
    // results are answered before the finish turn.
    let llm_calls = provider_probe.calls();
    assert_eq!(llm_calls.len(), 2);
    for c in &llm_calls {
        assert!(c.tools.iter().any(|t| t.name == "finish"));
    }
    let final_messages = &llm_calls[1].messages;
    assert!(
        final_messages
            .iter()
            .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c1"))
    );
    assert!(
        final_messages
            .iter()
            .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("c2"))
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn iteration_limit_degrades_gracefully() {
    // The model only ever sends the same single grep call: rejected twice by
    // batch enforcement, executed on the third turn, then the loop hits the
    // iteration limit, the forced-finish attempt returns another non-finish
    // call, and the loop synthesizes a degraded result.
    let provider =
        MockLlmProvider::new().with_fallback(ok_calls(vec![tc("c", "grep", r#"{"pattern":"x"}"#)]));
    let provider_probe = provider.clone();
    let search = MockSearchBackend::new().with_search_result(Ok(vec![finding("src/x.rs")]));
    let search_probe = search.clone();
    let agent = AgentLoop::new(
        MockMemoryBackend::new(),
        search,
        single_router(provider),
        MockRepoStateProbe::new(),
        AgentSettings {
            max_fallback_iterations: 3,
            ..fallback_only()
        },
        CacheSettings::default(),
    );
    let result = agent
        .run(&PathBuf::from("/repo"), &query("q"))
        .await
        .unwrap();

    assert!(result.summary.contains("iteration limit"));
    assert_eq!(result.findings, vec![finding("src/x.rs")]);
    // 2 rejected turns + 1 executed turn = one real search.
    assert_eq!(search_probe.calls().len(), 1);
    // 3 loop turns + 1 forced-finish attempt.
    assert_eq!(provider_probe.calls().len(), 4);
    // The forced-finish call offers only `finish` and forces it.
    let last = provider_probe.calls().pop().unwrap();
    assert_eq!(last.tools.len(), 1);
    assert_eq!(last.tools[0].name, "finish");
    assert_eq!(last.options.force_tool.as_deref(), Some("finish"));
}

#[tokio::test]
async fn mid_exploration_failover_across_providers() {
    let primary = MockLlmProvider::new().with_responses(vec![
        ok_calls(vec![tc("c1", "search_code", r#"{"query":"widget"}"#)]),
        Err(ProviderError::RateLimited {
            provider: "primary".to_string(),
            message: "429".to_string(),
        }),
    ]);
    let secondary = MockLlmProvider::new().with_responses(vec![
        ok_calls(vec![tc("c2", "get_architecture", r#"{"depth":2}"#)]),
        ok_calls(vec![tc(
            "c3",
            "finish",
            r#"{"findings":[],"summary":"done via secondary"}"#,
        )]),
    ]);
    let primary_probe = primary.clone();
    let secondary_probe = secondary.clone();
    let agent = AgentLoop::new(
        MockMemoryBackend::new(),
        MockSearchBackend::new(),
        router(vec![
            ("primary".to_string(), vec![("m1".to_string(), primary)]),
            ("secondary".to_string(), vec![("m2".to_string(), secondary)]),
        ]),
        MockRepoStateProbe::new(),
        fallback_only(),
        CacheSettings::default(),
    );
    let result = agent
        .run(&PathBuf::from("/repo"), &query("widget"))
        .await
        .unwrap();

    assert_eq!(result.summary, "done via secondary");
    // Turn 1: primary (single call -> rejected). Turn 2: primary rate-limited
    // -> secondary (single call -> rejected). Turn 3: primary cooling (clock
    // not advanced) -> secondary finish.
    assert_eq!(primary_probe.calls().len(), 2);
    assert_eq!(secondary_probe.calls().len(), 2);
}

#[tokio::test]
async fn exact_symbol_early_exit_makes_zero_llm_calls() {
    let provider = MockLlmProvider::new();
    let provider_probe = provider.clone();
    let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
        findings: vec![symbol_finding(
            "crates/x/src/freshness.rs",
            "decide_freshness",
        )],
        summary: "1 row".to_string(),
    }));
    let agent = AgentLoop::new(
        memory,
        MockSearchBackend::new(),
        single_router(provider),
        MockRepoStateProbe::new(),
        AgentSettings::default(),
        CacheSettings::default(),
    );
    let result = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();

    assert!(provider_probe.calls().is_empty(), "no LLM call may happen");
    assert!(result.summary.contains("Resolved deterministically"));
    assert_eq!(
        result.findings[0].location.path,
        PathBuf::from("crates/x/src/freshness.rs")
    );
}

/// Two rival exact symbols in different files: strong but ambiguous, so the
/// verification stage (not early exit, not the fallback loop) must decide.
fn ambiguous_memory() -> MockMemoryBackend {
    MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
        findings: vec![
            symbol_finding("src/fresh_a.rs", "decide_freshness"),
            symbol_finding("src/fresh_b.rs", "decide_freshness"),
        ],
        summary: "2 rows".to_string(),
    }))
}

#[tokio::test]
async fn medium_confidence_verifies_in_one_turn() {
    let provider = MockLlmProvider::new().with_responses(vec![ok_calls(vec![tc(
        "v1",
        "finish",
        r#"{"findings":[{"location":{"path":"src/fresh_a.rs","line_start":10,"line_end":20},"note":"the one"}],"summary":"verified"}"#,
    )])]);
    let provider_probe = provider.clone();
    let agent = AgentLoop::new(
        ambiguous_memory(),
        MockSearchBackend::new(),
        single_router(provider),
        MockRepoStateProbe::new(),
        AgentSettings::default(),
        CacheSettings::default(),
    );
    let dir = temp_repo("medium");
    let result = agent.run(&dir, &query("decide_freshness")).await.unwrap();

    assert_eq!(result.summary, "verified");
    let calls = provider_probe.calls();
    assert_eq!(calls.len(), 1, "exactly one verification turn");
    // Verification catalog: expand + finish only.
    let tool_names: Vec<&str> = calls[0].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tool_names, vec!["expand", "finish"]);
    // The user message carries the numbered candidate list.
    assert!(
        calls[0]
            .messages
            .iter()
            .any(|m| m.role == Role::User && m.content.contains("Candidates:"))
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn verify_finish_is_capped_by_max_results() {
    // Regression: `max_results` must bound the verify stage's finish, not
    // just the deterministic early-exit path.
    let provider = MockLlmProvider::new().with_responses(vec![ok_calls(vec![tc(
        "v1",
        "finish",
        r#"{"findings":[{"location":{"path":"src/fresh_a.rs","line_start":10,"line_end":20},"note":"one"},{"location":{"path":"src/fresh_b.rs","line_start":10,"line_end":20},"note":"two"}],"summary":"verified"}"#,
    )])]);
    let agent = AgentLoop::new(
        ambiguous_memory(),
        MockSearchBackend::new(),
        single_router(provider),
        MockRepoStateProbe::new(),
        AgentSettings::default(),
        CacheSettings::default(),
    );
    let mut q = query("decide_freshness");
    q.max_results = Some(1);
    let dir = temp_repo("verify_capped");
    let result = agent.run(&dir, &q).await.unwrap();

    assert_eq!(result.findings.len(), 1, "capped to max_results");
    assert_eq!(
        result.findings[0].location.path,
        PathBuf::from("src/fresh_a.rs")
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn verify_expand_turn_then_forced_finish() {
    let provider = MockLlmProvider::new().with_responses(vec![
        ok_calls(vec![tc("v1", "expand", r#"{"candidate_ids":[1,2]}"#)]),
        ok_calls(vec![tc(
            "v2",
            "finish",
            r#"{"findings":[],"summary":"after expand"}"#,
        )]),
    ]);
    let provider_probe = provider.clone();
    let agent = AgentLoop::new(
        ambiguous_memory(),
        MockSearchBackend::new(),
        single_router(provider),
        MockRepoStateProbe::new(),
        AgentSettings::default(),
        CacheSettings::default(),
    );
    let result = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();

    assert_eq!(result.summary, "after expand");
    let calls = provider_probe.calls();
    assert_eq!(calls.len(), 2);
    // Turn 1 free choice, turn 2 (the last verify turn) forces finish.
    assert_eq!(calls[0].options.force_tool, None);
    assert_eq!(calls[1].options.force_tool.as_deref(), Some("finish"));
    // The expand call got a tool response.
    assert!(
        calls[1]
            .messages
            .iter()
            .any(|m| m.role == Role::Tool && m.tool_call_id.as_deref() == Some("v1"))
    );
}

#[tokio::test]
async fn failed_verification_escalates_to_fallback_loop() {
    let provider = MockLlmProvider::new().with_responses(vec![
        // Verification refuses to call tools twice -> escalate.
        Ok(Completion::from(ProviderResponse::Text("hm".to_string()))),
        Ok(Completion::from(ProviderResponse::Text("hm".to_string()))),
        // Fallback loop finishes immediately.
        ok_calls(vec![tc(
            "f1",
            "finish",
            r#"{"findings":[],"summary":"via fallback"}"#,
        )]),
    ]);
    let provider_probe = provider.clone();
    let agent = AgentLoop::new(
        ambiguous_memory(),
        MockSearchBackend::new(),
        single_router(provider),
        MockRepoStateProbe::new(),
        AgentSettings::default(),
        CacheSettings::default(),
    );
    let result = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();

    assert_eq!(result.summary, "via fallback");
    let calls = provider_probe.calls();
    assert_eq!(calls.len(), 3);
    // The fallback turn offers the full 10-tool catalog and seeds candidates.
    assert_eq!(calls[2].tools.len(), 10);
    assert!(calls[2].messages.iter().any(|m| m.role == Role::User
        && m.content.contains("Starting points")
        && m.content.contains("src/fresh_a.rs")));
}

#[tokio::test]
async fn token_budget_exhaustion_forces_final_finish() {
    let provider = MockLlmProvider::new().with_responses(vec![
        // Turn 1 burns the whole budget (single call also gets rejected).
        Ok(Completion {
            response: ProviderResponse::ToolCalls(vec![tc("c1", "grep", r#"{"pattern":"x"}"#)]),
            usage: Some(TokenUsage {
                prompt_tokens: 8,
                completion_tokens: 5,
            }),
        }),
        // The forced final call yields a finish.
        ok_calls(vec![tc(
            "c2",
            "finish",
            r#"{"findings":[],"summary":"budget done"}"#,
        )]),
    ]);
    let provider_probe = provider.clone();
    let agent = AgentLoop::new(
        MockMemoryBackend::new(),
        MockSearchBackend::new(),
        single_router(provider),
        MockRepoStateProbe::new(),
        AgentSettings {
            token_budget: 10,
            ..fallback_only()
        },
        CacheSettings::default(),
    );
    let result = agent
        .run(&PathBuf::from("/repo"), &query("q"))
        .await
        .unwrap();

    assert_eq!(result.summary, "budget done");
    let calls = provider_probe.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].options.force_tool.as_deref(), Some("finish"));
    let tool_names: Vec<&str> = calls[1].tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tool_names, vec!["finish"]);
}

fn fp(sha: &str) -> RepoFingerprint {
    RepoFingerprint {
        head_sha: sha.to_string(),
        dirty_hash: "d".to_string(),
    }
}

#[tokio::test]
async fn repeated_query_is_served_from_cache() {
    let provider = MockLlmProvider::new();
    let provider_probe = provider.clone();
    let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
        findings: vec![symbol_finding("src/fresh.rs", "decide_freshness")],
        summary: "1 row".to_string(),
    }));
    let mem_probe = memory.clone();
    let probe = MockRepoStateProbe::new().with_fingerprint(Some(fp("aaa")));
    let agent = AgentLoop::new(
        memory,
        MockSearchBackend::new(),
        single_router(provider),
        probe,
        AgentSettings::default(),
        CacheSettings::default(),
    );

    let first = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    let calls_after_first = mem_probe.calls().len();

    let second = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(
        mem_probe.calls().len(),
        calls_after_first,
        "cache hit must not touch the backends"
    );
    assert!(provider_probe.calls().is_empty());
}

#[tokio::test]
async fn fingerprint_change_with_no_diff_keeps_cache_entry() {
    let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
        findings: vec![symbol_finding("src/fresh.rs", "decide_freshness")],
        summary: "1 row".to_string(),
    }));
    let mem_probe = memory.clone();
    // An empty diff is the only fingerprint change a cache hit can prove safe
    // to reuse without rerunning retrieval.
    let probe = MockRepoStateProbe::new()
        .with_fingerprint(Some(fp("aaa")))
        .with_changed_paths(Some(Vec::new()));
    let probe_handle = probe.clone();
    let agent = AgentLoop::new(
        memory,
        MockSearchBackend::new(),
        single_router(MockLlmProvider::new()),
        probe,
        AgentSettings::default(),
        CacheSettings::default(),
    );

    let first = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    let calls_after_first = mem_probe.calls().len();

    probe_handle.set_fingerprint(Some(fp("bbb")));
    let second = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(mem_probe.calls().len(), calls_after_first);
}

#[tokio::test]
async fn fingerprint_change_touching_unrelated_path_recomputes() {
    // A path outside the cached result's own paths can still turn into a
    // better match (e.g. a newly added file) that the stale entry never saw,
    // so any actual diff — even to a path the old result never mentioned —
    // must invalidate the entry rather than being assumed safe.
    let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
        findings: vec![symbol_finding("src/fresh.rs", "decide_freshness")],
        summary: "1 row".to_string(),
    }));
    let mem_probe = memory.clone();
    let probe = MockRepoStateProbe::new()
        .with_fingerprint(Some(fp("aaa")))
        .with_changed_paths(Some(vec![PathBuf::from("docs/readme.md")]));
    let probe_handle = probe.clone();
    let agent = AgentLoop::new(
        memory,
        MockSearchBackend::new(),
        single_router(MockLlmProvider::new()),
        probe,
        AgentSettings::default(),
        CacheSettings::default(),
    );

    let _ = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    let calls_after_first = mem_probe.calls().len();

    probe_handle.set_fingerprint(Some(fp("bbb")));
    let _ = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    assert!(
        mem_probe.calls().len() > calls_after_first,
        "a change to any path, even one unrelated to the old answer, must recompute"
    );
}

#[tokio::test]
async fn fingerprint_change_touching_result_paths_recomputes() {
    let memory = MockMemoryBackend::new().with_search_graph_result(Ok(ExplorationResult {
        findings: vec![symbol_finding("src/fresh.rs", "decide_freshness")],
        summary: "1 row".to_string(),
    }));
    let mem_probe = memory.clone();
    let probe = MockRepoStateProbe::new()
        .with_fingerprint(Some(fp("aaa")))
        .with_changed_paths(Some(vec![PathBuf::from("src/fresh.rs")]));
    let probe_handle = probe.clone();
    let agent = AgentLoop::new(
        memory,
        MockSearchBackend::new(),
        single_router(MockLlmProvider::new()),
        probe,
        AgentSettings::default(),
        CacheSettings::default(),
    );

    let _ = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    let calls_after_first = mem_probe.calls().len();

    probe_handle.set_fingerprint(Some(fp("bbb")));
    let _ = agent
        .run(&PathBuf::from("/repo"), &query("decide_freshness"))
        .await
        .unwrap();
    assert!(
        mem_probe.calls().len() > calls_after_first,
        "a change to a contributing path must recompute"
    );
}
