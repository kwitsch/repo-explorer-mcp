//! Library-agnostic LLM boundary: the `LlmProvider` async trait, serde-free
//! domain value types, the comparable `ProviderError`/`RouterError` types, the
//! generic `ProviderRouter` with an injectable `Clock`, and a gated mock.
//!
//! Core stays free of any LLM SDK: the sole production impl (`GenaiProvider`)
//! and the `genai` dependency live in the `repo-explorer-llm` crate, which
//! flattens every non-comparable `genai` cause into a `String` message here —
//! mirroring how `repo-explorer-memory` keeps `rmcp` out of `memory.rs`.

/// Chat role of a `Message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One chat message in a tool-use conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Set on Assistant messages that request tool execution.
    pub tool_calls: Vec<ToolCall>,
    /// Set on Tool-role messages: which call this result answers.
    pub tool_call_id: Option<String>,
}

impl Message {
    /// A `System` message: plain content, no tool fields.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// A `User` message: plain content, no tool fields.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// An `Assistant` message carrying plain text (no tool calls).
    pub fn assistant_text(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    /// An `Assistant` message requesting tool execution (empty content).
    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: String::new(),
            tool_calls,
            tool_call_id: None,
        }
    }

    /// A `Tool` message answering `call_id` with `content`.
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.into()),
        }
    }
}

/// A single tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON arguments (kept as text so the type stays `Eq`).
    pub arguments_json: String,
    /// Opaque provider continuation blob(s) (e.g. Gemini 3 "thought
    /// signatures") that must be replayed verbatim on the next request to
    /// validate this call; `None` for providers that don't use them.
    pub thought_signatures: Option<Vec<String>>,
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON Schema for the parameters, as raw text.
    pub parameters_schema_json: String,
}

/// The result of a single provider turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderResponse {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

/// Token counts a provider reported for one completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl TokenUsage {
    pub fn total(&self) -> u64 {
        self.prompt_tokens.saturating_add(self.completion_tokens)
    }
}

/// One provider turn plus the token usage it reported (`None` when the
/// provider reports none).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completion {
    pub response: ProviderResponse,
    pub usage: Option<TokenUsage>,
}

impl From<ProviderResponse> for Completion {
    fn from(response: ProviderResponse) -> Self {
        Self {
            response,
            usage: None,
        }
    }
}

/// Per-call knobs for `complete_with_tools`. `Default` means "no constraint".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CallOptions {
    /// Force the model to call this tool (by name) instead of choosing freely.
    pub force_tool: Option<String>,
    /// Cap the completion length, where the provider supports it.
    pub max_tokens: Option<u32>,
}

/// Provider-call failures. Fully comparable so mock-based tests can `assert_eq!`
/// on error values. Every variant carries the provider `name` for attribution;
/// any non-comparable `genai`/HTTP cause is flattened into `message` at the
/// `repo-explorer-llm` boundary (never stored as a `#[source]`, never a secret).
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum ProviderError {
    #[error("provider `{provider}` rate limited: {message}")]
    RateLimited { provider: String, message: String },
    #[error("provider `{provider}` quota exceeded: {message}")]
    QuotaExceeded { provider: String, message: String },
    #[error("provider `{provider}` authentication failed: {message}")]
    Authentication { provider: String, message: String },
    #[error("provider `{provider}` request invalid: {message}")]
    InvalidRequest { provider: String, message: String },
    #[error("provider `{provider}` transport error: {message}")]
    Transport { provider: String, message: String },
    #[error("provider `{provider}` returned an unusable response: {message}")]
    InvalidResponse { provider: String, message: String },
    /// Construction/config-level failure for this provider entry (unrecognized
    /// `kind`, a call-time-missing env var distinct from config-load validation,
    /// or a client-build failure).
    #[error("provider `{provider}` configuration error: {message}")]
    Configuration { provider: String, message: String },
}

impl ProviderError {
    /// Exactly the limit situations that trigger router failover.
    pub fn is_failover_trigger(&self) -> bool {
        matches!(self, Self::RateLimited { .. } | Self::QuotaExceeded { .. })
    }

    /// Provider name this error is attributed to.
    pub fn provider(&self) -> &str {
        match self {
            Self::RateLimited { provider, .. }
            | Self::QuotaExceeded { provider, .. }
            | Self::Authentication { provider, .. }
            | Self::InvalidRequest { provider, .. }
            | Self::Transport { provider, .. }
            | Self::InvalidResponse { provider, .. }
            | Self::Configuration { provider, .. } => provider,
        }
    }
}

/// Router-level failures.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum RouterError {
    #[error("no LLM providers are configured")]
    NoProviders,
    /// Every provider is exhausted (fresh limit error this pass) or still within
    /// its cooldown window. The string summarizes attempts/skips (names only).
    #[error("all configured LLM providers are exhausted or cooling down: {0}")]
    AllExhausted(String),
    /// A non-failover provider error, surfaced immediately (fail fast).
    #[error(transparent)]
    Provider(ProviderError),
}

/// The LLM contract implemented by a concrete provider.
///
/// Native `async fn` in trait (AFIT) — no `async-trait` dependency in core,
/// mirroring `MemoryBackend`. Static dispatch suffices; the `allow` silences the
/// warn-by-default `async_fn_in_trait` lint that `-D warnings` would reject.
#[allow(async_fn_in_trait)]
pub trait LlmProvider {
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &CallOptions,
    ) -> Result<Completion, ProviderError>;
}

/// Monotonic time source, injectable so cooldown-expiry tests need no real
/// sleeps. Production uses `SystemClock`; tests use `mock::FakeClock`.
pub trait Clock {
    fn now(&self) -> std::time::Instant;
}

/// Real monotonic clock.
#[derive(Debug, Clone, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

/// One model within a provider entry, plus its interior-mutable cooldown.
struct ModelSlot<P> {
    model: String,
    provider: P,
    cooling_until: std::sync::Mutex<Option<std::time::Instant>>,
}

impl<P> ModelSlot<P> {
    /// Lock `cooling_until`, recovering from poisoning instead of propagating
    /// the panic: a prior panic while holding this slot's cooldown state must
    /// not permanently break the whole failover router.
    fn lock_cooling(&self) -> std::sync::MutexGuard<'_, Option<std::time::Instant>> {
        self.cooling_until
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// One provider entry: a name plus its ordered model slots.
struct Entry<P> {
    name: String,
    models: Vec<ModelSlot<P>>,
}

/// Failover router over an ordered set of providers of one concrete type.
///
/// Generic over the provider `P` (static dispatch, no `async-trait`) and a
/// `Clock` `C`. Cooldown state lives behind a `Mutex` because
/// `complete_with_tools` takes `&self`.
pub struct ProviderRouter<P: LlmProvider, C: Clock = SystemClock> {
    entries: Vec<Entry<P>>,
    cooldown: std::time::Duration,
    clock: C,
}

impl<P: LlmProvider> ProviderRouter<P, SystemClock> {
    /// Pair each ordered `(name, models)` group with the shared cooldown window
    /// and the real system clock. File (= config) order is failover order for
    /// entries; list order is failover order for models within an entry.
    pub fn new(providers: Vec<(String, Vec<(String, P)>)>, cooldown_seconds: u64) -> Self {
        Self::with_clock(providers, cooldown_seconds, SystemClock)
    }
}

impl<P: LlmProvider, C: Clock> ProviderRouter<P, C> {
    /// Construct with an explicit clock (tests inject a `FakeClock`).
    pub fn with_clock(
        providers: Vec<(String, Vec<(String, P)>)>,
        cooldown_seconds: u64,
        clock: C,
    ) -> Self {
        let entries = providers
            .into_iter()
            .map(|(name, models)| Entry {
                name,
                models: models
                    .into_iter()
                    .map(|(model, provider)| ModelSlot {
                        model,
                        provider,
                        cooling_until: std::sync::Mutex::new(None),
                    })
                    .collect(),
            })
            .collect();
        Self {
            entries,
            cooldown: std::time::Duration::from_secs(cooldown_seconds),
            clock,
        }
    }

    /// One pass over the entries in configured order. Returns the first success;
    /// on a limit error the entry is put on cooldown and the next is tried; a
    /// non-limit error is surfaced immediately (fail fast); if every entry is
    /// cooling or freshly limited, returns `AllExhausted`.
    pub async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &CallOptions,
    ) -> Result<Completion, RouterError> {
        // Emptiness can live one level down now: an entry with no model slots
        // is skipped by the loop below and contributes nothing to `cooling`/
        // `limited`, which would surface as `AllExhausted("")`. Treat "no slot
        // to call at all" as `NoProviders` instead.
        if !self.entries.iter().any(|e| !e.models.is_empty()) {
            return Err(RouterError::NoProviders);
        }

        let now = self.clock.now();
        let mut cooling: Vec<String> = Vec::new();
        let mut limited: Vec<String> = Vec::new();

        for entry in &self.entries {
            for slot in &entry.models {
                {
                    let mut guard = slot.lock_cooling();
                    match *guard {
                        Some(until) if now < until => {
                            cooling.push(format!("{}/{}", entry.name, slot.model));
                            continue;
                        }
                        _ => {
                            // Expired (or never set): clear and proceed to call.
                            *guard = None;
                        }
                    }
                }

                match slot
                    .provider
                    .complete_with_tools(messages, tools, options)
                    .await
                {
                    Ok(resp) => return Ok(resp),
                    Err(e) if e.is_failover_trigger() => {
                        let mut guard = slot.lock_cooling();
                        *guard = Some(self.clock.now() + self.cooldown);
                        limited.push(format!("{}/{}", entry.name, slot.model));
                        continue;
                    }
                    Err(e) => return Err(RouterError::Provider(e)),
                }
            }
        }

        let mut parts: Vec<String> = Vec::new();
        if !limited.is_empty() {
            parts.push(format!("limited: {}", limited.join(", ")));
        }
        if !cooling.is_empty() {
            parts.push(format!("cooling: {}", cooling.join(", ")));
        }
        Err(RouterError::AllExhausted(parts.join("; ")))
    }
}

/// Test harness: a scripted, call-recording `LlmProvider` and a controllable
/// `Clock`. Gated so it compiles for core's own tests and for downstream crates
/// that enable `features = ["test-support"]`.
#[cfg(any(test, feature = "test-support"))]
pub mod mock {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// One recorded `complete_with_tools` invocation.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct MockCall {
        pub messages: Vec<Message>,
        pub tools: Vec<Tool>,
        pub options: CallOptions,
    }

    /// A programmable `LlmProvider`: pops a scripted response per call, falling
    /// back to `fallback` when the queue is empty, and records every call.
    #[derive(Clone)]
    pub struct MockLlmProvider {
        responses: Arc<Mutex<VecDeque<Result<Completion, ProviderError>>>>,
        fallback: Arc<Result<Completion, ProviderError>>,
        calls: Arc<Mutex<Vec<MockCall>>>,
    }

    impl Default for MockLlmProvider {
        fn default() -> Self {
            Self {
                responses: Arc::new(Mutex::new(VecDeque::new())),
                fallback: Arc::new(Ok(ProviderResponse::Text(String::new()).into())),
                calls: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl MockLlmProvider {
        pub fn new() -> Self {
            Self::default()
        }

        /// Queue a scripted sequence, consumed front-to-back across calls.
        /// Build usage-free items with `Completion::from(ProviderResponse)`.
        pub fn with_responses(self, r: Vec<Result<Completion, ProviderError>>) -> Self {
            *self.responses.lock().expect("mock responses poisoned") = r.into();
            self
        }

        /// Set the response returned once the scripted queue is empty.
        pub fn with_fallback(self, r: Result<Completion, ProviderError>) -> Self {
            Self {
                fallback: Arc::new(r),
                ..self
            }
        }

        /// Snapshot of recorded calls, in order.
        pub fn calls(&self) -> Vec<MockCall> {
            self.calls.lock().expect("mock calls poisoned").clone()
        }
    }

    impl LlmProvider for MockLlmProvider {
        async fn complete_with_tools(
            &self,
            messages: &[Message],
            tools: &[Tool],
            options: &CallOptions,
        ) -> Result<Completion, ProviderError> {
            self.calls
                .lock()
                .expect("mock calls poisoned")
                .push(MockCall {
                    messages: messages.to_vec(),
                    tools: tools.to_vec(),
                    options: options.clone(),
                });
            let popped = self
                .responses
                .lock()
                .expect("mock responses poisoned")
                .pop_front();
            match popped {
                Some(r) => r,
                None => (*self.fallback).clone(),
            }
        }
    }

    /// A `Clock` whose `now()` only moves when `advance` is called. Starts at
    /// `Instant::now()`; `Instant + Duration` avoids constructing arbitrary
    /// instants.
    #[derive(Clone)]
    pub struct FakeClock {
        now: Arc<Mutex<Instant>>,
    }

    impl Default for FakeClock {
        fn default() -> Self {
            Self {
                now: Arc::new(Mutex::new(Instant::now())),
            }
        }
    }

    impl FakeClock {
        pub fn new() -> Self {
            Self::default()
        }

        /// Move the clock forward by `d`.
        pub fn advance(&self, d: Duration) {
            let mut guard = self.now.lock().expect("fake clock poisoned");
            *guard += d;
        }
    }

    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            *self.now.lock().expect("fake clock poisoned")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_types_clone_and_eq() {
        let call = ToolCall {
            id: "c1".to_string(),
            name: "search_code".to_string(),
            arguments_json: r#"{"q":"main"}"#.to_string(),
            thought_signatures: None,
        };
        assert_eq!(call, call.clone());

        let msg = Message {
            role: Role::Assistant,
            content: "hi".to_string(),
            tool_calls: vec![call.clone()],
            tool_call_id: None,
        };
        assert_eq!(msg, msg.clone());
        assert_ne!(msg.role, Role::User);

        let tool = Tool {
            name: "search_code".to_string(),
            description: "search".to_string(),
            parameters_schema_json: r#"{"type":"object"}"#.to_string(),
        };
        assert_eq!(tool, tool.clone());

        let resp = ProviderResponse::ToolCalls(vec![call]);
        assert_eq!(resp, resp.clone());
        assert_ne!(resp, ProviderResponse::Text("x".to_string()));
    }

    #[test]
    fn provider_error_display_eq_and_provider_accessor() {
        let rl = ProviderError::RateLimited {
            provider: "primary".to_string(),
            message: "429".to_string(),
        };
        assert_eq!(rl, rl.clone());
        assert_eq!(rl.to_string(), "provider `primary` rate limited: 429");
        assert_eq!(rl.provider(), "primary");

        let qe = ProviderError::QuotaExceeded {
            provider: "p2".to_string(),
            message: "no credit".to_string(),
        };
        assert_eq!(qe.to_string(), "provider `p2` quota exceeded: no credit");

        let auth = ProviderError::Authentication {
            provider: "p3".to_string(),
            message: "bad key".to_string(),
        };
        assert_eq!(
            auth.to_string(),
            "provider `p3` authentication failed: bad key"
        );
        assert_eq!(auth.provider(), "p3");

        let inv = ProviderError::InvalidRequest {
            provider: "p4".to_string(),
            message: "bad".to_string(),
        };
        assert_eq!(inv.to_string(), "provider `p4` request invalid: bad");

        let tr = ProviderError::Transport {
            provider: "p5".to_string(),
            message: "conn".to_string(),
        };
        assert_eq!(tr.to_string(), "provider `p5` transport error: conn");

        let ir = ProviderError::InvalidResponse {
            provider: "p6".to_string(),
            message: "empty".to_string(),
        };
        assert_eq!(
            ir.to_string(),
            "provider `p6` returned an unusable response: empty"
        );

        let cfg = ProviderError::Configuration {
            provider: "p7".to_string(),
            message: "unknown kind".to_string(),
        };
        assert_eq!(
            cfg.to_string(),
            "provider `p7` configuration error: unknown kind"
        );
        assert_ne!(rl, qe);
    }

    #[test]
    fn only_rate_and_quota_are_failover_triggers() {
        let p = "x".to_string();
        assert!(
            ProviderError::RateLimited {
                provider: p.clone(),
                message: String::new()
            }
            .is_failover_trigger()
        );
        assert!(
            ProviderError::QuotaExceeded {
                provider: p.clone(),
                message: String::new()
            }
            .is_failover_trigger()
        );
        for e in [
            ProviderError::Authentication {
                provider: p.clone(),
                message: String::new(),
            },
            ProviderError::InvalidRequest {
                provider: p.clone(),
                message: String::new(),
            },
            ProviderError::Transport {
                provider: p.clone(),
                message: String::new(),
            },
            ProviderError::InvalidResponse {
                provider: p.clone(),
                message: String::new(),
            },
            ProviderError::Configuration {
                provider: p.clone(),
                message: String::new(),
            },
        ] {
            assert!(!e.is_failover_trigger());
        }
    }

    #[test]
    fn router_error_display_and_eq() {
        assert_eq!(
            RouterError::NoProviders.to_string(),
            "no LLM providers are configured"
        );
        let ex = RouterError::AllExhausted("primary cooling; secondary limited".to_string());
        assert_eq!(
            ex.to_string(),
            "all configured LLM providers are exhausted or cooling down: primary cooling; secondary limited"
        );
        let prov = RouterError::Provider(ProviderError::Authentication {
            provider: "p".to_string(),
            message: "bad".to_string(),
        });
        // `#[error(transparent)]` forwards the inner Display.
        assert_eq!(prov.to_string(), "provider `p` authentication failed: bad");
        assert_eq!(ex, ex.clone());
        assert_ne!(ex, RouterError::NoProviders);
    }

    #[test]
    fn llm_provider_trait_is_object_usable_via_generics() {
        // Compile-time check that the trait is usable as a generic bound.
        // Instantiating the generic fn is what makes this an assertion (the
        // trait uses `async fn`, so it is deliberately not dyn-compatible).
        fn assert_generic<P: LlmProvider>() {}
        assert_generic::<mock::MockLlmProvider>();
    }

    use mock::{FakeClock, MockCall, MockLlmProvider};
    use std::time::Duration;

    fn text(s: &str) -> Completion {
        Completion::from(ProviderResponse::Text(s.to_string()))
    }
    fn rate_limited(p: &str) -> ProviderError {
        ProviderError::RateLimited {
            provider: p.to_string(),
            message: "429".to_string(),
        }
    }

    #[tokio::test]
    async fn empty_provider_list_is_no_providers() {
        let router: ProviderRouter<MockLlmProvider> = ProviderRouter::new(vec![], 60);
        let got = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert_eq!(got, Err(RouterError::NoProviders));
    }

    #[tokio::test]
    async fn first_limited_falls_over_to_second() {
        let p1 = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let p2 = MockLlmProvider::new().with_fallback(Ok(text("from p2")));
        let router = ProviderRouter::new(
            vec![
                ("primary".to_string(), vec![("m1".to_string(), p1.clone())]),
                (
                    "secondary".to_string(),
                    vec![("m2".to_string(), p2.clone())],
                ),
            ],
            60,
        );
        let msgs = vec![Message {
            role: Role::User,
            content: "hi".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        }];
        let got = router
            .complete_with_tools(&msgs, &[], &CallOptions::default())
            .await;
        assert_eq!(got, Ok(text("from p2")));
        assert_eq!(
            p1.calls(),
            vec![MockCall {
                messages: msgs.clone(),
                tools: vec![],
                options: CallOptions::default()
            }]
        );
        assert_eq!(p2.calls().len(), 1);
    }

    #[tokio::test]
    async fn all_limited_is_all_exhausted() {
        let p1 = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let p2 = MockLlmProvider::new().with_fallback(Err(ProviderError::QuotaExceeded {
            provider: "secondary".to_string(),
            message: "no credit".to_string(),
        }));
        let router = ProviderRouter::new(
            vec![
                ("primary".to_string(), vec![("m1".to_string(), p1)]),
                ("secondary".to_string(), vec![("m2".to_string(), p2)]),
            ],
            60,
        );
        let got = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        match got {
            Err(RouterError::AllExhausted(summary)) => {
                assert!(summary.contains("primary"));
                assert!(summary.contains("secondary"));
            }
            other => panic!("expected AllExhausted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_limit_error_fails_fast_without_trying_next() {
        let p1 = MockLlmProvider::new().with_fallback(Err(ProviderError::Authentication {
            provider: "primary".to_string(),
            message: "bad key".to_string(),
        }));
        let p2 = MockLlmProvider::new().with_fallback(Ok(text("unused")));
        let router = ProviderRouter::new(
            vec![
                ("primary".to_string(), vec![("m1".to_string(), p1)]),
                (
                    "secondary".to_string(),
                    vec![("m2".to_string(), p2.clone())],
                ),
            ],
            60,
        );
        let got = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert_eq!(
            got,
            Err(RouterError::Provider(ProviderError::Authentication {
                provider: "primary".to_string(),
                message: "bad key".to_string(),
            }))
        );
        assert_eq!(p2.calls().len(), 0);
    }

    #[tokio::test]
    async fn cooldown_skips_then_retries_after_window() {
        let p1 = MockLlmProvider::new()
            .with_responses(vec![Err(rate_limited("primary")), Ok(text("p1 recovered"))]);
        let p2 = MockLlmProvider::new().with_fallback(Err(rate_limited("secondary")));
        let clock = FakeClock::new();
        let router = ProviderRouter::with_clock(
            vec![
                ("primary".to_string(), vec![("m1".to_string(), p1.clone())]),
                (
                    "secondary".to_string(),
                    vec![("m2".to_string(), p2.clone())],
                ),
            ],
            60,
            clock.clone(),
        );

        let first = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert!(matches!(first, Err(RouterError::AllExhausted(_))));

        clock.advance(Duration::from_secs(30));
        let second = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert!(matches!(second, Err(RouterError::AllExhausted(_))));
        assert_eq!(p1.calls().len(), 1, "p1 must be skipped while cooling");

        clock.advance(Duration::from_secs(60));
        let third = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert_eq!(third, Ok(text("p1 recovered")));
        assert_eq!(p1.calls().len(), 2, "p1 retried after cooldown elapsed");
    }

    #[tokio::test]
    async fn model_limit_falls_over_within_entry() {
        // Entry `primary` has two model slots: `a` limited, `b` ok. Entry
        // `secondary` (model `c`) must never be reached.
        let a = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let b = MockLlmProvider::new().with_fallback(Ok(text("from b")));
        let c = MockLlmProvider::new().with_fallback(Ok(text("from c")));
        let router = ProviderRouter::new(
            vec![
                (
                    "primary".to_string(),
                    vec![("a".to_string(), a.clone()), ("b".to_string(), b.clone())],
                ),
                ("secondary".to_string(), vec![("c".to_string(), c.clone())]),
            ],
            60,
        );
        let got = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert_eq!(got, Ok(text("from b")));
        assert_eq!(a.calls().len(), 1);
        assert_eq!(b.calls().len(), 1);
        assert_eq!(c.calls().len(), 0, "next entry must not be reached");
    }

    #[tokio::test]
    async fn entry_exhausted_falls_over_to_next_entry() {
        // Both models of `primary` are limited; `secondary` succeeds.
        let a = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let b = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let c = MockLlmProvider::new().with_fallback(Ok(text("from c")));
        let router = ProviderRouter::new(
            vec![
                (
                    "primary".to_string(),
                    vec![("a".to_string(), a.clone()), ("b".to_string(), b.clone())],
                ),
                ("secondary".to_string(), vec![("c".to_string(), c.clone())]),
            ],
            60,
        );
        let got = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert_eq!(got, Ok(text("from c")));
        assert_eq!(a.calls().len(), 1);
        assert_eq!(b.calls().len(), 1);
        assert_eq!(c.calls().len(), 1);
    }

    #[tokio::test]
    async fn all_models_all_entries_exhausted() {
        let a = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let b = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let c = MockLlmProvider::new().with_fallback(Err(rate_limited("secondary")));
        let router = ProviderRouter::new(
            vec![
                (
                    "primary".to_string(),
                    vec![("a".to_string(), a), ("b".to_string(), b)],
                ),
                ("secondary".to_string(), vec![("c".to_string(), c)]),
            ],
            60,
        );
        let got = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        match got {
            Err(RouterError::AllExhausted(summary)) => {
                assert!(summary.contains("primary/a"));
                assert!(summary.contains("primary/b"));
                assert!(summary.contains("secondary/c"));
            }
            other => panic!("expected AllExhausted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn per_model_cooldown_recovers() {
        // Model `a` limited on first call, ok on the second (post-cooldown);
        // model `b` always limited so the router falls through to AllExhausted
        // on the first pass and re-reaches `a` after the window.
        let a = MockLlmProvider::new()
            .with_responses(vec![Err(rate_limited("primary")), Ok(text("a recovered"))]);
        let b = MockLlmProvider::new().with_fallback(Err(rate_limited("primary")));
        let clock = FakeClock::new();
        let router = ProviderRouter::with_clock(
            vec![(
                "primary".to_string(),
                vec![("a".to_string(), a.clone()), ("b".to_string(), b.clone())],
            )],
            60,
            clock.clone(),
        );

        let first = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert!(matches!(first, Err(RouterError::AllExhausted(_))));
        assert_eq!(a.calls().len(), 1);

        // Within cooldown: `a` skipped, `b` still limited.
        clock.advance(Duration::from_secs(30));
        let second = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert!(matches!(second, Err(RouterError::AllExhausted(_))));
        assert_eq!(a.calls().len(), 1, "a must be skipped while cooling");

        // Past cooldown: `a` retried and succeeds.
        clock.advance(Duration::from_secs(60));
        let third = router
            .complete_with_tools(&[], &[], &CallOptions::default())
            .await;
        assert_eq!(third, Ok(text("a recovered")));
        assert_eq!(a.calls().len(), 2, "a retried after cooldown elapsed");
    }
}
