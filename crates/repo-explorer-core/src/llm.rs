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

/// A single tool invocation requested by the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON arguments (kept as text so the type stays `Eq`).
    pub arguments_json: String,
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
    ) -> Result<ProviderResponse, ProviderError>;
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
        // Compile-time check that the trait exists with the intended signature.
        fn _assert_impl<P: LlmProvider>() {}
        // no runtime assertion needed
    }
}
