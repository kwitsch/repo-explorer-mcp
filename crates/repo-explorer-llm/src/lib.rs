//! `genai`-backed `LlmProvider` (`GenaiProvider`), the domain↔`genai` mapping,
//! `genai`-error classification, and the `build_router` convenience constructor.
//!
//! This crate is the ONLY place in the workspace that names a `genai::*` type:
//! it flattens every non-comparable `genai` cause into a `String` message on the
//! comparable `ProviderError`, mirroring how `repo-explorer-memory` keeps `rmcp`
//! out of `repo-explorer-core`.

use repo_explorer_core::config::{LlmConfig, ProviderConfig};
use repo_explorer_core::llm::{
    LlmProvider, Message, ProviderError, ProviderResponse, ProviderRouter, Role, Tool, ToolCall,
};

/// Genai-independent view of an error, used by the pure classifier so its tests
/// need no `genai` value construction.
pub(crate) struct GenaiErrorFacts {
    /// HTTP status if `genai` exposed one.
    pub status: Option<u16>,
    /// Provider-specific structured error code, if available (e.g. OpenAI
    /// `insufficient_quota`, `rate_limit_exceeded`).
    pub code: Option<String>,
    /// Human-readable, secret-free message.
    pub message: String,
}

/// Substrings that indicate quota/billing exhaustion rather than transient rate
/// limiting, checked (lowercased) when no structured code disambiguates.
fn message_indicates_quota(message: &str) -> bool {
    let m = message.to_lowercase();
    m.contains("insufficient_quota")
        || m.contains("quota exceeded")
        || m.contains("exceeded your current quota")
        || m.contains("exceeded your quota")
        || m.contains("out of quota")
        || m.contains("credit balance")
        || m.contains("billing")
}

/// Pure classification of error facts into a `ProviderError`. Best-effort and
/// `kind`-aware per the source plan; both `RateLimited` and `QuotaExceeded` are
/// failover triggers, so the finer split need not be exact for correctness.
pub(crate) fn classify_error_facts(
    provider: &str,
    _kind: &str,
    facts: &GenaiErrorFacts,
) -> ProviderError {
    let provider = provider.to_string();
    let message = facts.message.clone();
    let code = facts.code.as_deref().unwrap_or("");
    let code_lower = code.to_lowercase();

    // Structured code wins when present.
    if code_lower.contains("insufficient_quota") || code_lower.contains("quota") {
        return ProviderError::QuotaExceeded { provider, message };
    }
    if code_lower.contains("rate_limit") {
        return ProviderError::RateLimited { provider, message };
    }

    match facts.status {
        Some(429) => {
            if message_indicates_quota(&message) {
                ProviderError::QuotaExceeded { provider, message }
            } else {
                ProviderError::RateLimited { provider, message }
            }
        }
        Some(401) | Some(403) => ProviderError::Authentication { provider, message },
        Some(400) | Some(404) | Some(422) => ProviderError::InvalidRequest { provider, message },
        // 5xx and any other status: transport-level.
        Some(_) => ProviderError::Transport { provider, message },
        // No status at all: connection/transport failure (unless the message
        // clearly names a quota/billing condition).
        None => {
            if message_indicates_quota(&message) {
                ProviderError::QuotaExceeded { provider, message }
            } else {
                ProviderError::Transport { provider, message }
            }
        }
    }
}

/// Adapt a `genai::Error` into genai-independent facts, then classify.
///
/// `genai` 0.6.x surfaces an HTTP status on `Error::HttpError` and, indirectly,
/// on the web-call variants whose `webc::Error::ResponseFailedStatus` carries
/// one. It does not preserve a structured provider error *code*, so `code` stays
/// `None` and classification relies on status plus substring matching. The
/// flattened `message` is the SDK's `Display`, which contains provider error
/// bodies but never our API key.
pub(crate) fn classify_genai_error(
    provider: &str,
    kind: &str,
    err: &genai::Error,
) -> ProviderError {
    let facts = GenaiErrorFacts {
        status: extract_status(err),
        code: None,
        message: err.to_string(),
    };
    classify_error_facts(provider, kind, &facts)
}

/// Return the HTTP status if the `genai::Error` (or its nested web error)
/// carries one, else `None` (degrading gracefully to substring classification).
fn extract_status(err: &genai::Error) -> Option<u16> {
    match err {
        genai::Error::HttpError { status, .. } => Some(status.as_u16()),
        genai::Error::WebModelCall { webc_error, .. }
        | genai::Error::WebAdapterCall { webc_error, .. } => webc_status(webc_error),
        _ => None,
    }
}

/// Extract the HTTP status from a `genai::webc::Error`, if it is a
/// failed-status response.
fn webc_status(err: &genai::webc::Error) -> Option<u16> {
    match err {
        genai::webc::Error::ResponseFailedStatus { status, .. } => Some(status.as_u16()),
        _ => None,
    }
}

/// Map a config `kind` string to a `genai` adapter. `google` is accepted as an
/// alias for the Gemini adapter. Returns `None` for unrecognized kinds.
///
/// Must recognize exactly the kinds in
/// `repo_explorer_core::config::KNOWN_PROVIDER_KINDS` — core can't depend on
/// `genai` to express that set as adapters directly, so this match is
/// re-declared here; `tests::adapter_kind_for_matches_known_provider_kinds`
/// guards the two lists against drifting apart.
fn adapter_kind_for(kind: &str) -> Option<genai::adapter::AdapterKind> {
    use genai::adapter::AdapterKind;
    match kind {
        "anthropic" => Some(AdapterKind::Anthropic),
        "openai" => Some(AdapterKind::OpenAI),
        "gemini" | "google" => Some(AdapterKind::Gemini),
        _ => None,
    }
}

/// The single production `LlmProvider`, backed by one `genai` client bound to a
/// specific model. `name`/`kind`/`model` come from one `ProviderConfig` entry.
pub struct GenaiProvider {
    name: String,
    kind: String,
    model: String,
    client: genai::Client,
}

impl GenaiProvider {
    /// Build one provider from a single config entry. The API key is read (by
    /// `genai`, at call time) from `provider.api_key_env`; this constructor only
    /// confirms the var is present, and its value is NEVER placed into any error
    /// message. `base_url`, when `Some`, overrides the adapter's endpoint. An
    /// unrecognized `kind`, or a missing key var at call-time (distinct from
    /// config-load validation), yields `ProviderError::Configuration`.
    pub fn from_config(provider: &ProviderConfig, model: &str) -> Result<Self, ProviderError> {
        let (name, kind, client) = Self::build_shared(provider)?;
        Ok(Self {
            name,
            kind,
            model: model.to_string(),
            client,
        })
    }

    /// Validate `provider` and build its `genai::Client` once. `genai::Client`
    /// wraps an `Arc` internally (cheap to clone), so callers with multiple
    /// models per entry (e.g. `build_router`) build this a single time per
    /// entry and clone the client for each `ModelSlot` instead of repeating
    /// the adapter-kind/env-var resolution and client construction per model.
    fn build_shared(
        provider: &ProviderConfig,
    ) -> Result<(String, String, genai::Client), ProviderError> {
        let name = provider.name.clone();

        let adapter_kind =
            adapter_kind_for(&provider.kind).ok_or_else(|| ProviderError::Configuration {
                provider: name.clone(),
                message: format!("unrecognized provider kind `{}`", provider.kind),
            })?;

        let env_name =
            provider
                .resolve_api_key_env()
                .ok_or_else(|| ProviderError::Configuration {
                    provider: name.clone(),
                    message: format!(
                        "no `api_key_env` set and no default env var for kind `{}`",
                        provider.kind
                    ),
                })?;

        // Fail with Configuration (never panic) if the key var is missing at
        // call-time, distinct from config-load validation. The value is read to
        // confirm presence; it is NEVER placed into any error message.
        if std::env::var(&env_name).is_err() {
            return Err(ProviderError::Configuration {
                provider: name,
                message: format!("environment variable `{env_name}` is not set"),
            });
        }

        let client = build_genai_client(adapter_kind, provider, &env_name);

        Ok((name, provider.kind.clone(), client))
    }
}

/// Construct a `genai::Client` bound to `adapter_kind`, authenticating from the
/// entry's custom `api_key_env` and overriding the endpoint when `base_url` is
/// set. This is the only place that names client-builder symbols.
fn build_genai_client(
    adapter_kind: genai::adapter::AdapterKind,
    provider: &ProviderConfig,
    env_name: &str,
) -> genai::Client {
    use genai::resolver::{AuthData, Endpoint};

    let env_name = env_name.to_string();
    let mut builder = genai::Client::builder()
        .with_adapter_kind(adapter_kind)
        .with_auth_resolver_fn(move |_model_iden: genai::ModelIden| {
            Ok(Some(AuthData::from_env(env_name.clone())))
        });

    if let Some(base_url) = provider.base_url.clone() {
        builder =
            builder.with_service_target_resolver_fn(move |mut target: genai::ServiceTarget| {
                target.endpoint = Endpoint::from_owned(base_url.clone());
                Ok(target)
            });
    }

    builder.build()
}

/// Map domain `Message`s onto genai chat messages, preserving role, content,
/// assistant `tool_calls`, and tool-result `tool_call_id`.
fn to_genai_messages(
    provider: &str,
    messages: &[Message],
) -> Result<Vec<genai::chat::ChatMessage>, ProviderError> {
    messages
        .iter()
        .map(|m| to_genai_message(provider, m))
        .collect()
}

/// Map a single domain `Message` onto a genai `ChatMessage`.
fn to_genai_message(
    provider: &str,
    message: &Message,
) -> Result<genai::chat::ChatMessage, ProviderError> {
    use genai::chat::{ChatMessage, ContentPart, MessageContent, ToolResponse};

    match message.role {
        Role::System => Ok(ChatMessage::system(message.content.clone())),
        Role::User => Ok(ChatMessage::user(message.content.clone())),
        Role::Assistant => {
            if message.tool_calls.is_empty() {
                return Ok(ChatMessage::assistant(message.content.clone()));
            }
            let mut parts: Vec<ContentPart> = Vec::new();
            if !message.content.is_empty() {
                parts.push(ContentPart::Text(message.content.clone()));
            }
            for tc in &message.tool_calls {
                parts.push(ContentPart::ToolCall(to_genai_tool_call(provider, tc)?));
            }
            Ok(ChatMessage::assistant(MessageContent::from_parts(parts)))
        }
        Role::Tool => {
            let call_id = message.tool_call_id.clone().unwrap_or_default();
            Ok(ChatMessage::tool(ToolResponse::new(
                call_id,
                message.content.clone(),
            )))
        }
    }
}

/// Map a domain `ToolCall` onto a genai `ToolCall`, parsing the JSON arguments
/// text (this crate owns `serde_json`); a parse failure is an `InvalidRequest`.
fn to_genai_tool_call(
    provider: &str,
    tc: &ToolCall,
) -> Result<genai::chat::ToolCall, ProviderError> {
    let fn_arguments: serde_json::Value =
        serde_json::from_str(&tc.arguments_json).map_err(|e| ProviderError::InvalidRequest {
            provider: provider.to_string(),
            message: format!("tool call `{}` has invalid arguments JSON: {e}", tc.name),
        })?;
    Ok(genai::chat::ToolCall {
        call_id: tc.id.clone(),
        fn_name: tc.name.clone(),
        fn_arguments,
        thought_signatures: None,
    })
}

/// Map domain `Tool`s onto genai tool definitions. The JSON-schema text is
/// parsed here at the crate boundary; a parse failure is an `InvalidRequest`.
fn to_genai_tools(provider: &str, tools: &[Tool]) -> Result<Vec<genai::chat::Tool>, ProviderError> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let schema: serde_json::Value =
            serde_json::from_str(&t.parameters_schema_json).map_err(|e| {
                ProviderError::InvalidRequest {
                    provider: provider.to_string(),
                    message: format!("tool `{}` has invalid parameter schema: {e}", t.name),
                }
            })?;
        out.push(
            genai::chat::Tool::new(t.name.clone())
                .with_description(t.description.clone())
                .with_schema(schema),
        );
    }
    Ok(out)
}

/// Map a genai chat response to a `ProviderResponse`: tool calls take priority,
/// then text; an empty/unusable response is an `InvalidResponse`.
fn from_genai_response(
    provider: &str,
    response: genai::chat::ChatResponse,
) -> Result<ProviderResponse, ProviderError> {
    let text = response.first_text().map(|s| s.to_string());
    let tool_calls = response.into_tool_calls();

    if !tool_calls.is_empty() {
        let mapped = tool_calls
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.call_id,
                name: tc.fn_name,
                arguments_json: tc.fn_arguments.to_string(),
            })
            .collect();
        return Ok(ProviderResponse::ToolCalls(mapped));
    }

    match text {
        Some(text) => Ok(ProviderResponse::Text(text)),
        None => Err(ProviderError::InvalidResponse {
            provider: provider.to_string(),
            message: "response contained neither text nor tool calls".to_string(),
        }),
    }
}

impl LlmProvider for GenaiProvider {
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<ProviderResponse, ProviderError> {
        let chat_messages = to_genai_messages(&self.name, messages)?;
        let genai_tools = to_genai_tools(&self.name, tools)?;

        let mut request = genai::chat::ChatRequest::new(chat_messages);
        if !genai_tools.is_empty() {
            request = request.with_tools(genai_tools);
        }

        match self
            .client
            .exec_chat(self.model.as_str(), request, None)
            .await
        {
            Ok(response) => from_genai_response(&self.name, response),
            Err(err) => Err(classify_genai_error(&self.name, &self.kind, &err)),
        }
    }
}

/// Build the production router from validated config: one `GenaiProvider` per
/// entry, in file (= failover) order, with the configured cooldown window.
///
/// The composition helper the `repo-explorer-mcp` binary calls in a later stage.
pub fn build_router(cfg: &LlmConfig) -> Result<ProviderRouter<GenaiProvider>, ProviderError> {
    let mut providers = Vec::with_capacity(cfg.providers.len());
    for entry in &cfg.providers {
        let (name, kind, client) = GenaiProvider::build_shared(entry)?;
        let mut models = Vec::with_capacity(entry.models.len());
        for model in &entry.models {
            let provider = GenaiProvider {
                name: name.clone(),
                kind: kind.clone(),
                model: model.clone(),
                client: client.clone(),
            };
            models.push((model.clone(), provider));
        }
        providers.push((entry.name.clone(), models));
    }
    Ok(ProviderRouter::new(providers, cfg.cooldown_seconds))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_kind_for_matches_known_provider_kinds() {
        for &kind in repo_explorer_core::config::KNOWN_PROVIDER_KINDS {
            assert!(
                adapter_kind_for(kind).is_some(),
                "kind `{kind}` is in KNOWN_PROVIDER_KINDS but adapter_kind_for returns None \
                 — the two lists have drifted apart"
            );
        }
        assert!(adapter_kind_for("not-a-real-kind").is_none());
    }

    fn facts(status: Option<u16>, code: Option<&str>, message: &str) -> GenaiErrorFacts {
        GenaiErrorFacts {
            status,
            code: code.map(|c| c.to_string()),
            message: message.to_string(),
        }
    }

    #[test]
    fn http_429_classifies_as_rate_limited() {
        let e = classify_error_facts("primary", "anthropic", &facts(Some(429), None, "slow down"));
        assert_eq!(
            e,
            ProviderError::RateLimited {
                provider: "primary".to_string(),
                message: "slow down".to_string(),
            }
        );
        assert!(e.is_failover_trigger());
    }

    #[test]
    fn openai_insufficient_quota_classifies_as_quota() {
        let e = classify_error_facts(
            "p",
            "openai",
            &facts(Some(429), Some("insufficient_quota"), "no credit"),
        );
        assert_eq!(
            e,
            ProviderError::QuotaExceeded {
                provider: "p".to_string(),
                message: "no credit".to_string(),
            }
        );
    }

    #[test]
    fn http_401_classifies_as_authentication_and_is_not_failover() {
        let e = classify_error_facts("p", "openai", &facts(Some(401), None, "bad key"));
        assert_eq!(
            e,
            ProviderError::Authentication {
                provider: "p".to_string(),
                message: "bad key".to_string(),
            }
        );
        assert!(!e.is_failover_trigger());
    }

    #[test]
    fn http_400_classifies_as_invalid_request() {
        let e = classify_error_facts("p", "openai", &facts(Some(400), None, "bad body"));
        assert_eq!(
            e,
            ProviderError::InvalidRequest {
                provider: "p".to_string(),
                message: "bad body".to_string(),
            }
        );
    }

    #[test]
    fn no_status_falls_back_to_transport() {
        let e = classify_error_facts("p", "anthropic", &facts(None, None, "connection reset"));
        assert_eq!(
            e,
            ProviderError::Transport {
                provider: "p".to_string(),
                message: "connection reset".to_string(),
            }
        );
    }

    #[test]
    fn quota_via_substring_when_status_only_429() {
        // Anthropic billing/overloaded phrasing without a structured code.
        let e = classify_error_facts(
            "p",
            "anthropic",
            &facts(Some(429), None, "credit balance is too low"),
        );
        assert_eq!(
            e,
            ProviderError::QuotaExceeded {
                provider: "p".to_string(),
                message: "credit balance is too low".to_string(),
            }
        );
    }

    #[test]
    fn error_message_never_echoes_a_key() {
        // The classifier only ever copies the provided message; assert it does
        // not fabricate secrets and preserves the given text verbatim.
        let e = classify_error_facts("p", "openai", &facts(Some(500), None, "server error"));
        assert_eq!(
            e,
            ProviderError::Transport {
                provider: "p".to_string(),
                message: "server error".to_string(),
            }
        );
    }

    #[tokio::test]
    #[ignore = "requires a live provider and real API key"]
    #[cfg(feature = "live-tests")]
    async fn live_genai_call_returns_a_response() {
        // Requires env vars for the chosen kind's key. Kept out of the default
        // suite via both `#[ignore]` and the `live-tests` feature.
        let cfg = ProviderConfig {
            name: "live".to_string(),
            kind: "openai".to_string(),
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            models: vec!["gpt-4o-mini".to_string()],
            base_url: None,
        };
        let provider = GenaiProvider::from_config(&cfg, &cfg.models[0]).expect("build provider");
        let msgs = vec![Message {
            role: Role::User,
            content: "Say hello.".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        }];
        let resp = provider.complete_with_tools(&msgs, &[]).await;
        assert!(resp.is_ok(), "live call failed: {resp:?}");
    }
}
