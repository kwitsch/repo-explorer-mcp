//! `genai`-backed `LlmProvider` (`GenaiProvider`), the domain↔`genai` mapping,
//! `genai`-error classification, and the `build_router` convenience constructor.
//!
//! This crate is the ONLY place in the workspace that names a `genai::*` type:
//! it flattens every non-comparable `genai` cause into a `String` message on the
//! comparable `ProviderError`, mirroring how `repo-explorer-memory` keeps `rmcp`
//! out of `repo-explorer-core`.

use repo_explorer_core::config::{LlmConfig, ProviderConfig, env_var_is_set};
use repo_explorer_core::llm::{
    CallOptions, Completion, LlmProvider, Message, ProviderError, ProviderResponse, ProviderRouter,
    Role, TokenUsage, Tool, ToolCall,
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

/// Pure classification of error facts into a `ProviderError`. Best-effort:
/// both `RateLimited` and `QuotaExceeded` are failover triggers, so the finer
/// split need not be exact for correctness.
pub(crate) fn classify_error_facts(provider: &str, facts: GenaiErrorFacts) -> ProviderError {
    let provider = provider.to_string();
    let message = facts.message;
    let code = facts.code.as_deref().unwrap_or("");
    let code_lower = code.to_lowercase();

    // Structured code wins when present.
    if code_lower.contains("quota") {
        return ProviderError::QuotaExceeded { provider, message };
    }
    if code_lower.contains("rate_limit") {
        return ProviderError::RateLimited { provider, message };
    }

    match facts.status {
        // 429 is standard rate-limiting; 529 is Anthropic's overloaded_error, a
        // transient-overload signal that should failover the same way; no
        // status at all is a connection/transport failure. All three report
        // QuotaExceeded instead when the message clearly names a quota/billing
        // condition.
        Some(429) | Some(529) | None if message_indicates_quota(&message) => {
            ProviderError::QuotaExceeded { provider, message }
        }
        Some(429) | Some(529) => ProviderError::RateLimited { provider, message },
        Some(401) | Some(403) => ProviderError::Authentication { provider, message },
        Some(400) | Some(404) | Some(422) => ProviderError::InvalidRequest { provider, message },
        // Other 5xx, any other status, and no status at all: transport-level.
        Some(_) | None => ProviderError::Transport { provider, message },
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
pub(crate) fn classify_genai_error(provider: &str, err: &genai::Error) -> ProviderError {
    let facts = GenaiErrorFacts {
        status: extract_status(err),
        code: None,
        message: err.to_string(),
    };
    classify_error_facts(provider, facts)
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
/// specific model. `name`/`model` come from one `ProviderConfig` entry.
pub struct GenaiProvider {
    name: String,
    model: String,
    client: genai::Client,
    /// Mark system messages with a provider-side prompt-cache hint. Only the
    /// Anthropic adapter supports the content-level `cache_control` marker.
    cache_system_prompt: bool,
}

impl GenaiProvider {
    /// Build one provider from a single config entry. The API key is read (by
    /// `genai`, at call time) from `provider.api_key_env`; this constructor only
    /// confirms the var is present, and its value is NEVER placed into any error
    /// message. `base_url`, when `Some`, overrides the adapter's endpoint. An
    /// unrecognized `kind`, or a missing key var at call-time (distinct from
    /// config-load validation), yields `ProviderError::Configuration`.
    pub fn from_config(
        provider: &ProviderConfig,
        model: &str,
        https_proxy: Option<&str>,
    ) -> Result<Self, ProviderError> {
        let shared = Self::build_shared(provider, https_proxy)?;
        Ok(Self::bind_model(&shared, model))
    }

    /// The one place a `GenaiProvider` is assembled from its parts, so
    /// `from_config` and `build_router` (which reuses one client across an
    /// entry's models) cannot drift as fields are added.
    fn bind_model(shared: &SharedProviderParts, model: &str) -> Self {
        Self {
            name: shared.name.clone(),
            model: model.to_string(),
            client: shared.client.clone(),
            cache_system_prompt: shared.adapter_kind == genai::adapter::AdapterKind::Anthropic,
        }
    }

    /// Validate `provider` and build its `genai::Client` once. `genai::Client`
    /// wraps an `Arc` internally (cheap to clone), so callers with multiple
    /// models per entry (e.g. `build_router`) build this a single time per
    /// entry and clone the client for each `ModelSlot` instead of repeating
    /// the adapter-kind/env-var resolution and client construction per model.
    /// `https_proxy`, when set, routes this provider's upstream requests
    /// through it (mirrors `llm.https_proxy` in config, applied to every
    /// provider uniformly).
    fn build_shared(
        provider: &ProviderConfig,
        https_proxy: Option<&str>,
    ) -> Result<SharedProviderParts, ProviderError> {
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
        if !env_var_is_set(|v| std::env::var(v).ok(), &env_name) {
            return Err(ProviderError::Configuration {
                provider: name,
                message: format!("environment variable `{env_name}` is not set"),
            });
        }

        let client = build_genai_client(adapter_kind, provider, env_name, https_proxy)?;

        Ok(SharedProviderParts {
            name,
            client,
            adapter_kind,
        })
    }
}

/// Per-entry parts shared by every model slot of one provider entry.
struct SharedProviderParts {
    name: String,
    client: genai::Client,
    adapter_kind: genai::adapter::AdapterKind,
}

/// Construct a `genai::Client` bound to `adapter_kind`, authenticating from the
/// entry's custom `api_key_env`, overriding the endpoint when `base_url` is
/// set, and routing through `https_proxy` when set. This is the only place
/// that names client-builder symbols.
///
/// `https_proxy` is applied via `reqwest::Proxy::https` (conventional
/// `HTTPS_PROXY` semantics): only requests to an `https://` destination are
/// proxied. A provider entry whose `base_url` is `http://` is not covered.
fn build_genai_client(
    adapter_kind: genai::adapter::AdapterKind,
    provider: &ProviderConfig,
    env_name: String,
    https_proxy: Option<&str>,
) -> Result<genai::Client, ProviderError> {
    use genai::WebConfig;
    use genai::resolver::{AuthData, Endpoint};

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

    if let Some(proxy_url) = https_proxy {
        // Never interpolate `proxy_url` (or the underlying parse error, which
        // may itself echo the URL) into this message: a proxy URL commonly
        // embeds basic-auth credentials, and `ProviderError`'s contract is
        // that `message` is never a secret (see the doc comment on
        // `ProviderError` in repo-explorer-core).
        let web_config = WebConfig::default()
            .with_https_proxy_url(proxy_url)
            .map_err(|_| ProviderError::Configuration {
                provider: provider.name.clone(),
                message: "llm.https_proxy is not a usable proxy URL (check scheme, host, \
                              and port); its value is withheld here in case it embeds credentials"
                    .to_string(),
            })?;
        builder = builder.with_web_config(web_config);
    }

    Ok(builder.build())
}

/// Map domain `Message`s onto genai chat messages, preserving role, content,
/// assistant `tool_calls`, and tool-result `tool_call_id`. With
/// `cache_system_prompt`, system messages are marked with the Anthropic
/// `cache_control` hint so a stable system prefix hits the provider's prompt
/// cache across turns.
fn to_genai_messages(
    provider: &str,
    messages: &[Message],
    cache_system_prompt: bool,
) -> Result<Vec<genai::chat::ChatMessage>, ProviderError> {
    // Tool-call id -> originating tool name, so a later Role::Tool message can
    // report its fn_name even when the provider that issued the call differs
    // from the one this conversation is now being replayed to (cross-provider
    // failover): call ids are provider-native and mean nothing to a different
    // provider's adapter, but the name is stable across providers.
    let call_id_to_fn_name: std::collections::HashMap<&str, &str> = messages
        .iter()
        .flat_map(|m| &m.tool_calls)
        .map(|tc| (tc.id.as_str(), tc.name.as_str()))
        .collect();

    messages
        .iter()
        .map(|m| {
            let mapped = to_genai_message(provider, m, &call_id_to_fn_name)?;
            Ok(if cache_system_prompt && m.role == Role::System {
                mapped.with_options(
                    genai::chat::MessageOptions::default()
                        .with_cache_control(genai::chat::CacheControl::Ephemeral),
                )
            } else {
                mapped
            })
        })
        .collect()
}

/// Map a single domain `Message` onto a genai `ChatMessage`. `call_id_to_fn_name`
/// resolves a `Role::Tool` message's originating tool name (see
/// `to_genai_messages`), independent of which provider issued the call id.
fn to_genai_message(
    provider: &str,
    message: &Message,
    call_id_to_fn_name: &std::collections::HashMap<&str, &str>,
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
            // Gemini stamps every captured thought signature onto the first
            // tool call only (see `from_genai_response`); re-emit them as
            // leading `ThoughtSignature` parts ahead of the tool-call parts —
            // the shape genai's own `ChatMessage::from(Vec<ToolCall>)` builds —
            // so the Gemini adapter's request builder reattaches them instead
            // of sending the continued turn with the signature missing.
            if let Some(signatures) = message
                .tool_calls
                .first()
                .and_then(|tc| tc.thought_signatures.as_ref())
            {
                parts.extend(
                    signatures
                        .iter()
                        .cloned()
                        .map(ContentPart::ThoughtSignature),
                );
            }
            for tc in &message.tool_calls {
                parts.push(ContentPart::ToolCall(to_genai_tool_call(provider, tc)?));
            }
            Ok(ChatMessage::assistant(MessageContent::from_parts(parts)))
        }
        Role::Tool => {
            let call_id = message.tool_call_id.as_deref().unwrap_or("");
            let mut response = ToolResponse::new(call_id.to_string(), message.content.clone());
            if let Some(fn_name) = call_id_to_fn_name.get(call_id) {
                response = response.with_fn_name(*fn_name);
            }
            Ok(ChatMessage::tool(response))
        }
    }
}

/// Parse `json` as a JSON value, mapping a parse failure to `InvalidRequest`
/// with a message built by `describe` from the underlying `serde_json` error.
fn parse_json_or_invalid_request(
    provider: &str,
    json: &str,
    describe: impl FnOnce(&serde_json::Error) -> String,
) -> Result<serde_json::Value, ProviderError> {
    serde_json::from_str(json).map_err(|e| ProviderError::InvalidRequest {
        provider: provider.to_string(),
        message: describe(&e),
    })
}

/// Map a domain `ToolCall` onto a genai `ToolCall`, parsing the JSON arguments
/// text (this crate owns `serde_json`); a parse failure is an `InvalidRequest`.
fn to_genai_tool_call(
    provider: &str,
    tc: &ToolCall,
) -> Result<genai::chat::ToolCall, ProviderError> {
    let fn_arguments = parse_json_or_invalid_request(provider, &tc.arguments_json, |e| {
        format!("tool call `{}` has invalid arguments JSON: {e}", tc.name)
    })?;
    Ok(genai::chat::ToolCall {
        call_id: tc.id.clone(),
        fn_name: tc.name.clone(),
        fn_arguments,
        thought_signatures: tc.thought_signatures.clone(),
    })
}

/// Map domain `Tool`s onto genai tool definitions. The JSON-schema text is
/// parsed here at the crate boundary; a parse failure is an `InvalidRequest`.
fn to_genai_tools(provider: &str, tools: &[Tool]) -> Result<Vec<genai::chat::Tool>, ProviderError> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let schema = parse_json_or_invalid_request(provider, &t.parameters_schema_json, |e| {
            format!("tool `{}` has invalid parameter schema: {e}", t.name)
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
    let has_tool_calls = response
        .content
        .iter()
        .any(|p| matches!(p, genai::chat::ContentPart::ToolCall(_)));
    if has_tool_calls {
        let mapped = response
            .into_tool_calls()
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.call_id,
                name: tc.fn_name,
                arguments_json: tc.fn_arguments.to_string(),
                thought_signatures: tc.thought_signatures,
            })
            .collect();
        return Ok(ProviderResponse::ToolCalls(mapped));
    }

    match response.into_first_text() {
        Some(text) => Ok(ProviderResponse::Text(text)),
        None => Err(ProviderError::InvalidResponse {
            provider: provider.to_string(),
            message: "response contained neither text nor tool calls".to_string(),
        }),
    }
}

/// Map genai's optional/signed token counts onto the domain `TokenUsage`.
/// `None` when the provider reported nothing at all.
fn usage_from(usage: &genai::chat::Usage) -> Option<TokenUsage> {
    if usage.prompt_tokens.is_none() && usage.completion_tokens.is_none() {
        return None;
    }
    let clamp = |n: Option<i32>| n.map(|n| n.max(0) as u64).unwrap_or(0);
    Some(TokenUsage {
        prompt_tokens: clamp(usage.prompt_tokens),
        completion_tokens: clamp(usage.completion_tokens),
    })
}

/// Build the genai `ChatRequest`/`ChatOptions` pair from already-mapped
/// messages/tools and the domain `CallOptions`: attaches tools when
/// non-empty, always captures usage, and maps `max_tokens` /
/// `force_tool` onto their genai counterparts (`force_tool` becomes a
/// `ToolChoice::tool`). Pure — no I/O — unlike `complete_with_tools`, which
/// also performs the network call.
fn build_chat_request(
    chat_messages: Vec<genai::chat::ChatMessage>,
    genai_tools: Vec<genai::chat::Tool>,
    options: &CallOptions,
) -> (genai::chat::ChatRequest, genai::chat::ChatOptions) {
    let mut request = genai::chat::ChatRequest::new(chat_messages);
    if !genai_tools.is_empty() {
        request = request.with_tools(genai_tools);
    }

    let mut chat_options = genai::chat::ChatOptions::default().with_capture_usage(true);
    if let Some(max_tokens) = options.max_tokens {
        chat_options = chat_options.with_max_tokens(max_tokens);
    }
    if let Some(tool) = &options.force_tool {
        chat_options = chat_options.with_tool_choice(genai::chat::ToolChoice::tool(tool.clone()));
    }

    (request, chat_options)
}

impl LlmProvider for GenaiProvider {
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &CallOptions,
    ) -> Result<Completion, ProviderError> {
        let chat_messages = to_genai_messages(&self.name, messages, self.cache_system_prompt)?;
        let genai_tools = to_genai_tools(&self.name, tools)?;
        let (request, chat_options) = build_chat_request(chat_messages, genai_tools, options);

        match self
            .client
            .exec_chat(self.model.as_str(), request, Some(&chat_options))
            .await
        {
            Ok(response) => {
                let usage = usage_from(&response.usage);
                let response = from_genai_response(&self.name, response)?;
                Ok(Completion { response, usage })
            }
            Err(err) => Err(classify_genai_error(&self.name, &err)),
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
        let shared = GenaiProvider::build_shared(entry, cfg.https_proxy.as_deref())?;
        let mut models = Vec::with_capacity(entry.models.len());
        for model in &entry.models {
            models.push((model.clone(), GenaiProvider::bind_model(&shared, model)));
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
        let e = classify_error_facts("primary", facts(Some(429), None, "slow down"));
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
            facts(Some(429), Some("insufficient_quota"), "no credit"),
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
        let e = classify_error_facts("p", facts(Some(401), None, "bad key"));
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
        let e = classify_error_facts("p", facts(Some(400), None, "bad body"));
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
        let e = classify_error_facts("p", facts(None, None, "connection reset"));
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
        let e = classify_error_facts("p", facts(Some(429), None, "credit balance is too low"));
        assert_eq!(
            e,
            ProviderError::QuotaExceeded {
                provider: "p".to_string(),
                message: "credit balance is too low".to_string(),
            }
        );
    }

    #[test]
    fn to_genai_messages_resolves_tool_response_fn_name_across_providers() {
        // The call id looks like an Anthropic-native id (as opposed to this
        // crate's own synthetic ids), simulating a history built against one
        // provider and replayed to another on cross-provider failover.
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "toolu_01ABC".to_string(),
                    name: "read_file".to_string(),
                    arguments_json: "{}".to_string(),
                    thought_signatures: None,
                }],
                tool_call_id: None,
            },
            Message {
                role: Role::Tool,
                content: "file contents".to_string(),
                tool_calls: vec![],
                tool_call_id: Some("toolu_01ABC".to_string()),
            },
        ];

        let chat_messages =
            to_genai_messages("gemini", &messages, false).expect("mapping succeeds");
        let responses = chat_messages[1].content.tool_responses();
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].fn_name.as_deref(), Some("read_file"));
    }

    #[test]
    fn build_chat_request_defaults_to_no_tools_no_cap_and_captures_usage() {
        let (request, chat_options) = build_chat_request(vec![], vec![], &CallOptions::default());
        assert!(request.tools.is_none());
        assert_eq!(chat_options.capture_usage, Some(true));
        assert_eq!(chat_options.max_tokens, None);
        assert_eq!(chat_options.tool_choice, None);
    }

    #[test]
    fn build_chat_request_attaches_tools_when_present() {
        let tools = vec![genai::chat::Tool::new("search")];
        let (request, _) = build_chat_request(vec![], tools, &CallOptions::default());
        assert_eq!(request.tools.map(|t| t.len()), Some(1));
    }

    #[test]
    fn build_chat_request_maps_max_tokens() {
        let options = CallOptions {
            max_tokens: Some(256),
            ..Default::default()
        };
        let (_, chat_options) = build_chat_request(vec![], vec![], &options);
        assert_eq!(chat_options.max_tokens, Some(256));
    }

    #[test]
    fn build_chat_request_maps_force_tool_to_tool_choice() {
        let options = CallOptions {
            force_tool: Some("search".to_string()),
            ..Default::default()
        };
        let (_, chat_options) = build_chat_request(vec![], vec![], &options);
        assert_eq!(
            chat_options.tool_choice,
            Some(genai::chat::ToolChoice::tool("search"))
        );
    }

    #[test]
    fn error_message_never_echoes_a_key() {
        // The classifier only ever copies the provided message; assert it does
        // not fabricate secrets and preserves the given text verbatim.
        let e = classify_error_facts("p", facts(Some(500), None, "server error"));
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
        let provider =
            GenaiProvider::from_config(&cfg, &cfg.models[0], None).expect("build provider");
        let msgs = vec![Message {
            role: Role::User,
            content: "Say hello.".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        }];
        let resp = provider
            .complete_with_tools(&msgs, &[], &CallOptions::default())
            .await;
        assert!(resp.is_ok(), "live call failed: {resp:?}");
    }
}
