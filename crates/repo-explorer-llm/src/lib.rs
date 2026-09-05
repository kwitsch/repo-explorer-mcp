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
    /// Human-readable, secret-free message.
    pub message: String,
}

/// Substrings that indicate quota/billing exhaustion rather than transient
/// rate limiting, checked (lowercased) against the flattened error message —
/// `genai` exposes no structured error code to disambiguate on instead.
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

    match facts.status {
        // 429 is standard rate-limiting; 529 is Anthropic's overloaded_error
        // and 503 UNAVAILABLE is Gemini's "experiencing high demand" — both
        // transient-overload signals that should failover the same way; no
        // status at all is a connection/transport failure. All of them report
        // QuotaExceeded instead when the message clearly names a quota/billing
        // condition.
        Some(429) | Some(503) | Some(529) | None if message_indicates_quota(&message) => {
            ProviderError::QuotaExceeded { provider, message }
        }
        Some(429) | Some(503) | Some(529) => ProviderError::RateLimited { provider, message },
        Some(401) | Some(403) => ProviderError::Authentication { provider, message },
        // A model-scoped completion endpoint returning 404 always means "this
        // model id doesn't exist / was retired" (never a malformed-body
        // issue — that's 400), across OpenAI, Anthropic and Gemini alike. So
        // unlike 400/422, it's specific to the one model slot: failover to
        // the next configured model instead of failing the whole request.
        Some(404) => ProviderError::ModelUnavailable { provider, message },
        Some(400) | Some(422) => ProviderError::InvalidRequest { provider, message },
        // Other 5xx, any other status, and no status at all: transport-level.
        Some(_) | None => ProviderError::Transport { provider, message },
    }
}

/// Adapt a `genai::Error` into genai-independent facts, then classify.
///
/// `genai` 0.6.x surfaces an HTTP status on `Error::HttpError` and, indirectly,
/// on the web-call variants whose `webc::Error::ResponseFailedStatus` carries
/// one. It does not preserve a structured provider error *code*, so
/// classification relies on status plus substring matching. The flattened
/// `message` is the SDK's `Display`, which contains provider error bodies but
/// never our API key.
pub(crate) fn classify_genai_error(provider: &str, err: &genai::Error) -> ProviderError {
    let facts = GenaiErrorFacts {
        status: extract_status(err),
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
    adapter_kind: genai::adapter::AdapterKind,
    /// Mark system messages with a provider-side prompt-cache hint. Only the
    /// Anthropic adapter supports the content-level `cache_control` marker.
    cache_system_prompt: bool,
    /// 1-based call counter for this provider/model slot, logged as
    /// `provider call`'s `attempt` field — exists purely to be logged.
    call_counter: std::sync::atomic::AtomicU64,
}

impl GenaiProvider {
    /// The one place a `GenaiProvider` is assembled from its parts, so
    /// `build_router` (which reuses one client across an entry's models)
    /// and its test-only single-provider callers cannot drift as fields are
    /// added.
    fn bind_model(shared: &SharedProviderParts, model: &str) -> Self {
        Self {
            name: shared.name.clone(),
            model: model.to_string(),
            client: shared.client.clone(),
            adapter_kind: shared.adapter_kind,
            cache_system_prompt: shared.adapter_kind == genai::adapter::AdapterKind::Anthropic,
            call_counter: std::sync::atomic::AtomicU64::new(0),
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

    let mapped = messages
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
        .collect::<Result<Vec<_>, ProviderError>>()?;

    Ok(merge_consecutive_tool_messages(mapped))
}

/// Merge runs of consecutive `ChatRole::Tool` messages into one, combining
/// their content parts (each is a single `ToolResponse` — see
/// `to_genai_message`). Unlike genai's Gemini adapter, which explicitly
/// merges consecutive tool-response entries before building its request, the
/// Anthropic adapter has no such step: `N` separate `Role::Tool` domain
/// messages in a row (one per tool call in a batched turn — see
/// `repo-explorer-agent`'s fallback loop and `verify` stage) would otherwise
/// become `N` standalone `role:user` JSON messages with no assistant message
/// between them, a shape Anthropic's Messages API rejects as malformed.
/// Merging here, before any adapter sees the messages, keeps every adapter's
/// request well-formed regardless of whether it merges on its own.
fn merge_consecutive_tool_messages(
    messages: Vec<genai::chat::ChatMessage>,
) -> Vec<genai::chat::ChatMessage> {
    use genai::chat::ChatRole;

    let mut out: Vec<genai::chat::ChatMessage> = Vec::with_capacity(messages.len());
    for msg in messages {
        if msg.role == ChatRole::Tool
            && let Some(prev) = out.last_mut()
            && prev.role == ChatRole::Tool
        {
            prev.content.extend(msg.content);
        } else {
            out.push(msg);
        }
    }
    out
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
            let call_id = message
                .tool_call_id
                .as_deref()
                .expect("Role::Tool message always carries tool_call_id");
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
        // Real signatures are carried by the leading `ThoughtSignature` parts
        // promoted in `to_genai_message`; no adapter this crate ever selects
        // reads this field off a wrapped `ContentPart::ToolCall`.
        thought_signatures: None,
    })
}

/// Map domain `Tool`s onto genai tool definitions. The JSON-schema text is
/// parsed here at the crate boundary; a parse failure is an `InvalidRequest`.
fn to_genai_tools(
    provider: &str,
    adapter_kind: genai::adapter::AdapterKind,
    tools: &[Tool],
) -> Result<Vec<genai::chat::Tool>, ProviderError> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        let mut schema = parse_json_or_invalid_request(provider, &t.parameters_schema_json, |e| {
            format!("tool `{}` has invalid parameter schema: {e}", t.name)
        })?;
        if adapter_kind == genai::adapter::AdapterKind::Gemini {
            strip_additional_properties(&mut schema);
        }
        out.push(
            genai::chat::Tool::new(t.name.clone())
                .with_description(t.description.clone())
                .with_schema(schema),
        );
    }
    Ok(out)
}

/// Gemini's `functionDeclarations[].parameters` is an OpenAPI-subset `Schema`
/// proto with no `additionalProperties` field: the API rejects the whole
/// request with HTTP 400 `Unknown name "additionalProperties"` when it appears
/// anywhere in the tree, and genai 0.6 forwards it verbatim. Every catalog
/// schema in `repo-explorer-agent` sets it (the dispatcher is
/// `deny_unknown_fields`), so it's dropped here for Gemini only — the other
/// adapters accept it and benefit from the hint.
fn strip_additional_properties(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(map) => {
            map.remove("additionalProperties");
            map.values_mut().for_each(strip_additional_properties);
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(strip_additional_properties),
        _ => {}
    }
}

/// Map a genai chat response to a `ProviderResponse`: tool calls take priority,
/// then text; an empty/unusable response is an `InvalidResponse`.
fn from_genai_response(
    provider: &str,
    response: genai::chat::ChatResponse,
) -> Result<ProviderResponse, ProviderError> {
    if response.content.contains_tool_call() {
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

/// Bucket a `ProviderError` into the six `provider call` outcome labels,
/// reusing the same rate/quota/auth/invalid classification `GenaiProvider`
/// already applies via `classify_genai_error` — never a new one.
fn outcome_label(err: &ProviderError) -> &'static str {
    match err {
        ProviderError::RateLimited { .. } => "rate_limited",
        ProviderError::QuotaExceeded { .. } => "quota",
        ProviderError::ModelUnavailable { .. } => "model_unavailable",
        ProviderError::Authentication { .. } => "auth",
        ProviderError::InvalidRequest { .. } => "invalid",
        ProviderError::Transport { .. }
        | ProviderError::InvalidResponse { .. }
        | ProviderError::Configuration { .. } => "other",
    }
}

impl LlmProvider for GenaiProvider {
    async fn complete_with_tools(
        &self,
        messages: &[Message],
        tools: &[Tool],
        options: &CallOptions,
    ) -> Result<Completion, ProviderError> {
        let chat_messages = to_genai_messages(&self.name, messages, self.cache_system_prompt)?;
        let genai_tools = to_genai_tools(&self.name, self.adapter_kind, tools)?;
        let (request, chat_options) = build_chat_request(chat_messages, genai_tools, options);

        let attempt = self
            .call_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let start = std::time::Instant::now();
        match self
            .client
            .exec_chat(self.model.as_str(), request, Some(&chat_options))
            .await
        {
            Ok(response) => {
                let latency_ms = start.elapsed().as_millis() as u64;
                let usage = usage_from(&response.usage);
                let reasoning_tokens = response
                    .usage
                    .completion_tokens_details
                    .as_ref()
                    .and_then(|d| d.reasoning_tokens);
                let model_served = response.provider_model_iden.model_name.as_str();
                let prompt_tokens = usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0);
                let completion_tokens = usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0);
                match reasoning_tokens {
                    Some(reasoning_tokens) => tracing::info!(
                        provider = %self.name,
                        model_requested = %self.model,
                        model_served,
                        attempt,
                        outcome = "ok",
                        latency_ms,
                        prompt_tokens,
                        completion_tokens,
                        reasoning_tokens,
                        "provider call"
                    ),
                    None => tracing::info!(
                        provider = %self.name,
                        model_requested = %self.model,
                        model_served,
                        attempt,
                        outcome = "ok",
                        latency_ms,
                        prompt_tokens,
                        completion_tokens,
                        "provider call"
                    ),
                }
                let response = from_genai_response(&self.name, response)?;
                Ok(Completion { response, usage })
            }
            Err(err) => {
                let latency_ms = start.elapsed().as_millis() as u64;
                let classified = classify_genai_error(&self.name, &err);
                tracing::info!(
                    provider = %self.name,
                    model_requested = %self.model,
                    attempt,
                    outcome = outcome_label(&classified),
                    latency_ms,
                    "provider call"
                );
                Err(classified)
            }
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

    #[test]
    fn to_genai_tools_strips_additional_properties_for_gemini_only() {
        use genai::adapter::AdapterKind;
        // Regression: Gemini answered 400 "Unknown name additionalProperties"
        // to every tool-bearing request, since its function-declaration
        // schema proto has no such field and genai forwards it verbatim.
        let tool = Tool {
            name: "grep".to_string(),
            description: String::new(),
            parameters_schema_json: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "opts": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "additionalProperties": false
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            })
            .to_string(),
        };

        let gemini = to_genai_tools("p", AdapterKind::Gemini, std::slice::from_ref(&tool)).unwrap();
        let schema = gemini[0].schema.as_ref().unwrap();
        assert!(schema.get("additionalProperties").is_none());
        assert!(
            schema["properties"]["opts"]
                .get("additionalProperties")
                .is_none()
        );
        assert_eq!(schema["required"], serde_json::json!(["pattern"]));

        let anthropic =
            to_genai_tools("p", AdapterKind::Anthropic, std::slice::from_ref(&tool)).unwrap();
        let schema = anthropic[0].schema.as_ref().unwrap();
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["opts"]["additionalProperties"], false);
    }

    fn facts(status: Option<u16>, message: &str) -> GenaiErrorFacts {
        GenaiErrorFacts {
            status,
            message: message.to_string(),
        }
    }

    #[test]
    fn http_429_classifies_as_rate_limited() {
        let e = classify_error_facts("primary", facts(Some(429), "slow down"));
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
        // `genai` preserves no structured error code, so this relies on the
        // SDK's flattened message text containing the quota phrasing.
        let e = classify_error_facts("p", facts(Some(429), "insufficient_quota: no credit"));
        assert_eq!(
            e,
            ProviderError::QuotaExceeded {
                provider: "p".to_string(),
                message: "insufficient_quota: no credit".to_string(),
            }
        );
    }

    #[test]
    fn http_401_classifies_as_authentication_and_is_not_failover() {
        let e = classify_error_facts("p", facts(Some(401), "bad key"));
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
        let e = classify_error_facts("p", facts(Some(400), "bad body"));
        assert_eq!(
            e,
            ProviderError::InvalidRequest {
                provider: "p".to_string(),
                message: "bad body".to_string(),
            }
        );
    }

    #[test]
    fn http_503_classifies_as_rate_limited_so_the_next_model_is_tried() {
        // Regression: Gemini's 503 UNAVAILABLE "experiencing high demand" was a
        // Transport error, which the router surfaces immediately — the other
        // configured models were never tried.
        let e = classify_error_facts("p", facts(Some(503), "high demand"));
        assert!(e.is_failover_trigger(), "{e:?}");
    }

    #[test]
    fn http_404_classifies_as_model_unavailable_so_the_next_model_is_tried() {
        // Regression: a retired/renamed model id (e.g. Gemini's
        // "gemini-2.5-flash is no longer available to new users") answered
        // every request with 404, which used to classify as InvalidRequest —
        // the router fails fast on that, so every model configured after the
        // dead one in the chain was never tried.
        let e = classify_error_facts("p", facts(Some(404), "model not found"));
        assert_eq!(
            e,
            ProviderError::ModelUnavailable {
                provider: "p".to_string(),
                message: "model not found".to_string(),
            }
        );
        assert!(e.is_failover_trigger(), "{e:?}");
    }

    #[test]
    fn no_status_falls_back_to_transport() {
        let e = classify_error_facts("p", facts(None, "connection reset"));
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
        let e = classify_error_facts("p", facts(Some(429), "credit balance is too low"));
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
    fn to_genai_messages_batches_consecutive_tool_results_into_one_message() {
        // Mirrors the shape the fallback loop's batched-tool-call turn
        // produces (agent.rs): one assistant message issuing 2+ tool calls,
        // followed by one `Role::Tool` domain message per call. Anthropic
        // requires every tool result following a multi-tool-call assistant
        // turn to land in a single `role:user` message, so the two
        // consecutive `Role::Tool` messages here must collapse into one
        // genai `ChatMessage` instead of staying two.
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: String::new(),
                tool_calls: vec![
                    ToolCall {
                        id: "toolu_01A".to_string(),
                        name: "search_code".to_string(),
                        arguments_json: "{}".to_string(),
                        thought_signatures: None,
                    },
                    ToolCall {
                        id: "toolu_01B".to_string(),
                        name: "search_graph".to_string(),
                        arguments_json: "{}".to_string(),
                        thought_signatures: None,
                    },
                ],
                tool_call_id: None,
            },
            Message {
                role: Role::Tool,
                content: "code results".to_string(),
                tool_calls: vec![],
                tool_call_id: Some("toolu_01A".to_string()),
            },
            Message {
                role: Role::Tool,
                content: "graph results".to_string(),
                tool_calls: vec![],
                tool_call_id: Some("toolu_01B".to_string()),
            },
        ];

        let chat_messages =
            to_genai_messages("anthropic", &messages, false).expect("mapping succeeds");

        // Assistant turn + exactly one merged tool-result message, never two.
        assert_eq!(chat_messages.len(), 2);
        assert_eq!(chat_messages[1].role, genai::chat::ChatRole::Tool);
        let responses = chat_messages[1].content.tool_responses();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].call_id, "toolu_01A");
        assert_eq!(responses[0].content, "code results");
        assert_eq!(responses[1].call_id, "toolu_01B");
        assert_eq!(responses[1].content, "graph results");
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
        let e = classify_error_facts("p", facts(Some(500), "server error"));
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
        let shared = GenaiProvider::build_shared(&cfg, None).expect("build shared parts");
        let provider = GenaiProvider::bind_model(&shared, &cfg.models[0]);
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
