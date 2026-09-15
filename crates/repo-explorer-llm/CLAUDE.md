# repo-explorer-llm (GenaiProvider)

`GenaiProvider` (the sole `LlmProvider` impl) backed by the `genai` crate;
provides `build_router(&LlmConfig)`. Owns the `genai` dependency — core does
not. Every `genai::*` reference must stay confined to this crate.

## Prompt caching

Anthropic-only and message-level: `bind_model` sets `cache_system_prompt` iff
the adapter is `AdapterKind::Anthropic`, and `to_genai_messages` marks every
`Role::System` message with `CacheControl::Ephemeral`. genai's Gemini adapter
ignores message-level cache control entirely and its OpenAI adapter honours
only request-level hints, so there is nothing to enable for either. Do not
route this through `ChatOptions::with_cache_control` (genai drops request-level
cache control for Anthropic with an `info!`) and do not move the system prompt
to `ChatRequest::with_system` (that field can never carry cache control).

The marker only pays off while the cached prefix — tools, then system prompt,
in Anthropic's prefix order — stays byte-identical across turns and queries.
That is why both agent system prompts are per-run-content-free `const &str`
and the tool catalogs are `LazyLock` statics in fixed order. Guarded by
`to_genai_messages_marks_only_the_system_message` /
`to_genai_messages_marks_nothing_when_caching_is_off` /
`bind_model_enables_prompt_caching_for_anthropic_only` here, and by
`verify::tests::verify_cache_prefix_is_byte_stable` /
`agent::tests::fallback_cache_prefix_is_byte_stable` in `repo-explorer-agent`.

Measured prefix sizes (those two tests pin the numbers): verify stage 1627
content bytes (~430-510 tokens), fallback loop 5387 (~1.5-1.7k tokens).
Anthropic ignores a cache breakpoint whose prefix is under 1024 tokens (2048
for Haiku), so caching currently does nothing on the verify stage — the hot
path — and only helps the fallback loop on Sonnet/Opus.
