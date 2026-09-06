# repo-explorer-llm (GenaiProvider)

`GenaiProvider` (the sole `LlmProvider` impl) backed by the `genai` crate;
provides `build_router(&LlmConfig)`. Owns the `genai` dependency — core does
not. Every `genai::*` reference must stay confined to this crate.
