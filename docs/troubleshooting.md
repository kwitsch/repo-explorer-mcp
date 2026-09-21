# Troubleshooting

Config loading fails fast with a named error. `repo-explorer-mcp config test`
prints the error plus the offending TOML key path as JSON.

| Error                                 | Cause                                                | Fix                                        |
| ------------------------------------- | ---------------------------------------------------- | ------------------------------------------ |
| `EmptyProviderList`                   | `llm.providers` is empty                             | Add at least one `[[llm.providers]]`.      |
| `DuplicateProviderName`               | Two providers share a `name`                         | Make each provider `name` unique.          |
| `EmptyModelsList`                     | A provider's `models` list is empty                  | List at least one model ID.                |
| `UnknownProviderKind`                 | `kind` is not `anthropic`/`openai`/`gemini`/`google` | Use one of the supported kinds.            |
| `MissingEnvVar`                       | An `api_key_env` names an unset or blank variable    | `export` the named variable before launch. |
| `MissingCodebaseMemoryConnection`     | Neither `command` nor `endpoint` set                 | Set exactly one under `[codebase_memory]`. |
| `ConflictingCodebaseMemoryConnection` | Both `command` and `endpoint` set                    | Keep exactly one.                          |

See [`smoke-test.md`](smoke-test.md) for verifying a downloaded release artifact end to end.
