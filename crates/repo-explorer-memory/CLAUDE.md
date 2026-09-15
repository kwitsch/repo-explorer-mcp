# repo-explorer-memory (MemoryBackend)

`MemoryBackend` implementation backed by an `rmcp` client to
`codebase-memory-mcp`. Owns the `rmcp` dependency — core does not.

`get_architecture_text` is the raw-text sibling of `get_architecture`:
identical wire call (`{project}` only, via `call_memory_tool_with`), but
mapped with `raw_text_summary` instead of `findings_and_summary`. It exists
because `text_table_findings` drops every section whose `(cols: ...)` list has
no `file`/`path` column — `node_labels:`, `edge_types:`, `packages:` — which
is precisely the data the agent's Stage-5 repo brief is rendered from.
`get_architecture` itself keeps the lossy mapper on purpose: it is also the
in-loop LLM tool, where the full architecture text as a tool result would cost
more tokens than the brief saves.
