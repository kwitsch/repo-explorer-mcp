# repo-explorer-memory (MemoryBackend)

`MemoryBackend` implementation backed by an `rmcp` client to
`codebase-memory-mcp` (CBM). Owns the `rmcp` dependency — core does not.

## Project resolution

CBM names projects itself from the path handed to `index_repository`
(`/home/k/repos/x` → `home-k-repos-x`, with its own normalization) and shares
one index across every client of a repo. The crate therefore never derives a
name: `cached_project_name` resolves a repo root by paging
`list_projects{format:"json"}` and matching `root_path` against the canonical
root (`MemoryClient::find_project_by_root`); a root with no project is indexed
first and the name read from `index_repository`'s response (`indexed_project`).
`index_repository` is sent without `name` on purpose — passing one would
build a private duplicate of the plugin's index. (Before this, a locally
derived `<basename>-<sha8>` name existed under no CBM project, so every call
re-indexed and every memory tool failed with "project not found".)

## Wire format

Every call that reads paths asks for `format:"json"`. CBM 0.11.0's default
tree text factors shared prefixes into `X_refs:` tables and `@N+suffix`
cells (`@2+pipeline.rs` is not a path), and the `index_status`/
`detect_changes` tree text decodes to a `Value::String` no field lookup can
read (which forced a reindex on every call). Shapes, all pinned by tests
with real payloads: `search_graph` columnar `{cols, groups}`; `search_code`
and `query_graph` flat `{cols|columns, rows:[[..]]}` (`flat_rows_findings`);
`index_status` `{status, indexed_at, root_path}` — `indexed_at` is the last
**full-generation** stamp (incremental refreshes, ours or the daemon's
watcher, do not move it), so it only seeds `decide_freshness` when this
process has not reindexed itself yet; `detect_changes{scope:"files"}`
`{changed_total, changed_files, changed_has_more}` — it diffs the working
tree against the **base branch**, not against the index (`since` takes git
revisions only), so a dirty tree reports the same files forever, and without
`scope:"files"` the impact analysis eats the output budget and the file list
comes back empty. Whether those files are already indexed is answered by
`check_index_coverage{paths}`: every row `freshness` of `metadata_match` (or
`not_tracked`, a file the index ignores anyway) means no reindex
(`coverage_covers_all`). `get_architecture` stays
tree text: the brief parses its sections. Probe any tool against the live
daemon with `codebase-memory-mcp cli --quiet [--json] <tool> '<json>'`.

`get_architecture_text` is the raw-text sibling of `get_architecture`:
identical wire call (`{project}` only, via `call_memory_tool_with`), but
mapped with `raw_text_summary` instead of `findings_and_summary`. It exists
because `text_table_findings` drops every section whose `(cols: ...)` list has
no `file`/`path` column — `node_labels:`, `edge_types:`, `packages:` — which
is precisely the data the agent's Stage-5 repo brief is rendered from.
`get_architecture` itself keeps the lossy mapper on purpose: it is also the
in-loop LLM tool, where the full architecture text as a tool result would cost
more tokens than the brief saves.
