# Response shape

`explore_repository` returns its answer as MCP `structuredContent` (the schema
is advertised as the tool's `outputSchema`, derived from the response type):

```json
{
  "findings": [
    {
      "location": {
        "path": "crates/repo-explorer-core/src/domain.rs",
        "line_start": 27,
        "line_end": 31
      },
      "snippet": "pub struct FileLocation {\n    pub path: PathBuf,\n    pub line_start: u32,\n    pub line_end: u32,\n}",
      "note": "exact symbol match: `FileLocation`",
      "symbol": "repo_explorer_core.domain.FileLocation"
    },
    {
      "location": { "path": "crates/repo-explorer-agent/src/render.rs" },
      "note": "graph row with no resolvable line"
    }
  ],
  "summary": "FileLocation is defined in core's domain module and re-used by every backend.",
  "retrieval_confidence": 92,
  "stage_exit": "early-exit"
}
```

- `retrieval_confidence` (0-100) scores the deterministic pre-stage's
  **candidate set**, not the answer: a low value means the answer came from the
  explorative fallback loop, not that it is wrong.
- `stage_exit` is one of `"early-exit"`, `"verify"`, `"fallback"`, `"cache"` —
  the same literal the server logs as `path` for that call.
- `line_start`/`line_end`, `snippet`, `note` and `symbol` are **omitted** (never
  `null`) when unknown. A missing `line_start` means the backend had no
  resolvable line; a missing `symbol` means no single ranked candidate
  overlapped that location.

## Parsing this from a client

`content[0].text` carries the same object as compact JSON, so a client without
`structuredContent` support loses nothing:

```python
payload = result.structuredContent or json.loads(result.content[0].text)
for f in payload["findings"]:
    loc = f["location"]
    print(loc["path"], loc.get("line_start"), f.get("symbol"), sep=":")
print(payload["summary"], payload["stage_exit"], payload["retrieval_confidence"])
```

Nothing in this response shape is configurable — there is no config key for it.
