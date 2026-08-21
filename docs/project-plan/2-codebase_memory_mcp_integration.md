# Stage 2 — codebase-memory-mcp Integration

## Ziel

Primäre Explorationsquelle anbinden: das upstream `codebase-memory-mcp`. Der Server stellt eigenständig sicher, dass das Arbeitsverzeichnis einen aktuellen Index hat, bevor exploriert wird.

## Architektur-Entscheidung

`repo-explorer-core` darf keine `rmcp`-Abhängigkeit bekommen (Konvention). Der eigentliche MCP-Client für `codebase-memory-mcp` gehört daher **nicht** in Core. Empfehlung: neues Crate `crates/repo-explorer-memory`, das:

- den `MemoryBackend`-Trait aus Core implementiert,
- als MCP-Client via `rmcp` gegen `codebase-memory-mcp` spricht.

Core definiert nur den Trait, gegen den der Rest des Systems (inkl. Stage 5) programmiert.

## Deliverables

- **Trait in Core**: `MemoryBackend` mit u. a.
  - `ensure_fresh_index(&self, repo_root: &Path) -> Result<IndexStatus, MemoryError>` (kombiniert `index_status` / `detect_changes`, triggert `index_repository` bei Staleness über dem konfigurierten Schwellenwert)
  - `search_code`, `search_graph`, `query_graph`, `trace_path`, `get_architecture`, `get_code_snippet` — jeweils schlanke Signaturen, die auf die Domänentypen aus Stage 1 abbilden.
- **Implementierung** in `crates/repo-explorer-memory` gegen die echten `codebase-memory-mcp`-Tools.
- **Staleness-Policy**: bei Sessionstart einmal `ensure_fresh_index` aufrufen; Ergebnis (indexiert/aktuell/fehlgeschlagen) wird nach oben durchgereicht statt die Exploration hart zu blockieren.
- **Mock-Implementierung** für Tests (`repo-explorer-core` test-utils oder eigenes `dev-dependencies`-Fixture).

## Out of Scope

ripgrep/rtk-Fallback (Stage 3), LLM-Loop (Stage 5).

## Abhängigkeiten

Stage 1 (Domänentypen, Fehlerkonvention).

## Abnahmekriterien

- Unit-Tests gegen Mock-`MemoryBackend`.
- Optionaler, standardmäßig `#[ignore]`d Integrationstest gegen eine laufende `codebase-memory-mcp`-Instanz.
- `cargo clippy`/`fmt` weiterhin sauber über den gesamten Workspace, inkl. neuem Crate.
