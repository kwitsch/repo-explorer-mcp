# Stage 1 — Workspace Foundations & Config

## Ziel

`repo-explorer-core` erhält die grundlegenden Domänentypen und die Konfigurationsschicht, ohne jede MCP-/Transport-Abhängigkeit (`rmcp` bleibt tabu in diesem Crate, siehe `.claude/rules/rust-conventions.md`).

## Deliverables

- **Config-Schema** (TOML, geladen via `serde` + `toml`):
  - `[[llm.providers]]` — geordnete Liste, je Eintrag: `name`, `kind` (`anthropic` | `openai` | `google` | …), `api_key_env`, `model`, optional `base_url`. Reihenfolge in der Datei = Failover-Reihenfolge.
  - `[llm]` — `cooldown_seconds` (wie lange ein Provider nach Limit-Fehler übersprungen wird).
  - `[codebase_memory]` — Verbindungsangaben zum upstream `codebase-memory-mcp` (Command/Args oder Endpoint), Staleness-Schwellenwert.
  - `[search]` — Pfade/Erkennung für `rtk` und `ripgrep`, Timeout, `prefer_rtk: bool`.
  - `[logging]` — Level.
- **Config-Loader** in `repo-explorer-core`: Parsing + Validierung (Provider-Liste nicht leer, eindeutige Namen, referenzierte Env-Vars vorhanden), mit Fixture-Tests.
- **Domänentypen** (kein I/O):
  - `FileLocation { path, line_start, line_end }`
  - `ExplorationFinding { location, snippet: Option<String>, note: Option<String> }`
  - `ExplorationQuery { text, scope_hint: Option<PathBuf>, max_results: Option<u32> }`
  - `ExplorationResult { findings: Vec<ExplorationFinding>, summary: String }`
- **Fehlerbehandlung-Entscheidung**: `thiserror` für `repo-explorer-core` (typisierte `ConfigError`, `ValidationError`), `anyhow` an der Binary-Grenze (`repo-explorer-mcp`). Entscheidung in `.claude/rules/rust-conventions.md` nachtragen, sobald umgesetzt.

## Out of Scope

Tatsächliche Provider-Calls, Suchausführung, MCP-Wiring — folgen in späteren Stufen.

## Abhängigkeiten

Keine — Startpunkt.

## Abnahmekriterien

- `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check` sind grün.
- Unit-Tests: gültige Config lädt korrekt; fehlende Env-Var/leere Provider-Liste/doppelte Namen schlagen mit sprechenden Fehlern fehl.
