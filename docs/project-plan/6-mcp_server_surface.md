# Stage 6 — MCP Server Surface

## Ziel

Alles aus Stage 1–5 als einen einzigen MCP-Tool-Aufruf (`explore_repository` o. ä.) über `rmcp` in `repo-explorer-mcp` verfügbar machen — der Toolcall, der den Explore-Agenten ersetzt.

## Deliverables

- **Tool-Definition**:
  - Input: `query: string` (Freitext-Explorationsauftrag), optional `scope_hint: string` (Pfad-Einschränkung), optional `max_results: number`.
  - Output: strukturiertes JSON passend zu `ExplorationResult` — Dateien mit Zeilennummern, optionalem Snippet/Kontext, plus `summary`.
- **Server-Bootstrapping** in `repo-explorer-mcp`:
  - Config laden (Pfad via Env-Var/CLI-Arg, Default `./repo-explorer.toml`).
  - `repo-explorer-memory`, Such-Layer, Provider-Router, `AgentLoop` verdrahten.
  - Tracing/Logging gemäß `[logging]`-Config.
  - Klare Startup-Fehler bei fehlender/ungültiger Config oder nicht erreichbarem `codebase-memory-mcp`.
- **`.mcp.json`**: jetzt sinnvoll (Server ist lauffähig) — Eintrag ergänzen, der `repo-explorer-mcp` registriert, damit Claude Code den neuen Tool-Call anstelle des eingebauten Explore-Agenten nutzen kann. Hinweis in `CLAUDE.md` („`.mcp.json` ist absichtlich abwesend, bis der Server lauffähig ist“) entsprechend auflösen/aktualisieren.

## Out of Scope

Cross-Platform-Packaging (Stage 7).

## Abhängigkeiten

Stage 1–5.

## Abnahmekriterien

- Manueller Smoke-Test: Server starten, Tool via MCP-Inspector oder Claude Code aufrufen, strukturierte Ausgabe prüfen.
- `cargo build/test/clippy/fmt` über den gesamten Workspace weiterhin grün.
