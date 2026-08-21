# Stage 5 — Exploration Agent Loop

## Ziel

Das eigentliche Kernstück, das den Claude Code Explore-Agenten durch einen einzigen Toolcall ersetzt: eine interne LLM-gesteuerte Schleife, die `MemoryBackend` (Stage 2) und `SearchBackend` (Stage 3) als Werkzeuge nutzt und am Ende ein strukturiertes `ExplorationResult` (Stage 1) liefert. Lose Inspiration: [fastcontext](https://github.com/manjunathshiva/fastcontext).

## Deliverables

- **Interne Tool-Menge**, die dem LLM angeboten wird (Namen vorläufig):
  - `search_code`, `search_graph`, `query_graph`, `trace_path`, `get_architecture`, `get_code_snippet` (→ `MemoryBackend`)
  - `grep`, `find`, `read_file` (→ `SearchBackend` / rtk-Read)
  - `finish` — erzwungener Abschluss-Tool-Call mit dem finalen `ExplorationResult` (Schema-Validierung analog „Forced Tool Call für strukturierte Ausgabe“).
- **`AgentLoop`** (in Core oder eigenem Crate, je nach Abhängigkeitsschnitt zu `LlmProvider`):
  1. `ensure_fresh_index` einmalig zu Beginn.
  2. Tool-Use-Turns über den `ProviderRouter` (Stage 4), bis `finish` aufgerufen wird oder ein Iterationslimit erreicht ist.
  3. Dispatch jedes Tool-Calls an `MemoryBackend`/`SearchBackend`, Ergebnis zurück in den Message-Verlauf.
  4. Bei Iterationslimit ohne `finish`: bestmögliches Zwischenergebnis + Hinweis im `summary`-Feld statt Hard-Fail.
- **Priorisierung**: `codebase-memory-mcp` ist die primäre Quelle; rtk/ripgrep wird vom Modell (per Tool-Beschreibung/Prompt-Guidance) als Ergänzung/Fallback genutzt, nicht als Ersatz.

## Out of Scope

MCP-Serverfläche nach außen (Stage 6).

## Abhängigkeiten

Stage 1, 2, 3, 4.

## Abnahmekriterien

- Integrationstest mit einem skriptbaren Fake-`LlmProvider` (fester Tool-Call-Ablauf), der Dispatch an Fake-Backends und korrekte Zusammensetzung von `ExplorationResult` verifiziert.
- Test für Iterationslimit-Verhalten.
- Test für Provider-Failover mitten in einer laufenden Exploration (Stage-4-Router wird tatsächlich durchlaufen).
