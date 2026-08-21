# Stage 3 — ripgrep/rtk Such-Layer

## Ziel

Schnelle, textbasierte Fallback-/Ergänzungssuche für Fälle, in denen `codebase-memory-mcp` keine Treffer liefert oder eine literale Muster-Suche gefragt ist.

## Deliverables

- **Trait in Core**: `SearchBackend` mit `search(pattern, scope: Option<PathBuf>, options) -> Result<Vec<ExplorationFinding>, SearchError>`.
- **Implementierung** (vermutlich eigenes Modul in `repo-explorer-memory` oder neues schlankes Crate `repo-explorer-search`, je nach Abhängigkeitsschnitt):
  - Primär: Subprocess gegen `rtk rg` / `rtk grep` / `rtk find` / `rtk read` (bereits token-optimierte Ausgabe, lokal vorhanden unter `~/.local/bin/rtk`).
  - Fallback: direktes `ripgrep` (`rg --json`) falls `rtk` nicht im `PATH` ist — Parsing des JSON-Line-Formats von `rg`.
- **Konfiguration** (Stage 1 Schema erweitern falls nötig): `search.prefer_rtk`, Binary-Pfade, Timeout, Auto-Detection via `which`.
- **Output-Mapping**: rtk/rg-Treffer → `FileLocation` + Snippet, konsistent mit `MemoryBackend`-Ergebnissen, damit Stage 5 beide Quellen uniform behandeln kann.

## Out of Scope

LLM-Entscheidung, wann welche Quelle genutzt wird (Stage 5).

## Abhängigkeiten

Stage 1 (Domänentypen).

## Abnahmekriterien

- Parser-Unit-Tests für `rtk`-Ausgabe und für `rg --json`-Ausgabe anhand von Fixtures.
- Test für den Fallback-Pfad (rtk „nicht gefunden“ → rg wird genutzt).
- Integrationstest gegen ein kleines Fixture-Repo im Testverzeichnis.
