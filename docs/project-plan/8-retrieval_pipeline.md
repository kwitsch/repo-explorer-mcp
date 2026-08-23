# Stage 8 — Deterministische Retrieval-Pipeline mit LLM-Verifikation

## Ziel

Alles Deterministische aus dem LLM-Loop (Stage 5) nach Rust verlagern: das LLM
soll nur noch **selektieren/verifizieren**, nicht **suchen**. Aus „Repo
explorieren" wird „Relevanz-Ranking auf vorgefilterten Snippets".

```text
Query → [Rust: Cache-Lookup → Symbol-Lookup + Grep-Fanout → Ranking/Konfidenz]
      → Konfidenz hoch    → Early Exit, 0 LLM-Calls
      → sonst             → Top-k-Kandidaten als Skeletons
                            → [LLM: 1 Verifikations-Turn, optional 1 expand-Turn]
      → Konfidenz niedrig / Verifikation gescheitert
                          → Fallback: explorativer Loop (Stage 5, gehärtet)
      → ExplorationResult (gecacht)
```

## Bausteine

- **Vorstufe** (`core::retrieval` pur + `agent::pipeline` Orchestrierung):
  Pattern-Ableitung aus der Query (Identifier, Literale, Pfad-Tokens),
  nebenläufiger Fanout über `search_graph`(Symbol-Lookup), `search_code`,
  Grep-Legs und Dateinamens-Legs; Merge/Ranking/Konfidenz (0–100) rein
  deterministisch. Backend-Fehler sind weich (Leg liefert nichts).
- **Early Exit**: Konfidenz ≥ `agent.early_exit_confidence` → Ergebnis direkt
  aus den Kandidaten, ohne LLM.
- **Skeleton-Views** (`agent::skeleton`): Datei-Outline (Symbolname +
  Zeilenbereich) aus `search_graph{file_pattern}` statt Full Reads; das
  Memory-Crate transportiert den Symbolnamen in `ExplorationFinding.note`.
- **Verifikations-Stage** (`agent::verify`): statischer System-Prompt,
  Katalog nur `expand` + `finish`; letzter Turn erzwingt `finish` via
  `CallOptions.force_tool`. Jeder Fehler eskaliert in den Fallback-Loop.
- **Fallback-Loop-Härtung** (Stage-5-Loop): Kandidaten als Startpunkte in der
  User-Message, gemeinsames Token-Budget (`agent.token_budget`, Usage-Capture
  über `Completion.usage`), Batch-Zwang (Einzel-Call-Turns werden abgewiesen,
  2-Strike-Akzeptanz als Deadlock-Schutz), nebenläufige Batch-Ausführung,
  Forced-Final-`finish` bei Budget-Ende mit deterministischer Synthese aus
  Kandidaten + gesammelten Findings als letzter Stufe.
- **Output-Kompression** (`agent::render`): Pfad-Normalisierung (kein `./`),
  Dedupe pro Location, Snippet-Cap (`agent.snippet_max_chars`),
  `read_file`-Zeilen-Cap mit explizitem Truncation-Marker.
- **Caching** (`agent::cache`, in-memory pro Prozess): Tool-Result- und
  Leg-Memoization plus Query→Result-Cache, geschlüsselt über den
  Repo-Fingerprint (`GitStateProbe` im Search-Crate: HEAD-SHA + Digest über
  `git status --porcelain` und `git diff HEAD`); Invalidierung pfadgenau über
  `git diff --name-only <alt> <neu>`, konservativ bei unbekanntem Dirty-Delta.
- **Statischer Prompt**: Index-Note wandert aus dem System- in die
  User-Message; für Anthropic-Provider setzt `repo-explorer-llm`
  `cache_control` auf die System-Message (Provider-Prompt-Caching).

## Konfiguration

Neue, voll defaultete Sektionen `[agent]` und `[cache]` (siehe README);
`AgentConfig`/`max_iterations` aus Stage 5 ist durch
`config::AgentSettings`/`CacheSettings` ersetzt.

## Out of Scope

BM25/Embeddings-Ranking (Heuristik reicht für v1), tree-sitter-Skeletons
(Graph-basiert genügt; gleiche Abstraktion erlaubt Nachrüstung), persistenter
On-Disk-Cache.

## Abhängigkeiten

Stage 1–6.

## Abnahmekriterien

- Exakter Symbol-Treffer → Early Exit mit **0** LLM-Calls.
- Mittlere Konfidenz → genau 1 Verifikations-Turn (`finish`) bzw. `expand` +
  erzwungener `finish`-Turn (`force_tool` belegt).
- Niedrige Konfidenz / gescheiterte Verifikation → Fallback-Loop mit
  Kandidaten-Seeding.
- Token-Budget-Ende → Forced-Final-`finish`, sonst deterministische Synthese.
- Batch-Zwang: Einzel-Call zweimal abgewiesen, danach ausgeführt; Batches
  laufen nebenläufig.
- Wiederholte Query → zweiter Lauf aus dem Cache (keine Backend-/LLM-Calls);
  Fingerprint-Wechsel invalidiert pfadgenau.
