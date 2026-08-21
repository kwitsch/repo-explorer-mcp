# Stage 4 — LLM Provider Abstraction & Multi-Provider Routing

## Ziel

Die eigentlichen LLM-Aufgaben (Steuerung der Exploration in Stage 5) sollen über mehrere Provider abgedeckt werden können. Der Nutzer konfiguriert die Reihenfolge; bei Limit-Fehlern eines Providers wird automatisch der nächste konfigurierte verwendet.

## Bibliotheksentscheidung (geprüft 2026-08-21)

litellm selbst ist Python — nicht direkt nutzbar. Rust-Äquivalente mit Multi-Provider-Abstraktion + Tool-Calling:

| Crate                            | Version                   | Downloads | Bewertung                                                                                                                                                                            |
| -------------------------------- | ------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `genai` (jeremychone/rust-genai) | 0.6.5 stable (0.7.0-beta) | 318k      | Reine Client-Abstraktion: OpenAI, Gemini, Anthropic, Ollama, Bedrock, Groq, DeepSeek u. v. m.; Tool-Calling; aktiv gepflegt (letztes Release 08/2026); schlank, kein Framework-Zwang |
| `llm` (graniet/llm)              | 1.3.8                     | 114k      | „Rust-litellm“: Provider per Feature-Flag (anthropic, openai, google, mistral, ollama, …), eingebaute Agent-/Chain-Features; letztes Release 04/2026                                 |
| `rig-core`                       | 0.42.0                    | 2.3M      | Vollwertiges Agent-Framework; bringt eigene Agent-/Tool-Abstraktionen mit — mehr als hier gebraucht wird, würde mit unserem eigenen Agent-Loop (Stage 5) konkurrieren                |

**Entscheidung: `genai`** (stabile 0.6.x-Linie).

- Deckt die geforderten Provider (Anthropic, OpenAI, Google) plus weitere ab — eine `ChatRequest`-API für alle.
- Tool-Calling wird unterstützt, was Stage 5 zwingend braucht.
- Kein Framework-Overhead: unser Agent-Loop bleibt selbst geschrieben, `genai` ersetzt nur die drei handgeschriebenen HTTP-Clients (`reqwest`-Ebene entfällt komplett).
- `rig-core` bleibt Fallback-Option, falls `genai` bei einem Provider-Detail klemmt; die eigene `LlmProvider`-Trait-Grenze (s. u.) macht den Austausch billig.

**Was `genai` NICHT liefert und selbst gebaut wird:** Failover-Routing bei Rate-Limits. Der `ProviderRouter` bleibt Eigenentwicklung — jetzt aber als dünne Schicht über `genai` statt über drei eigene HTTP-Clients.

## Deliverables

- **Trait**: `LlmProvider` mit z. B. `complete_with_tools(&self, messages, tool_defs) -> Result<ProviderResponse, ProviderError>`. `ProviderResponse` unterscheidet Text-Antwort vs. Tool-Call(s). Der Trait bleibt trotz `genai` bestehen: er entkoppelt Router und Agent-Loop von der Bibliothek und ermöglicht Fake-Provider in Tests.
- **`GenaiProvider`**: einzige produktive Trait-Implementierung; kapselt einen `genai::Client` pro konfiguriertem Provider-Eintrag (Modell, `api_key_env`, optional `base_url`) und mappt zwischen unseren Domänentypen und `genai`-Typen (`ChatRequest`, `Tool`, `ChatResponse`).
- **Fehlerklassifikation**: Mapping von `genai`-Fehlern auf `ProviderError::RateLimited` / `::QuotaExceeded` vs. andere Fehler — nötig, um Failover korrekt auszulösen (nur bei Limit-Situationen zum nächsten Provider springen; HTTP 429 sowie providerspezifische Fehlerkörper berücksichtigen, ggf. anhand des `genai`-Fehlertyps/-Status).
- **`ProviderRouter`** (Eigenbau): iteriert die konfigurierte Reihenfolge (Stage 1 Config), überspringt Provider innerhalb des `cooldown_seconds`-Fensters nach einem Limit-Fehler, liefert erst dann einen Gesamtfehler, wenn alle konfigurierten Provider erschöpft sind.
- Kein Streaming in v1 — synchrone Tool-Use-Turns reichen für den Agent-Loop aus Stage 5.

## Auswirkung auf Stage 1

Das Config-Schema bleibt unverändert gültig; `kind` mappt auf den `genai`-Adapter-Namen (z. B. `anthropic`, `openai`, `gemini`). Prüfen, ob `genai`s Modell-zu-Adapter-Auflösung die explizite `kind`-Angabe teilweise überflüssig macht — Feld trotzdem behalten für Eindeutigkeit bei custom `base_url`.

## Out of Scope

Die eigentliche Such-/Tool-Logik der Exploration (Stage 5).

## Abhängigkeiten

Stage 1 (Config-Schema für `llm.providers`).

## Abnahmekriterien

- Unit-Tests für den Router mit Fake-Providern (erschöpfter erster Provider → zweiter wird genutzt; alle erschöpft → Fehler; Cooldown-Ablauf → Provider wird wieder versucht).
- Mapping-Tests: `genai`-Fehler → `ProviderError`-Klassifikation.
- Keine Live-Netzwerkaufrufe im Standard-Testlauf; echte Provider-Calls nur hinter einem Feature-Flag/`#[ignore]`.
