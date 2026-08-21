# Stage 7 — Packaging, CI, Setup-Script & Docs

## Ziel

Auslieferung für Linux und Windows, ein npx-fähiges Setup-Script für Abhängigkeitsprüfung und Installation, sowie Abschlussdokumentation.

## Deliverables

- **Release-Workflow** (`.github/workflows/release.yml`): Cross-Builds für Linux (`x86_64-unknown-linux-gnu`) und Windows (`x86_64-pc-windows-msvc` oder `-gnu`), Artefakt-Checksums, Versionierung über `workspace.package.version`. Artefakt-Namensschema muss stabil und vom Setup-Script (s. u.) ableitbar sein (z. B. `repo-explorer-mcp-<version>-<target>.{tar.gz,zip}`).
- **CI-Workflow** (`.github/workflows/ci.yml`): sicherstellen, dass `cargo build/test/clippy/fmt` weiterhin für beide Zielplattformen läuft (Matrix-Build).
- **Setup-Script** (`setup/`, ESM `.mjs`, ausführbar via `npx`):
  - `setup/package.json` mit `bin`-Eintrag, sodass `npx github:<owner>/repo-explorer-mcp` bzw. nach Publish `npx repo-explorer-mcp-setup` das Script startet; keine Runtime-Dependencies — nur Node-Builtins (`os`, `fs`, `path`, `child_process`, `fetch`), damit npx ohne Install-Overhead läuft.
  - **OS-/Arch-Erkennung**: `process.platform` + `process.arch` → Release-Target (`linux`/`win32` × `x64`; andere Plattformen mit klarer Fehlermeldung ablehnen).
  - **Abhängigkeitsprüfung**: prüft im `PATH` (Windows: zusätzlich `where`, Linux: `which`):
    - `ripgrep` (`rg --version`)
    - `rtk` (`rtk --version`)
    - `codebase-memory-mcp` (Startbarkeit des upstream MCP-Servers gemäß dessen Distributionsform)
  - **Nachinstallation bei Bedarf**: fehlende Abhängigkeiten pro OS installieren — Linux: Paketmanager-Erkennung (`apt`/`dnf`/`pacman`) für ripgrep, sonst Binary-Download von den offiziellen GitHub-Releases; Windows: `winget` (Fallback: Binary-Download). Vor jeder Installation Ausgabe, was installiert wird; `--yes`-Flag für nicht-interaktive Läufe, ohne Flag Bestätigung per Prompt.
    - **Binary-Download**: lädt das passende `repo-explorer-mcp`-Release-Artefakt von GitHub Releases (latest oder `--version`-Flag), verifiziert die Checksum, entpackt nach `~/.local/bin` (Linux) bzw. `%LOCALAPPDATA%\repo-explorer-mcp` (Windows) und prüft/meldet, ob das Zielverzeichnis im `PATH` liegt.
  - **Idempotenz**: erneuter Lauf erkennt vorhandene, aktuelle Installationen und tut nichts; `--force` erzwingt Neuinstallation.
  - Abschlussausgabe: gefundene/installierte Versionen + fertiges `.mcp.json`-Snippet zum Kopieren.
- **README**: Installationsanleitung (primär: das npx-Setup-Script; manueller Weg als Fallback), Beispiel-`repo-explorer.toml` (Provider-Reihenfolge, `codebase-memory-mcp`-Verbindung, Suchoptionen), `.mcp.json`-Snippet zur Einbindung in Claude Code.
- **CLAUDE.md-Nachpflege**: finalisierte Fehlerbehandlungs-Konvention (aus Stage 1) eintragen, Scaffolding-Hinweis („Currently scaffolding only“) entfernen/aktualisieren.
- **Smoke-Test-Dokumentation**: kurze Anleitung, wie ein Release-Artefakt manuell gegen ein Testrepo verifiziert wird.

## Out of Scope

Weitere Provider/Backends über den in Stage 2–4 definierten Rahmen hinaus.

## Abhängigkeiten

Stage 1–6.

## Abnahmekriterien

- CI grün für einen getaggten Release-Build auf beiden Zielplattformen.
- Setup-Script läuft auf frischem Linux und Windows durch: erkennt OS, meldet fehlende Abhängigkeiten, installiert sie nach Bestätigung, lädt die korrekte Binary inkl. Checksum-Verifikation; zweiter Lauf ist ein No-op.
- README deckt Konfiguration, Provider-Setup, Setup-Script-Nutzung und MCP-Einbindung vollständig ab.
