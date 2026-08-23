# Smoke test: verifying a release artifact

A manual checklist to confirm a published release archive works end to end on a
supported platform (`x86_64-unknown-linux-gnu` / `x86_64-pc-windows-msvc`).

## 1. Download and verify the checksum

Download the archive and its `.sha256` sidecar for your platform from the
GitHub Release, then verify:

Linux:

```bash
sha256sum -c repo-explorer-mcp- < version > -x86_64-unknown-linux-gnu.tar.gz.sha256
```

Windows (PowerShell):

```powershell
$expected = (Get-Content repo-explorer-mcp-<version>-x86_64-pc-windows-msvc.zip.sha256).Split(' ')[0]
(Get-FileHash -Algorithm SHA256 repo-explorer-mcp-<version>-x86_64-pc-windows-msvc.zip).Hash -ieq $expected
```

## 2. Extract and check the version

Extract the archive, then:

```bash
./repo-explorer-mcp --version # expect: repo-explorer-mcp <version>
```

This confirms the versioned build without loading any config.

## 3. Prepare a test repo

In a scratch repository, create a minimal `repo-explorer.toml` with a single
provider (`models = [...]`, and its `api_key_env` variable set in the
environment) and a reachable `codebase-memory-mcp` (stdio `command` or network
`endpoint`). Ensure `rg` is on PATH; `rtk` is optional.

Validate it before launching:

```bash
./repo-explorer-mcp --config ./repo-explorer.toml config test
```

Expect `"status": "valid"` and exit code 0. (Alternatively run
`./repo-explorer-mcp setup` to have the wizard write a per-user config.)

## 4. Launch the binary

Run the binary from the test repo root, passing `--config ./repo-explorer.toml`
explicitly so the resolution order can't pick up a pre-existing per-user config.
Confirm it logs `repo-explorer-mcp serving on stdio` to **stderr** and that
**stdout** carries the JSON-RPC MCP stream (do not expect human-readable logs on
stdout).

## 5. Register in Claude Code

Add the installed-binary `.mcp.json` snippet (see README), then invoke the
`explore_repository` tool and confirm a response.

## 6. Idempotent re-run

Re-run the setup script:

```bash
npx github:kwitsch/repo-explorer-mcp
```

Confirm it reports `already up to date` and performs no download — the
idempotency acceptance check. `--force` reinstalls unconditionally.
