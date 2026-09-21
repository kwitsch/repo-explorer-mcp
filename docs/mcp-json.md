# `.mcp.json`

Installed-binary form (what the setup script prints):

```json
{
  "mcpServers": {
    "repo-explorer": {
      "command": "/home/you/.local/bin/repo-explorer-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

On Windows the `command` is
`%LOCALAPPDATA%\\repo-explorer-mcp\\repo-explorer-mcp.exe`. With no `--config`
in `args`, the config is resolved by the precedence above (per-user file first,
then a `repo-explorer.toml` in the launch directory); add
`"--config", "<path>"` to `args` to point at a specific file. The in-repo
development form instead launches via
`cargo run --release --quiet -p repo-explorer-mcp --`.
