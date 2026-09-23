# judge-serve

A small launcher that serves a locally fine-tuned Laya checkpoint through
upstream `laya.serve.create_app`, for `repo-explorer-mcp`'s `[judge]`
config (`judge.mode = "laya"` or `"shadow"`).

Upstream `laya-serve` (and its NixOS module) has no environment variable or
option to point at a custom, locally fine-tuned checkpoint directory — it
only knows its own published model names. This launcher exists solely to
bridge that gap: it maps the checkpoint directory to the `typed-decisions`
model name that the client sends, then hands off to the unmodified upstream
`create_app`.

## Run

```bash
REPO_EXPLORER_LAYA_CHECKPOINT=~/laya-ckpt/v1 LAYA_DEVICE=cuda \
  uv run --project judge-serve python judge-serve/serve.py
```

## Environment variables

| Variable                        | Default     | Meaning                                                                                                                                                                                                                                     |
| ------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `REPO_EXPLORER_LAYA_CHECKPOINT` | unset       | Path to a checkpoint directory calibrated by `train/laya/calibrate.py` (must contain `repo_explorer_judge.json`). When unset, the zero-shot upstream `typed-decisions` checkpoint is served instead — smoke tests only, not for production. |
| `LAYA_DEVICE`                   | auto        | Passed to `laya.router.Router(device=...)`, e.g. `cuda` or `cpu`.                                                                                                                                                                           |
| `LAYA_THREADS`                  | `0` (auto)  | When set to a positive integer, passed to `torch.set_num_threads`.                                                                                                                                                                          |
| `LAYA_HOST`                     | `127.0.0.1` | Bind host for uvicorn.                                                                                                                                                                                                                      |
| `LAYA_PORT`                     | `8765`      | Bind port for uvicorn.                                                                                                                                                                                                                      |
| `LAYA_LOG_LEVEL`                | `info`      | uvicorn log level.                                                                                                                                                                                                                          |
| `LAYA_API_KEY`                  | unset       | Optional bearer token, honoured directly by upstream `laya.serve.create_app`. When set, clients must send `authorization: Bearer <key>`.                                                                                                    |

## Matching `[judge]` config

```toml
[judge]
mode = "laya"                        # or "shadow"
base_url = "http://127.0.0.1:8765"
model = "typed-decisions"
select_threshold = 50                # the value serve.py prints at startup
# api_key_env = "REPO_EXPLORER_JUDGE_API_KEY"  # only if LAYA_API_KEY is set
```

## systemd user unit example

```ini
# ~/.config/systemd/user/repo-explorer-judge-serve.service
[Unit]
Description=repo-explorer-mcp local Laya judge

[Service]
Environment=REPO_EXPLORER_LAYA_CHECKPOINT=%h/laya-ckpt/v1
Environment=LAYA_DEVICE=cuda
WorkingDirectory=%h/repos/repo-explorer-mcp
ExecStart=uv run --project judge-serve python judge-serve/serve.py
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now repo-explorer-judge-serve.service
```
