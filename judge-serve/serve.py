"""Serve a Laya checkpoint for repo-explorer-mcp's [judge] via laya-serve's /v1/systemone."""
import json, os, sys

EXPECTED_JUDGE_STATE_VERSION = 1  # keep equal to repo_explorer_core::judge::JUDGE_STATE_VERSION

def main() -> None:
    import torch, uvicorn
    from laya.router import Router
    from laya.serve import create_app

    threads = int(os.environ.get("LAYA_THREADS", "0") or "0")
    if threads > 0:
        torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    ckpt = os.environ.get("REPO_EXPLORER_LAYA_CHECKPOINT", "").strip()
    if ckpt:
        meta = os.path.join(ckpt, "repo_explorer_judge.json")
        if not os.path.exists(meta):
            sys.exit(f"{meta} missing: calibrate the checkpoint first (train/laya/calibrate.py)")
        with open(meta) as f:
            info = json.load(f)
        if info.get("judge_state_version") != EXPECTED_JUDGE_STATE_VERSION:
            sys.exit(f"checkpoint judge_state_version {info.get('judge_state_version')} != {EXPECTED_JUDGE_STATE_VERSION}")
        print(f"[judge-serve] checkpoint {ckpt}; recommended judge.select_threshold = {info.get('select_threshold')}", flush=True)
        models = {"typed-decisions": ckpt}
    else:
        print("[judge-serve] WARNING: REPO_EXPLORER_LAYA_CHECKPOINT unset - serving the zero-shot upstream "
              "typed-decisions checkpoint (smoke tests only, not for production)", flush=True)
        models = None

    router = Router(models=models, device=os.environ.get("LAYA_DEVICE") or None, max_loaded=1)
    router.preload(["typed-decisions"])
    uvicorn.run(create_app(router),
                host=os.environ.get("LAYA_HOST", "127.0.0.1"),
                port=int(os.environ.get("LAYA_PORT", "8765")),
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))

if __name__ == "__main__":
    main()
