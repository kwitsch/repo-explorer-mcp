"""Raw logits + checkpoint resolution. torch/laya imported inside functions."""

import os

# The pinned D1 judge question (byte-parity with repo_explorer_core::judge).
QUESTION = {
    "type": "choice",
    "instructions": "Does this code location answer the repository search query?",
    "criteria": {
        "A": "yes, this location answers the query",
        "B": "no, this location does not answer the query",
    },
}


def resolve_checkpoint(name_or_dir):
    """Return a local checkpoint directory, downloading from the Hub if needed.

    Matches how `laya.Agent` resolves a repo-root checkpoint.
    """
    if os.path.isdir(name_or_dir):
        return name_or_dir
    from huggingface_hub import snapshot_download
    import laya

    directory = snapshot_download(
        name_or_dir,
        allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
    )
    laya.agent._fix_tokenizer_config(directory)  # laya 0.3.7
    return directory


def raw_logits(agent, states):
    """Pre-temperature logits [n, 2] of the judge question for each state.

    Avoids `Agent.predict`'s 4-decimal probability rounding.
    """
    import numpy as np
    import torch
    from laya.common import build_sequence, collate_items  # laya 0.3.7

    q = agent._to_internal(QUESTION)  # laya 0.3.7
    max_len = agent.cfg["max_len"]
    head_max_len = agent.cfg["head_max_len"]
    use_cuda = torch.cuda.is_available() and str(agent.device).startswith("cuda")

    out = []
    for start in range(0, len(states), 16):
        items = []
        for state in states[start:start + 16]:
            seq, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            items.append((seq, markers))
        batch = collate_items(items, agent.tok.pad_token_id)
        batch = {k: (v.to(agent.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.no_grad():
            if use_cuda:
                with torch.autocast("cuda", dtype=agent.dtype):  # same rule as Agent.system_one
                    logits = agent.model(**batch)
            else:
                logits = agent.model(**batch)
        out.append(logits[:, :2].float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2), dtype="float32")
