"""Row loading + validation (the D4.8 schema). No torch/laya import here."""

import json

REQUIRED_KEYS = [
    "id", "workflow", "repo", "lang", "split", "query_id", "query", "query_lang",
    "template", "candidate_rank", "candidate_kind", "label", "state", "questions", "gold",
]


def load_rows(path):
    """Return the validated rows of a JSONL file.

    Raises ValueError naming the file, the 1-based line and the offending key
    on any invalid row.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: invalid JSON: {exc}") from exc
            for key in REQUIRED_KEYS:
                if key not in obj:
                    raise ValueError(f"{path}:{i}: missing key `{key}`")
            if obj["label"] not in ("A", "B"):
                raise ValueError(f"{path}:{i}: key `label` must be A or B, got {obj['label']!r}")
            try:
                gold = json.loads(obj["gold"])
                gold_label = gold["relevant"]["label"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{i}: key `gold` is malformed: {exc}") from exc
            if gold_label != obj["label"]:
                raise ValueError(
                    f"{path}:{i}: key `gold` label {gold_label!r} disagrees with `label` {obj['label']!r}"
                )
            rows.append(obj)
    return rows
