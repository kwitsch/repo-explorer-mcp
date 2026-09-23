import json
import os
import tempfile
import unittest

from judge_train import data


def row(**over):
    base = {
        "id": "r:000001:01", "workflow": "repo_explorer_verify", "repo": "r", "lang": "rust",
        "split": "train", "query_id": "r:000001", "query": "q", "query_lang": "en",
        "template": "define-en", "candidate_rank": 1, "candidate_kind": "exact symbol match",
        "label": "A",
        "state": json.dumps("state text"),
        "questions": json.dumps({"relevant": {"type": "choice"}}),
        "gold": json.dumps({"relevant": {"label": "A", "probabilities": {"A": 1.0, "B": 0.0}}}),
    }
    base.update(over)
    return base


def write(rows):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


class TestData(unittest.TestCase):
    def test_accepts_valid_row(self):
        path = write([row()])
        try:
            rows = data.load_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "A")
        finally:
            os.remove(path)

    def test_rejects_missing_key(self):
        r = row()
        del r["template"]
        path = write([r])
        try:
            with self.assertRaises(ValueError):
                data.load_rows(path)
        finally:
            os.remove(path)

    def test_rejects_bad_label(self):
        path = write([row(label="C")])
        try:
            with self.assertRaises(ValueError):
                data.load_rows(path)
        finally:
            os.remove(path)

    def test_rejects_gold_inconsistent_with_label(self):
        r = row(gold=json.dumps({"relevant": {"label": "B", "probabilities": {"A": 0.0, "B": 1.0}}}))
        path = write([r])
        try:
            with self.assertRaises(ValueError):
                data.load_rows(path)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
