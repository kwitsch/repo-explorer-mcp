#!/usr/bin/env python3
"""Assert-based self-check for eval/score.py's snippet_found_at window (F-17).

No test framework, no fixtures, no results/ directory, no external corpus:
writes a synthetic file to a tempdir and asserts the classifications directly.
Run with:  python eval/test_score.py   (prints OK; non-zero exit on failure)
"""
import sys
import tempfile
import types
from pathlib import Path

# score.py imports pyyaml at module top for load_queries(), which this check
# never calls; stub it so the file imports under a bare `python` with no deps.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import snippet_found_at


def _write_file(dir_path: Path, name: str, n_lines: int) -> None:
    body = "\n".join(f"SENTINEL_{i:03d}_line_content" for i in range(n_lines))
    (dir_path / name).write_text(body + "\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        _write_file(repo, "sample.py", 40)

        # 1. Widened window: in-range hit far past line_start+3 but inside
        #    line_end+4 (chunk at file index 32; claimed span 5..35).
        #    Pre-fix -> "misaligned"; post-fix must be "ok".
        assert snippet_found_at(
            repo, "sample.py", 5, 35, "SENTINEL_032_line_content"
        ) == "ok"

        # 2. No over-correction: chunk genuinely outside the widened window
        #    (chunk at index 37; claimed span 5..10, window upper bound 14).
        assert snippet_found_at(
            repo, "sample.py", 5, 10, "SENTINEL_037_line_content"
        ) == "misaligned"

        # 3. Fabricated chunk: not present anywhere in the file.
        assert snippet_found_at(
            repo, "sample.py", 5, 35, "TOTALLY_ABSENT_SENTINEL_STRING"
        ) == "not_found"

        # 4. line_end is None must not raise TypeError; chunk (index 5) sits
        #    inside the fallback window (end := line_start).
        assert snippet_found_at(
            repo, "sample.py", 5, None, "SENTINEL_005_line_content"
        ) == "ok"

    print("OK")


if __name__ == "__main__":
    main()
