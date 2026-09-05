#!/usr/bin/env bash
# Builds the R-18 robustness fixture (docs/eval/real-world-test-plan.md §9): a repo with a
# submodule, a symlink loop, and a large binary blob, to confirm explore_repository doesn't hang
# or crash on any of the three. Not part of the pilot (Phase 0-1) scope — used in Phase 3.
#
# Usage: eval/fixtures/make_r18.sh <target-dir> <path-to-a-real-git-repo-to-use-as-submodule>
set -euo pipefail

TARGET="${1:?usage: make_r18.sh <target-dir> <submodule-source-repo>}"
SUBMODULE_SRC="${2:?usage: make_r18.sh <target-dir> <submodule-source-repo>}"

if [ -e "$TARGET" ]; then
  echo "refusing to overwrite existing path: $TARGET" >&2
  exit 1
fi

mkdir -p "$TARGET"
cd "$TARGET"
git init --quiet
git -c protocol.file.allow=always submodule add --quiet "$SUBMODULE_SRC" sub
ln -s . loop
head -c 50M /dev/urandom > blob.bin
git add -A
git -c user.email="eval@localhost" -c user.name="eval" commit --quiet -m "R-18 fixture: submodule + symlink loop + 50MB blob"

echo "R-18 fixture ready at: $TARGET"
echo "Suggested probe query (scope_hint: sub): \"where is resolve_redirects defined\""
echo "Pass condition: completes in under 300s and no finding path is under blob.bin"
