# eval/queries/ (ground-truth query fixtures)

Ground-truth `span`s for symbol _definitions_ must cover the full
function/class body (brace/indent-aware), not just the `def`/`fn` line —
the tool correctly returns whole functions, and `eval/score.py`'s
`range_hit` will flag a too-narrow hand-authored span as a miss.
