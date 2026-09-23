//! Candidate labelling against a symbol's known ground-truth location, and the
//! per-query row selection (all positives + a capped tail of negatives).

use repo_explorer_core::domain::Candidate;
use repo_explorer_core::retrieval::normalize_rel_path;

/// The known-good location a query points at: the source symbol.
pub struct Ground<'a> {
    pub path: &'a str,
    pub line_start: u32,
    pub line_end: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selected {
    /// Index into the input candidate slice.
    pub index: usize,
    /// 1-based retrieval rank (= `index + 1`).
    pub rank: usize,
    pub label: &'static str,
}

#[derive(Debug)]
pub struct LabelOutcome {
    pub selected: Vec<Selected>,
    pub has_positive: bool,
}

/// True when `c` overlaps the ground truth in the same file (edge-touching
/// counts): `c.line_start <= G.line_end && max(c.line_end, c.line_start) >= G.line_start`.
pub fn is_positive(c: &Candidate, ground: &Ground) -> bool {
    let normalized = normalize_rel_path(c.location.path.clone());
    let path = normalized.to_string_lossy().replace('\\', "/");
    if path != ground.path {
        return false;
    }
    let (start, end) = (c.location.line_start, c.location.line_end);
    start <= ground.line_end && end.max(start) >= ground.line_start
}

/// Select rows for one query: every positive, plus the first
/// `max_negatives_per_query` negatives in rank order (at most 2 when there is
/// no positive). Only candidates with a `Some` state are considered; skipped
/// (`None`) candidates keep their rank slot so numbering stays the true
/// retrieval rank. The output is sorted by `rank` ascending.
pub fn label_candidates(
    candidates: &[Candidate],
    states: &[Option<String>],
    ground: &Ground,
    max_negatives_per_query: usize,
) -> LabelOutcome {
    let mut positives: Vec<Selected> = Vec::new();
    let mut negatives: Vec<Selected> = Vec::new();
    for (i, (c, state)) in candidates.iter().zip(states).enumerate() {
        if state.is_none() {
            continue;
        }
        let positive = is_positive(c, ground);
        let sel = Selected {
            index: i,
            rank: i + 1,
            label: if positive { "A" } else { "B" },
        };
        if positive {
            positives.push(sel);
        } else {
            negatives.push(sel);
        }
    }
    let has_positive = !positives.is_empty();
    let neg_cap = if has_positive {
        max_negatives_per_query
    } else {
        2
    };
    let mut selected = positives;
    selected.extend(negatives.into_iter().take(neg_cap));
    selected.sort_by_key(|s| s.rank);
    LabelOutcome {
        selected,
        has_positive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::{Candidate, CandidateKind, FileLocation};
    use std::path::PathBuf;

    fn cand(path: &str, s: u32, e: u32) -> Candidate {
        Candidate {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: s,
                line_end: e,
            },
            symbol: None,
            kind: CandidateKind::ContentHit,
            score: 0,
            snippet: None,
        }
    }

    #[test]
    fn overlap_rule_including_edges() {
        let g = Ground {
            path: "src/a.rs",
            line_start: 10,
            line_end: 20,
        };
        assert!(is_positive(&cand("src/a.rs", 10, 20), &g));
        assert!(is_positive(&cand("src/a.rs", 12, 15), &g));
        assert!(is_positive(&cand("src/a.rs", 5, 10), &g)); // touches start
        assert!(is_positive(&cand("src/a.rs", 20, 25), &g)); // touches end
        assert!(!is_positive(&cand("src/a.rs", 1, 9), &g));
        assert!(!is_positive(&cand("src/a.rs", 21, 30), &g));
        assert!(!is_positive(&cand("src/b.rs", 10, 20), &g));
        assert!(is_positive(&cand("./src/a.rs", 10, 20), &g)); // ./ normalized
    }

    #[test]
    fn selection_caps_negatives_with_positive() {
        let g = Ground {
            path: "src/a.rs",
            line_start: 10,
            line_end: 20,
        };
        let cands = vec![
            cand("src/a.rs", 10, 20),
            cand("src/b.rs", 1, 5),
            cand("src/c.rs", 1, 5),
            cand("src/d.rs", 1, 5),
            cand("src/e.rs", 1, 5),
            cand("src/f.rs", 1, 5),
        ];
        let states: Vec<Option<String>> = cands.iter().map(|_| Some("s".to_string())).collect();
        let out = label_candidates(&cands, &states, &g, 4);
        assert!(out.has_positive);
        assert_eq!(
            out.selected.iter().map(|s| s.rank).collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5]
        );
        assert_eq!(out.selected[0].label, "A");
        assert!(out.selected[1..].iter().all(|s| s.label == "B"));
    }

    #[test]
    fn no_positive_keeps_at_most_two_negatives() {
        let g = Ground {
            path: "src/z.rs",
            line_start: 10,
            line_end: 20,
        };
        let cands = vec![
            cand("src/a.rs", 1, 5),
            cand("src/b.rs", 1, 5),
            cand("src/c.rs", 1, 5),
        ];
        let states: Vec<Option<String>> = cands.iter().map(|_| Some("s".into())).collect();
        let out = label_candidates(&cands, &states, &g, 4);
        assert!(!out.has_positive);
        assert_eq!(out.selected.len(), 2);
        assert!(out.selected.iter().all(|s| s.label == "B"));
    }

    #[test]
    fn none_state_candidates_are_skipped_but_ranks_are_preserved() {
        let g = Ground {
            path: "src/a.rs",
            line_start: 10,
            line_end: 20,
        };
        let cands = vec![cand("src/x.rs", 1, 5), cand("src/a.rs", 10, 20)];
        let states = vec![None, Some("s".into())];
        let out = label_candidates(&cands, &states, &g, 4);
        assert_eq!(out.selected.len(), 1);
        assert_eq!(out.selected[0].rank, 2);
        assert_eq!(out.selected[0].label, "A");
    }
}
