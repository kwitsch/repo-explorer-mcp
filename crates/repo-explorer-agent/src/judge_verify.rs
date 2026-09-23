//! Stage-4 verification by the local candidate judge (`judge.mode = "laya"`
//! or `"shadow"`). Renders per-candidate states with the 10a renderer
//! (train/serve parity), judges them, selects `p >= select_threshold`,
//! disk-verifies the selection, and builds a deterministic result with 0 LLM
//! calls. See `crates/repo-explorer-agent/CLAUDE.md`.

use repo_explorer_core::config::JudgeSettings;
use repo_explorer_core::domain::{
    Candidate, ExplorationFinding, ExplorationQuery, ExplorationResult,
};
use repo_explorer_core::judge::{CandidateJudge, JudgeError};
use repo_explorer_core::memory::MemoryBackend;
use repo_explorer_core::retrieval::normalize_rel_path;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::agent::verified_candidate_findings;
use crate::judge_input::render_judge_states;

#[derive(Debug)]
pub(crate) enum JudgeVerifyOutcome {
    Finished {
        result: ExplorationResult,
        scores: Vec<Option<u32>>,
        selected: Vec<usize>,
        judged: u32,
        elapsed_ms: u64,
    },
    Escalate {
        scores: Vec<Option<u32>>,
        judged: u32,
        elapsed_ms: u64,
    },
    Failed {
        error: JudgeError,
        elapsed_ms: u64,
    },
}

/// Candidate indices with `score >= select_threshold * 10` permille, sorted by
/// score descending then index ascending.
pub(crate) fn select_indices(scores: &[Option<u32>], select_threshold: u32) -> Vec<usize> {
    let cutoff = select_threshold * 10;
    let mut chosen: Vec<(usize, u32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.filter(|&p| p >= cutoff).map(|p| (i, p)))
        .collect();
    chosen.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    chosen.into_iter().map(|(i, _)| i).collect()
}

/// Candidate indices whose location overlaps any result finding: same
/// `normalize_rel_path` path and `c.line_start <= f.line_end &&
/// c.line_end.max(c.line_start) >= f.line_start`, both locations known.
pub(crate) fn llm_overlap_set(
    candidates: &[Candidate],
    findings: &[ExplorationFinding],
) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        let cpath = normalize_rel_path(c.location.path.clone());
        let (cs, ce) = (
            c.location.line_start,
            c.location.line_end.max(c.location.line_start),
        );
        if cs == 0 {
            continue; // unknown location
        }
        let hit = findings.iter().any(|f| {
            f.location.line_start != 0
                && normalize_rel_path(f.location.path.clone()) == cpath
                && crate::ranges_overlap(cs, ce, f.location.line_start, f.location.line_end)
        });
        if hit {
            out.push(i);
        }
    }
    out
}

/// D3.4. `llm`/`judge` are the overlap/selected index sets when that side
/// finished, `None` when it escalated. Judge `Failed` is handled by the caller
/// (returns `None`), never reaching here.
pub(crate) fn classify_agreement(
    llm: Option<&[usize]>,
    judge: Option<&[usize]>,
) -> Option<&'static str> {
    match (llm, judge) {
        (Some(l), Some(j)) => {
            let ls: HashSet<usize> = l.iter().copied().collect();
            let js: HashSet<usize> = j.iter().copied().collect();
            if ls == js {
                Some("exact")
            } else if ls.is_disjoint(&js) {
                Some("disjoint")
            } else {
                Some("overlap")
            }
        }
        (Some(_), None) => Some("judge-escalated"),
        (None, Some(_)) => Some("llm-escalated"),
        (None, None) => Some("both-escalated"),
    }
}

pub(crate) async fn judge_verify<M: MemoryBackend, J: CandidateJudge>(
    memory: &M,
    judge: &J,
    repo_root: &Path,
    query: &ExplorationQuery,
    note: Option<&str>,
    candidates: &[Candidate],
    settings: &JudgeSettings,
) -> JudgeVerifyOutcome {
    let rendered = render_judge_states(memory, repo_root, &query.text, candidates).await;
    // (index, state) for every Some, moved out of `rendered` once — no clones.
    let (idx_states, state_texts): (Vec<usize>, Vec<String>) = rendered
        .into_iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|text| (i, text)))
        .unzip();
    let mut scores: Vec<Option<u32>> = vec![None; candidates.len()];
    if idx_states.is_empty() {
        // No judge call was made at all, so there is no latency to report.
        return JudgeVerifyOutcome::Escalate {
            scores,
            judged: 0,
            elapsed_ms: 0,
        };
    }
    let judged = state_texts.len() as u32;
    // Timed narrowly around the judge call itself (what `judge.timeout_ms`
    // bounds) — never `render_judge_states`'s local file reads before it, or
    // `verified_candidate_findings`'s disk verification after it, so
    // `judge_ms` stays a true judge-latency metric for `eval/score.py`.
    let start = Instant::now();
    let result = judge.judge(&state_texts).await;
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let judgements = match result {
        Ok(v) if v.len() == state_texts.len() => v,
        Ok(v) => {
            return JudgeVerifyOutcome::Failed {
                error: JudgeError::Protocol {
                    message: format!(
                        "judge returned {} results for {} states",
                        v.len(),
                        state_texts.len()
                    ),
                },
                elapsed_ms,
            };
        }
        Err(error) => return JudgeVerifyOutcome::Failed { error, elapsed_ms },
    };
    for (idx, j) in idx_states.iter().zip(&judgements) {
        scores[*idx] = Some(j.p_relevant_permille);
    }
    let selected = select_indices(&scores, settings.select_threshold);
    if selected.is_empty() {
        return JudgeVerifyOutcome::Escalate {
            scores,
            judged,
            elapsed_ms,
        };
    }
    // Disk-verify the selection in selection order.
    let selected_refs: Vec<&Candidate> = selected.iter().map(|&i| &candidates[i]).collect();
    let verified = verified_candidate_findings(repo_root, &selected_refs).await;
    if verified.is_empty() {
        return JudgeVerifyOutcome::Escalate {
            scores,
            judged,
            elapsed_ms,
        };
    }
    // Map each survivor back to its candidate index and attach the judge suffix.
    let mut findings = Vec::with_capacity(verified.len());
    let mut survivor_indices = Vec::with_capacity(verified.len());
    for (pos_in_selected, mut finding) in verified {
        let cand_index = selected[pos_in_selected];
        let score = scores[cand_index].unwrap_or(0);
        let base = finding.note.take().unwrap_or_default();
        finding.note = Some(format!("{base}; judge p={:.2}", score as f64 / 1000.0));
        findings.push(finding);
        survivor_indices.push(cand_index);
    }
    let n = findings.len();
    let mut summary = format!(
        "Selected by the local judge (no LLM involved): {n} of {judged} candidate(s) for \"{}\".",
        query.text
    );
    if let Some(note) = note {
        summary.push(' ');
        summary.push_str(note);
    }
    JudgeVerifyOutcome::Finished {
        result: ExplorationResult { findings, summary },
        scores,
        selected: survivor_indices,
        judged,
        elapsed_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_repo_with;
    use repo_explorer_core::config::{JudgeMode, JudgeSettings};
    use repo_explorer_core::domain::{
        Candidate, CandidateKind, ExplorationFinding, ExplorationQuery, FileLocation,
    };
    use repo_explorer_core::judge::mock::MockJudge;
    use repo_explorer_core::judge::{JudgeError, Judgement};
    use repo_explorer_core::memory::mock::MockMemoryBackend;
    use std::path::PathBuf;

    fn q(text: &str) -> ExplorationQuery {
        ExplorationQuery {
            text: text.to_string(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        }
    }

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

    fn settings(threshold: u32) -> JudgeSettings {
        JudgeSettings {
            mode: JudgeMode::Laya,
            select_threshold: threshold,
            ..JudgeSettings::default()
        }
    }

    #[test]
    fn selection_orders_by_score_desc_then_index() {
        // score >= threshold*10 permille selected; ties broken by index asc.
        let scores = vec![Some(700u32), Some(500u32), Some(700u32), Some(100u32), None];
        let selected = select_indices(&scores, 50); // threshold 50 => 500 permille
        assert_eq!(selected, vec![0, 2, 1]);
    }

    #[test]
    fn agreement_classification_matrix() {
        assert_eq!(
            classify_agreement(Some(&[1, 2]), Some(&[2, 1])),
            Some("exact")
        );
        assert_eq!(
            classify_agreement(Some(&[1, 2]), Some(&[2, 3])),
            Some("overlap")
        );
        assert_eq!(classify_agreement(Some(&[1]), Some(&[3])), Some("disjoint"));
        assert_eq!(
            classify_agreement(Some(&[1]), None),
            Some("judge-escalated")
        );
        assert_eq!(classify_agreement(None, Some(&[1])), Some("llm-escalated"));
        assert_eq!(classify_agreement(None, None), Some("both-escalated"));
    }

    #[test]
    fn overlap_set_matches_same_path_and_range() {
        let cands = vec![cand("a.rs", 10, 20), cand("b.rs", 1, 5)];
        let findings = vec![ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from("a.rs"),
                line_start: 15,
                line_end: 15,
            },
            snippet: None,
            note: None,
        }];
        assert_eq!(llm_overlap_set(&cands, &findings), vec![0]);
    }

    #[tokio::test]
    async fn judge_verify_selects_above_threshold_with_suffix() {
        let dir = temp_repo_with(
            "judge_verify",
            "select",
            &[("a.rs", "l1\nl2\nl3\nl4\nl5\n")],
        );
        let memory = MockMemoryBackend::new();
        let cands = vec![cand("a.rs", 1, 1), cand("a.rs", 3, 3)];
        let judge = MockJudge::new().with_responses(vec![Ok(vec![
            Judgement {
                p_relevant_permille: 900,
            },
            Judgement {
                p_relevant_permille: 100,
            },
        ])]);
        let out = judge_verify(
            &memory,
            &judge,
            &dir,
            &q("find"),
            None,
            &cands,
            &settings(50),
        )
        .await;
        std::fs::remove_dir_all(&dir).ok();
        match out {
            JudgeVerifyOutcome::Finished {
                result,
                selected,
                judged,
                scores,
                ..
            } => {
                assert_eq!(judged, 2);
                assert_eq!(selected, vec![0]);
                assert_eq!(scores, vec![Some(900), Some(100)]);
                assert_eq!(result.findings.len(), 1);
                assert!(
                    result.findings[0]
                        .note
                        .as_ref()
                        .unwrap()
                        .contains("judge p=0.90")
                );
                assert!(result.summary.contains("Selected by the local judge"));
            }
            other => panic!("expected Finished, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn judge_verify_escalates_when_all_below() {
        let dir = temp_repo_with("judge_verify", "escalate", &[("a.rs", "l1\nl2\n")]);
        let memory = MockMemoryBackend::new();
        let cands = vec![cand("a.rs", 1, 1)];
        let judge = MockJudge::new().with_responses(vec![Ok(vec![Judgement {
            p_relevant_permille: 100,
        }])]);
        let out = judge_verify(&memory, &judge, &dir, &q("f"), None, &cands, &settings(50)).await;
        std::fs::remove_dir_all(&dir).ok();
        assert!(matches!(
            out,
            JudgeVerifyOutcome::Escalate { judged: 1, .. }
        ));
    }

    #[tokio::test]
    async fn judge_verify_failure_is_failed() {
        let dir = temp_repo_with("judge_verify", "fail", &[("a.rs", "l1\n")]);
        let memory = MockMemoryBackend::new();
        let cands = vec![cand("a.rs", 1, 1)];
        let judge = MockJudge::new().with_responses(vec![Err(JudgeError::Unavailable {
            message: "down".into(),
        })]);
        let out = judge_verify(&memory, &judge, &dir, &q("f"), None, &cands, &settings(50)).await;
        std::fs::remove_dir_all(&dir).ok();
        assert!(matches!(out, JudgeVerifyOutcome::Failed { .. }));
    }

    #[tokio::test]
    async fn no_known_states_escalates_without_calling_judge() {
        let dir = temp_repo_with("judge_verify", "unknown", &[("a.rs", "x\n")]);
        let memory = MockMemoryBackend::new();
        // Unknown-location candidate → render_judge_states yields None → judged 0.
        let cands = vec![cand("a.rs", 0, 0)];
        let judge = MockJudge::new();
        let out = judge_verify(&memory, &judge, &dir, &q("f"), None, &cands, &settings(50)).await;
        std::fs::remove_dir_all(&dir).ok();
        assert!(matches!(
            out,
            JudgeVerifyOutcome::Escalate { judged: 0, .. }
        ));
        assert!(
            judge.calls().is_empty(),
            "judge must not be called with no states"
        );
    }
}
