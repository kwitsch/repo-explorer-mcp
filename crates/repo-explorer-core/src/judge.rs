//! The local candidate judge (Stage 10): the fixed question every judge
//! backend asks about ONE retrieval candidate. Constants only; the
//! `CandidateJudge` trait lives here too once Stage 10b adds it.

/// Version of the plain-text state rendered by
/// `repo_explorer_agent::judge_input::render_judge_state`. Bump on ANY change
/// to that rendering or to the constants below: a checkpoint is valid only
/// for the version it was trained on.
pub const JUDGE_STATE_VERSION: u32 = 1;
pub const QUESTION_ID: &str = "relevant";
pub const QUESTION_TYPE: &str = "choice";
pub const QUESTION_INSTRUCTIONS: &str =
    "Does this code location answer the repository search query?";
pub const POSITIVE_KEY: &str = "A";
pub const POSITIVE_CRITERION: &str = "yes, this location answers the query";
pub const NEGATIVE_KEY: &str = "B";
pub const NEGATIVE_CRITERION: &str = "no, this location does not answer the query";

/// One judged candidate. Per-mille, keeping the domain integer-only / `Eq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Judgement {
    /// P(relevant) × 1000, rounded, clamped to 0..=1000.
    pub p_relevant_permille: u32,
}

/// Errors a [`CandidateJudge`] can return. The API key is never included.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum JudgeError {
    #[error("judge unavailable: {message}")]
    Unavailable { message: String },
    #[error("judge protocol error: {message}")]
    Protocol { message: String },
    #[error("judge timed out after {timeout_ms} ms")]
    Timeout { timeout_ms: u64 },
}

/// A local candidate judge: scores retrieval candidates by relevance
/// probability. Static dispatch (AFIT), matching the crate's no-`async-trait`
/// convention.
#[allow(async_fn_in_trait)]
pub trait CandidateJudge {
    /// Exactly one judgement per state, same order. Any per-state failure
    /// fails the whole call (no partial results).
    async fn judge(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError>;

    /// Best-effort readiness step run once in the background at startup
    /// (HTTP: health probe; 10c candle: model load). Default: nothing.
    async fn warm_up(&self) -> Result<(), JudgeError> {
        Ok(())
    }
}

/// The judge of a loop built without `with_judge`: every call is `Unavailable`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoJudge;

impl CandidateJudge for NoJudge {
    async fn judge(&self, _states: &[String]) -> Result<Vec<Judgement>, JudgeError> {
        Err(JudgeError::Unavailable {
            message: "no judge configured".to_string(),
        })
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod mock {
    use super::{CandidateJudge, JudgeError, Judgement};
    use std::sync::Mutex;

    type Response = Result<Vec<Judgement>, JudgeError>;
    type Scorer = Box<dyn Fn(&str) -> u32 + Send + Sync + 'static>;

    /// Canned responses in order (then a per-state scorer if set, else
    /// `Unavailable`); records every `states` slice it was called with.
    pub struct MockJudge {
        responses: Mutex<std::collections::VecDeque<Response>>,
        scorer: Option<Scorer>,
        calls: Mutex<Vec<Vec<String>>>,
    }

    /// `Default` mirrors the other core mocks so `clippy::new_without_default`
    /// stays clean.
    impl Default for MockJudge {
        fn default() -> Self {
            Self::new()
        }
    }

    impl MockJudge {
        pub fn new() -> Self {
            Self {
                responses: Mutex::new(std::collections::VecDeque::new()),
                scorer: None,
                calls: Mutex::new(Vec::new()),
            }
        }

        pub fn with_responses(mut self, r: Vec<Response>) -> Self {
            self.responses = Mutex::new(r.into());
            self
        }

        /// Score every call by a closure over the state text (used after the
        /// canned queue drains, and for tests that don't know the call count).
        pub fn with_scorer(mut self, f: impl Fn(&str) -> u32 + Send + Sync + 'static) -> Self {
            self.scorer = Some(Box::new(f));
            self
        }

        pub fn calls(&self) -> Vec<Vec<String>> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl CandidateJudge for MockJudge {
        async fn judge(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError> {
            self.calls.lock().unwrap().push(states.to_vec());
            if let Some(r) = self.responses.lock().unwrap().pop_front() {
                return r;
            }
            match &self.scorer {
                Some(f) => Ok(states
                    .iter()
                    .map(|s| Judgement {
                        p_relevant_permille: f(s).min(1000),
                    })
                    .collect()),
                None => Err(JudgeError::Unavailable {
                    message: "mock judge exhausted".to_string(),
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn judge_constants_are_pinned() {
        assert_eq!(JUDGE_STATE_VERSION, 1);
        assert_eq!(QUESTION_ID, "relevant");
        assert_eq!(QUESTION_TYPE, "choice");
        assert_eq!(
            QUESTION_INSTRUCTIONS,
            "Does this code location answer the repository search query?"
        );
        assert_eq!(POSITIVE_KEY, "A");
        assert_eq!(POSITIVE_CRITERION, "yes, this location answers the query");
        assert_eq!(NEGATIVE_KEY, "B");
        assert_eq!(
            NEGATIVE_CRITERION,
            "no, this location does not answer the query"
        );
        // Order invariant: "A" sorts before "B", so serde_json's sorted map keeps A then B.
        assert!(POSITIVE_KEY < NEGATIVE_KEY);
    }

    use super::mock::MockJudge;

    #[tokio::test]
    async fn no_judge_is_unavailable() {
        let j = NoJudge;
        let err = j.judge(&["s".to_string()]).await.unwrap_err();
        assert_eq!(
            err,
            JudgeError::Unavailable {
                message: "no judge configured".to_string()
            }
        );
        // warm_up default succeeds.
        assert!(j.warm_up().await.is_ok());
    }

    #[tokio::test]
    async fn mock_judge_replays_and_records() {
        let j = MockJudge::new().with_responses(vec![
            Ok(vec![Judgement {
                p_relevant_permille: 900,
            }]),
            Err(JudgeError::Timeout { timeout_ms: 5 }),
        ]);
        assert_eq!(
            j.judge(&["a".to_string()]).await.unwrap(),
            vec![Judgement {
                p_relevant_permille: 900
            }]
        );
        assert_eq!(
            j.judge(&["b".to_string(), "c".to_string()])
                .await
                .unwrap_err(),
            JudgeError::Timeout { timeout_ms: 5 }
        );
        // exhausted queue falls back to Unavailable.
        assert!(matches!(
            j.judge(&["d".to_string()]).await.unwrap_err(),
            JudgeError::Unavailable { .. }
        ));
        assert_eq!(
            j.calls(),
            vec![
                vec!["a".to_string()],
                vec!["b".to_string(), "c".to_string()],
                vec!["d".to_string()],
            ]
        );
    }

    #[tokio::test]
    async fn mock_judge_scorer_scores_by_state_text() {
        let j = MockJudge::new().with_scorer(|s| if s.contains("hit") { 950 } else { 10 });
        let out = j
            .judge(&["a hit".to_string(), "miss".to_string()])
            .await
            .unwrap();
        assert_eq!(out[0].p_relevant_permille, 950);
        assert_eq!(out[1].p_relevant_permille, 10);
    }
}
