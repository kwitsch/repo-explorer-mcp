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
}
