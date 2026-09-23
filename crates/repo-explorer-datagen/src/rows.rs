//! Output row and manifest types, plus the `state`/`questions`/`gold`
//! JSON-string encoding (the `LocalLLaMA/typed-decisions` schema): each is a
//! string whose content is JSON, so the Python consumer calls `json.loads` on it.

use repo_explorer_core::judge::{
    NEGATIVE_CRITERION, NEGATIVE_KEY, POSITIVE_CRITERION, POSITIVE_KEY, QUESTION_ID,
    QUESTION_INSTRUCTIONS, QUESTION_TYPE,
};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Row {
    pub id: String,
    pub workflow: String,
    pub repo: String,
    pub lang: String,
    pub split: String,
    pub query_id: String,
    pub query: String,
    pub query_lang: String,
    pub template: String,
    pub candidate_rank: u32,
    pub candidate_kind: String,
    pub label: String,
    /// JSON-encoded string of the rendered state.
    pub state: String,
    /// JSON-encoded questions object.
    pub questions: String,
    /// JSON-encoded gold object.
    pub gold: String,
}

/// Inputs to `build_row`. `state` is the raw rendered state text (not yet
/// JSON-encoded).
pub struct RowInput<'a> {
    pub repo: &'a str,
    pub lang: &'a str,
    pub split: &'a str,
    pub query_index: usize,
    pub query: &'a str,
    pub query_lang: &'a str,
    pub template: &'a str,
    pub candidate_rank: u32,
    pub candidate_kind: &'a str,
    pub label: &'a str,
    pub state: &'a str,
}

/// Per-repo counters accumulated by the generator core.
#[derive(Debug, Clone, Default)]
pub struct RepoStats {
    pub files: u32,
    pub outline_errors: u32,
    pub symbols_total: u32,
    pub symbols_sampled: u32,
    pub queries_generated: u32,
    pub filtered_early_exit: u32,
    pub filtered_fallback: u32,
    pub queries_with_positive: u32,
    pub queries_without_positive: u32,
    pub rows_positive: u32,
    pub rows_negative: u32,
    /// template_id -> number of accepted queries using it.
    pub templates: BTreeMap<String, u32>,
}

/// One manifest `repos[]` entry (fields serialize in this exact order).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RepoEntry {
    pub name: String,
    pub rev: String,
    pub split: String,
    pub lang: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub files: u32,
    pub outline_errors: u32,
    pub symbols_total: u32,
    pub symbols_sampled: u32,
    pub queries_generated: u32,
    pub filtered_early_exit: u32,
    pub filtered_fallback: u32,
    pub queries_with_positive: u32,
    pub queries_without_positive: u32,
    pub rows_positive: u32,
    pub rows_negative: u32,
    pub templates: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ManifestSettings {
    pub max_queries_per_repo: usize,
    pub max_negatives_per_query: usize,
    pub top_k: u32,
    pub fallback_confidence: u32,
    pub early_exit_confidence: u32,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SplitTotals {
    pub files: u32,
    pub outline_errors: u32,
    pub symbols_total: u32,
    pub symbols_sampled: u32,
    pub queries_generated: u32,
    pub filtered_early_exit: u32,
    pub filtered_fallback: u32,
    pub queries_with_positive: u32,
    pub queries_without_positive: u32,
    pub rows_positive: u32,
    pub rows_negative: u32,
}

pub type Totals = BTreeMap<String, SplitTotals>;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Manifest {
    pub datagen_version: String,
    pub judge_state_version: u32,
    pub seed: u64,
    pub corpus_sha256: String,
    pub eval_repos_sha256: String,
    pub settings: ManifestSettings,
    pub repos: Vec<RepoEntry>,
    pub totals: Totals,
}

/// The pinned judge question object (D1 constants). serde_json's default sorted
/// map keeps `criteria`/`instructions`/`type` and `A`/`B` in order.
pub fn questions_value() -> Value {
    let mut criteria = Map::new();
    criteria.insert(POSITIVE_KEY.to_string(), json!(POSITIVE_CRITERION));
    criteria.insert(NEGATIVE_KEY.to_string(), json!(NEGATIVE_CRITERION));

    let mut inner = Map::new();
    inner.insert("criteria".to_string(), Value::Object(criteria));
    inner.insert("instructions".to_string(), json!(QUESTION_INSTRUCTIONS));
    inner.insert("type".to_string(), json!(QUESTION_TYPE));

    let mut outer = Map::new();
    outer.insert(QUESTION_ID.to_string(), Value::Object(inner));
    Value::Object(outer)
}

/// The gold object for a hard label (`A` -> {A:1.0,B:0.0}, else {A:0.0,B:1.0}).
pub fn gold_value(label: &str) -> Value {
    let (pa, pb) = if label == POSITIVE_KEY {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let mut probs = Map::new();
    probs.insert(POSITIVE_KEY.to_string(), json!(pa));
    probs.insert(NEGATIVE_KEY.to_string(), json!(pb));

    let mut inner = Map::new();
    inner.insert("label".to_string(), json!(label));
    inner.insert("probabilities".to_string(), Value::Object(probs));

    let mut outer = Map::new();
    outer.insert(QUESTION_ID.to_string(), Value::Object(inner));
    Value::Object(outer)
}

/// Assemble one output row. `state`/`questions`/`gold` become JSON-in-a-string.
pub fn build_row(input: &RowInput) -> Row {
    let query_id = format!("{}:{:06}", input.repo, input.query_index);
    let id = format!("{query_id}:{:02}", input.candidate_rank);
    Row {
        id,
        workflow: "repo_explorer_verify".to_string(),
        repo: input.repo.to_string(),
        lang: input.lang.to_string(),
        split: input.split.to_string(),
        query_id,
        query: input.query.to_string(),
        query_lang: input.query_lang.to_string(),
        template: input.template.to_string(),
        candidate_rank: input.candidate_rank,
        candidate_kind: input.candidate_kind.to_string(),
        label: input.label.to_string(),
        state: serde_json::to_string(input.state).expect("state string serializes"),
        questions: serde_json::to_string(&questions_value()).expect("questions serialize"),
        gold: serde_json::to_string(&gold_value(input.label)).expect("gold serialize"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> RowInput<'static> {
        RowInput {
            repo: "ripgrep",
            lang: "rust",
            split: "train",
            query_index: 17,
            query: "where is the code that searches a single file",
            query_lang: "en",
            template: "doc-en",
            candidate_rank: 3,
            candidate_kind: "text match",
            label: "B",
            state: "query: x\ncandidate: a.rs:1-2 (text match)\ncode:\n  y",
        }
    }

    #[test]
    fn ids_are_formatted_with_zero_padding() {
        let row = build_row(&input());
        assert_eq!(row.query_id, "ripgrep:000017");
        assert_eq!(row.id, "ripgrep:000017:03");
        assert_eq!(row.workflow, "repo_explorer_verify");
    }

    #[test]
    fn state_questions_gold_are_json_strings() {
        let row = build_row(&input());

        let state_text: String = serde_json::from_str(&row.state).unwrap();
        assert_eq!(state_text, input().state);

        let questions: serde_json::Value = serde_json::from_str(&row.questions).unwrap();
        assert_eq!(questions["relevant"]["type"], "choice");
        assert_eq!(
            questions["relevant"]["instructions"],
            "Does this code location answer the repository search query?"
        );
        assert_eq!(
            questions["relevant"]["criteria"]["A"],
            "yes, this location answers the query"
        );
        assert_eq!(
            questions["relevant"]["criteria"]["B"],
            "no, this location does not answer the query"
        );

        let gold: serde_json::Value = serde_json::from_str(&row.gold).unwrap();
        assert_eq!(gold["relevant"]["label"], "B");
        assert_eq!(gold["relevant"]["probabilities"]["A"], 0.0);
        assert_eq!(gold["relevant"]["probabilities"]["B"], 1.0);
    }

    #[test]
    fn row_serializes_with_id_first_and_round_trips() {
        let row = build_row(&input());
        let serialized = serde_json::to_string(&row).unwrap();
        assert!(
            serialized.starts_with("{\"id\":\"ripgrep:000017:03\","),
            "{serialized}"
        );
        let back: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(back["candidate_rank"], 3);
        assert_eq!(back["candidate_kind"], "text match");
        assert_eq!(back["label"], "B");
    }

    #[test]
    fn gold_matches_positive_label() {
        let g = gold_value("A");
        assert_eq!(g["relevant"]["label"], "A");
        assert_eq!(g["relevant"]["probabilities"]["A"], 1.0);
        assert_eq!(g["relevant"]["probabilities"]["B"], 0.0);
    }
}
