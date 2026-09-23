//! The `stats` subcommand: read the three JSONL split files, validate every
//! row, and print per-split counts as JSON to stdout.

use crate::rows::Row;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Default, serde::Serialize)]
struct SplitStats {
    rows: u32,
    positive: u32,
    negative: u32,
    queries: u32,
    queries_with_positive: u32,
    per_template: BTreeMap<String, u32>,
    per_lang: BTreeMap<String, u32>,
    per_query_lang: BTreeMap<String, u32>,
}

/// Run `stats`. Returns 0, or 1 on the first invalid row (naming file/line).
pub fn run_stats(data: &Path) -> u8 {
    let mut result: BTreeMap<String, SplitStats> = BTreeMap::new();
    for split in ["train", "val", "test"] {
        let path = data.join(format!("{split}.jsonl"));
        let text = std::fs::read_to_string(&path).unwrap_or_default();
        let mut stats = SplitStats::default();
        let mut qids: BTreeSet<String> = BTreeSet::new();
        let mut qids_pos: BTreeSet<String> = BTreeSet::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: Row = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{}:{}: invalid row: {e}", path.display(), i + 1);
                    return 1;
                }
            };
            if row.label != "A" && row.label != "B" {
                eprintln!(
                    "{}:{}: label must be A or B, got `{}`",
                    path.display(),
                    i + 1,
                    row.label
                );
                return 1;
            }
            let gold: serde_json::Value = match serde_json::from_str(&row.gold) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{}:{}: gold is not valid JSON: {e}", path.display(), i + 1);
                    return 1;
                }
            };
            if gold["relevant"]["label"].as_str() != Some(row.label.as_str()) {
                eprintln!(
                    "{}:{}: gold label disagrees with row label `{}`",
                    path.display(),
                    i + 1,
                    row.label
                );
                return 1;
            }
            stats.rows += 1;
            if row.label == "A" {
                stats.positive += 1;
                qids_pos.insert(row.query_id.clone());
            } else {
                stats.negative += 1;
            }
            *stats.per_template.entry(row.template.clone()).or_default() += 1;
            *stats.per_lang.entry(row.lang.clone()).or_default() += 1;
            *stats
                .per_query_lang
                .entry(row.query_lang.clone())
                .or_default() += 1;
            qids.insert(row.query_id.clone());
        }
        stats.queries = qids.len() as u32;
        stats.queries_with_positive = qids_pos.len() as u32;
        result.insert(split.to_string(), stats);
    }
    match serde_json::to_string_pretty(&result) {
        Ok(json) => {
            println!("{json}");
            0
        }
        Err(e) => {
            eprintln!("failed to render stats: {e}");
            1
        }
    }
}
