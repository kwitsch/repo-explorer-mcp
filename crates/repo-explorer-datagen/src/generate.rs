//! The generator core: file walk -> symbol candidates -> queries -> retrieval
//! snapshot -> labelled rows. Steps 3-7 of D4.4 only; git, index refresh and
//! file output live in `run_generate`. No LLM, no provider call.

use crate::corpus::CorpusRepo;
use crate::label::{Ground, label_candidates};
use crate::rng::{Rng, fnv1a64, splitmix64};
use crate::rows::{RepoStats, Row, RowInput, build_row};
use crate::symbols::{
    SymbolCandidate, doc_sentence, error_literal, symbols_from_outline, walk_source_files,
};
use crate::templates::build_query;
use repo_explorer_agent::judge_input::render_judge_states;
use repo_explorer_agent::{SnapshotStage, retrieval_snapshot};
use repo_explorer_core::config::AgentSettings;
use repo_explorer_core::domain::ExplorationQuery;
use repo_explorer_core::memory::MemoryBackend;
use repo_explorer_core::retrieval::kind_label;
use repo_explorer_core::search::SearchBackend;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub struct GenerateOptions {
    pub seed: u64,
    pub max_queries_per_repo: usize,
    pub max_negatives_per_query: usize,
}

pub struct RepoOutput {
    pub rows: Vec<Row>,
    pub stats: RepoStats,
}

/// Steps 3-7 for one already-checked-out, already-indexed repo.
pub async fn generate_repo<M: MemoryBackend, S: SearchBackend>(
    memory: &M,
    search: &S,
    repo: &CorpusRepo,
    repo_root: &Path,
    opts: &GenerateOptions,
) -> RepoOutput {
    let settings = AgentSettings::default();
    let mut stats = RepoStats::default();
    let mut rows: Vec<Row> = Vec::new();

    // Step 3: file walk.
    let files = walk_source_files(repo_root);
    stats.files = files.len() as u32;

    // Step 4: symbols.
    let mut all_symbols: Vec<SymbolCandidate> = Vec::new();
    for rel in &files {
        match memory
            .file_outline(repo_root, Path::new(rel), Some(200))
            .await
        {
            Ok(res) => all_symbols.extend(symbols_from_outline(rel, &res.findings)),
            Err(_) => stats.outline_errors += 1,
        }
    }
    let mut seen: HashSet<(String, u32, u32)> = HashSet::new();
    all_symbols.retain(|s| seen.insert((s.path.clone(), s.line_start, s.line_end)));
    stats.symbols_total = all_symbols.len() as u32;

    // Step 5: sampling.
    let target = 3 * opts.max_queries_per_repo;
    let sampled = sample_symbols(all_symbols, target, opts.seed, &repo.name);
    stats.symbols_sampled = sampled.len() as u32;

    // Steps 6-7: queries + snapshot.
    let mut file_cache: HashMap<String, Option<String>> = HashMap::new();
    let mut query_index = 0usize;
    for symbol in &sampled {
        if query_index >= opts.max_queries_per_repo {
            break;
        }
        let is_python = symbol.path.ends_with(".py");
        let file_text = read_file_cached(&mut file_cache, repo_root, &symbol.path);
        let doc = file_text
            .as_deref()
            .and_then(|t| doc_sentence(t, symbol.line_start, symbol.line_end, is_python));
        let literal = file_text
            .as_deref()
            .and_then(|t| error_literal(t, symbol.line_start, symbol.line_end));

        let Some(query) = build_query(
            &repo.name,
            symbol,
            doc.as_deref(),
            literal.as_deref(),
            opts.seed,
        ) else {
            continue;
        };
        let this_index = query_index;
        query_index += 1;
        stats.queries_generated += 1;
        *stats
            .templates
            .entry(query.template.to_string())
            .or_default() += 1;

        let exploration = ExplorationQuery {
            text: query.text.clone(),
            scope_hint: None,
            max_results: None,
            detailed_snippets: false,
        };
        let snapshot = retrieval_snapshot(memory, search, repo_root, &exploration, &settings).await;

        match snapshot.stage {
            SnapshotStage::EarlyExit => stats.filtered_early_exit += 1,
            SnapshotStage::Fallback => stats.filtered_fallback += 1,
            SnapshotStage::Verify => {
                let states =
                    render_judge_states(memory, repo_root, &query.text, &snapshot.candidates).await;
                let ground = Ground {
                    path: &symbol.path,
                    line_start: symbol.line_start,
                    line_end: symbol.line_end,
                };
                let outcome = label_candidates(
                    &snapshot.candidates,
                    &states,
                    &ground,
                    opts.max_negatives_per_query,
                );
                if outcome.has_positive {
                    stats.queries_with_positive += 1;
                } else {
                    stats.queries_without_positive += 1;
                }
                for sel in &outcome.selected {
                    let state = states[sel.index]
                        .as_deref()
                        .expect("selected candidate has a state");
                    let candidate = &snapshot.candidates[sel.index];
                    let row = build_row(&RowInput {
                        repo: &repo.name,
                        lang: &repo.lang,
                        split: &repo.split,
                        query_index: this_index,
                        query: &query.text,
                        query_lang: query.query_lang,
                        template: query.template,
                        candidate_rank: sel.rank as u32,
                        candidate_kind: kind_label(candidate.kind),
                        label: sel.label,
                        state,
                    });
                    if sel.label == "A" {
                        stats.rows_positive += 1;
                    } else {
                        stats.rows_negative += 1;
                    }
                    rows.push(row);
                }
            }
        }
    }

    RepoOutput { rows, stats }
}

/// Read `rel` under `repo_root` once, caching the result (including a failed
/// read as `None`).
fn read_file_cached(
    cache: &mut HashMap<String, Option<String>>,
    repo_root: &Path,
    rel: &str,
) -> Option<String> {
    if let Some(hit) = cache.get(rel) {
        return hit.clone();
    }
    let content = std::fs::read_to_string(repo_root.join(rel)).ok();
    cache.insert(rel.to_string(), content.clone());
    content
}

/// Partial Fisher-Yates selection of `target` symbols (seeded per repo), then
/// restore their original order. Returns all of `symbols` when it already fits.
fn sample_symbols(
    symbols: Vec<SymbolCandidate>,
    target: usize,
    seed: u64,
    repo_name: &str,
) -> Vec<SymbolCandidate> {
    if symbols.len() <= target {
        return symbols;
    }
    let mut rng = Rng::new(splitmix64(seed ^ fnv1a64(repo_name.as_bytes())));
    let n = symbols.len();
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..target {
        let j = i + (rng.next_u64() % (n - i) as u64) as usize;
        idx.swap(i, j);
    }
    let mut selected: Vec<usize> = idx[..target].to_vec();
    selected.sort_unstable();
    selected.into_iter().map(|i| symbols[i].clone()).collect()
}
