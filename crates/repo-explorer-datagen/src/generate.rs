//! The generator core: file walk -> symbol candidates -> queries -> retrieval
//! snapshot -> labelled rows. Steps 3-7 of D4.4 only; git, index refresh and
//! file output live in `run_generate`. No LLM, no provider call.

use crate::cli::GenerateArgs;
use crate::corpus::{
    CorpusRepo, check_eval_exclusion, load_corpus, load_eval_urls, validate_corpus,
};
use crate::label::{Ground, label_candidates};
use crate::rng::{Rng, fnv1a64, splitmix64};
use crate::rows::{
    Manifest, ManifestSettings, RepoEntry, RepoStats, Row, RowInput, Totals, build_row,
};
use crate::symbols::{
    SymbolCandidate, doc_sentence, error_literal, symbols_from_outline, walk_source_files,
};
use crate::templates::build_query;
use repo_explorer_agent::judge_input::render_judge_states;
use repo_explorer_agent::{SnapshotStage, retrieval_snapshot};
use repo_explorer_core::config::{AgentSettings, CodebaseMemoryConfig, SearchConfig};
use repo_explorer_core::domain::ExplorationQuery;
use repo_explorer_core::judge::JUDGE_STATE_VERSION;
use repo_explorer_core::memory::{IndexStatus, MemoryBackend};
use repo_explorer_core::retrieval::kind_label;
use repo_explorer_core::search::SearchBackend;
use repo_explorer_memory::MemoryClientBackend;
use repo_explorer_search::NativeSearchBackend;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
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

/// Run the full `generate` pipeline. Returns 0, or 1 on invalid corpus/config
/// or when at least one repo failed (partial output is still written).
pub async fn run_generate(args: &GenerateArgs) -> u8 {
    let corpus = match load_corpus(&args.corpus) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if let Err(e) = validate_corpus(&corpus) {
        eprintln!("{e}");
        return 1;
    }
    let eval_urls = match load_eval_urls(&args.eval_repos) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if let Err(e) = check_eval_exclusion(&corpus, &eval_urls) {
        eprintln!("{e}");
        return 1;
    }

    let corpus_sha256 = match sha256_hex(&args.corpus) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let eval_repos_sha256 = match sha256_hex(&args.eval_repos) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    let mem_config = CodebaseMemoryConfig {
        command: Some(args.memory_command.clone()),
        args: args.memory_args.clone(),
        endpoint: None,
        staleness_seconds: 3600,
    };
    let memory = match MemoryClientBackend::connect(&mem_config).await {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to connect to codebase-memory: {e}");
            return 1;
        }
    };
    let search = NativeSearchBackend::new(&SearchConfig::default());
    let settings = AgentSettings::default();
    let opts = GenerateOptions {
        seed: args.seed,
        max_queries_per_repo: args.max_queries_per_repo,
        max_negatives_per_query: args.max_negatives_per_query,
    };

    if let Err(e) = std::fs::create_dir_all(&args.out) {
        eprintln!("failed to create {}: {e}", args.out.display());
        return 1;
    }
    let mut writers = match SplitWriters::open(&args.out) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    let mut entries: Vec<RepoEntry> = Vec::new();
    let mut ok_names: Vec<String> = Vec::new();
    let mut failed: Vec<(String, String)> = Vec::new();

    for repo in &corpus.repos {
        if !args.only.is_empty() && !args.only.contains(&repo.name) {
            continue;
        }
        let repo_root = args.checkouts.join(&repo.name);

        match git_head(&repo_root) {
            Some(head) if head == repo.rev => {}
            _ => {
                let err = "checkout not at pinned rev; run fetch".to_string();
                entries.push(failed_entry(repo, err.clone()));
                failed.push((repo.name.clone(), err));
                continue;
            }
        }

        match memory.ensure_fresh_index(&repo_root).await {
            Ok(IndexStatus::UpToDate | IndexStatus::Reindexed) => {}
            Ok(IndexStatus::IndexingFailed { reason }) => {
                let err = format!("index refresh failed: {reason}");
                entries.push(failed_entry(repo, err.clone()));
                failed.push((repo.name.clone(), err));
                continue;
            }
            Err(e) => {
                let err = format!("index refresh error: {e}");
                entries.push(failed_entry(repo, err.clone()));
                failed.push((repo.name.clone(), err));
                continue;
            }
        }

        let output = generate_repo(&memory, &search, repo, &repo_root, &opts).await;
        if let Err(e) = writers.write_rows(&repo.split, &output.rows) {
            let err = format!("failed to write rows: {e}");
            entries.push(failed_entry(repo, err.clone()));
            failed.push((repo.name.clone(), err));
            continue;
        }
        tracing::info!(repo = %repo.name, rows = output.rows.len(), "generated");
        entries.push(ok_entry(repo, &output.stats));
        ok_names.push(repo.name.clone());
    }

    if let Err(e) = writers.finalize(&args.out) {
        eprintln!("{e}");
        return 1;
    }

    let manifest = Manifest {
        datagen_version: env!("CARGO_PKG_VERSION").to_string(),
        judge_state_version: JUDGE_STATE_VERSION,
        seed: args.seed,
        corpus_sha256,
        eval_repos_sha256,
        settings: ManifestSettings {
            max_queries_per_repo: args.max_queries_per_repo,
            max_negatives_per_query: args.max_negatives_per_query,
            top_k: settings.top_k,
            fallback_confidence: settings.fallback_confidence,
            early_exit_confidence: settings.early_exit_confidence,
        },
        totals: totals_from(&entries),
        repos: entries,
    };
    if let Err(e) = write_manifest(&args.out, &manifest) {
        eprintln!("{e}");
        return 1;
    }

    let summary = serde_json::json!({
        "generated": ok_names,
        "failed": failed.iter().map(|(n, e)| serde_json::json!({"name": n, "error": e})).collect::<Vec<_>>(),
    });
    println!("{summary}");
    if failed.is_empty() { 0 } else { 1 }
}

struct SplitWriters {
    train: BufWriter<File>,
    val: BufWriter<File>,
    test: BufWriter<File>,
}

impl SplitWriters {
    fn open(out: &std::path::Path) -> Result<Self, String> {
        Ok(Self {
            train: BufWriter::new(create_tmp(out, "train.tmp")?),
            val: BufWriter::new(create_tmp(out, "val.tmp")?),
            test: BufWriter::new(create_tmp(out, "test.tmp")?),
        })
    }

    fn write_rows(&mut self, split: &str, rows: &[Row]) -> std::io::Result<()> {
        let writer = match split {
            "train" => &mut self.train,
            "val" => &mut self.val,
            _ => &mut self.test,
        };
        for row in rows {
            let line = serde_json::to_string(row).expect("row serializes");
            writeln!(writer, "{line}")?;
        }
        Ok(())
    }

    fn finalize(self, out: &std::path::Path) -> Result<(), String> {
        // Flush and CLOSE every writer before renaming (Windows can't rename an
        // open file).
        let SplitWriters { train, val, test } = self;
        for mut writer in [train, val, test] {
            writer.flush().map_err(|e| e.to_string())?;
            drop(writer);
        }
        rename_tmp(out, "train.tmp", "train.jsonl")?;
        rename_tmp(out, "val.tmp", "val.jsonl")?;
        rename_tmp(out, "test.tmp", "test.jsonl")?;
        Ok(())
    }
}

fn create_tmp(out: &std::path::Path, name: &str) -> Result<File, String> {
    File::create(out.join(name))
        .map_err(|e| format!("failed to create {}: {e}", out.join(name).display()))
}

fn rename_tmp(out: &std::path::Path, from: &str, to: &str) -> Result<(), String> {
    let dest = out.join(to);
    let _ = std::fs::remove_file(&dest);
    std::fs::rename(out.join(from), &dest)
        .map_err(|e| format!("failed to rename {from} -> {to}: {e}"))
}

fn git_head(dir: &std::path::Path) -> Option<String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn sha256_hex(path: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let bytes =
        std::fs::read(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

fn ok_entry(repo: &CorpusRepo, stats: &RepoStats) -> RepoEntry {
    RepoEntry {
        name: repo.name.clone(),
        rev: repo.rev.clone(),
        split: repo.split.clone(),
        lang: repo.lang.clone(),
        status: "ok".to_string(),
        error: None,
        files: stats.files,
        outline_errors: stats.outline_errors,
        symbols_total: stats.symbols_total,
        symbols_sampled: stats.symbols_sampled,
        queries_generated: stats.queries_generated,
        filtered_early_exit: stats.filtered_early_exit,
        filtered_fallback: stats.filtered_fallback,
        queries_with_positive: stats.queries_with_positive,
        queries_without_positive: stats.queries_without_positive,
        rows_positive: stats.rows_positive,
        rows_negative: stats.rows_negative,
        templates: stats.templates.clone(),
    }
}

fn failed_entry(repo: &CorpusRepo, error: String) -> RepoEntry {
    RepoEntry {
        name: repo.name.clone(),
        rev: repo.rev.clone(),
        split: repo.split.clone(),
        lang: repo.lang.clone(),
        status: "failed".to_string(),
        error: Some(error),
        files: 0,
        outline_errors: 0,
        symbols_total: 0,
        symbols_sampled: 0,
        queries_generated: 0,
        filtered_early_exit: 0,
        filtered_fallback: 0,
        queries_with_positive: 0,
        queries_without_positive: 0,
        rows_positive: 0,
        rows_negative: 0,
        templates: BTreeMap::new(),
    }
}

fn totals_from(entries: &[RepoEntry]) -> Totals {
    let mut totals: Totals = Totals::new();
    for e in entries {
        let t = totals.entry(e.split.clone()).or_default();
        t.files += e.files;
        t.outline_errors += e.outline_errors;
        t.symbols_total += e.symbols_total;
        t.symbols_sampled += e.symbols_sampled;
        t.queries_generated += e.queries_generated;
        t.filtered_early_exit += e.filtered_early_exit;
        t.filtered_fallback += e.filtered_fallback;
        t.queries_with_positive += e.queries_with_positive;
        t.queries_without_positive += e.queries_without_positive;
        t.rows_positive += e.rows_positive;
        t.rows_negative += e.rows_negative;
    }
    totals
}

fn write_manifest(out: &std::path::Path, manifest: &Manifest) -> Result<(), String> {
    let json = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
    std::fs::write(out.join("manifest.json"), json)
        .map_err(|e| format!("failed to write manifest: {e}"))
}
