//! The verification stage: the primary (and normally only) LLM involvement.
//! The deterministic pre-stage already found the candidates; the model merely
//! selects/verifies them — one turn, plus an optional expand turn whose last
//! iteration forces `finish`. Any failure escalates to the explorative
//! fallback loop instead of erroring.

use futures_util::future::join_all;
use repo_explorer_core::domain::{Candidate, ExplorationQuery, ExplorationResult};
use repo_explorer_core::llm::{
    CallOptions, Clock, LlmProvider, Message, ProviderResponse, ProviderRouter,
};
use repo_explorer_core::memory::MemoryBackend;
use repo_explorer_core::retrieval::{is_unknown_location, kind_label};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::agent::{TokenBudget, force_finish_options, push_nudge};
use crate::dispatch::{canonical_repo_root, parse_args, read_file_canonical};
use crate::render::{RenderCaps, cap_file_lines, cap_snippet};
use crate::skeleton::skeleton_for;
use crate::tools::{ExpandArgs, resolve_finish, verify_catalog};

/// Lines of body context fetched around a candidate on `expand`.
const EXPAND_CONTEXT_BEFORE: u32 = 10;
const EXPAND_CONTEXT_AFTER: u32 = 20;

pub(crate) enum VerifyOutcome {
    Finished(ExplorationResult),
    /// Verification could not conclude — run the explorative fallback loop.
    Escalate,
}

/// Static system prompt — deliberately free of any per-run content so the
/// provider-side prompt cache gets a stable prefix.
const VERIFY_SYSTEM_PROMPT: &str = "You are a code-location verification assistant. \
A deterministic retrieval stage has already searched the repository and produced a ranked, \
numbered list of candidate locations for the user's query. Your only job is to select the \
candidates that actually answer the query. If a candidate's outline/snippet is not enough to \
judge it, call the expand tool ONCE, batching every id you need in a single call. \
Then conclude by calling finish: copy the selected candidates' locations verbatim, add a short \
note per finding, and write a one-paragraph summary. Do not attempt broad exploration; if the \
candidates cannot answer the query, call finish with an empty findings list and say so in the \
summary.";

#[allow(clippy::too_many_arguments)]
pub(crate) async fn verify<M, P, C>(
    memory: &M,
    router: &ProviderRouter<P, C>,
    repo_root: &Path,
    query: &ExplorationQuery,
    index_note: Option<&str>,
    candidates: &[Candidate],
    max_verify_iterations: u32,
    budget: &mut TokenBudget,
    caps: &RenderCaps,
) -> VerifyOutcome
where
    M: MemoryBackend,
    P: LlmProvider,
    C: Clock,
{
    let block = candidates_block(memory, repo_root, candidates, caps).await;
    let mut messages = vec![
        Message::system(VERIFY_SYSTEM_PROMPT),
        Message::user(verify_user_prompt(query, index_note, &block)),
    ];

    let turns = max_verify_iterations.max(1);
    for turn in 0..turns {
        let last = turn + 1 == turns || budget.exhausted();
        let options = if last {
            force_finish_options()
        } else {
            CallOptions::default()
        };
        match router
            .complete_with_tools(&messages, verify_catalog(), &options)
            .await
        {
            Ok(completion) => {
                budget.add(completion.usage);
                match completion.response {
                    ProviderResponse::ToolCalls(calls) if !calls.is_empty() => {
                        // Check for a successful finish before cloning anything: the
                        // common case (finish on the first turn) returns right here.
                        let mut responses = match resolve_finish(&calls) {
                            Ok(result) => {
                                let action = if last { "forced_finish" } else { "finish" };
                                tracing::debug!(turn, action, "verify action");
                                return VerifyOutcome::Finished(result);
                            }
                            Err(rejections) => rejections,
                        };
                        tracing::debug!(turn, action = "expand", "verify action");
                        for call in calls.iter().filter(|c| c.name != "finish") {
                            let content = match call.name.as_str() {
                                "expand" => {
                                    expand_content(
                                        repo_root,
                                        candidates,
                                        &call.arguments_json,
                                        caps,
                                    )
                                    .await
                                }
                                other => format!("unknown tool: {other}"),
                            };
                            responses.push(Message::tool(&call.id, content));
                        }
                        messages.push(Message::assistant_tool_calls(calls));
                        messages.extend(responses);
                    }
                    ProviderResponse::ToolCalls(_) => {
                        tracing::debug!(turn, action = "nudge", "verify action");
                        push_nudge(
                            &mut messages,
                            Message::assistant_tool_calls(Vec::new()),
                            "Respond with a tool call: expand or finish.",
                        );
                    }
                    ProviderResponse::Text(text) => {
                        tracing::debug!(turn, action = "nudge", "verify action");
                        push_nudge(
                            &mut messages,
                            Message::assistant_text(text),
                            "Respond with a tool call: expand or finish.",
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "verification stage provider call failed; escalating");
                return VerifyOutcome::Escalate;
            }
        }
    }
    VerifyOutcome::Escalate
}

fn verify_user_prompt(query: &ExplorationQuery, index_note: Option<&str>, block: &str) -> String {
    let mut s = crate::agent::query_preamble(query, index_note);
    s.push_str("\n\nCandidates:\n");
    s.push_str(block);
    s
}

/// The numbered candidate list: header per candidate, plus the file's symbol
/// outline (shown once per file, on its first candidate) or the candidate's
/// capped snippet.
async fn candidates_block<M: MemoryBackend>(
    memory: &M,
    repo_root: &Path,
    candidates: &[Candidate],
    caps: &RenderCaps,
) -> String {
    let unique_paths: HashSet<&PathBuf> = candidates.iter().map(|c| &c.location.path).collect();
    let skeletons: HashMap<&PathBuf, String> =
        join_all(unique_paths.into_iter().map(|path| async move {
            let outline = skeleton_for(memory, repo_root, path).await?;
            Some((path, outline))
        }))
        .await
        .into_iter()
        .flatten()
        .collect();

    let mut shown_outline: HashSet<&PathBuf> = HashSet::new();
    let mut out = String::new();
    for (idx, c) in candidates.iter().enumerate() {
        let symbol = c
            .symbol
            .as_deref()
            .map(|s| format!(", symbol `{s}`"))
            .unwrap_or_default();
        if is_unknown_location(&c.location) {
            out.push_str(&format!(
                "[{}] {} (location unknown, {}{symbol})\n",
                idx + 1,
                c.location.path.display(),
                kind_label(c.kind),
            ));
        } else {
            out.push_str(&format!(
                "[{}] {}:{}-{} ({}{symbol})\n",
                idx + 1,
                c.location.path.display(),
                c.location.line_start,
                c.location.line_end,
                kind_label(c.kind),
            ));
        }
        if let Some(outline) = skeletons.get(&c.location.path) {
            if shown_outline.insert(&c.location.path) {
                out.push_str(outline);
                out.push('\n');
            }
        } else if let Some(snippet) = &c.snippet {
            out.push_str("  ");
            out.push_str(
                &cap_snippet(snippet.clone(), caps.snippet_max_chars).replace('\n', "\n  "),
            );
            out.push('\n');
        }
    }
    out
}

/// Bodies for the requested candidate ids, read deterministically from disk
/// with context around each candidate's range. Ids that don't resolve to a
/// candidate become immediate errors; the rest are read concurrently (like
/// every other fanout in this crate) against `repo_root` canonicalized once
/// up front, then reassembled in the model's requested order.
async fn expand_content(
    repo_root: &Path,
    candidates: &[Candidate],
    arguments_json: &str,
    caps: &RenderCaps,
) -> String {
    let args: ExpandArgs = match parse_args(arguments_json) {
        Ok(args) => args,
        Err(e) => return e,
    };
    if args.candidate_ids.is_empty() {
        return "invalid arguments: candidate_ids must be non-empty".to_string();
    }

    let mut sections: Vec<Option<String>> = vec![None; args.candidate_ids.len()];
    let mut reads = Vec::new();
    for (pos, id) in args.candidate_ids.into_iter().enumerate() {
        match (id as usize).checked_sub(1).and_then(|i| candidates.get(i)) {
            None => sections[pos] = Some(format!("[{id}] no such candidate")),
            Some(candidate) if is_unknown_location(&candidate.location) => {
                let path = candidate.location.path.to_string_lossy().into_owned();
                sections[pos] = Some(format!(
                    "[{id}] {path}: location unknown, no snippet available"
                ));
            }
            Some(candidate) => {
                let path = candidate.location.path.to_string_lossy().into_owned();
                let start = candidate
                    .location
                    .line_start
                    .saturating_sub(EXPAND_CONTEXT_BEFORE)
                    .max(1);
                let end = candidate
                    .location
                    .line_end
                    .saturating_add(EXPAND_CONTEXT_AFTER);
                reads.push((pos, id, path, start, end));
            }
        }
    }

    if !reads.is_empty() {
        let canonical_root = canonical_repo_root(repo_root).await;
        let results = join_all(reads.iter().map(|(_, id, path, start, end)| {
            let (id, start, end) = (*id, *start, *end);
            let canonical_root = &canonical_root;
            async move {
                let result = match canonical_root {
                    Ok(r) => read_file_canonical(repo_root, r, path, Some(start), Some(end)).await,
                    Err(e) => Err(e.clone()),
                };
                match result {
                    Ok(body) => format!(
                        "[{id}] {path}:{start}-{end}\n{}",
                        cap_file_lines(body, caps.read_file_max_lines)
                    ),
                    Err(e) => format!("[{id}] {e}"),
                }
            }
        }))
        .await;
        for ((pos, ..), section) in reads.iter().zip(results) {
            sections[*pos] = Some(section);
        }
    }

    sections
        .into_iter()
        .map(|s| s.expect("every position filled by an error or a read"))
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::{CandidateKind, FileLocation};
    use repo_explorer_core::memory::mock::MockMemoryBackend;

    fn candidate(path: &str, start: u32, end: u32) -> Candidate {
        Candidate {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: start,
                line_end: end,
            },
            symbol: Some("sym".to_string()),
            kind: CandidateKind::SymbolExact,
            score: 700,
            snippet: Some("fn sym() {}".to_string()),
        }
    }

    #[tokio::test]
    async fn candidates_block_numbers_and_falls_back_to_snippets() {
        let memory = MockMemoryBackend::new(); // no graph → no outlines
        let caps = RenderCaps::default();
        let block = candidates_block(
            &memory,
            Path::new("/repo"),
            &[candidate("a.rs", 5, 9), candidate("b.rs", 1, 2)],
            &caps,
        )
        .await;
        assert!(block.contains("[1] a.rs:5-9 (exact symbol match, symbol `sym`)"));
        assert!(block.contains("[2] b.rs:1-2"));
        assert!(block.contains("fn sym() {}"));
    }

    #[tokio::test]
    async fn expand_reports_unknown_ids_and_bad_args() {
        let caps = RenderCaps::default();
        let out = expand_content(
            Path::new("/repo"),
            &[candidate("a.rs", 1, 2)],
            r#"{"candidate_ids":[7]}"#,
            &caps,
        )
        .await;
        assert!(out.contains("[7] no such candidate"));
        let out = expand_content(Path::new("/repo"), &[], r#"{"bogus":1}"#, &caps).await;
        assert!(out.contains("invalid arguments"));
        let out = expand_content(Path::new("/repo"), &[], r#"{"candidate_ids":[]}"#, &caps).await;
        assert!(out.contains("must be non-empty"));
    }

    #[tokio::test]
    async fn expand_reports_unknown_location_instead_of_fabricating_a_range() {
        let caps = RenderCaps::default();
        // (0, *) is core's "unknown location" sentinel — see
        // `is_unknown_location` — reachable here via a SymbolFuzzy/SymbolExact
        // candidate whose retrieval leg never resolved a line number.
        let out = expand_content(
            Path::new("/repo"),
            &[candidate("a.rs", 0, 0)],
            r#"{"candidate_ids":[1]}"#,
            &caps,
        )
        .await;
        assert!(out.contains("[1] a.rs: location unknown, no snippet available"));
        assert!(!out.contains(":1-20"));
    }

    #[tokio::test]
    async fn expand_id_zero_is_not_a_candidate() {
        let caps = RenderCaps::default();
        let out = expand_content(
            Path::new("/repo"),
            &[candidate("a.rs", 1, 2)],
            r#"{"candidate_ids":[0]}"#,
            &caps,
        )
        .await;
        assert!(out.contains("[0] no such candidate"));
    }
}
