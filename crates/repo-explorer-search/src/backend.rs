//! `NativeSearchBackend`: in-process text/filename search implementing
//! `repo_explorer_core::search::SearchBackend` on ripgrep's own library
//! crates — `ignore` for traversal + glob overrides, `grep` for matching — so
//! no external `rg` binary is ever spawned. Traversal uses rg's defaults
//! (`.gitignore`/`.ignore`/hidden/binary skips, no symlink follow,
//! single-threaded); determinism (F-03) comes from an explicit
//! `(path, line)` sort before the client-side `max_results` truncation.

use grep::regex::{RegexMatcher, RegexMatcherBuilder};
use grep::searcher::{
    BinaryDetection, Searcher, SearcherBuilder, Sink, SinkContext, SinkContextKind, SinkMatch,
};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;
use repo_explorer_core::config::SearchConfig;
use repo_explorer_core::domain::{ExplorationFinding, FileLocation, saturate_u32};
use repo_explorer_core::search::{SearchBackend, SearchError, SearchOptions};
use std::path::Path;
use std::time::{Duration, Instant};

pub struct NativeSearchBackend {
    timeout_seconds: u64,
}

impl NativeSearchBackend {
    /// Infallible, synchronous — there is no binary to resolve, so a native
    /// backend is always available. `timeout_seconds` (`0` = no timeout) is
    /// honored as a soft per-file deadline in the walk loop.
    pub fn new(config: &SearchConfig) -> Self {
        Self {
            timeout_seconds: config.timeout_seconds,
        }
    }
}

/// Per-file line collector implementing `grep`'s `Sink`. Matched lines become
/// entries; before-context is buffered and prepended to the next match,
/// trailing context is appended to the previous match — the same grouping the
/// deleted `rg`-output parser produced.
#[derive(Default)]
struct Collector {
    entries: Vec<(u64, String)>,
    pending_before: Vec<String>,
}

/// A matched/context line's bytes as UTF-8 (lossy), with any trailing newline
/// trimmed. Multiline mode is off, so a match is exactly one line.
fn trim_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_end_matches(['\r', '\n'])
        .to_string()
}

impl Sink for Collector {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &Searcher,
        mat: &SinkMatch<'_>,
    ) -> Result<bool, std::io::Error> {
        let line = mat.line_number().unwrap_or(0);
        let text = trim_line(mat.bytes());
        let snippet = if self.pending_before.is_empty() {
            text
        } else {
            let mut combined = self.pending_before.join("\n");
            combined.push('\n');
            combined.push_str(&text);
            self.pending_before.clear();
            combined
        };
        self.entries.push((line, snippet));
        Ok(true)
    }

    fn context(
        &mut self,
        _searcher: &Searcher,
        ctx: &SinkContext<'_>,
    ) -> Result<bool, std::io::Error> {
        let text = trim_line(ctx.bytes());
        match ctx.kind() {
            // Leading context precedes a not-yet-created match: buffer it.
            SinkContextKind::Before => self.pending_before.push(text),
            // Trailing (or other) context follows a match: append to it.
            _ => {
                if let Some(last) = self.entries.last_mut() {
                    last.1.push('\n');
                    last.1.push_str(&text);
                }
            }
        }
        Ok(true)
    }
}

impl SearchBackend for NativeSearchBackend {
    async fn search(
        &self,
        repo_root: &Path,
        pattern: &str,
        scope: Option<&Path>,
        options: &SearchOptions,
    ) -> Result<Vec<ExplorationFinding>, SearchError> {
        if pattern.is_empty() {
            return Err(SearchError::InvalidInput(
                "empty search pattern".to_string(),
            ));
        }

        // Build the matcher on the async side so a malformed pattern returns
        // without spawning a blocking thread. `case_smart(true)` reproduces rg
        // `-S` when the caller wants insensitive/default; a caller-forced
        // `case_sensitive` maps to `case_smart(false)` + `case_insensitive(false)`.
        let matcher = RegexMatcherBuilder::new()
            .case_smart(!options.case_sensitive)
            .case_insensitive(false)
            .build(pattern)
            .map_err(|e| SearchError::InvalidInput(e.to_string()))?;

        // `scope` is always repo-root-relative and non-escaping
        // (`dispatch::validate_scope` rejects absolute/`..` paths upstream).
        let base = scope
            .map(|s| repo_root.join(s))
            .unwrap_or_else(|| repo_root.to_path_buf());
        let repo_root = repo_root.to_path_buf();
        let file_glob = options.file_glob.clone();
        let context_lines = options.context_lines.unwrap_or(0) as usize;
        let max_results = options.max_results;
        let timeout_seconds = self.timeout_seconds;

        // The walk and line search are blocking; the pre-stage fans out many
        // concurrent `search()` calls, so keep this off the tokio workers —
        // mirrors `git_probe`'s and `dispatch::read_file`'s `spawn_blocking`.
        // Do NOT wrap this in `tokio::time::timeout`: that would stop awaiting
        // without stopping the running thread (leaked work). The timeout is a
        // soft per-file deadline checked inside `run_walk` instead.
        let mut findings = tokio::task::spawn_blocking(move || {
            run_walk(
                &base,
                &repo_root,
                &matcher,
                file_glob.as_deref(),
                context_lines,
                timeout_seconds,
            )
        })
        .await
        .map_err(|e| SearchError::BackendFailed {
            backend: "native",
            message: format!("search task failed: {e}"),
        })??;

        // F-03: `ignore`'s walk is readdir-ordered, so an explicit sort is
        // what makes the client-side truncation deterministic across runs.
        findings.sort_by(|a, b| {
            a.location
                .path
                .cmp(&b.location.path)
                .then(a.location.line_start.cmp(&b.location.line_start))
                .then(a.location.line_end.cmp(&b.location.line_end))
        });
        if let Some(max) = max_results {
            findings.truncate(max as usize);
        }
        Ok(findings)
    }
}

/// Synchronous walk + per-file line search. Per-entry walk/read errors are
/// skipped (rg's partial-walk tolerance) and never fail the whole search.
///
/// ponytail: the timeout is a soft, per-file-granularity deadline — a single
/// pathological huge file can overrun it; upgrade to an intra-file counting
/// `Sink` only if that ever bites (in-process, line-bounded reads make it
/// unlikely).
fn run_walk(
    base: &Path,
    repo_root: &Path,
    matcher: &RegexMatcher,
    file_glob: Option<&str>,
    context_lines: usize,
    timeout_seconds: u64,
) -> Result<Vec<ExplorationFinding>, SearchError> {
    let mut builder = WalkBuilder::new(base);
    if let Some(glob) = file_glob {
        // rg `-g`: a whitelist-only override restricts the walk to matching
        // files; a no-slash glob (`name.ext` or `*name*`) matches the basename
        // anywhere — identical to `file_glob_for`'s contract.
        let mut ov = OverrideBuilder::new(base);
        ov.add(glob)
            .map_err(|e| SearchError::InvalidInput(e.to_string()))?;
        let ov = ov
            .build()
            .map_err(|e| SearchError::InvalidInput(e.to_string()))?;
        builder.overrides(ov);
    }
    // Single-threaded (`build`, not `build_parallel`) to avoid an
    // N-searches x M-threads explosion under the concurrent leg fanout;
    // determinism comes from the caller's sort, not walk order. Defaults keep
    // git-ignore/hidden/parent-ignore all on and do not follow symlinks.
    let walker = builder.build();

    let mut searcher = SearcherBuilder::new()
        .binary_detection(BinaryDetection::quit(0))
        .line_number(true)
        .before_context(context_lines)
        .after_context(context_lines)
        .build();

    let deadline = Instant::now() + Duration::from_secs(timeout_seconds);
    let mut findings: Vec<ExplorationFinding> = Vec::new();
    for result in walker {
        if timeout_seconds > 0 && Instant::now() >= deadline {
            return Err(SearchError::Timeout {
                backend: "native",
                seconds: timeout_seconds,
            });
        }
        let Ok(entry) = result else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let mut collector = Collector::default();
        if searcher
            .search_path(matcher, entry.path(), &mut collector)
            .is_err()
        {
            continue;
        }
        // Repo-root-relative, no `./` prefix. `base` is under `repo_root`, so
        // strip_prefix succeeds; the fallback keeps a path that somehow isn't.
        let rel = entry.path().strip_prefix(repo_root).unwrap_or(entry.path());
        for (line, snippet) in collector.entries {
            let n = saturate_u32(line);
            findings.push(ExplorationFinding {
                location: FileLocation {
                    path: rel.to_path_buf(),
                    line_start: n,
                    line_end: n,
                },
                snippet: Some(snippet),
                note: None,
            });
        }
    }
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rex_native_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn backend() -> NativeSearchBackend {
        NativeSearchBackend::new(&SearchConfig::default())
    }

    async fn run(dir: &Path, pattern: &str, opts: SearchOptions) -> Vec<ExplorationFinding> {
        backend()
            .search(dir, pattern, None, &opts)
            .await
            .expect("search should succeed")
    }

    fn paths(f: &[ExplorationFinding]) -> Vec<PathBuf> {
        f.iter().map(|x| x.location.path.clone()).collect()
    }

    #[tokio::test]
    async fn smart_case_lowercase_matches_mixed_case() {
        let dir = tmp("smart_lower");
        fs::write(dir.join("a.txt"), "Needle up\nneedle down\n").unwrap();
        let findings = run(&dir, "needle", SearchOptions::default()).await;
        // Lowercase pattern -> case-insensitive (rg -S): both lines match.
        assert_eq!(findings.len(), 2);
        // Repo-root-relative path, no `./` prefix.
        assert_eq!(findings[0].location.path, PathBuf::from("a.txt"));
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn smart_case_uppercase_pattern_is_case_sensitive() {
        let dir = tmp("smart_upper");
        fs::write(dir.join("a.txt"), "Needle up\nneedle down\n").unwrap();
        let findings = run(&dir, "Needle", SearchOptions::default()).await;
        // An uppercase letter in the pattern -> case-sensitive: only line 1.
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].location.line_start, 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn case_sensitive_flag_forces_sensitivity() {
        let dir = tmp("case_sensitive");
        fs::write(dir.join("a.txt"), "Needle up\nneedle down\n").unwrap();
        let opts = SearchOptions {
            case_sensitive: true,
            ..Default::default()
        };
        let findings = run(&dir, "needle", opts).await;
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].location.line_start, 2);
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn gitignore_hidden_and_dotgit_are_skipped() {
        let dir = tmp("gitignore");
        // An empty `.git` dir activates git-ignore semantics (rg's
        // `require_git` default). `.git/` is itself hidden, so it is never
        // descended.
        fs::create_dir_all(dir.join(".git")).unwrap();
        fs::write(dir.join(".git").join("config"), "needle\n").unwrap();
        fs::write(dir.join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(dir.join("kept.txt"), "needle\n").unwrap();
        fs::write(dir.join("ignored.txt"), "needle\n").unwrap();
        fs::write(dir.join(".hidden.txt"), "needle\n").unwrap();
        let findings = run(&dir, "needle", SearchOptions::default()).await;
        assert_eq!(paths(&findings), vec![PathBuf::from("kept.txt")]);
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn binary_files_are_skipped() {
        let dir = tmp("binary");
        // A NUL before the literal: BinaryDetection::quit stops before the
        // match, so the binary file yields nothing.
        fs::write(dir.join("bin.dat"), b"\x00needle\n").unwrap();
        fs::write(dir.join("plain.txt"), "needle\n").unwrap();
        let findings = run(&dir, "needle", SearchOptions::default()).await;
        assert_eq!(paths(&findings), vec![PathBuf::from("plain.txt")]);
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn file_glob_restricts_by_basename() {
        let dir = tmp("glob");
        fs::write(dir.join("target.txt"), "needle\n").unwrap();
        fs::write(dir.join("other.log"), "needle\n").unwrap();
        // `name.ext` form (file_glob_for output for a dotted token).
        let opts = SearchOptions {
            file_glob: Some("target.txt".to_string()),
            ..Default::default()
        };
        assert_eq!(
            paths(&run(&dir, "needle", opts).await),
            vec![PathBuf::from("target.txt")]
        );
        // `*name*` form (file_glob_for output for an extensionless token).
        let opts = SearchOptions {
            file_glob: Some("*other*".to_string()),
            ..Default::default()
        };
        assert_eq!(
            paths(&run(&dir, "needle", opts).await),
            vec![PathBuf::from("other.log")]
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn context_lines_fold_into_the_snippet() {
        let dir = tmp("context");
        fs::write(dir.join("a.txt"), "line one\nneedle two\nline three\n").unwrap();
        let opts = SearchOptions {
            context_lines: Some(1),
            ..Default::default()
        };
        let findings = run(&dir, "needle", opts).await;
        assert_eq!(findings.len(), 1);
        assert_eq!(
            findings[0].snippet.as_deref(),
            Some("line one\nneedle two\nline three")
        );
        assert_eq!(findings[0].location.line_start, 2);
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn scope_narrows_the_walk_and_paths_stay_repo_relative() {
        let dir = tmp("scope");
        fs::create_dir_all(dir.join("sub")).unwrap();
        fs::write(dir.join("top.txt"), "needle\n").unwrap();
        fs::write(dir.join("sub").join("inner.txt"), "needle\n").unwrap();
        let findings = backend()
            .search(
                &dir,
                "needle",
                Some(Path::new("sub")),
                &SearchOptions::default(),
            )
            .await
            .expect("scoped search should succeed");
        assert_eq!(
            paths(&findings),
            vec![PathBuf::from("sub").join("inner.txt")]
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn empty_pattern_is_invalid_input() {
        let dir = tmp("empty");
        let err = backend()
            .search(&dir, "", None, &SearchOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, SearchError::InvalidInput(_)));
        fs::remove_dir_all(&dir).ok();
    }
}
