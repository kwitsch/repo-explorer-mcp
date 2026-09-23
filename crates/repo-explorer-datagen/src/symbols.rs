//! Repository file walk, outline filtering, and doc/literal extraction. Pure
//! string logic (plus the `ignore` walk); no LLM, no network.

use ignore::WalkBuilder;
use repo_explorer_core::domain::ExplorationFinding;
use std::collections::HashSet;
use std::path::Path;

/// File extensions kept by the walk.
pub const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "mjs", "go", "java", "kt",
];

/// Directory components dropped (compared case-insensitively).
const EXCLUDED_DIRS: &[&str] = &[
    "test",
    "tests",
    "testdata",
    "testing",
    "__tests__",
    "vendor",
    "node_modules",
    "third_party",
    "dist",
    "build",
    "target",
    "examples",
    "benches",
    "fixtures",
    "docs",
];

/// Lowercased short names never turned into queries (too generic).
const SHORT_NAME_STOPLIST: &[&str] = &[
    "main",
    "init",
    "__init__",
    "new",
    "default",
    "from",
    "into",
    "fmt",
    "drop",
    "clone",
    "hash",
    "test",
    "setup",
    "run",
    "get",
    "set",
    "len",
    "call",
    "next",
    "build",
    "close",
    "open",
    "read",
    "write",
    "apply",
    "invoke",
    "equals",
    "hashcode",
    "tostring",
    "compareto",
    "iter",
    "item",
    "value",
    "values",
    "keys",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolCandidate {
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub qualified_name: String,
}

/// Repo-relative (`/`-separated) source paths, `.gitignore`-honouring, sorted.
pub fn walk_source_files(repo_root: &Path) -> Vec<String> {
    let mut out = Vec::new();
    for entry in WalkBuilder::new(repo_root).build().flatten() {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let Ok(rel) = entry.path().strip_prefix(repo_root) else {
            continue;
        };
        let rel = rel.to_string_lossy().replace('\\', "/");
        if is_included_path(&rel) {
            out.push(rel);
        }
    }
    out.sort();
    out
}

/// True for a repo-relative path kept by the walk's extension + exclusion rules.
pub fn is_included_path(rel: &str) -> bool {
    let name = rel.rsplit('/').next().unwrap_or(rel);
    let ext = name
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .unwrap_or_default();
    if !SOURCE_EXTENSIONS.contains(&ext.as_str()) {
        return false;
    }
    let mut components: Vec<&str> = rel.split('/').collect();
    components.pop(); // drop the file name; check directory components only
    for c in components {
        if EXCLUDED_DIRS.contains(&c.to_ascii_lowercase().as_str()) {
            return false;
        }
    }
    let lname = name.to_ascii_lowercase();
    !(lname.ends_with("_test.go")
        || lname.ends_with("_test.py")
        || lname.ends_with(".min.js")
        || lname.ends_with(".d.ts")
        || (lname.starts_with("test_") && lname.ends_with(".py"))
        || lname.contains(".test.")
        || lname.contains(".spec."))
}

/// The last non-empty segment after splitting on `.`, `:`, `/`, `#`, `$`.
pub fn short_name(qualified: &str) -> &str {
    qualified
        .split(['.', ':', '/', '#', '$'])
        .rfind(|s| !s.is_empty())
        .unwrap_or(qualified)
}

/// Symbol candidates for one file: keep known-location findings with a
/// qualified `note`, `line_end > line_start`, a >=4-char non-stoplisted short
/// name; dedupe on the line range (first wins).
pub fn symbols_from_outline(rel: &str, findings: &[ExplorationFinding]) -> Vec<SymbolCandidate> {
    let mut out = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    for f in findings {
        let Some(qualified) = f.note.as_deref() else {
            continue;
        };
        let (line_start, line_end) = (f.location.line_start, f.location.line_end);
        if line_start == 0 || line_end <= line_start {
            continue;
        }
        let short = short_name(qualified);
        if short.chars().count() < 4 {
            continue;
        }
        if SHORT_NAME_STOPLIST.contains(&short.to_ascii_lowercase().as_str()) {
            continue;
        }
        if !seen.insert((line_start, line_end)) {
            continue;
        }
        out.push(SymbolCandidate {
            path: rel.to_string(),
            line_start,
            line_end,
            qualified_name: qualified.to_string(),
        });
    }
    out
}

fn strip_doc_marker(line: &str) -> &str {
    for marker in ["///", "//!", "/**", "//", "/*", "*", "#"] {
        if let Some(rest) = line.strip_prefix(marker) {
            return rest;
        }
    }
    line
}

fn split_lines(file_text: &str) -> Vec<&str> {
    file_text
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .collect()
}

/// The doc sentence for a symbol (D4.5.1), or `None` if none is acceptable.
pub fn doc_sentence(
    file_text: &str,
    line_start: u32,
    line_end: u32,
    is_python: bool,
) -> Option<String> {
    let lines = split_lines(file_text);
    let def = line_start as usize; // 1-based def line

    // Step 1: comment block above the definition.
    let mut collected: Vec<String> = Vec::new();
    if def >= 2 {
        for i in (1..=def - 1).rev() {
            let trimmed = lines.get(i - 1).copied().unwrap_or("").trim();
            if trimmed.starts_with("#[") || trimmed.starts_with("#!") || trimmed.starts_with('@') {
                continue; // skipped: never a stop, never collected
            }
            let is_comment = trimmed.starts_with("///")
                || trimmed.starts_with("//!")
                || trimmed.starts_with("//")
                || trimmed.starts_with("/**")
                || trimmed.starts_with("/*")
                || trimmed.starts_with('*')
                || trimmed.starts_with('#');
            if is_comment {
                collected.push(trimmed.to_string());
            } else {
                break;
            }
        }
    }
    collected.reverse();

    let mut cleaned: Vec<String> = Vec::new();
    for line in &collected {
        let no_close = line.strip_suffix("*/").unwrap_or(line);
        let t = strip_doc_marker(no_close.trim()).trim();
        if t.is_empty() || t.starts_with('@') {
            continue;
        }
        cleaned.push(t.to_string());
    }

    // Step 2: Python docstring fallback (only when step 1 found nothing).
    if cleaned.is_empty() && is_python {
        for ln in (def + 1)..=(line_end as usize) {
            let trimmed = lines.get(ln - 1).copied().unwrap_or("").trim();
            if trimmed.is_empty() {
                continue;
            }
            let opener = if trimmed.starts_with("\"\"\"") {
                Some("\"\"\"")
            } else if trimmed.starts_with("'''") {
                Some("'''")
            } else {
                None
            };
            if let Some(q) = opener {
                let rest = &trimmed[q.len()..];
                let text = match rest.find(q) {
                    Some(idx) => &rest[..idx],
                    None => rest,
                };
                let t = text.trim();
                if !t.is_empty() {
                    cleaned.push(t.to_string());
                }
            }
            break; // only the first non-empty line
        }
    }

    if cleaned.is_empty() {
        return None;
    }

    // Step 3: join and collapse whitespace.
    let collapsed = cleaned
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    // Step 4: acceptance is judged on the whole doc, not the cut sentence.
    let words = collapsed.split_whitespace().count();
    if !(4..=25).contains(&words)
        || collapsed.contains("TODO")
        || collapsed.contains("FIXME")
        || collapsed.starts_with('@')
        || collapsed.starts_with('#')
    {
        return None;
    }

    // Step 5: cut at the first ". ".
    let sentence = match collapsed.find(". ") {
        Some(idx) => collapsed[..idx + 1].to_string(),
        None => collapsed,
    };
    Some(sentence)
}

/// The first acceptable error literal in `[line_start, line_end]` (D4.5.2).
pub fn error_literal(file_text: &str, line_start: u32, line_end: u32) -> Option<String> {
    const MARKERS: &[&str] = &[
        "panic!(",
        "bail!(",
        "anyhow!(",
        "ensure!(",
        "Err(",
        "raise ",
        "throw ",
        "errors.New(",
        "fmt.Errorf(",
        "Exception(",
        "Error(",
    ];
    let lines = split_lines(file_text);
    for ln in (line_start as usize)..=(line_end as usize) {
        let raw = lines.get(ln.saturating_sub(1)).copied().unwrap_or("");
        if !raw.contains('"') || !MARKERS.iter().any(|m| raw.contains(m)) {
            continue;
        }
        let bytes = raw.as_bytes();
        let Some(first) = raw.find('"') else {
            continue;
        };
        let mut end_quote = None;
        for idx in (first + 1)..raw.len() {
            if bytes[idx] == b'"' && bytes[idx - 1] != b'\\' {
                end_quote = Some(idx);
                break;
            }
        }
        let Some(end_q) = end_quote else {
            continue;
        };
        let literal = &raw[first + 1..end_q];
        let chars = literal.chars().count();
        if (12..=120).contains(&chars) && literal.split_whitespace().count() >= 2 {
            return Some(literal.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::domain::{ExplorationFinding, FileLocation};
    use std::path::PathBuf;

    fn fnote(path: &str, s: u32, e: u32, note: &str) -> ExplorationFinding {
        ExplorationFinding {
            location: FileLocation {
                path: PathBuf::from(path),
                line_start: s,
                line_end: e,
            },
            snippet: None,
            note: Some(note.to_string()),
        }
    }

    #[test]
    fn short_name_splits_on_all_separators() {
        assert_eq!(short_name("a::b::Foo"), "Foo");
        assert_eq!(short_name("pkg.Class$Inner"), "Inner");
        assert_eq!(short_name("mod/fn"), "fn");
        assert_eq!(short_name("bare"), "bare");
    }

    #[test]
    fn outline_filter_applies_stoplist_length_and_location_rules() {
        let findings = vec![
            fnote("a.rs", 10, 20, "module::new"), // stoplisted short name
            fnote("a.rs", 30, 40, "module::do"),  // < 4 chars
            fnote("a.rs", 50, 60, "module::resolve_url"), // keep
            fnote("a.rs", 70, 70, "module::widen"), // line_end == line_start
            fnote("a.rs", 0, 5, "module::early"), // unknown location
            fnote("a.rs", 55, 58, "other::plain"), // keep (distinct range)
        ];
        let out = symbols_from_outline("a.rs", &findings);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].qualified_name, "module::resolve_url");
        assert_eq!((out[0].line_start, out[0].line_end), (50, 60));
        assert_eq!(out[1].qualified_name, "other::plain");
    }

    #[test]
    fn outline_filter_dedupes_on_line_range() {
        let findings = vec![
            fnote("a.rs", 10, 20, "module::alpha_fn"),
            fnote("a.rs", 10, 20, "module::beta_fn"), // same range -> dropped
        ];
        let out = symbols_from_outline("a.rs", &findings);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].qualified_name, "module::alpha_fn");
    }

    #[test]
    fn doc_slash_slash_slash() {
        let f = "/// Parse the config file into settings\nfn parse_config() {}";
        assert_eq!(
            doc_sentence(f, 2, 3, false).as_deref(),
            Some("Parse the config file into settings")
        );
    }

    #[test]
    fn doc_block_comment() {
        let f = "/**\n * Builds the request pipeline\n */\nclass Pipeline {}";
        assert_eq!(
            doc_sentence(f, 4, 5, false).as_deref(),
            Some("Builds the request pipeline")
        );
    }

    #[test]
    fn doc_hash_comment() {
        let f = "# Compute the moving average window\ndef average():\n    pass";
        assert_eq!(
            doc_sentence(f, 2, 3, true).as_deref(),
            Some("Compute the moving average window")
        );
    }

    #[test]
    fn doc_slash_comment_skips_attribute_and_annotation() {
        let f = "#[derive(Debug)]\n@Override\n// Resolves the redirect target\nfn resolve() {}";
        assert_eq!(
            doc_sentence(f, 4, 5, false).as_deref(),
            Some("Resolves the redirect target")
        );
    }

    #[test]
    fn doc_python_docstring() {
        let f = "def handler():\n    \"\"\"Handle the incoming websocket frame\"\"\"\n    pass";
        assert_eq!(
            doc_sentence(f, 1, 3, true).as_deref(),
            Some("Handle the incoming websocket frame")
        );
    }

    #[test]
    fn doc_cuts_at_first_sentence() {
        let f = "/// Parse the file. Then validate every field before use\nfn parse() {}";
        assert_eq!(
            doc_sentence(f, 2, 3, false).as_deref(),
            Some("Parse the file.")
        );
    }

    #[test]
    fn doc_rejects_too_short_or_todo() {
        assert_eq!(doc_sentence("/// short doc\nfn x() {}", 2, 3, false), None); // 2 words
        assert_eq!(
            doc_sentence(
                "/// TODO fix the broken retry loop later\nfn x() {}",
                2,
                3,
                false
            ),
            None
        );
    }

    #[test]
    fn literal_between_first_unescaped_quotes() {
        let f = "fn x() {\n    return Err(\"config file was not found here\");\n}";
        assert_eq!(
            error_literal(f, 1, 3).as_deref(),
            Some("config file was not found here")
        );
    }

    #[test]
    fn literal_keeps_escaped_quotes() {
        let f = r#"raise ValueError("path \"x\" is invalid here")"#;
        let l = error_literal(f, 1, 1).unwrap();
        assert!(l.starts_with("path"), "{l}");
        assert!(l.contains("invalid here"), "{l}");
    }

    #[test]
    fn literal_length_and_marker_bounds() {
        assert_eq!(error_literal(r#"throw new Error("short")"#, 1, 1), None); // < 12 chars
        let long = "x ".repeat(80);
        assert_eq!(error_literal(&format!("bail!(\"{long}\")"), 1, 1), None); // > 120 chars
        assert_eq!(
            error_literal(r#"let name = "hello world foo";"#, 1, 1),
            None
        ); // no marker
    }

    #[test]
    fn path_exclusion_rules() {
        assert!(is_included_path("src/lib.rs"));
        assert!(is_included_path("pkg/foo.go"));
        assert!(is_included_path("a/b/c.kt"));
        assert!(is_included_path("src/Foo.java"));
        assert!(!is_included_path("src/tests/foo.rs"));
        assert!(!is_included_path("node_modules/x/a.js"));
        assert!(!is_included_path("vendor/y/z.go"));
        assert!(!is_included_path("src/foo_test.go"));
        assert!(!is_included_path("utils_test.py"));
        assert!(!is_included_path("src/test_foo.py"));
        assert!(!is_included_path("src/bundle.min.js"));
        assert!(!is_included_path("types/index.d.ts"));
        assert!(!is_included_path("src/foo.test.ts"));
        assert!(!is_included_path("src/foo.spec.tsx"));
        assert!(!is_included_path("README.md"));
    }
}
