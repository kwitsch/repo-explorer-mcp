//! Corpus schema, validation and eval-exclusion. Every violation is an error
//! naming the offending repo and field. No LLM, no network — pure parsing.

use std::collections::HashSet;
use std::path::Path;

/// Allowed SPDX license strings (permissive only).
pub const LICENSES: &[&str] = &[
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MIT OR Apache-2.0",
    "MIT OR Unlicense",
];
/// Allowed `lang` values.
pub const LANGS: &[&str] = &[
    "rust",
    "python",
    "typescript",
    "javascript",
    "go",
    "java",
    "kotlin",
];
/// Allowed `split` values.
pub const SPLITS: &[&str] = &["train", "val", "test"];

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Corpus {
    pub version: u32,
    #[serde(default, rename = "repo")]
    pub repos: Vec<CorpusRepo>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CorpusRepo {
    pub name: String,
    pub url: String,
    pub rev: String,
    pub license: String,
    pub lang: String,
    pub split: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct EvalRepos {
    #[serde(default, rename = "repo")]
    repos: Vec<EvalRepo>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct EvalRepo {
    url: String,
}

/// Lowercase, strip one trailing `/`, then strip one trailing `.git`.
pub fn normalize_url(url: &str) -> String {
    let mut u = url.to_ascii_lowercase();
    if let Some(s) = u.strip_suffix('/') {
        u = s.to_string();
    }
    if let Some(s) = u.strip_suffix(".git") {
        u = s.to_string();
    }
    u
}

fn is_valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

fn is_40_lower_hex(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn load_corpus(path: &Path) -> Result<Corpus, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read corpus {}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("failed to parse corpus {}: {e}", path.display()))
}

pub fn load_eval_urls(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read eval-repos {}: {e}", path.display()))?;
    let parsed: EvalRepos = toml::from_str(&text)
        .map_err(|e| format!("failed to parse eval-repos {}: {e}", path.display()))?;
    Ok(parsed.repos.iter().map(|r| normalize_url(&r.url)).collect())
}

pub fn validate_corpus(corpus: &Corpus) -> Result<(), String> {
    if corpus.version != 1 {
        return Err(format!("corpus version must be 1, got {}", corpus.version));
    }
    let mut seen = HashSet::new();
    for repo in &corpus.repos {
        let name = &repo.name;
        if !is_valid_name(name) {
            return Err(format!(
                "repo `{name}`: field `name` must match ^[a-z0-9._-]+$"
            ));
        }
        if !seen.insert(name.clone()) {
            return Err(format!("repo `{name}`: field `name` is duplicated"));
        }
        if !repo.url.starts_with("https://") {
            return Err(format!(
                "repo `{name}`: field `url` must start with https://"
            ));
        }
        if !is_40_lower_hex(&repo.rev) {
            return Err(format!(
                "repo `{name}`: field `rev` must be exactly 40 lowercase hex characters"
            ));
        }
        if !LICENSES.contains(&repo.license.as_str()) {
            return Err(format!(
                "repo `{name}`: field `license` `{}` is not in the allowlist",
                repo.license
            ));
        }
        if !LANGS.contains(&repo.lang.as_str()) {
            return Err(format!(
                "repo `{name}`: field `lang` `{}` is not supported",
                repo.lang
            ));
        }
        if !SPLITS.contains(&repo.split.as_str()) {
            return Err(format!(
                "repo `{name}`: field `split` `{}` must be train|val|test",
                repo.split
            ));
        }
    }
    Ok(())
}

/// Fail if any corpus URL, normalized, equals an eval URL or is this repository.
pub fn check_eval_exclusion(corpus: &Corpus, eval_urls: &[String]) -> Result<(), String> {
    for repo in &corpus.repos {
        let norm = normalize_url(&repo.url);
        if eval_urls.contains(&norm) {
            return Err(format!(
                "repo `{}`: url matches an eval-corpus repo ({norm}); excluded from generation",
                repo.name
            ));
        }
        if norm.ends_with("/repo-explorer-mcp") {
            return Err(format!(
                "repo `{}`: url is this repository ({norm}); excluded from generation",
                repo.name
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo(
        name: &str,
        url: &str,
        rev: &str,
        license: &str,
        lang: &str,
        split: &str,
    ) -> CorpusRepo {
        CorpusRepo {
            name: name.into(),
            url: url.into(),
            rev: rev.into(),
            license: license.into(),
            lang: lang.into(),
            split: split.into(),
        }
    }

    fn good() -> CorpusRepo {
        repo(
            "ripgrep",
            "https://github.com/BurntSushi/ripgrep",
            "3fce3b5bb0236da2df6d99672afb8a719642eca7",
            "MIT OR Unlicense",
            "rust",
            "train",
        )
    }

    fn corpus(repos: Vec<CorpusRepo>) -> Corpus {
        Corpus { version: 1, repos }
    }

    #[test]
    fn valid_corpus_passes() {
        assert!(validate_corpus(&corpus(vec![good()])).is_ok());
    }

    #[test]
    fn bad_version_fails() {
        let mut c = corpus(vec![good()]);
        c.version = 2;
        assert!(validate_corpus(&c).unwrap_err().contains("version"));
    }

    #[test]
    fn each_invalid_field_is_named() {
        let mut bad_name = good();
        bad_name.name = "Ripgrep".into();
        assert!(
            validate_corpus(&corpus(vec![bad_name]))
                .unwrap_err()
                .contains("name")
        );

        let mut bad_url = good();
        bad_url.url = "http://github.com/x/y".into();
        assert!(
            validate_corpus(&corpus(vec![bad_url]))
                .unwrap_err()
                .contains("url")
        );

        let mut short_rev = good();
        short_rev.rev = "abc123".into();
        assert!(
            validate_corpus(&corpus(vec![short_rev]))
                .unwrap_err()
                .contains("rev")
        );

        let mut upper_rev = good();
        upper_rev.rev = "3FCE3B5BB0236DA2DF6D99672AFB8A719642ECA7".into();
        assert!(
            validate_corpus(&corpus(vec![upper_rev]))
                .unwrap_err()
                .contains("rev")
        );

        let mut bad_lic = good();
        bad_lic.license = "GPL-3.0".into();
        assert!(
            validate_corpus(&corpus(vec![bad_lic]))
                .unwrap_err()
                .contains("license")
        );

        let mut bad_lang = good();
        bad_lang.lang = "cobol".into();
        assert!(
            validate_corpus(&corpus(vec![bad_lang]))
                .unwrap_err()
                .contains("lang")
        );

        let mut bad_split = good();
        bad_split.split = "holdout".into();
        assert!(
            validate_corpus(&corpus(vec![bad_split]))
                .unwrap_err()
                .contains("split")
        );
    }

    #[test]
    fn duplicate_name_fails() {
        let err = validate_corpus(&corpus(vec![good(), good()])).unwrap_err();
        assert!(err.contains("duplicated"), "{err}");
    }

    #[test]
    fn normalize_url_lowercases_and_strips() {
        assert_eq!(
            normalize_url("https://GitHub.com/A/B/"),
            "https://github.com/a/b"
        );
        assert_eq!(
            normalize_url("https://github.com/a/b.git"),
            "https://github.com/a/b"
        );
        assert_eq!(
            normalize_url("https://github.com/a/b.git/"),
            "https://github.com/a/b"
        );
    }

    #[test]
    fn eval_url_match_after_normalization_fails() {
        let c = corpus(vec![repo(
            "dup",
            "https://github.com/Encode/HTTPX.git",
            "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
            "BSD-3-Clause",
            "python",
            "train",
        )]);
        let eval = vec![normalize_url("https://github.com/encode/httpx")];
        assert!(
            check_eval_exclusion(&c, &eval)
                .unwrap_err()
                .contains("eval")
        );
    }

    #[test]
    fn self_repo_suffix_fails() {
        let c = corpus(vec![repo(
            "selfish",
            "https://github.com/kwitsch/repo-explorer-mcp",
            "b5addb64f0161ff6bfe94c124ef76f6a1fba5254",
            "MIT",
            "rust",
            "train",
        )]);
        assert!(
            check_eval_exclusion(&c, &[])
                .unwrap_err()
                .contains("this repository")
        );
    }
}
