//! Rule-based query templates over a symbol. Six template ids in two languages,
//! chosen by a seeded weighted draw so a run is byte-reproducible.

use crate::rng::{Rng, fnv1a64, splitmix64};
use crate::symbols::{SymbolCandidate, short_name};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedQuery {
    pub text: String,
    pub query_lang: &'static str,
    pub template: &'static str,
}

/// Split an identifier into lowercased words: break on `_`/`-`, lower->upper
/// transitions, and acronym ends (`HTTPServer` -> `http`, `server`). Digits
/// stay attached to the preceding word. Empty words are dropped.
pub fn words(name: &str) -> Vec<String> {
    let chars: Vec<char> = name.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for i in 0..chars.len() {
        let c = chars[i];
        if c == '_' || c == '-' {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if !cur.is_empty() {
            let prev = chars[i - 1];
            let boundary = if c.is_ascii_digit() || prev.is_ascii_digit() {
                false
            } else if prev.is_ascii_lowercase() && c.is_ascii_uppercase() {
                true
            } else if prev.is_ascii_uppercase() && c.is_ascii_uppercase() {
                chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase())
            } else {
                false
            };
            if boundary {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.push(c.to_ascii_lowercase());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|w| !w.is_empty());
    out
}

fn lower_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[derive(Clone, Copy)]
enum Kind {
    DefineEn,
    WordsEn,
    DocEn,
    LiteralEn,
    WordsDe,
    DefineDe,
}

/// Build one query for a symbol, or `None` if none applies (never happens in
/// practice — `define-en`/`define-de` always apply).
pub fn build_query(
    repo: &str,
    symbol: &SymbolCandidate,
    doc: Option<&str>,
    literal: Option<&str>,
    seed: u64,
) -> Option<GeneratedQuery> {
    let n = short_name(&symbol.qualified_name);
    let split = words(n);
    let w = split.join(" ");
    let two_words = split.len() >= 2;

    // Applicable templates in table order, each with its integer weight.
    let mut applicable: Vec<(u64, Kind)> = Vec::new();
    applicable.push((2, Kind::DefineEn));
    if two_words {
        applicable.push((6, Kind::WordsEn));
    }
    if doc.is_some() {
        applicable.push((6, Kind::DocEn));
    }
    if literal.is_some() {
        applicable.push((4, Kind::LiteralEn));
    }
    if two_words {
        applicable.push((2, Kind::WordsDe));
    }
    applicable.push((1, Kind::DefineDe));

    let total: u64 = applicable.iter().map(|(w, _)| *w).sum();
    if total == 0 {
        return None;
    }

    let key = format!(
        "{repo}\0{}\0{}\0{}",
        symbol.path, symbol.line_start, symbol.qualified_name
    );
    let mut rng = Rng::new(splitmix64(seed ^ fnv1a64(key.as_bytes())));

    let mut pick = rng.next_u64() % total;
    let mut chosen = applicable[0].1;
    for (weight, kind) in &applicable {
        if pick < *weight {
            chosen = *kind;
            break;
        }
        pick -= *weight;
    }

    let (text, query_lang, template) = match chosen {
        Kind::DefineEn => (format!("where is {n} defined"), "en", "define-en"),
        Kind::WordsEn => {
            let text = match rng.next_u64() % 3 {
                0 => format!("where is the code that handles {w}"),
                1 => format!("how does this project {w}"),
                _ => format!("where do we {w}"),
            };
            (text, "en", "words-en")
        }
        Kind::DocEn => {
            let d = lower_first(doc.unwrap());
            let d = d.strip_suffix('.').unwrap_or(&d);
            (format!("where is the code that {d}"), "en", "doc-en")
        }
        Kind::LiteralEn => (
            format!("where is the error \"{}\" raised", literal.unwrap()),
            "en",
            "literal-en",
        ),
        Kind::WordsDe => {
            let text = if rng.next_u64().is_multiple_of(2) {
                format!("wo wird {w} behandelt")
            } else {
                format!("wo ist die Logik für {w}")
            };
            (text, "de", "words-de")
        }
        Kind::DefineDe => (format!("wo ist {n} definiert"), "de", "define-de"),
    };

    Some(GeneratedQuery {
        text,
        query_lang,
        template,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::SymbolCandidate;

    fn symbol(q: &str) -> SymbolCandidate {
        SymbolCandidate {
            path: "src/a.rs".into(),
            line_start: 10,
            line_end: 20,
            qualified_name: q.into(),
        }
    }

    #[test]
    fn words_splits_identifiers() {
        assert_eq!(
            words("parse_finish_lenient"),
            vec!["parse", "finish", "lenient"]
        );
        assert_eq!(words("HTTPServer"), vec!["http", "server"]);
        assert_eq!(words("getHTTPResponse2"), vec!["get", "http", "response2"]);
        assert_eq!(words("__init__"), vec!["init"]);
        assert_eq!(words("kebab-case-name"), vec!["kebab", "case", "name"]);
    }

    #[test]
    fn single_word_symbol_only_defines() {
        let sym = symbol("m::fizzbuzz");
        let q = build_query("ripgrep", &sym, None, None, 20260923).unwrap();
        assert!(
            q.text == "where is fizzbuzz defined" || q.text == "wo ist fizzbuzz definiert",
            "{}",
            q.text
        );
        assert!(q.template == "define-en" || q.template == "define-de");
    }

    #[test]
    fn selection_is_deterministic_for_a_seed() {
        let sym = symbol("m::render_widget");
        let a = build_query(
            "repo",
            &sym,
            Some("Renders the widget."),
            Some("bad widget state here"),
            12345,
        )
        .unwrap();
        let b = build_query(
            "repo",
            &sym,
            Some("Renders the widget."),
            Some("bad widget state here"),
            12345,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn doc_en_lowercases_first_char_and_drops_trailing_period() {
        let sym = symbol("m::render_widget");
        let mut found = None;
        for seed in 0..2000u64 {
            let q = build_query("repo", &sym, Some("Renders the widget."), None, seed).unwrap();
            if q.template == "doc-en" {
                found = Some(q);
                break;
            }
        }
        let q = found.expect("doc-en should be reachable across seeds");
        assert_eq!(q.text, "where is the code that renders the widget");
        assert_eq!(q.query_lang, "en");
    }

    #[test]
    fn literal_en_wraps_the_message() {
        let sym = symbol("m::do_thing_now");
        let mut found = None;
        for seed in 0..2000u64 {
            let q =
                build_query("repo", &sym, None, Some("connection refused by peer"), seed).unwrap();
            if q.template == "literal-en" {
                found = Some(q);
                break;
            }
        }
        let q = found.expect("literal-en should be reachable");
        assert_eq!(
            q.text,
            "where is the error \"connection refused by peer\" raised"
        );
    }

    #[test]
    fn words_templates_use_split_words() {
        let sym = symbol("m::resolveRedirects");
        let (mut en, mut de) = (None, None);
        for seed in 0..5000u64 {
            let q = build_query("repo", &sym, None, None, seed).unwrap();
            match q.template {
                "words-en" if en.is_none() => en = Some(q.text.clone()),
                "words-de" if de.is_none() => de = Some(q.text.clone()),
                _ => {}
            }
            if en.is_some() && de.is_some() {
                break;
            }
        }
        let en = en.expect("words-en reachable");
        assert!(
            en == "where is the code that handles resolve redirects"
                || en == "how does this project resolve redirects"
                || en == "where do we resolve redirects",
            "{en}"
        );
        let de = de.expect("words-de reachable");
        assert!(
            de == "wo wird resolve redirects behandelt"
                || de == "wo ist die Logik für resolve redirects",
            "{de}"
        );
    }
}
