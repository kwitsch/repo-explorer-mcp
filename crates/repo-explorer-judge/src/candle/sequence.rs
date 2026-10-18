//! Exact port of laya 0.3.7 `common.build_sequence` for a `choice` question
//! with identity option order and `truncate_left = false`.

use repo_explorer_core::judge::JudgeError;
use tokenizers::Tokenizer;

/// A tokenized judge sequence and the two option-marker token positions.
#[derive(Debug)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

/// Resolved ids of the four special tokens the sequence build needs.
pub struct SpecialIds {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    pub pad: u32,
}

pub fn build_sequence(
    tok: &Tokenizer,
    sp: &SpecialIds,
    state: &str,
    instructions: &str,
    options: &[String],
    max_len: usize,
    head_max_len: usize,
) -> Result<Encoded, JudgeError> {
    let enc = |s: &str| -> Result<Vec<u32>, JudgeError> {
        Ok(tok
            .encode(s, false)
            .map_err(|e| JudgeError::Protocol {
                message: format!("tokenizer encode failed: {e}"),
            })?
            .get_ids()
            .to_vec())
    };

    let ins = instructions.replace("[MASK]", " ");
    let mut head_ids = enc(&format!("choice question: {ins}"))?;

    let n = options.len();
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(n);
    for o in options {
        let mut v = vec![sp.mask];
        let body = enc(&(" ".to_string() + &o.replace("[MASK]", " ")))?;
        v.extend(body.into_iter().take(48));
        opt_ids.push(v);
    }

    let sum_opt: usize = opt_ids.iter().map(|v| v.len()).sum();
    let mut opt_budget = head_max_len as isize - sum_opt as isize;
    if opt_budget < 16 {
        let per = std::cmp::max(4, head_max_len.saturating_sub(16) / std::cmp::max(1, n));
        for v in &mut opt_ids {
            v.truncate(per);
        }
        let sum_opt: usize = opt_ids.iter().map(|v| v.len()).sum();
        opt_budget = head_max_len as isize - sum_opt as isize;
    }

    let keep_head = std::cmp::max(8, opt_budget.max(0) as usize);
    head_ids.truncate(keep_head);

    let mut ids: Vec<u32> = Vec::new();
    let mut markers: Vec<usize> = Vec::new();
    ids.push(sp.cls);
    ids.extend_from_slice(&head_ids);
    ids.push(sp.sep);
    for v in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(v);
    }
    ids.push(sp.sep);

    let room = max_len.saturating_sub(ids.len() + 1);
    let st = enc(&state.replace("[MASK]", " "))?;
    ids.extend(st.into_iter().take(room));
    ids.push(sp.sep);

    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    if markers.len() != options.len() {
        return Err(JudgeError::Protocol {
            message: "options exceed head_max_len".to_string(),
        });
    }
    Ok(Encoded { ids, markers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// A WordLevel tokenizer with whitespace pre-tokenization. Every word is one
    /// id; `[UNK]` catches the rest; the specials are fixed ids. This makes the
    /// D5 layout hand-derivable.
    fn toy() -> (Tokenizer, SpecialIds) {
        let mut vocab = String::from("{");
        let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];
        let words = [
            "choice",
            "question",
            "Does",
            "this",
            "code",
            "location",
            "answer",
            "the",
            "repository",
            "search",
            "query",
            "A",
            "B",
            "yes",
            "no",
            "does",
            "not",
            "answers",
            "state",
            "x",
            "let",
            "v",
            "1",
            "foo",
            "bar",
            "baz",
            "qux",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "eta",
            "theta",
        ];
        let mut id = 0usize;
        let mut first = true;
        let push = |vocab: &mut String, tok: &str, id: &mut usize, first: &mut bool| {
            if !*first {
                vocab.push(',');
            }
            *first = false;
            vocab.push_str(&format!("\"{tok}\":{}", *id));
            *id += 1;
        };
        for s in specials {
            push(&mut vocab, s, &mut id, &mut first);
        }
        for w in words {
            push(&mut vocab, w, &mut id, &mut first);
        }
        vocab.push('}');
        let json = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
"normalizer":null,
"pre_tokenizer":{{"type":"Whitespace"}},
"post_processor":null,"decoder":null,
"model":{{"type":"WordLevel","vocab":{vocab},"unk_token":"[UNK]"}}}}"#
        );
        let tok = Tokenizer::from_str(&json).unwrap();
        let sp = SpecialIds {
            cls: tok.token_to_id("[CLS]").unwrap(),
            sep: tok.token_to_id("[SEP]").unwrap(),
            mask: tok.token_to_id("[MASK]").unwrap(),
            pad: tok.token_to_id("[PAD]").unwrap(),
        };
        (tok, sp)
    }

    fn opts() -> Vec<String> {
        vec![
            "A: yes, this location answers the query".to_string(),
            "B: no, this location does not answer the query".to_string(),
        ]
    }

    #[test]
    fn short_state_layout_and_markers() {
        let (tok, sp) = toy();
        let out = build_sequence(&tok, &sp, "state x", "Does this code", &opts(), 128, 64).unwrap();
        // Layout: [CLS] head [SEP] [MASK] optA [MASK] optB [SEP] state [SEP]
        assert_eq!(out.ids[0], sp.cls);
        assert_eq!(out.markers.len(), 2);
        assert_eq!(out.ids[out.markers[0]], sp.mask);
        assert_eq!(out.ids[out.markers[1]], sp.mask);
        assert_eq!(*out.ids.last().unwrap(), sp.sep);
        // markers are strictly increasing
        assert!(out.markers[0] < out.markers[1]);
    }

    #[test]
    fn long_state_truncated_to_max_len() {
        let (tok, sp) = toy();
        let long = "x ".repeat(500);
        let out = build_sequence(&tok, &sp, &long, "Does this code", &opts(), 64, 48).unwrap();
        assert_eq!(out.ids.len(), 64);
        assert_eq!(out.markers.len(), 2);
    }

    #[test]
    fn tiny_head_max_len_truncates_options_and_head() {
        let (tok, sp) = toy();
        // head_max_len = 20 -> per = max(4, (20-16)/2) = 4 -> each opt_ids <= 4
        let out = build_sequence(&tok, &sp, "state x", "Does this code", &opts(), 128, 20).unwrap();
        let opt_a_len = out.markers[1] - out.markers[0];
        assert!(opt_a_len <= 4, "opt A len {opt_a_len}");
        assert_eq!(out.markers.len(), 2);
    }

    #[test]
    fn mask_in_state_and_instructions_becomes_space() {
        let (tok, sp) = toy();
        let out = build_sequence(
            &tok,
            &sp,
            "state [MASK] x",
            "Does [MASK] this",
            &opts(),
            128,
            64,
        )
        .unwrap();
        // The only [MASK] ids are the two option markers.
        let mask_count = out.ids.iter().filter(|&&i| i == sp.mask).count();
        assert_eq!(mask_count, 2);
    }

    #[test]
    fn marker_outside_max_len_is_protocol_error() {
        let (tok, sp) = toy();
        // max_len tiny so the second option marker cannot fit.
        let err =
            build_sequence(&tok, &sp, "state x", "Does this code", &opts(), 6, 64).unwrap_err();
        assert!(matches!(err, JudgeError::Protocol { .. }));
    }
}
