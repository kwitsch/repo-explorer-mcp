//! Checkpoint file resolution, config parsing and the transformers-version
//! encoder-config fix-up. No weight loading here (that lives in `mod.rs`).
// Wired into `LoadedModel` in a later task; unused in the lib target until then.
#![allow(dead_code)]

use crate::candle::sequence::SpecialIds;
use candle_core::Device;
use candle_transformers::models::modernbert::Config as ModernBertConfig;
use repo_explorer_core::config::JudgeDevice;
use repo_explorer_core::judge::{JUDGE_STATE_VERSION, JudgeError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub(crate) struct AgentConfig {
    pub max_len: usize,
    pub head_max_len: usize,
    pub head_layers: usize,
    pub temperature: [f32; 3],
    pub temperature_by_options: BTreeMap<String, f32>,
}

#[derive(Debug)]
pub(crate) struct JudgeMeta {
    pub judge_state_version: u32,
    pub select_threshold: u32,
}

pub(crate) fn required_file(dir: &Path, name: &str) -> Result<PathBuf, JudgeError> {
    let p = dir.join(name);
    if p.exists() {
        Ok(p)
    } else {
        Err(JudgeError::Unavailable {
            message: format!("checkpoint {}: missing {name}", dir.display()),
        })
    }
}

pub(crate) fn parse_judge_meta(v: &serde_json::Value) -> Result<JudgeMeta, JudgeError> {
    let ver = v
        .get("judge_state_version")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u32;
    if ver != JUDGE_STATE_VERSION {
        return Err(JudgeError::Unavailable {
            message: format!(
                "checkpoint was calibrated for judge state v{ver}, this binary renders v{JUDGE_STATE_VERSION}"
            ),
        });
    }
    let select_threshold = v
        .get("select_threshold")
        .and_then(|x| x.as_u64())
        .unwrap_or(50) as u32;
    Ok(JudgeMeta {
        judge_state_version: ver,
        select_threshold,
    })
}

pub(crate) fn parse_agent_config(v: &serde_json::Value) -> Result<AgentConfig, JudgeError> {
    let usize_or = |key: &str, def: usize| -> usize {
        v.get(key)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(def)
    };
    let max_len = usize_or("max_len", 512);
    let head_max_len = usize_or("head_max_len", 192);
    let head_layers = usize_or("head_layers", 2);

    let temperature = match v.get("temperature").and_then(|x| x.as_array()) {
        Some(arr) if arr.len() == 3 => {
            let mut t = [1.0f32; 3];
            for (i, e) in arr.iter().enumerate() {
                t[i] = e.as_f64().map(|x| x as f32).unwrap_or(f32::NAN);
            }
            t
        }
        _ => [1.0, 1.0, 1.0],
    };

    let mut temperature_by_options = BTreeMap::new();
    if let Some(map) = v.get("temperature_by_options").and_then(|x| x.as_object()) {
        for (k, val) in map {
            if let Some(f) = val.as_f64() {
                temperature_by_options.insert(k.clone(), f as f32);
            }
        }
    }

    if max_len <= head_max_len + 16 {
        return Err(JudgeError::Unavailable {
            message: format!(
                "rl_agent_config: max_len ({max_len}) must be > head_max_len + 16 ({})",
                head_max_len + 16
            ),
        });
    }

    Ok(AgentConfig {
        max_len,
        head_max_len,
        head_layers,
        temperature,
        temperature_by_options,
    })
}

fn missing2(target: &str, source: &str) -> JudgeError {
    JudgeError::Unavailable {
        message: format!("encoder/config.json: missing `{target}` and its source `{source}`"),
    }
}

pub(crate) fn fixup_encoder_config(
    mut v: serde_json::Value,
) -> Result<ModernBertConfig, JudgeError> {
    let rope_theta = |v: &serde_json::Value, branch: &str| -> Option<serde_json::Value> {
        v.get("rope_parameters")
            .and_then(|r| r.get(branch))
            .and_then(|f| f.get("rope_theta"))
            .cloned()
    };
    let layer_types = v.get("layer_types").and_then(|l| l.as_array()).cloned();

    let obj = v.as_object_mut().ok_or_else(|| JudgeError::Unavailable {
        message: "encoder/config.json is not a JSON object".to_string(),
    })?;

    if !obj.contains_key("layer_norm_eps") {
        match obj.get("norm_eps").cloned() {
            Some(x) => {
                obj.insert("layer_norm_eps".into(), x);
            }
            None => return Err(missing2("layer_norm_eps", "norm_eps")),
        }
    }
    if !obj.contains_key("global_rope_theta") {
        match rope_theta(&serde_json::Value::Object(obj.clone()), "full_attention") {
            Some(x) => {
                obj.insert("global_rope_theta".into(), x);
            }
            None => {
                return Err(missing2(
                    "global_rope_theta",
                    "rope_parameters.full_attention.rope_theta",
                ));
            }
        }
    }
    if !obj.contains_key("local_rope_theta") {
        match rope_theta(&serde_json::Value::Object(obj.clone()), "sliding_attention") {
            Some(x) => {
                obj.insert("local_rope_theta".into(), x);
            }
            None => {
                return Err(missing2(
                    "local_rope_theta",
                    "rope_parameters.sliding_attention.rope_theta",
                ));
            }
        }
    }
    if !obj.contains_key("global_attn_every_n_layers") {
        let idx = layer_types.as_ref().and_then(|arr| {
            arr.iter()
                .enumerate()
                .skip(1)
                .find(|(_, t)| t.as_str() == Some("full_attention"))
                .map(|(i, _)| i)
        });
        match idx {
            Some(i) => {
                obj.insert("global_attn_every_n_layers".into(), serde_json::json!(i));
            }
            None => return Err(missing2("global_attn_every_n_layers", "layer_types")),
        }
    }

    // Consistency check when layer_types is present.
    if let Some(arr) = &layer_types {
        let every = obj
            .get("global_attn_every_n_layers")
            .and_then(|x| x.as_u64())
            .unwrap_or(1)
            .max(1) as usize;
        for (i, t) in arr.iter().enumerate() {
            let is_full = t.as_str() == Some("full_attention");
            let should = i % every == 0;
            if is_full != should {
                return Err(JudgeError::Unavailable {
                    message: format!(
                        "encoder/config.json: layer_types[{i}] inconsistent with global_attn_every_n_layers = {every}"
                    ),
                });
            }
        }
    }

    serde_json::from_value(v).map_err(|e| JudgeError::Unavailable {
        message: format!("encoder/config.json is not a valid ModernBert config: {e}"),
    })
}

pub(crate) fn load_tokenizer(path: &Path) -> Result<(Tokenizer, SpecialIds), JudgeError> {
    let mut tok = Tokenizer::from_file(path).map_err(|e| JudgeError::Unavailable {
        message: format!("loading tokenizer failed: {e}"),
    })?;
    tok.with_truncation(None)
        .map_err(|e| JudgeError::Unavailable {
            message: format!("tokenizer truncation reset failed: {e}"),
        })?;
    tok.with_padding(None);
    let id = |t: &str| -> Result<u32, JudgeError> {
        tok.token_to_id(t).ok_or_else(|| JudgeError::Unavailable {
            message: format!("tokenizer missing special token {t}"),
        })
    };
    let sp = SpecialIds {
        cls: id("[CLS]")?,
        sep: id("[SEP]")?,
        mask: id("[MASK]")?,
        pad: id("[PAD]")?,
    };
    Ok((tok, sp))
}

pub(crate) fn resolve_device(device: JudgeDevice) -> Result<Device, JudgeError> {
    match device {
        JudgeDevice::Cpu => Ok(Device::Cpu),
        JudgeDevice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|e| JudgeError::Unavailable {
                    message: format!("loading checkpoint failed: {e}"),
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(JudgeError::Unavailable {
                    message: "judge.device = \"cuda\" requires a build with the cuda-judge feature"
                        .to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("judge_ckpt_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn required_file_names_the_missing_one() {
        let d = td("missing");
        let err = required_file(&d, "model.safetensors").unwrap_err();
        match err {
            JudgeError::Unavailable { message } => {
                assert!(message.contains("model.safetensors"), "{message}");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn version_mismatch_message() {
        let v = serde_json::json!({ "judge_state_version": 2, "select_threshold": 40 });
        let err = parse_judge_meta(&v).unwrap_err();
        match err {
            JudgeError::Unavailable { message } => {
                assert!(message.contains("v2"), "{message}");
                assert!(
                    message.contains(&format!("v{JUDGE_STATE_VERSION}")),
                    "{message}"
                );
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn agent_config_defaults_and_max_len_rule() {
        let cfg = parse_agent_config(&serde_json::json!({})).unwrap();
        assert_eq!(cfg.max_len, 512);
        assert_eq!(cfg.head_max_len, 192);
        assert_eq!(cfg.head_layers, 2);
        assert_eq!(cfg.temperature, [1.0, 1.0, 1.0]);
        assert!(cfg.temperature_by_options.is_empty());

        let bad = serde_json::json!({ "max_len": 200, "head_max_len": 192 });
        assert!(parse_agent_config(&bad).is_err());
    }

    #[test]
    fn transformers5_config_fixup() {
        let v = serde_json::json!({
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 100,
            "max_position_embeddings": 1024,
            "pad_token_id": 0,
            "norm_eps": 1e-5,
            "global_attn_every_n_layers": 2,
            "local_attention": 128,
            "rope_parameters": {
                "full_attention": { "rope_theta": 160000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            },
            "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
        });
        // Should deserialize without error (exact field set validated by candle).
        let _cfg = fixup_encoder_config(v).unwrap();
    }

    #[test]
    fn hub_style_config_passes_through() {
        let v = serde_json::json!({
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 100,
            "max_position_embeddings": 1024,
            "pad_token_id": 0,
            "layer_norm_eps": 1e-5,
            "global_rope_theta": 160000.0,
            "local_rope_theta": 10000.0,
            "global_attn_every_n_layers": 2,
            "local_attention": 128
        });
        let _cfg = fixup_encoder_config(v).unwrap();
    }

    #[test]
    fn missing_source_key_names_both() {
        let v = serde_json::json!({
            "hidden_size": 128,
            "global_rope_theta": 1.0,
            "local_rope_theta": 1.0,
            "global_attn_every_n_layers": 2
            // no layer_norm_eps AND no norm_eps
        });
        let err = fixup_encoder_config(v).unwrap_err();
        match err {
            JudgeError::Unavailable { message } => {
                assert!(message.contains("layer_norm_eps"), "{message}");
                assert!(message.contains("norm_eps"), "{message}");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn inconsistent_layer_types_names_index() {
        let v = serde_json::json!({
            "hidden_size": 128,
            "norm_eps": 1e-5,
            "global_attn_every_n_layers": 2,
            "local_attention": 128,
            "rope_parameters": {
                "full_attention": { "rope_theta": 1.0 },
                "sliding_attention": { "rope_theta": 1.0 }
            },
            // index 1 must be sliding for every_n=2, but is full -> error at 1
            "layer_types": ["full_attention", "full_attention", "full_attention", "sliding_attention"]
        });
        let err = fixup_encoder_config(v).unwrap_err();
        match err {
            JudgeError::Unavailable { message } => assert!(message.contains("[1]"), "{message}"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn tokenizer_truncation_is_reset_on_load() {
        // A tokenizer.json with a stored truncation of max_length 4.
        let dir = td("tok");
        let vocab = r#""[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"a":5,"b":6"#;
        let json = format!(
            r#"{{"version":"1.0",
"truncation":{{"max_length":4,"strategy":"LongestFirst","stride":0,"direction":"Right"}},
"padding":null,"added_tokens":[],"normalizer":null,
"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,
"model":{{"type":"WordLevel","vocab":{{{vocab}}},"unk_token":"[UNK]"}}}}"#
        );
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, json).unwrap();
        let (tok, sp) = load_tokenizer(&path).unwrap();
        assert_eq!(sp.cls, 2);
        let ids = tok
            .encode("a b a b a b a b a b", false)
            .unwrap()
            .get_ids()
            .len();
        assert_eq!(ids, 10, "truncation was not reset");
    }
}
