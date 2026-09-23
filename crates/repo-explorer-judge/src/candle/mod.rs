//! In-process candle judge backend (feature `candle`). A faithful port of
//! Laya 0.3.7 inference for the one fixed judge question.

pub mod calib;
pub(crate) mod checkpoint;
pub(crate) mod head;
mod sequence;

pub use sequence::{Encoded, SpecialIds, build_sequence};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::ModernBert;
use head::DecisionHead;
use repo_explorer_core::config::JudgeDevice;
use repo_explorer_core::judge::{self, JudgeError, Judgement};
use std::collections::HashMap;
use std::path::Path;

/// A fully loaded checkpoint: tokenizer, encoder, head and calibration.
pub struct LoadedModel {
    tokenizer: tokenizers::Tokenizer,
    special: SpecialIds,
    agent: checkpoint::AgentConfig,
    encoder: ModernBert,
    head: DecisionHead,
    device: Device,
    temperature: f32,
    select_threshold: u32,
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("temperature", &self.temperature)
            .field("select_threshold", &self.select_threshold)
            .finish_non_exhaustive()
    }
}

fn cerr(e: candle_core::Error) -> JudgeError {
    JudgeError::Unavailable {
        message: format!("loading checkpoint failed: {e}"),
    }
}

impl LoadedModel {
    /// Blocking. Full D4 load including the state-version check.
    pub fn load(dir: &Path, device: JudgeDevice) -> Result<LoadedModel, JudgeError> {
        let agent_p = checkpoint::required_file(dir, "rl_agent_config.json")?;
        let weights_p = checkpoint::required_file(dir, "model.safetensors")?;
        let tok_p = checkpoint::required_file(dir, "tokenizer/tokenizer.json")?;
        let enc_cfg_p = checkpoint::required_file(dir, "encoder/config.json")?;
        let meta_p = checkpoint::required_file(dir, "repo_explorer_judge.json")?;

        let read_json = |p: &Path| -> Result<serde_json::Value, JudgeError> {
            let s = std::fs::read_to_string(p).map_err(|e| JudgeError::Unavailable {
                message: format!("checkpoint {}: {e}", p.display()),
            })?;
            serde_json::from_str(&s).map_err(|e| JudgeError::Unavailable {
                message: format!("checkpoint {}: invalid JSON: {e}", p.display()),
            })
        };

        let meta = checkpoint::parse_judge_meta(&read_json(&meta_p)?)?;
        let agent = checkpoint::parse_agent_config(&read_json(&agent_p)?)?;
        let enc_cfg = checkpoint::fixup_encoder_config(read_json(&enc_cfg_p)?)?;
        let (tokenizer, special) = checkpoint::load_tokenizer(&tok_p)?;
        let device = checkpoint::resolve_device(device)?;

        let mut tensors = candle_core::safetensors::load(&weights_p, &device).map_err(cerr)?;
        let renamed: HashMap<String, Tensor> = tensors
            .drain()
            .map(|(k, v)| {
                let k = match k.strip_prefix("encoder.") {
                    Some(rest) => format!("model.{rest}"),
                    None => k,
                };
                (k, v)
            })
            .collect();
        let vb = VarBuilder::from_tensors(renamed, DType::F32, &device);
        let encoder = ModernBert::load(vb.clone(), &enc_cfg).map_err(cerr)?;
        let hidden = enc_cfg.hidden_size;
        let head = DecisionHead::load(&vb, hidden, agent.head_layers).map_err(cerr)?;
        let temperature = calib::select_temperature(
            agent.temperature_by_options.get("choice:2").copied(),
            agent.temperature[0],
        );

        Ok(LoadedModel {
            tokenizer,
            special,
            agent,
            encoder,
            head,
            device,
            temperature,
            select_threshold: meta.select_threshold,
        })
    }

    /// Build the judge sequence for one state (D5 with the fixed question).
    pub fn encode(&self, state: &str) -> Result<Encoded, JudgeError> {
        let options = [
            format!("{}: {}", judge::POSITIVE_KEY, judge::POSITIVE_CRITERION),
            format!("{}: {}", judge::NEGATIVE_KEY, judge::NEGATIVE_CRITERION),
        ];
        build_sequence(
            &self.tokenizer,
            &self.special,
            state,
            judge::QUESTION_INSTRUCTIONS,
            &options,
            self.agent.max_len,
            self.agent.head_max_len,
        )
    }

    /// D6 steps 1-6: pre-temperature logits per sequence, in input order.
    /// Chunks of at most 4 sequences per forward pass.
    pub fn raw_logits(&self, batch: &[Encoded]) -> Result<Vec<[f32; 2]>, JudgeError> {
        let mut out = Vec::with_capacity(batch.len());
        for chunk in batch.chunks(4) {
            let l = chunk.iter().map(|e| e.ids.len()).max().unwrap_or(0);
            let b = chunk.len();
            let mut ids = Vec::with_capacity(b * l);
            let mut mask = Vec::with_capacity(b * l);
            let mut markers = Vec::with_capacity(b * 2);
            for e in chunk {
                for &id in &e.ids {
                    ids.push(id);
                    mask.push(1u32);
                }
                for _ in e.ids.len()..l {
                    ids.push(self.special.pad);
                    mask.push(0u32);
                }
                markers.push(e.markers[0] as u32);
                markers.push(e.markers[1] as u32);
            }
            let ids_t = Tensor::from_vec(ids, (b, l), &self.device).map_err(cerr)?;
            let mask_t = Tensor::from_vec(mask, (b, l), &self.device).map_err(cerr)?;
            let marker_t = Tensor::from_vec(markers, (b, 2), &self.device).map_err(cerr)?;
            let h = self.encoder.forward(&ids_t, &mask_t).map_err(cerr)?;
            let h = h.to_dtype(DType::F32).map_err(cerr)?;
            let logits = self.head.forward(&h, &mask_t, &marker_t).map_err(cerr)?;
            let rows = logits.to_vec2::<f32>().map_err(cerr)?;
            for row in rows {
                out.push([row[0], row[1]]);
            }
        }
        Ok(out)
    }

    /// D6 steps 7-8: encode each state, batch the forward pass, apply the
    /// clamped temperature and produce a `Judgement` per state, in order.
    pub fn judge_states(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError> {
        if states.is_empty() {
            return Ok(vec![]);
        }
        let encoded: Vec<Encoded> = states
            .iter()
            .map(|s| self.encode(s))
            .collect::<Result<_, _>>()?;
        let raw = self.raw_logits(&encoded)?;
        Ok(raw
            .iter()
            .map(|&logits| {
                let p = calib::softmax2(logits, self.temperature)[0];
                Judgement {
                    p_relevant_permille: calib::permille(p),
                }
            })
            .collect())
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn select_threshold(&self) -> u32 {
        self.select_threshold
    }
}

#[cfg(test)]
mod load_tests {
    use super::LoadedModel;
    use repo_explorer_core::config::JudgeDevice;
    use repo_explorer_core::judge::JudgeError;
    use std::path::Path;

    #[test]
    fn load_names_the_first_missing_file() {
        let dir = std::env::temp_dir().join(format!("judge_load_missing_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let err = LoadedModel::load(&dir, JudgeDevice::Cpu).unwrap_err();
        match err {
            JudgeError::Unavailable { message } => {
                assert!(message.contains("missing"), "{message}")
            }
            _ => panic!("wrong variant: {err:?}"),
        }
    }

    #[test]
    fn cuda_without_feature_is_unavailable() {
        // resolve_device is only reached after files exist; test it via a dir
        // whose files are all present but weights are absent is complex, so we
        // assert the device resolver directly.
        let err = super::checkpoint::resolve_device(JudgeDevice::Cuda).unwrap_err();
        match err {
            JudgeError::Unavailable { message } => {
                assert!(
                    message.contains("cuda-judge") || message.contains("cuda"),
                    "{message}"
                );
            }
            _ => panic!("wrong variant"),
        }
    }

    // Sanity: the path type is what callers pass.
    fn _typecheck(p: &Path) {
        let _ = LoadedModel::load(p, JudgeDevice::Cpu);
    }
}
