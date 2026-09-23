//! The local candidate judge over HTTP: the only crate that knows the upstream
//! `laya-serve` wire format (`POST /v1/systemone`). Owns `reqwest` for the
//! judge and never proxies (the judge is a local/LAN service).

#[cfg(feature = "candle")]
pub mod candle;

use futures_util::future::try_join_all;
use repo_explorer_core::config::{JudgeMode, JudgeSettings};
use repo_explorer_core::judge::{
    self, CandidateJudge, JudgeError, Judgement, NoJudge, POSITIVE_KEY,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// A judge backed by an upstream `laya-serve` instance.
#[derive(Debug)]
pub struct LayaHttpJudge {
    client: reqwest::Client,
    endpoint: String,
    health_url: String,
    api_key: Option<String>,
    model: String,
    timeout: Duration,
    semaphore: Arc<Semaphore>,
}

impl LayaHttpJudge {
    /// `new_with_env(settings, |v| std::env::var(v).ok())`.
    pub fn new(settings: &JudgeSettings) -> Result<Self, JudgeError> {
        Self::new_with_env(settings, |v| std::env::var(v).ok())
    }

    /// Reads the bearer token from `api_key_env` (if set) once, here, via the
    /// injected accessor (the `Config::validate_with_env` convention). A
    /// set-but-blank or unset variable → `Unavailable`.
    pub fn new_with_env(
        settings: &JudgeSettings,
        get_env: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, JudgeError> {
        let api_key = match &settings.api_key_env {
            Some(var) => {
                let val = get_env(var).filter(|v| !v.trim().is_empty());
                match val {
                    Some(v) => Some(v),
                    None => {
                        return Err(JudgeError::Unavailable {
                            message: format!("judge.api_key_env `{var}` is not set"),
                        });
                    }
                }
            }
            None => None,
        };
        // The judge is a local/LAN service and is never proxied.
        let client =
            reqwest::Client::builder()
                .no_proxy()
                .build()
                .map_err(|e| JudgeError::Unavailable {
                    message: format!("failed to build judge HTTP client: {e}"),
                })?;
        let base = settings.base_url.trim_end_matches('/');
        Ok(Self {
            client,
            endpoint: format!("{base}/v1/systemone"),
            health_url: format!("{base}/health"),
            api_key,
            model: settings.model.clone(),
            timeout: Duration::from_millis(settings.timeout_ms),
            semaphore: Arc::new(Semaphore::new(settings.max_concurrency.max(1) as usize)),
        })
    }

    async fn judge_one(&self, state: &str) -> Result<Judgement, JudgeError> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| JudgeError::Unavailable {
                message: format!("judge semaphore closed: {e}"),
            })?;
        let body = serde_json::json!({
            "model": self.model,
            "state": state,
            "questions": {
                judge::QUESTION_ID: {
                    "type": judge::QUESTION_TYPE,
                    "instructions": judge::QUESTION_INSTRUCTIONS,
                    "criteria": {
                        judge::POSITIVE_KEY: judge::POSITIVE_CRITERION,
                        judge::NEGATIVE_KEY: judge::NEGATIVE_CRITERION,
                    }
                }
            }
        });
        let mut req = self
            .client
            .post(&self.endpoint)
            .header("content-type", "application/json")
            .json(&body);
        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        let resp = req.send().await.map_err(|e| JudgeError::Unavailable {
            message: format!("judge request failed: {}", sanitize(&e.to_string())),
        })?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(JudgeError::Unavailable {
                message: format!("judge rejected the bearer token (HTTP {})", status.as_u16()),
            });
        }
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            let head: String = text.chars().take(200).collect();
            return Err(JudgeError::Protocol {
                message: format!("HTTP {}: {head}", status.as_u16()),
            });
        }
        let text = resp.text().await.map_err(|e| JudgeError::Protocol {
            message: format!("failed to read judge body: {e}"),
        })?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| JudgeError::Protocol {
                message: format!("judge body is not JSON: {e}"),
            })?;
        let p = v
            .get("answers")
            .and_then(|a| a.get(judge::QUESTION_ID))
            .and_then(|q| q.get("probabilities"))
            .and_then(|p| p.get(POSITIVE_KEY))
            .and_then(|p| p.as_f64())
            .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
            .ok_or_else(|| JudgeError::Protocol {
                message: "judge answer missing a finite answers.relevant.probabilities.A in [0,1]"
                    .to_string(),
            })?;
        Ok(Judgement {
            p_relevant_permille: ((p * 1000.0).round() as u32).min(1000),
        })
    }
}

/// Never let a transport error string leak an authorization header.
fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(idx) = rest.find("Bearer ") {
        out.push_str(&rest[..idx]);
        out.push_str("Bearer <redacted>");
        rest = &rest["Bearer ".len() + idx..];
        let token_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        rest = &rest[token_end..];
    }
    out.push_str(rest);
    out
}

impl CandidateJudge for LayaHttpJudge {
    async fn judge(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError> {
        if states.is_empty() {
            return Ok(vec![]);
        }
        let all = try_join_all(states.iter().map(|s| self.judge_one(s)));
        let out = tokio::time::timeout(self.timeout, all)
            .await
            .map_err(|_| JudgeError::Timeout {
                timeout_ms: self.timeout.as_millis() as u64,
            })??;
        if out.len() != states.len() {
            return Err(JudgeError::Protocol {
                message: format!(
                    "judge returned {} results for {} states",
                    out.len(),
                    states.len()
                ),
            });
        }
        Ok(out)
    }

    async fn warm_up(&self) -> Result<(), JudgeError> {
        let resp = self
            .client
            .get(&self.health_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map_err(|e| JudgeError::Unavailable {
                message: format!("judge health probe failed: {}", sanitize(&e.to_string())),
            })?;
        if resp.status().is_success() {
            Ok(())
        } else {
            Err(JudgeError::Unavailable {
                message: format!("judge health returned HTTP {}", resp.status().as_u16()),
            })
        }
    }
}

/// Runtime choice of judge without `dyn`: the crate convention is static
/// dispatch.
#[derive(Debug)]
pub enum ConfiguredJudge {
    Disabled(NoJudge),
    Laya(LayaHttpJudge),
    #[cfg(feature = "candle")]
    Candle(candle::LayaCandleJudge),
}

impl ConfiguredJudge {
    pub fn from_settings(settings: &JudgeSettings) -> Result<Self, JudgeError> {
        use repo_explorer_core::config::JudgeBackend;
        match settings.mode {
            JudgeMode::Off => Ok(ConfiguredJudge::Disabled(NoJudge)),
            JudgeMode::Laya | JudgeMode::Shadow => match settings.backend {
                JudgeBackend::Http => Ok(ConfiguredJudge::Laya(LayaHttpJudge::new(settings)?)),
                JudgeBackend::Candle => {
                    #[cfg(feature = "candle")]
                    {
                        Ok(ConfiguredJudge::Candle(candle::LayaCandleJudge::new(
                            settings,
                        )?))
                    }
                    #[cfg(not(feature = "candle"))]
                    {
                        Err(JudgeError::Unavailable {
                            message: "this build has no candle judge backend (built without the candle-judge feature)".to_string(),
                        })
                    }
                }
            },
        }
    }
}

impl CandidateJudge for ConfiguredJudge {
    async fn judge(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError> {
        match self {
            ConfiguredJudge::Disabled(j) => j.judge(states).await,
            ConfiguredJudge::Laya(j) => j.judge(states).await,
            #[cfg(feature = "candle")]
            ConfiguredJudge::Candle(j) => j.judge(states).await,
        }
    }

    async fn warm_up(&self) -> Result<(), JudgeError> {
        match self {
            ConfiguredJudge::Disabled(j) => j.warm_up().await,
            ConfiguredJudge::Laya(j) => j.warm_up().await,
            #[cfg(feature = "candle")]
            ConfiguredJudge::Candle(j) => j.warm_up().await,
        }
    }
}

#[cfg(all(test, feature = "candle"))]
mod configured_candle_tests {
    use super::ConfiguredJudge;
    use repo_explorer_core::config::{JudgeBackend, JudgeMode, JudgeSettings};

    #[test]
    fn candle_backend_selects_candle_variant() {
        let s = JudgeSettings {
            mode: JudgeMode::Laya,
            backend: JudgeBackend::Candle,
            checkpoint_dir: "/nonexistent/ckpt".to_string(),
            ..JudgeSettings::default()
        };
        let j = ConfiguredJudge::from_settings(&s).unwrap();
        assert!(matches!(j, ConfiguredJudge::Candle(_)));
    }
}

#[cfg(all(test, not(feature = "candle")))]
mod configured_no_candle_tests {
    use super::ConfiguredJudge;
    use repo_explorer_core::config::{JudgeBackend, JudgeMode, JudgeSettings};
    use repo_explorer_core::judge::JudgeError;

    #[test]
    fn candle_backend_without_feature_is_unavailable() {
        let s = JudgeSettings {
            mode: JudgeMode::Laya,
            backend: JudgeBackend::Candle,
            checkpoint_dir: "/x".to_string(),
            ..JudgeSettings::default()
        };
        let err = ConfiguredJudge::from_settings(&s).unwrap_err();
        match err {
            JudgeError::Unavailable { message } => {
                assert!(message.contains("candle-judge"), "{message}")
            }
            _ => panic!("wrong variant"),
        }
    }
}

#[cfg(test)]
mod sanitize_tests {
    use super::sanitize;

    #[test]
    fn strips_the_token_not_just_marks_it() {
        let s = sanitize("request failed: Bearer sk-live-abc123 rejected by proxy");
        assert!(!s.contains("sk-live-abc123"), "token leaked: {s}");
        assert_eq!(s, "request failed: Bearer <redacted> rejected by proxy");
    }
}
