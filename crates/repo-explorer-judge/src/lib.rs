//! The local candidate judge over HTTP: the model-neutral `/v1/systemone`
//! client (any server speaking that wire schema, e.g. the H2O shim). Owns
//! `reqwest` for the judge and never proxies (the judge is a local/LAN
//! service).

use futures_util::future::try_join_all;
use repo_explorer_core::config::{JudgeMode, JudgeSettings};
use repo_explorer_core::judge::{
    self, CandidateJudge, JudgeError, Judgement, NoJudge, POSITIVE_KEY,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

/// A judge backed by any server speaking `POST /v1/systemone`.
#[derive(Debug)]
pub struct SystemOneHttpJudge {
    client: reqwest::Client,
    endpoint: String,
    models_url: String,
    api_key: Option<String>,
    model: String,
    timeout: Duration,
    semaphore: Arc<Semaphore>,
}

impl SystemOneHttpJudge {
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
            models_url: format!("{base}/v1/models"),
            api_key,
            model: settings.model.clone(),
            timeout: Duration::from_millis(settings.timeout_ms),
            semaphore: Arc::new(Semaphore::new(settings.max_concurrency.max(1) as usize)),
        })
    }

    /// Attach the bearer header when a key is configured (never otherwise).
    fn authorized(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => req.header("authorization", format!("Bearer {key}")),
            None => req,
        }
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
        let req = self.authorized(
            self.client
                .post(&self.endpoint)
                .header("content-type", "application/json")
                .json(&body),
        );
        let resp = req.send().await.map_err(|e| JudgeError::Unavailable {
            message: format!("judge request failed: {}", sanitize(&e.to_string())),
        })?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(JudgeError::Unavailable {
                message: format!("judge rejected the bearer token (HTTP {})", status.as_u16()),
            });
        }
        if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
            // Model still loading or overloaded: a readiness problem, not a
            // protocol violation.
            let text = resp.text().await.unwrap_or_default();
            let head: String = text.chars().take(200).collect();
            return Err(JudgeError::Unavailable {
                message: format!("HTTP 503: {head}"),
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
        // Optional server timing; missing or garbled is `None`, never an error.
        let latency_ms = v
            .get("usage")
            .and_then(|u| u.get("latency_ms"))
            .and_then(|x| x.as_f64())
            .filter(|x| x.is_finite() && *x >= 0.0)
            .map(|x| x.round() as u64);
        Ok(Judgement {
            p_relevant_permille: ((p * 1000.0).round() as u32).min(1000),
            latency_ms,
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

impl CandidateJudge for SystemOneHttpJudge {
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

    /// `GET /v1/models` (server up?), then one 1-state probe request, which
    /// exercises the model name and warms the server's prefix cache.
    async fn warm_up(&self) -> Result<(), JudgeError> {
        let resp = self
            .authorized(
                self.client
                    .get(&self.models_url)
                    .timeout(Duration::from_secs(2)),
            )
            .send()
            .await
            .map_err(|e| JudgeError::Unavailable {
                message: format!("judge models probe failed: {}", sanitize(&e.to_string())),
            })?;
        if !resp.status().is_success() {
            return Err(JudgeError::Unavailable {
                message: format!(
                    "judge models probe returned HTTP {}",
                    resp.status().as_u16()
                ),
            });
        }
        self.judge(&["query: warm-up probe".to_string()])
            .await
            .map(|_| ())
    }
}

/// Runtime choice of judge without `dyn`: the crate convention is static
/// dispatch.
#[derive(Debug)]
pub enum ConfiguredJudge {
    Disabled(NoJudge),
    Http(SystemOneHttpJudge),
}

impl ConfiguredJudge {
    pub fn from_settings(settings: &JudgeSettings) -> Result<Self, JudgeError> {
        match settings.mode {
            JudgeMode::Off => Ok(ConfiguredJudge::Disabled(NoJudge)),
            JudgeMode::On | JudgeMode::Shadow => {
                Ok(ConfiguredJudge::Http(SystemOneHttpJudge::new(settings)?))
            }
        }
    }
}

impl CandidateJudge for ConfiguredJudge {
    async fn judge(&self, states: &[String]) -> Result<Vec<Judgement>, JudgeError> {
        match self {
            ConfiguredJudge::Disabled(j) => j.judge(states).await,
            ConfiguredJudge::Http(j) => j.judge(states).await,
        }
    }

    async fn warm_up(&self) -> Result<(), JudgeError> {
        match self {
            ConfiguredJudge::Disabled(j) => j.warm_up().await,
            ConfiguredJudge::Http(j) => j.warm_up().await,
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
