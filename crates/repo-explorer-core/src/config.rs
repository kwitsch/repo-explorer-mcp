//! Strongly-typed configuration model loaded and validated from TOML.
//!
//! The single public entry point is [`load`], which reads a file, parses it as
//! TOML into a [`Config`], and runs [`Config::validate`]. Validation is also
//! public so it can be unit-tested on hand-built `Config` values.

use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Top-level configuration.
///
/// `#[serde(deny_unknown_fields)]` is intentionally NOT used so later stages can
/// add fields without breaking existing configs.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub codebase_memory: CodebaseMemoryConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlmConfig {
    /// Order in the file = failover order. Preserved by using `Vec`.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    /// Open string (anthropic | openai | google | …) — kept extensible.
    pub kind: String,
    /// Name of the env var holding the API key (never the key itself).
    /// When omitted, derived from `kind` via `default_api_key_env`.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Ordered model list. First model = first tried; on a usage-limit error
    /// the router advances to the next model, then to the next provider entry.
    pub models: Vec<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

/// Default API-key env var name for a known provider kind, mirroring the
/// `genai` crate's own adapter defaults. `None` for unrecognized kinds.
/// Keyed on the same `kind` strings as `adapter_kind_for` in repo-explorer-llm.
pub fn default_api_key_env(kind: &str) -> Option<&'static str> {
    match kind {
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "openai" => Some("OPENAI_API_KEY"),
        "gemini" | "google" => Some("GEMINI_API_KEY"),
        _ => None,
    }
}

impl ProviderConfig {
    /// Effective API-key env var name: the explicit `api_key_env` when set,
    /// otherwise the default derived from `kind`. `None` when neither is
    /// available (unknown kind and no explicit override).
    pub fn resolve_api_key_env(&self) -> Option<String> {
        self.api_key_env
            .clone()
            .or_else(|| default_api_key_env(&self.kind).map(str::to_owned))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodebaseMemoryConfig {
    /// Stdio transport: process command to launch.
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Network transport: endpoint URL. Mutually exclusive with `command`.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Index considered stale after this many seconds (consumed by Stage 2).
    #[serde(default = "default_staleness_seconds")]
    pub staleness_seconds: u64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchConfig {
    /// Explicit path to the `rtk` binary; `None` → auto-detect in Stage 3.
    #[serde(default)]
    pub rtk_path: Option<PathBuf>,
    /// Explicit path to the `ripgrep` binary; `None` → auto-detect in Stage 3.
    #[serde(default)]
    pub ripgrep_path: Option<PathBuf>,
    #[serde(default = "default_search_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default = "default_prefer_rtk")]
    pub prefer_rtk: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoggingConfig {
    #[serde(default)]
    pub level: LogLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

fn default_cooldown_seconds() -> u64 {
    60
}

fn default_staleness_seconds() -> u64 {
    3600
}

fn default_search_timeout_seconds() -> u64 {
    30
}

fn default_prefer_rtk() -> bool {
    true
}

/// Errors returned by [`load`].
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config as TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error(transparent)]
    Validation(#[from] ValidationError),
}

/// Semantic validation failures, independent of parsing.
///
/// Messages name the offending provider/variable/section but never a secret value.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("llm.providers must contain at least one provider")]
    EmptyProviderList,
    #[error("duplicate provider name `{name}` in llm.providers")]
    DuplicateProviderName { name: String },
    #[error("provider `{provider}` references environment variable `{var}`, which is not set")]
    MissingEnvVar { provider: String, var: String },
    #[error(
        "codebase_memory must set either `command` (stdio) or `endpoint` (network), but neither is present"
    )]
    MissingCodebaseMemoryConnection,
    #[error("codebase_memory sets both `command` and `endpoint`; exactly one is allowed")]
    ConflictingCodebaseMemoryConnection,
    #[error("provider `{provider}` must list at least one model in `models`")]
    EmptyModelList { provider: String },
    #[error(
        "provider `{provider}`: cannot derive an API key environment variable \
         for unknown kind `{kind}`; set `api_key_env` explicitly"
    )]
    UnknownProviderKind { provider: String, kind: String },
}

/// Read, parse, and validate a config file. The single public entry point.
pub fn load(path: &Path) -> Result<Config, ConfigError> {
    let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let config: Config = toml::from_str(&contents)?;
    config.validate()?;
    Ok(config)
}

impl Config {
    /// Semantic validation, independent of parsing. Public so it is unit-testable
    /// on hand-built `Config` values.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.llm.providers.is_empty() {
            return Err(ValidationError::EmptyProviderList);
        }

        let mut seen = std::collections::HashSet::new();
        for provider in &self.llm.providers {
            if !seen.insert(provider.name.as_str()) {
                return Err(ValidationError::DuplicateProviderName {
                    name: provider.name.clone(),
                });
            }
        }

        for provider in &self.llm.providers {
            if provider.models.is_empty() {
                return Err(ValidationError::EmptyModelList {
                    provider: provider.name.clone(),
                });
            }
            let var = provider.resolve_api_key_env().ok_or_else(|| {
                ValidationError::UnknownProviderKind {
                    provider: provider.name.clone(),
                    kind: provider.kind.clone(),
                }
            })?;
            if std::env::var(&var).is_err() {
                return Err(ValidationError::MissingEnvVar {
                    provider: provider.name.clone(),
                    var,
                });
            }
        }

        match (
            self.codebase_memory.command.is_some(),
            self.codebase_memory.endpoint.is_some(),
        ) {
            (false, false) => return Err(ValidationError::MissingCodebaseMemoryConnection),
            (true, true) => return Err(ValidationError::ConflictingCodebaseMemoryConnection),
            _ => {}
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fixture_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    /// Build a minimal valid `Config` in memory with a single provider whose
    /// `api_key_env` names `env_var`, and a stdio codebase_memory connection.
    fn config_with_provider(name: &str, env_var: &str) -> Config {
        Config {
            llm: LlmConfig {
                providers: vec![ProviderConfig {
                    name: name.to_string(),
                    kind: "anthropic".to_string(),
                    api_key_env: Some(env_var.to_string()),
                    models: vec!["m".to_string()],
                    base_url: None,
                }],
                cooldown_seconds: 60,
            },
            codebase_memory: CodebaseMemoryConfig {
                command: Some("cmd".to_string()),
                args: vec![],
                endpoint: None,
                staleness_seconds: 3600,
            },
            search: SearchConfig::default(),
            logging: LoggingConfig::default(),
        }
    }

    #[test]
    fn valid_config_loads() {
        let var = "REPO_EXPLORER_TEST_KEY_VALID";
        // Edition 2024: env mutation is unsafe.
        unsafe {
            std::env::set_var(var, "not-a-real-key");
        }

        let config = load(&fixture_path("valid.toml")).expect("valid config should load");

        // Failover order is file order.
        assert_eq!(config.llm.providers.len(), 2);
        assert_eq!(config.llm.providers[0].name, "primary");
        assert_eq!(config.llm.providers[0].kind, "anthropic");
        assert_eq!(config.llm.providers[1].name, "secondary");
        assert_eq!(
            config.llm.providers[0].models,
            vec!["claude-sonnet-4".to_string(), "claude-haiku-4".to_string()]
        );
        assert_eq!(config.llm.providers[1].models, vec!["gpt-4o".to_string()]);
        assert_eq!(config.llm.cooldown_seconds, 90);

        // Explicit values from the fixture.
        assert_eq!(config.search.timeout_seconds, 45);
        assert!(!config.search.prefer_rtk);
        assert_eq!(config.logging.level, LogLevel::Debug);

        // Defaults for values omitted in the fixture.
        assert_eq!(config.codebase_memory.staleness_seconds, 3600);
        assert_eq!(
            config.codebase_memory.command.as_deref(),
            Some("codebase-memory-mcp")
        );
        assert_eq!(config.search.rtk_path, None);

        unsafe {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn empty_provider_list_fails() {
        let mut config = config_with_provider("p", "REPO_EXPLORER_TEST_KEY_EMPTY");
        config.llm.providers.clear();
        assert_eq!(config.validate(), Err(ValidationError::EmptyProviderList));
    }

    #[test]
    fn duplicate_provider_names_fail() {
        let mut config = config_with_provider("dup", "REPO_EXPLORER_TEST_KEY_DUP");
        let second = config.llm.providers[0].clone();
        config.llm.providers.push(second);
        assert_eq!(
            config.validate(),
            Err(ValidationError::DuplicateProviderName {
                name: "dup".to_string()
            })
        );
    }

    #[test]
    fn missing_env_var_fails() {
        let var = "REPO_EXPLORER_TEST_KEY_MISSING";
        unsafe {
            std::env::remove_var(var);
        }
        let config = config_with_provider("primary", var);
        let err = config.validate().unwrap_err();
        assert_eq!(
            err,
            ValidationError::MissingEnvVar {
                provider: "primary".to_string(),
                var: var.to_string(),
            }
        );
        // The message names the variable and provider but no secret value.
        let msg = err.to_string();
        assert!(msg.contains(var));
        assert!(msg.contains("primary"));
        assert!(!msg.contains("not-a-real-key"));
    }

    #[test]
    fn codebase_memory_connection() {
        // Neither command nor endpoint.
        let mut config = config_with_provider("p", "REPO_EXPLORER_TEST_KEY_CM");
        unsafe {
            std::env::set_var("REPO_EXPLORER_TEST_KEY_CM", "x");
        }
        config.codebase_memory.command = None;
        config.codebase_memory.endpoint = None;
        assert_eq!(
            config.validate(),
            Err(ValidationError::MissingCodebaseMemoryConnection)
        );

        // Both command and endpoint.
        config.codebase_memory.command = Some("cmd".to_string());
        config.codebase_memory.endpoint = Some("http://localhost:1234".to_string());
        assert_eq!(
            config.validate(),
            Err(ValidationError::ConflictingCodebaseMemoryConnection)
        );
        unsafe {
            std::env::remove_var("REPO_EXPLORER_TEST_KEY_CM");
        }
    }

    #[test]
    fn parse_error_is_reported() {
        let malformed = "this is = = not valid toml";
        let err = toml::from_str::<Config>(malformed).unwrap_err();
        let config_err: ConfigError = err.into();
        assert!(matches!(config_err, ConfigError::Parse(_)));
    }

    #[test]
    fn missing_file_is_read_error() {
        let err = load(Path::new("does-not-exist-42.toml")).unwrap_err();
        assert!(matches!(err, ConfigError::Read { .. }));
    }

    #[test]
    fn default_api_key_env_maps_known_kinds() {
        assert_eq!(default_api_key_env("anthropic"), Some("ANTHROPIC_API_KEY"));
        assert_eq!(default_api_key_env("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(default_api_key_env("gemini"), Some("GEMINI_API_KEY"));
        assert_eq!(default_api_key_env("google"), Some("GEMINI_API_KEY"));
        assert_eq!(default_api_key_env("mystery"), None);
    }

    #[test]
    fn resolve_api_key_env_prefers_explicit_then_kind_default() {
        let mut config = config_with_provider("p", "EXPLICIT_VAR");
        let p = &mut config.llm.providers[0];
        // Explicit `api_key_env` wins.
        assert_eq!(p.resolve_api_key_env(), Some("EXPLICIT_VAR".to_string()));
        // None + known kind -> derived default.
        p.api_key_env = None;
        p.kind = "openai".to_string();
        assert_eq!(p.resolve_api_key_env(), Some("OPENAI_API_KEY".to_string()));
        // None + unknown kind -> None.
        p.kind = "mystery".to_string();
        assert_eq!(p.resolve_api_key_env(), None);
    }

    #[test]
    fn empty_model_list_fails() {
        let mut config = config_with_provider("p", "REPO_EXPLORER_TEST_KEY_EMPTY_MODELS");
        config.llm.providers[0].models.clear();
        assert_eq!(
            config.validate(),
            Err(ValidationError::EmptyModelList {
                provider: "p".to_string()
            })
        );
    }

    #[test]
    fn unknown_kind_without_env_fails() {
        let mut config = config_with_provider("p", "unused");
        config.llm.providers[0].api_key_env = None;
        config.llm.providers[0].kind = "mystery".to_string();
        assert_eq!(
            config.validate(),
            Err(ValidationError::UnknownProviderKind {
                provider: "p".to_string(),
                kind: "mystery".to_string(),
            })
        );
    }

    #[test]
    fn derives_env_var_and_reports_missing() {
        let mut config = config_with_provider("primary", "unused");
        config.llm.providers[0].api_key_env = None;
        config.llm.providers[0].kind = "anthropic".to_string();
        // No other test reads/writes ANTHROPIC_API_KEY.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
        }
        let err = config.validate().unwrap_err();
        assert_eq!(
            err,
            ValidationError::MissingEnvVar {
                provider: "primary".to_string(),
                var: "ANTHROPIC_API_KEY".to_string(),
            }
        );
    }
}
