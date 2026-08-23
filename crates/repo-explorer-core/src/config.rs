//! Strongly-typed configuration model loaded and validated from TOML.
//!
//! The single public entry point is [`load`], which reads a file, parses it as
//! TOML into a [`Config`], and runs [`Config::validate`]. Validation is also
//! public so it can be unit-tested on hand-built `Config` values.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level configuration.
///
/// `#[serde(deny_unknown_fields)]` is intentionally NOT used so later stages can
/// add fields without breaking existing configs.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub codebase_memory: CodebaseMemoryConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlmConfig {
    /// Order in the file = failover order. Preserved by using `Vec`.
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    #[serde(default = "default_cooldown_seconds")]
    pub cooldown_seconds: u64,
    /// HTTPS proxy URL used for model upstream requests when set. Applied to
    /// every provider entry uniformly, but — matching the conventional
    /// `HTTPS_PROXY` env var semantics — only requests to an `https://`
    /// destination are routed through it; a provider entry whose `base_url`
    /// is `http://` is not covered. Unset means "no proxy".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub https_proxy: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
    pub name: String,
    /// Open string (anthropic | openai | google | …) — kept extensible.
    pub kind: String,
    /// Name of the env var holding the API key (never the key itself).
    /// When omitted, derived from `kind` via `default_api_key_env`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Ordered model list. First model = first tried; on a usage-limit error
    /// the router advances to the next model, then to the next provider entry.
    pub models: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

/// The single definition of "is this API-key env var actually set": present
/// and non-blank once trimmed. A blank value is worse than a missing one — it
/// would pass a bare `env::var(..).is_ok()` check and then fail as a 401 at
/// call time — so every layer that asks this question must ask it the same
/// way. The accessor is injected so callers can supply a test double instead
/// of mutating the process environment.
pub fn env_var_is_set(get: impl Fn(&str) -> Option<String>, var: &str) -> bool {
    get(var).is_some_and(|v| !v.trim().is_empty())
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

/// True when `url` looks like a usable http(s) proxy URL: a (case-insensitive)
/// `http://`/`https://` scheme followed by a non-empty host. Shallow — no full
/// RFC 3986 parse, matching the validation depth already applied to
/// `base_url`/`endpoint` elsewhere in this module — but it catches the two
/// mistakes users actually make: wrong/missing scheme, and a scheme with
/// nothing after it (e.g. `"https://"`). The single source of truth for this
/// rule: both [`Config::validate`] and the setup wizard call this instead of
/// each re-implementing the scheme check.
pub fn is_valid_https_proxy_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    let host_and_rest = lower
        .strip_prefix("http://")
        .or_else(|| lower.strip_prefix("https://"));
    match host_and_rest {
        Some(rest) => !rest.is_empty() && !rest.starts_with('/'),
        None => false,
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodebaseMemoryConfig {
    /// Stdio transport: process command to launch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Network transport: endpoint URL. Mutually exclusive with `command`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// Index considered stale after this many seconds (consumed by Stage 2).
    #[serde(default = "default_staleness_seconds")]
    pub staleness_seconds: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchConfig {
    /// Explicit path to the `rtk` binary; `None` → auto-detect in Stage 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rtk_path: Option<PathBuf>,
    /// Explicit path to the `ripgrep` binary; `None` → auto-detect in Stage 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ripgrep_path: Option<PathBuf>,
    /// Per-search subprocess timeout. `0` means "no timeout" — the explicit
    /// opt-out, not a stand-in for the default.
    #[serde(default = "default_search_timeout_seconds")]
    pub timeout_seconds: u64,
    #[serde(default = "default_prefer_rtk")]
    pub prefer_rtk: bool,
}

/// Hand-written (not derived) so that `SearchConfig::default()` and the serde
/// field defaults are the *same* values: a derived `Default` would yield
/// `timeout_seconds: 0` / `prefer_rtk: false`, silently disagreeing with what
/// loading an empty `[search]` section produces.
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            rtk_path: None,
            ripgrep_path: None,
            timeout_seconds: default_search_timeout_seconds(),
            prefer_rtk: default_prefer_rtk(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct LoggingConfig {
    #[serde(default)]
    pub level: LogLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    #[default]
    Info,
    Warn,
    Error,
}

/// Default `llm.cooldown_seconds` (also the serde default for that field).
/// Public so callers building a `Config` programmatically (e.g. the setup
/// wizard) can reuse the same value instead of duplicating the literal.
pub fn default_cooldown_seconds() -> u64 {
    60
}

/// Default `codebase_memory.staleness_seconds` (also the serde default for
/// that field). Public for the same reason as `default_cooldown_seconds`.
pub fn default_staleness_seconds() -> u64 {
    3600
}

fn default_search_timeout_seconds() -> u64 {
    30
}

fn default_prefer_rtk() -> bool {
    true
}

/// Provider `kind` strings the LLM boundary can map to a genai adapter.
/// Kept in sync with `repo_explorer_llm::adapter_kind_for`; core cannot depend
/// on genai, so the set is re-declared here as plain strings.
pub const KNOWN_PROVIDER_KINDS: &[&str] = &["anthropic", "openai", "gemini", "google"];

/// Errors returned by [`load`].
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config file `{path}`: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config as TOML: {source}")]
    Parse {
        #[source]
        source: toml::de::Error,
        location: Option<String>,
    },
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error("failed to serialize config as TOML: {source}")]
    Serialize {
        #[source]
        source: toml::ser::Error,
    },
}

impl ConfigError {
    /// TOML location of the error, when one is known.
    /// `None` for whole-file read errors.
    pub fn toml_path(&self) -> Option<String> {
        match self {
            ConfigError::Read { .. } => None,
            ConfigError::Parse { location, .. } => location.clone(),
            ConfigError::Validation(v) => Some(v.toml_path()),
            ConfigError::Serialize { .. } => None,
        }
    }

    /// True when the underlying cause is a missing config file.
    pub fn is_not_found(&self) -> bool {
        matches!(self, ConfigError::Read { source, .. }
            if source.kind() == std::io::ErrorKind::NotFound)
    }
}

/// Semantic validation failures, independent of parsing.
///
/// Messages name the offending provider/variable/section but never a secret value.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("llm.providers must contain at least one provider")]
    EmptyProviderList,
    #[error("duplicate provider name `{name}` in llm.providers")]
    DuplicateProviderName { index: usize, name: String },
    #[error("provider `{provider}` has an empty `models` list; at least one model is required")]
    EmptyModelsList { index: usize, provider: String },
    #[error("provider `{provider}` references environment variable `{var}`, which is not set")]
    MissingEnvVar {
        index: usize,
        provider: String,
        var: String,
        /// Whether `var` came from an explicit `api_key_env` in the file
        /// (vs. the kind-derived default, which has no literal TOML key).
        explicit: bool,
    },
    #[error(
        "provider `{provider}` has unknown kind `{kind}` (expected one of: {})",
        KNOWN_PROVIDER_KINDS.join(", ")
    )]
    UnknownProviderKind {
        index: usize,
        provider: String,
        kind: String,
    },
    #[error(
        "codebase_memory must set either `command` (stdio) or `endpoint` (network), but neither is present"
    )]
    MissingCodebaseMemoryConnection,
    #[error("codebase_memory sets both `command` and `endpoint`; exactly one is allowed")]
    ConflictingCodebaseMemoryConnection,
    #[error("llm.https_proxy `{url}` is not a valid http(s):// URL")]
    InvalidHttpsProxyUrl { url: String },
}

impl ValidationError {
    /// Dotted TOML key path where this error occurred.
    pub fn toml_path(&self) -> String {
        match self {
            ValidationError::EmptyProviderList => "llm.providers".to_string(),
            ValidationError::DuplicateProviderName { index, .. } => {
                format!("llm.providers[{index}].name")
            }
            ValidationError::EmptyModelsList { index, .. } => {
                format!("llm.providers[{index}].models")
            }
            ValidationError::MissingEnvVar {
                index, explicit, ..
            } => {
                if *explicit {
                    format!("llm.providers[{index}].api_key_env")
                } else {
                    format!("llm.providers[{index}].kind")
                }
            }
            ValidationError::UnknownProviderKind { index, .. } => {
                format!("llm.providers[{index}].kind")
            }
            ValidationError::MissingCodebaseMemoryConnection
            | ValidationError::ConflictingCodebaseMemoryConnection => "codebase_memory".to_string(),
            ValidationError::InvalidHttpsProxyUrl { .. } => "llm.https_proxy".to_string(),
        }
    }
}

/// Read, parse, and validate a config file. The single public entry point.
pub fn load(path: &Path) -> Result<Config, ConfigError> {
    let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let config: Config = toml::from_str(&contents).map_err(|source| {
        let location = source.span().map(|span| line_col(&contents, span.start));
        ConfigError::Parse { source, location }
    })?;
    config.validate()?;
    Ok(config)
}

/// Render a byte offset into `text` as a `line N, column M` string (1-based).
fn line_col(text: &str, byte: usize) -> String {
    let byte = byte.min(text.len());
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, ch) in text.char_indices() {
        if i >= byte {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    format!("line {line}, column {col}")
}

/// Serialize a `Config` to a pretty TOML string. Inverse of `load`'s parse
/// step; the single serialization entry point (core owns the schema).
pub fn to_toml_string(config: &Config) -> Result<String, ConfigError> {
    toml::to_string_pretty(config).map_err(|source| ConfigError::Serialize { source })
}

impl Config {
    /// Semantic validation, independent of parsing. Public so it is unit-testable
    /// on hand-built `Config` values.
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.llm.providers.is_empty() {
            return Err(ValidationError::EmptyProviderList);
        }

        // One pass per provider: name uniqueness, then the per-entry checks in
        // increasing order of specificity.
        let mut seen = std::collections::HashSet::new();
        for (index, provider) in self.llm.providers.iter().enumerate() {
            if !seen.insert(provider.name.as_str()) {
                return Err(ValidationError::DuplicateProviderName {
                    index,
                    name: provider.name.clone(),
                });
            }
            if provider.models.is_empty() {
                return Err(ValidationError::EmptyModelsList {
                    index,
                    provider: provider.name.clone(),
                });
            }
            if !KNOWN_PROVIDER_KINDS.contains(&provider.kind.as_str()) {
                return Err(ValidationError::UnknownProviderKind {
                    index,
                    provider: provider.name.clone(),
                    kind: provider.kind.clone(),
                });
            }
            let var = provider.resolve_api_key_env().unwrap_or_default();
            if !env_var_is_set(|v| std::env::var(v).ok(), &var) {
                return Err(ValidationError::MissingEnvVar {
                    index,
                    provider: provider.name.clone(),
                    var,
                    explicit: provider.api_key_env.is_some(),
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

        if let Some(proxy) = &self.llm.https_proxy
            && !is_valid_https_proxy_url(proxy)
        {
            return Err(ValidationError::InvalidHttpsProxyUrl { url: proxy.clone() });
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
                cooldown_seconds: default_cooldown_seconds(),
                https_proxy: None,
            },
            codebase_memory: CodebaseMemoryConfig {
                command: Some("cmd".to_string()),
                args: vec![],
                endpoint: None,
                staleness_seconds: default_staleness_seconds(),
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
        assert_eq!(
            config.codebase_memory.staleness_seconds,
            default_staleness_seconds()
        );
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
        unsafe {
            std::env::set_var("REPO_EXPLORER_TEST_KEY_DUP", "x");
        }
        let second = config.llm.providers[0].clone();
        config.llm.providers.push(second);
        assert_eq!(
            config.validate(),
            Err(ValidationError::DuplicateProviderName {
                index: 1,
                name: "dup".to_string()
            })
        );
        unsafe {
            std::env::remove_var("REPO_EXPLORER_TEST_KEY_DUP");
        }
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
                index: 0,
                provider: "primary".to_string(),
                var: var.to_string(),
                explicit: true,
            }
        );
        // The message names the variable and provider but no secret value.
        let msg = err.to_string();
        assert!(msg.contains(var));
        assert!(msg.contains("primary"));
        assert!(!msg.contains("not-a-real-key"));
        assert_eq!(err.toml_path(), "llm.providers[0].api_key_env");
    }

    #[test]
    fn missing_env_var_implicit_default_points_at_kind() {
        let var = "ANTHROPIC_API_KEY";
        let had = std::env::var(var).ok();
        unsafe {
            std::env::remove_var(var);
        }
        let mut config = config_with_provider("primary", var);
        config.llm.providers[0].api_key_env = None; // rely on kind-derived default
        let err = config.validate().unwrap_err();
        assert_eq!(
            err,
            ValidationError::MissingEnvVar {
                index: 0,
                provider: "primary".to_string(),
                var: var.to_string(),
                explicit: false,
            }
        );
        assert_eq!(err.toml_path(), "llm.providers[0].kind");
        if let Some(v) = had {
            unsafe {
                std::env::set_var(var, v);
            }
        }
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
        let config_err = ConfigError::Parse {
            source: err,
            location: None,
        };
        assert!(matches!(config_err, ConfigError::Parse { .. }));
    }

    #[test]
    fn missing_file_is_read_error() {
        let err = load(Path::new("does-not-exist-42.toml")).unwrap_err();
        assert!(matches!(err, ConfigError::Read { .. }));
    }

    #[test]
    fn unknown_provider_kind_fails() {
        let var = "REPO_EXPLORER_TEST_KEY_UNKNOWN_KIND";
        unsafe {
            std::env::set_var(var, "x");
        }
        let err = load(&fixture_path("unknown_kind.toml")).unwrap_err();
        match &err {
            ConfigError::Validation(ValidationError::UnknownProviderKind {
                index,
                provider,
                kind,
            }) => {
                assert_eq!(*index, 0);
                assert_eq!(provider, "primary");
                assert_eq!(kind, "bogus");
            }
            other => panic!("expected UnknownProviderKind, got {other:?}"),
        }
        assert_eq!(err.toml_path(), Some("llm.providers[0].kind".to_string()));
        unsafe {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn toml_path_for_each_variant() {
        assert_eq!(
            ValidationError::EmptyProviderList.toml_path(),
            "llm.providers"
        );
        assert_eq!(
            ValidationError::DuplicateProviderName {
                index: 2,
                name: "d".to_string()
            }
            .toml_path(),
            "llm.providers[2].name"
        );
        assert_eq!(
            ValidationError::MissingEnvVar {
                index: 1,
                provider: "p".to_string(),
                var: "V".to_string(),
                explicit: true,
            }
            .toml_path(),
            "llm.providers[1].api_key_env"
        );
        assert_eq!(
            ValidationError::MissingEnvVar {
                index: 1,
                provider: "p".to_string(),
                var: "V".to_string(),
                explicit: false,
            }
            .toml_path(),
            "llm.providers[1].kind"
        );
        assert_eq!(
            ValidationError::EmptyModelsList {
                index: 3,
                provider: "p".to_string(),
            }
            .toml_path(),
            "llm.providers[3].models"
        );
        assert_eq!(
            ValidationError::UnknownProviderKind {
                index: 0,
                provider: "p".to_string(),
                kind: "k".to_string()
            }
            .toml_path(),
            "llm.providers[0].kind"
        );
        assert_eq!(
            ValidationError::MissingCodebaseMemoryConnection.toml_path(),
            "codebase_memory"
        );
        assert_eq!(
            ValidationError::ConflictingCodebaseMemoryConnection.toml_path(),
            "codebase_memory"
        );
        assert_eq!(
            ValidationError::InvalidHttpsProxyUrl {
                url: "ftp://x".to_string()
            }
            .toml_path(),
            "llm.https_proxy"
        );
    }

    #[test]
    fn https_proxy_accepts_http_and_https_schemes() {
        let mut config = config_with_provider("p", "REPO_EXPLORER_TEST_KEY_PROXY_OK");
        unsafe {
            std::env::set_var("REPO_EXPLORER_TEST_KEY_PROXY_OK", "x");
        }
        config.llm.https_proxy = Some("https://proxy.example.com:8443".to_string());
        assert!(config.validate().is_ok());
        config.llm.https_proxy = Some("http://proxy.example.com:8080".to_string());
        assert!(config.validate().is_ok());
        unsafe {
            std::env::remove_var("REPO_EXPLORER_TEST_KEY_PROXY_OK");
        }
    }

    #[test]
    fn https_proxy_rejects_non_http_scheme() {
        let mut config = config_with_provider("p", "REPO_EXPLORER_TEST_KEY_PROXY_BAD");
        unsafe {
            std::env::set_var("REPO_EXPLORER_TEST_KEY_PROXY_BAD", "x");
        }
        config.llm.https_proxy = Some("proxy.example.com:8080".to_string());
        assert_eq!(
            config.validate(),
            Err(ValidationError::InvalidHttpsProxyUrl {
                url: "proxy.example.com:8080".to_string()
            })
        );
        unsafe {
            std::env::remove_var("REPO_EXPLORER_TEST_KEY_PROXY_BAD");
        }
    }

    #[test]
    fn is_valid_https_proxy_url_accepts_uppercase_scheme() {
        // URL schemes are case-insensitive (RFC 3986); the check must not
        // reject a proxy URL just because its scheme isn't lowercase.
        assert!(is_valid_https_proxy_url("HTTPS://proxy.example.com:8443"));
        assert!(is_valid_https_proxy_url("Http://proxy.example.com"));
    }

    #[test]
    fn is_valid_https_proxy_url_rejects_empty_host() {
        // A bare scheme with no host must not pass as "a valid http(s) URL" —
        // it will fail later at client-build time regardless.
        assert!(!is_valid_https_proxy_url("https://"));
        assert!(!is_valid_https_proxy_url("http://"));
        assert!(!is_valid_https_proxy_url("https:///path"));
    }

    #[test]
    fn https_proxy_case_insensitive_scheme_validates() {
        let mut config = config_with_provider("p", "REPO_EXPLORER_TEST_KEY_PROXY_CASE");
        unsafe {
            std::env::set_var("REPO_EXPLORER_TEST_KEY_PROXY_CASE", "x");
        }
        config.llm.https_proxy = Some("HTTPS://proxy.example.com:8443".to_string());
        assert!(config.validate().is_ok());
        unsafe {
            std::env::remove_var("REPO_EXPLORER_TEST_KEY_PROXY_CASE");
        }
    }

    #[test]
    fn https_proxy_empty_host_fails_validation() {
        let mut config = config_with_provider("p", "REPO_EXPLORER_TEST_KEY_PROXY_EMPTY");
        unsafe {
            std::env::set_var("REPO_EXPLORER_TEST_KEY_PROXY_EMPTY", "x");
        }
        config.llm.https_proxy = Some("https://".to_string());
        assert_eq!(
            config.validate(),
            Err(ValidationError::InvalidHttpsProxyUrl {
                url: "https://".to_string()
            })
        );
        unsafe {
            std::env::remove_var("REPO_EXPLORER_TEST_KEY_PROXY_EMPTY");
        }
    }

    #[test]
    fn parse_error_has_location() {
        let err = load(&fixture_path("malformed.toml")).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
        let path = err.toml_path();
        assert!(path.is_some());
        let s = path.unwrap();
        assert!(s.contains("line"), "expected a line/column string, got {s}");
        assert!(
            s.contains("column"),
            "expected a line/column string, got {s}"
        );
    }

    #[test]
    fn not_found_is_detected() {
        let err = load(Path::new("does-not-exist-99.toml")).unwrap_err();
        assert!(err.is_not_found());
        assert_eq!(err.toml_path(), None);
    }

    #[test]
    fn to_toml_string_round_trips_and_skips_none() {
        let var = "REPO_EXPLORER_TEST_KEY_ROUNDTRIP";
        unsafe {
            std::env::set_var(var, "x");
        }
        let config = config_with_provider("primary", var);
        let toml = to_toml_string(&config).expect("serialize should succeed");

        // None fields must be omitted (no `key = ` line) — guards skip_serializing_if.
        assert!(
            !toml.contains("base_url"),
            "None base_url must be skipped, got:\n{toml}"
        );
        assert!(
            !toml.contains("endpoint"),
            "None endpoint must be skipped, got:\n{toml}"
        );

        // Round-trips back into an equivalent, valid Config.
        let parsed: Config = toml::from_str(&toml).expect("serialized TOML should parse");
        parsed
            .validate()
            .expect("round-tripped config should validate");
        assert_eq!(parsed.llm.providers[0].name, "primary");
        assert_eq!(parsed.llm.providers[0].api_key_env.as_deref(), Some(var));
        assert_eq!(parsed.llm.providers[0].models, vec!["m".to_string()]);
        assert_eq!(parsed.codebase_memory.command.as_deref(), Some("cmd"));

        unsafe {
            std::env::remove_var(var);
        }
    }

    #[test]
    fn search_config_default_matches_serde_defaults() {
        // A derived `Default` would silently disagree with the serde field
        // defaults, and the setup wizard writes `SearchConfig::default()` —
        // producing configs with no search timeout and rtk preference off.
        let from_default = SearchConfig::default();
        let from_empty_section: SearchConfig =
            toml::from_str("").expect("an empty [search] section must parse");
        assert_eq!(
            from_default.timeout_seconds,
            from_empty_section.timeout_seconds
        );
        assert_eq!(from_default.prefer_rtk, from_empty_section.prefer_rtk);
        assert_eq!(
            from_default.timeout_seconds,
            default_search_timeout_seconds()
        );
        assert!(from_default.prefer_rtk);
    }

    #[test]
    fn to_toml_string_omits_implicit_default_api_key_env() {
        // A wizard-shaped gemini config: api_key_env = None (implicit default).
        unsafe {
            std::env::set_var("GEMINI_API_KEY", "x");
        }
        let mut config = config_with_provider("gemini", "GEMINI_API_KEY");
        config.llm.providers[0].kind = "gemini".to_string();
        config.llm.providers[0].api_key_env = None;
        config.llm.providers[0].models = vec!["gemini-2.5-flash".to_string()];
        let toml = to_toml_string(&config).expect("serialize should succeed");
        assert!(
            !toml.contains("api_key_env"),
            "implicit-default api_key_env must be omitted, got:\n{toml}"
        );
        let parsed: Config = toml::from_str(&toml).expect("parse back");
        parsed.validate().expect("validate");
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
        }
    }
}
