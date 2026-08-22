//! Interactive first-run setup wizard for `repo-explorer-mcp`.
//!
//! All interactive IO lives here at the binary boundary. Prompts, echoes, and
//! progress messages go to STDERR; stdin is read line-by-line. stdout is NEVER
//! written by the wizard (it is reserved for the MCP protocol stream). The
//! wizard detects provider API-key env vars, builds a `Config`, serializes it
//! via `repo_explorer_core::config::to_toml_string`, writes it to the resolved
//! XDG path, then self-verifies via `repo_explorer_core::config::load`.

use anyhow::Context;
use repo_explorer_core::config::{
    self, CodebaseMemoryConfig, Config, KNOWN_PROVIDER_KINDS, LlmConfig, LoggingConfig,
    ProviderConfig, SearchConfig, default_api_key_env, default_cooldown_seconds,
    default_staleness_seconds,
};
use std::io::{BufRead, Write};
use std::path::Path;
use std::process::ExitCode;

/// True for the single-token subcommand `setup` (mirrors `wants_config_test`).
pub fn wants_setup(args: &[String]) -> bool {
    args.iter().any(|a| a == "setup")
}

/// A provider inferred from a set env var. `kind` is one of KNOWN_PROVIDER_KINDS.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DetectedProvider {
    kind: &'static str,
    api_key_env: String,
}

/// Candidate `(env var, kind)` table. Iteration order defines both detection
/// order and the emitted provider (failover) order.
///
/// Derived from core's `KNOWN_PROVIDER_KINDS`/`default_api_key_env` — the
/// single source of truth for which kinds exist and their canonical env var —
/// plus one wizard-specific extra alias (`GOOGLE_API_KEY`) that core has no
/// reason to know about. `GEMINI_API_KEY` precedes `GOOGLE_API_KEY` so it wins
/// the per-kind dedup when both are set.
fn candidate_env_vars() -> Vec<(&'static str, &'static str)> {
    let mut out: Vec<(&'static str, &'static str)> = Vec::new();
    let mut seen_vars = std::collections::HashSet::new();
    for &kind in KNOWN_PROVIDER_KINDS {
        // `KNOWN_PROVIDER_KINDS` lists `gemini` and `google` as distinct kind
        // strings that both resolve to `GEMINI_API_KEY` — keep only the first
        // (canonical `gemini`) so `google` doesn't surface as a detectable kind.
        if let Some(var) = default_api_key_env(kind)
            && seen_vars.insert(var)
        {
            out.push((var, kind));
        }
    }
    out.push(("GOOGLE_API_KEY", "gemini"));
    out
}

/// The single definition of "is this env var actually set": present and
/// non-blank once trimmed. Shared by `detect_providers` and any other check
/// that needs to ask the same question about a specific var (e.g. the
/// both-Gemini-vars-set notice in `run_setup_inner`).
fn env_var_is_set(get: &impl Fn(&str) -> Option<String>, var: &str) -> bool {
    get(var).map(|v| !v.trim().is_empty()).unwrap_or(false)
}

/// Scan candidate env vars via the injected accessor and return deduped
/// detected providers, at most one per `kind`, in order of first detection.
/// A var counts as set only when the accessor returns `Some(non-blank)`.
/// Injecting the accessor keeps this pure and unit-testable.
fn detect_providers(get: impl Fn(&str) -> Option<String>) -> Vec<DetectedProvider> {
    let mut out: Vec<DetectedProvider> = Vec::new();
    for &(var, kind) in &candidate_env_vars() {
        let present = env_var_is_set(&get, var);
        if !present {
            continue;
        }
        if out.iter().any(|p| p.kind == kind) {
            continue; // already detected this kind via an earlier (canonical) var
        }
        out.push(DetectedProvider {
            kind,
            api_key_env: var.to_string(),
        });
    }
    out
}

/// Hand-maintained free-tier model catalog, keyed on the same `kind` strings as
/// `repo_explorer_core::config::KNOWN_PROVIDER_KINDS`. An empty slice means
/// "no standing free tier".
///
/// Research date: 2026. As of this date only Gemini/Google offers a genuine
/// ongoing free API tier (Flash / Flash-Lite class); Anthropic and OpenAI
/// provide only one-time trial credit or opt-in allowances, i.e. no stable
/// free-tier model list, so their slices are empty.
///
/// CAVEATS:
///  - Hand-maintained: there is NO compile-time or automated check against
///    live provider APIs, so this table can drift/stale over time.
///  - Gemini free-tier terms are rate-limit based (RPM/RPD/TPM) per model, not
///    merely a model allowlist, so a user may still hit HTTP 429 at runtime.
///  - Pro-class Gemini models are deliberately excluded (not free tier).
///  - `genai` forwards these model IDs verbatim; any string is syntactically
///    legal, so there is no validation of the IDs here.
fn free_tier_models(kind: &str) -> &'static [&'static str] {
    match kind {
        "gemini" | "google" => &["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "anthropic" => &[],
        "openai" => &[],
        _ => &[],
    }
}

/// Split a comma-separated list into trimmed, non-empty model IDs.
fn parse_models(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Return `base` if unused, else `base-2`, `base-3`, ... until unique.
fn unique_name(base: &str, used: &std::collections::HashSet<String>) -> String {
    if !used.contains(base) {
        return base.to_string();
    }
    let mut i = 2usize;
    loop {
        let candidate = format!("{base}-{i}");
        if !used.contains(&candidate) {
            return candidate;
        }
        i += 1;
    }
}

/// Read one line from stdin, stripping the trailing newline. `None` on EOF.
fn read_line() -> anyhow::Result<Option<String>> {
    let mut line = String::new();
    let n = std::io::stdin()
        .lock()
        .read_line(&mut line)
        .context("failed to read from stdin")?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\r', '\n']).to_string()))
}

/// Prompt (showing `default`) on stderr; empty input or EOF returns `default`.
fn prompt_default(prompt: &str, default: &str) -> anyhow::Result<String> {
    eprint!("{prompt} [{default}]: ");
    let _ = std::io::stderr().flush();
    match read_line()? {
        Some(s) if !s.trim().is_empty() => Ok(s.trim().to_string()),
        _ => Ok(default.to_string()),
    }
}

/// Prompt on stderr and return the raw trimmed line (may be empty). EOF bails.
fn prompt_raw(prompt: &str) -> anyhow::Result<String> {
    eprint!("{prompt}: ");
    let _ = std::io::stderr().flush();
    match read_line()? {
        Some(s) => Ok(s.trim().to_string()),
        None => anyhow::bail!("unexpected end of input"),
    }
}

/// Top-level wizard entry: detect -> prompt -> build Config -> write ->
/// self-verify. Returns `SUCCESS` on a written+validated config, `FAILURE`
/// otherwise (with a stderr diagnostic).
pub fn run_setup(config_path: &Path) -> ExitCode {
    match run_setup_inner(config_path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("repo-explorer-mcp setup: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run_setup_inner(config_path: &Path) -> anyhow::Result<()> {
    if config_path.exists() {
        eprintln!("A config already exists at {}.", config_path.display());
        let answer = prompt_default("Overwrite it? (y/N)", "n")?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            eprintln!("Aborted; existing config left unchanged.");
            return Ok(());
        }
    }

    eprintln!("repo-explorer-mcp interactive setup");
    eprintln!("Detecting provider API-key environment variables...");

    let get_env = |k: &str| std::env::var(k).ok();
    let detected = detect_providers(get_env);

    // If both Gemini vars are set, note that GEMINI_API_KEY was preferred.
    let gemini_set = env_var_is_set(&get_env, "GEMINI_API_KEY");
    let google_set = env_var_is_set(&get_env, "GOOGLE_API_KEY");
    if gemini_set && google_set {
        eprintln!("  note: both GEMINI_API_KEY and GOOGLE_API_KEY are set; using GEMINI_API_KEY.");
    }

    if detected.is_empty() {
        anyhow::bail!(
            "no provider API-key environment variable detected. Set one of \
             ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY \
             and re-run `repo-explorer-mcp setup`."
        );
    }
    for p in &detected {
        eprintln!("  detected `{}` provider via {}", p.kind, p.api_key_env);
    }

    let mut providers: Vec<ProviderConfig> = Vec::new();
    let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for dp in &detected {
        eprintln!();
        eprintln!("Configuring `{}` provider:", dp.kind);

        // Provider name (kept unique for DuplicateProviderName).
        let default_name = unique_name(dp.kind, &used_names);
        let mut name = prompt_default("  provider name", &default_name)?;
        if used_names.contains(&name) {
            let deduped = unique_name(&name, &used_names);
            eprintln!("  name `{name}` already used; using `{deduped}`.");
            name = deduped;
        }
        used_names.insert(name.clone());

        // Models.
        let catalog = free_tier_models(dp.kind);
        let models: Vec<String> = if catalog.is_empty() {
            eprintln!(
                "  `{}` has no persistent free tier (trial credit / paid only).",
                dp.kind
            );
            eprintln!("  Enter at least one model ID to use (comma-separated).");
            loop {
                let raw = prompt_raw("  models")?;
                let parsed = parse_models(&raw);
                if parsed.is_empty() {
                    eprintln!("  at least one model ID is required.");
                    continue;
                }
                break parsed;
            }
        } else {
            let default_models = catalog.join(", ");
            let raw = prompt_default("  models (comma-separated)", &default_models)?;
            let parsed = parse_models(&raw);
            if parsed.is_empty() {
                catalog.iter().map(|s| s.to_string()).collect()
            } else {
                parsed
            }
        };

        // api_key_env: None (implicit default) when the detected var equals the
        // default for this kind; else explicit (the GOOGLE_API_KEY case).
        let api_key_env = match default_api_key_env(dp.kind) {
            Some(def) if def == dp.api_key_env => None,
            _ => Some(dp.api_key_env.clone()),
        };

        providers.push(ProviderConfig {
            name,
            kind: dp.kind.to_string(),
            api_key_env,
            models,
            base_url: None,
        });
    }

    // Codebase-memory connection (XOR command/endpoint enforced by the prompt).
    eprintln!();
    eprintln!("Codebase memory connection:");
    eprintln!("  1) stdio: launch `codebase-memory-mcp --stdio` (default)");
    eprintln!("  2) network endpoint (URL)");
    let choice = prompt_default("  choose 1 or 2", "1")?;
    let codebase_memory = if choice == "2" {
        let endpoint = loop {
            let e = prompt_raw("  endpoint URL")?;
            if !e.is_empty() {
                break e;
            }
            eprintln!("  an endpoint URL is required.");
        };
        CodebaseMemoryConfig {
            command: None,
            args: Vec::new(),
            endpoint: Some(endpoint),
            staleness_seconds: default_staleness_seconds(),
        }
    } else {
        CodebaseMemoryConfig {
            command: Some("codebase-memory-mcp".to_string()),
            args: vec!["--stdio".to_string()],
            endpoint: None,
            staleness_seconds: default_staleness_seconds(),
        }
    };

    // search / logging left at defaults (fully defaulted in core); not prompted.
    let cfg = Config {
        llm: LlmConfig {
            providers,
            cooldown_seconds: default_cooldown_seconds(),
        },
        codebase_memory,
        search: SearchConfig::default(),
        logging: LoggingConfig::default(),
    };

    let toml_string =
        config::to_toml_string(&cfg).context("failed to serialize the generated config as TOML")?;

    if let Some(parent) = config_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }
    std::fs::write(config_path, &toml_string)
        .with_context(|| format!("failed to write config to {}", config_path.display()))?;

    // Self-check: the written config must pass the exact `config test` path.
    config::load(config_path).map_err(|e| {
        anyhow::anyhow!(
            "written config at {} failed validation: {e} (at {})",
            config_path.display(),
            e.toml_path().unwrap_or_else(|| "<unknown>".to_string())
        )
    })?;

    eprintln!();
    eprintln!("wrote config to {}", config_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build an env accessor closure over a fixed (var -> value) table.
    fn acc(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn detect_anthropic_only() {
        let got = detect_providers(acc(&[("ANTHROPIC_API_KEY", "k")]));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, "anthropic");
        assert_eq!(got[0].api_key_env, "ANTHROPIC_API_KEY");
    }

    #[test]
    fn detect_openai_only() {
        let got = detect_providers(acc(&[("OPENAI_API_KEY", "k")]));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, "openai");
        assert_eq!(got[0].api_key_env, "OPENAI_API_KEY");
    }

    #[test]
    fn detect_gemini_via_gemini_var() {
        let got = detect_providers(acc(&[("GEMINI_API_KEY", "k")]));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, "gemini");
        assert_eq!(got[0].api_key_env, "GEMINI_API_KEY");
    }

    #[test]
    fn detect_gemini_via_google_var_only_uses_explicit_env() {
        let got = detect_providers(acc(&[("GOOGLE_API_KEY", "k")]));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, "gemini");
        assert_eq!(got[0].api_key_env, "GOOGLE_API_KEY");
    }

    #[test]
    fn detect_both_gemini_vars_prefers_canonical_single_entry() {
        let got = detect_providers(acc(&[("GEMINI_API_KEY", "k"), ("GOOGLE_API_KEY", "k2")]));
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].kind, "gemini");
        assert_eq!(got[0].api_key_env, "GEMINI_API_KEY");
    }

    #[test]
    fn detect_multiple_kinds_is_order_stable() {
        let got = detect_providers(acc(&[
            ("OPENAI_API_KEY", "k"),
            ("ANTHROPIC_API_KEY", "k"),
            ("GEMINI_API_KEY", "k"),
        ]));
        // Order follows the candidate table (anthropic, openai, gemini).
        let kinds: Vec<&str> = got.iter().map(|p| p.kind).collect();
        assert_eq!(kinds, vec!["anthropic", "openai", "gemini"]);
    }

    #[test]
    fn detect_none_is_empty() {
        let got = detect_providers(acc(&[]));
        assert!(got.is_empty());
    }

    #[test]
    fn detect_empty_string_var_is_not_detected() {
        let got = detect_providers(acc(&[("ANTHROPIC_API_KEY", "   ")]));
        assert!(got.is_empty());
    }

    #[test]
    fn free_tier_gemini_non_empty_excludes_pro() {
        let m = free_tier_models("gemini");
        assert!(!m.is_empty());
        assert!(m.iter().all(|id| !id.contains("pro")));
        assert_eq!(free_tier_models("google"), free_tier_models("gemini"));
    }

    #[test]
    fn free_tier_anthropic_openai_and_unknown_empty() {
        assert!(free_tier_models("anthropic").is_empty());
        assert!(free_tier_models("openai").is_empty());
        assert!(free_tier_models("something-else").is_empty());
    }

    #[test]
    fn wants_setup_truth_table() {
        assert!(wants_setup(&["setup".to_string()]));
        assert!(!wants_setup(&["config".to_string(), "test".to_string()]));
        assert!(!wants_setup(&[]));
        assert!(!wants_setup(&["--version".to_string()]));
    }

    #[test]
    fn parse_models_trims_and_drops_empties() {
        assert_eq!(
            parse_models(" a , b ,, c "),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert!(parse_models("  ,  ,").is_empty());
    }

    #[test]
    fn unique_name_appends_index_on_collision() {
        let mut used = std::collections::HashSet::new();
        used.insert("gemini".to_string());
        assert_eq!(unique_name("gemini", &used), "gemini-2");
        used.insert("gemini-2".to_string());
        assert_eq!(unique_name("gemini", &used), "gemini-3");
        assert_eq!(unique_name("openai", &used), "openai");
    }
}
