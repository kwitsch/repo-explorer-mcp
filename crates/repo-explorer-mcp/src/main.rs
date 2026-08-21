//! `repo-explorer-mcp` binary entrypoint (Stage 6 bootstrap).
//!
//! Resolves the config path, loads config, initializes `tracing` to stderr, and
//! (completed in a later task) wires the exploration pipeline over rmcp stdio.
//! stdout is reserved for the MCP protocol stream; every diagnostic goes to
//! stderr.

use anyhow::Context;
use repo_explorer_core::config::LogLevel;
use std::path::PathBuf;
use std::process::ExitCode;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // stderr only: stdout carries the MCP protocol stream.
            eprintln!("repo-explorer-mcp: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let config_path = resolve_config_path(
        std::env::args().skip(1),
        std::env::var("REPO_EXPLORER_CONFIG").ok(),
    );
    let config = repo_explorer_core::config::load(&config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    init_tracing(config.logging.level);
    tracing::info!("repo-explorer-mcp configuration loaded");
    Ok(())
}

/// Resolve the config path. Precedence: `--config <path>` / `--config=<path>`
/// CLI arg, then the `REPO_EXPLORER_CONFIG` env var, then `./repo-explorer.toml`.
fn resolve_config_path(args: impl Iterator<Item = String>, env_var: Option<String>) -> PathBuf {
    let mut args = args;
    while let Some(arg) = args.next() {
        if let Some(rest) = arg.strip_prefix("--config=") {
            return PathBuf::from(rest);
        }
        if arg == "--config"
            && let Some(value) = args.next()
        {
            return PathBuf::from(value);
        }
    }
    match env_var {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from("./repo-explorer.toml"),
    }
}

/// Map a config `LogLevel` onto a `tracing` `LevelFilter`.
fn tracing_level_filter(level: LogLevel) -> tracing::level_filters::LevelFilter {
    use tracing::level_filters::LevelFilter;
    match level {
        LogLevel::Trace => LevelFilter::TRACE,
        LogLevel::Debug => LevelFilter::DEBUG,
        LogLevel::Info => LevelFilter::INFO,
        LogLevel::Warn => LevelFilter::WARN,
        LogLevel::Error => LevelFilter::ERROR,
    }
}

/// Initialize `tracing` to **stderr** at the configured level. Best-effort:
/// a `try_init` failure (e.g. a subscriber already set) is ignored so this is
/// safe to call more than once.
fn init_tracing(level: LogLevel) {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing_level_filter(level))
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::config::LogLevel;
    use tracing::level_filters::LevelFilter;

    fn args(v: &[&str]) -> impl Iterator<Item = String> {
        v.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn config_flag_space_form_wins() {
        let p = resolve_config_path(args(&["--config", "custom.toml"]), Some("env.toml".into()));
        assert_eq!(p, PathBuf::from("custom.toml"));
    }

    #[test]
    fn config_flag_equals_form() {
        let p = resolve_config_path(args(&["--config=eq.toml"]), None);
        assert_eq!(p, PathBuf::from("eq.toml"));
    }

    #[test]
    fn env_var_used_when_no_flag() {
        let p = resolve_config_path(args(&["--other", "x"]), Some("env.toml".into()));
        assert_eq!(p, PathBuf::from("env.toml"));
    }

    #[test]
    fn default_when_neither() {
        let p = resolve_config_path(args(&[]), None);
        assert_eq!(p, PathBuf::from("./repo-explorer.toml"));
    }

    #[test]
    fn cli_arg_precedence_over_env() {
        let p = resolve_config_path(args(&["--config", "cli.toml"]), Some("env.toml".into()));
        assert_eq!(p, PathBuf::from("cli.toml"));
    }

    #[test]
    fn level_filter_mapping() {
        assert_eq!(tracing_level_filter(LogLevel::Trace), LevelFilter::TRACE);
        assert_eq!(tracing_level_filter(LogLevel::Debug), LevelFilter::DEBUG);
        assert_eq!(tracing_level_filter(LogLevel::Info), LevelFilter::INFO);
        assert_eq!(tracing_level_filter(LogLevel::Warn), LevelFilter::WARN);
        assert_eq!(tracing_level_filter(LogLevel::Error), LevelFilter::ERROR);
    }
}
