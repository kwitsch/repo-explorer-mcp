//! `repo-explorer-mcp` binary entrypoint (Stage 6 bootstrap).
//!
//! Resolves the config path, loads config, initializes `tracing` to stderr, and
//! wires the exploration pipeline over rmcp stdio. stdout is reserved for the
//! MCP protocol stream; every diagnostic goes to stderr.

mod server;
mod setup;
mod update;

use anyhow::Context;
use repo_explorer_agent::AgentLoop;
use repo_explorer_core::config::LogLevel;
use repo_explorer_memory::MemoryClientBackend;
use repo_explorer_search::{CliSearchBackend, GitStateProbe};
use rmcp::ServiceExt;
use server::RepoExplorerServer;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

const USAGE: &str = "\
repo-explorer-mcp — MCP server exposing the `explore_repository` tool over stdio.

Usage:
  repo-explorer-mcp [--config <path>]   Serve on stdio (default)
  repo-explorer-mcp setup               Run the interactive first-run wizard
  repo-explorer-mcp config test         Validate the resolved config only
  repo-explorer-mcp --update            Check for and install updates
  repo-explorer-mcp --version           Print the version
  repo-explorer-mcp --help              Print this help

Config path precedence: --config <path> -> REPO_EXPLORER_CONFIG -> the
per-user config dir -> ./repo-explorer.toml (when it exists).";

#[tokio::main]
async fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // Every subsequent predicate scans only the tokens that are NOT the value
    // of a `--config <path>` pair, so a config file named `setup` (or `test`)
    // can never be mistaken for a subcommand.
    let subcommand_args = args_without_config_value(&argv);
    if has_flag(&subcommand_args, &["--version", "-V"]) {
        // stdout: a one-shot CLI query that exits before the MCP transport starts.
        println!("repo-explorer-mcp {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if has_flag(&subcommand_args, &["--help", "-h"]) {
        // stdout: a one-shot CLI query that exits before the MCP transport starts.
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if update::wants_update(&subcommand_args) {
        return update::run_update().await;
    }
    let config_path = resolve_config_path(
        &argv,
        std::env::var("REPO_EXPLORER_CONFIG").ok(),
        xdg_default_config_path(),
        |p| p.exists(),
    );
    if wants_config_test(&subcommand_args) {
        return run_config_test(&config_path);
    }
    if setup::wants_setup(&subcommand_args) {
        return setup::run_setup(&config_path);
    }
    // One load, and `is_not_found` (not a second `Path::exists` probe) decides
    // whether this is a first run: a config that exists but is unreadable or
    // malformed must report its real error, not "no config".
    let config = match repo_explorer_core::config::load(&config_path) {
        Ok(config) => config,
        Err(e) if e.is_not_found() => {
            if std::io::stdin().is_terminal() {
                return setup::run_setup(&config_path);
            }
            eprintln!(
                "repo-explorer-mcp: no config at {}. Run `repo-explorer-mcp setup` in a terminal to create one.",
                config_path.display()
            );
            return ExitCode::FAILURE;
        }
        Err(e) => {
            // stderr only: stdout carries the MCP protocol stream.
            eprintln!(
                "repo-explorer-mcp: failed to load config from {}: {e}",
                config_path.display()
            );
            return ExitCode::FAILURE;
        }
    };
    match run(config).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // stderr only: stdout carries the MCP protocol stream.
            eprintln!("repo-explorer-mcp: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(config: repo_explorer_core::config::Config) -> anyhow::Result<()> {
    init_tracing(config.logging.level);

    // The directory the server is launched in = the project root to explore.
    let repo_root = std::env::current_dir().context("failed to determine current directory")?;

    let memory = MemoryClientBackend::connect(&config.codebase_memory)
        .await
        .context("failed to connect to codebase-memory-mcp")?;
    let search = CliSearchBackend::new(&config.search);
    let router = repo_explorer_llm::build_router(&config.llm)
        .context("failed to build LLM provider router")?;
    let probe = GitStateProbe::new(config.search.timeout_seconds);
    let agent = AgentLoop::new(memory, search, router, probe, config.agent, config.cache);

    let server = RepoExplorerServer::new(Arc::new(agent), repo_root);
    tracing::info!("repo-explorer-mcp serving on stdio");
    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .context("failed to start MCP server on stdio")?;
    service
        .waiting()
        .await
        .context("MCP server terminated with an error")?;
    Ok(())
}

/// Path of the last-resort, launch-directory config.
const CWD_CONFIG_PATH: &str = "./repo-explorer.toml";

/// Copy of `args` with the value token of every `--config <path>` pair removed,
/// so a subcommand/flag scan can never mistake a config *path* for a
/// subcommand (`--config setup` must not launch the wizard).
fn args_without_config_value(args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(args.len());
    let mut skip_next = false;
    for arg in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--config" {
            skip_next = true;
        }
        out.push(arg.clone());
    }
    out
}

/// True when any of `flags` appears verbatim in `args`.
fn has_flag(args: &[String], flags: &[&str]) -> bool {
    args.iter().any(|a| flags.contains(&a.as_str()))
}

/// Resolve the config path. Precedence: `--config <path>` / `--config=<path>`
/// CLI arg -> `REPO_EXPLORER_CONFIG` env var -> the per-user config dir (when
/// resolvable) -> `./repo-explorer.toml`.
///
/// The last two tiers are `exists`-gated so the launch-directory fallback is
/// actually reachable: without that check the per-user default — which
/// resolves on essentially every machine — would always win, even when no file
/// is there, making the documented `./repo-explorer.toml` fallback dead and
/// silently ignoring an in-repo config. When neither file exists the per-user
/// default is still returned, since that is where the setup wizard writes.
fn resolve_config_path(
    args: &[String],
    env_var: Option<String>,
    xdg_default: Option<PathBuf>,
    exists: impl Fn(&Path) -> bool,
) -> PathBuf {
    let mut args = args.iter();
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
    if let Some(v) = env_var {
        return PathBuf::from(v);
    }
    let Some(xdg_default) = xdg_default else {
        return PathBuf::from(CWD_CONFIG_PATH);
    };
    if exists(&xdg_default) {
        return xdg_default;
    }
    let cwd = PathBuf::from(CWD_CONFIG_PATH);
    if exists(&cwd) {
        return cwd;
    }
    xdg_default
}

/// `<config dir>/repo-explorer/repo-explorer.toml`, or `None` when no config
/// dir can be determined (e.g. no HOME). On Linux the config dir honors
/// `XDG_CONFIG_HOME` (falling back to `~/.config`); on Windows it is `%APPDATA%`.
fn xdg_default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("repo-explorer").join("repo-explorer.toml"))
}

/// True for `--config-test`, or the two-token subcommand `config test`
/// (adjacent tokens). Callers pass [`args_without_config_value`] output, so a
/// `--config <path>` value can never supply either token.
fn wants_config_test(args: &[String]) -> bool {
    if has_flag(args, &["--config-test"]) {
        return true;
    }
    args.windows(2).any(|w| w[0] == "config" && w[1] == "test")
}

#[derive(serde::Serialize)]
struct ConfigTestReport {
    status: &'static str,
    config_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ConfigTestError>,
}

#[derive(serde::Serialize)]
struct ConfigTestError {
    message: String,
    toml_path: Option<String>,
}

/// Validate-only mode: load + parse + validate the config and print a
/// structured JSON report to stdout. No runtime subsystem is started (no
/// tracing, memory/search/LLM connections, or rmcp stdio transport). Because
/// no MCP session exists here, stdout is used for the report; exits non-zero
/// on any load/parse/validation failure.
fn run_config_test(config_path: &Path) -> ExitCode {
    match repo_explorer_core::config::load(config_path) {
        Ok(_) => {
            let report = ConfigTestReport {
                status: "valid",
                config_path: config_path.display().to_string(),
                error: None,
            };
            print_report(&report, "config-test");
            ExitCode::SUCCESS
        }
        Err(e) => {
            let report = ConfigTestReport {
                status: "invalid",
                config_path: config_path.display().to_string(),
                error: Some(ConfigTestError {
                    message: format!("{e}"),
                    toml_path: e.toml_path(),
                }),
            };
            print_report(&report, "config-test");
            ExitCode::FAILURE
        }
    }
}

/// Print any of this binary's one-shot CLI reports (config-test, update) as a
/// single pretty-printed JSON object to stdout. `kind` names the report in
/// the error message printed on the (rare) serialization failure.
pub(crate) fn print_report(report: &impl serde::Serialize, kind: &str) {
    match serde_json::to_string_pretty(report) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("repo-explorer-mcp: failed to serialize {kind} report: {e}"),
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

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// `exists` double: only the listed paths exist.
    fn only(paths: &[&str]) -> impl Fn(&Path) -> bool {
        let owned: Vec<PathBuf> = paths.iter().map(PathBuf::from).collect();
        move |p: &Path| owned.iter().any(|o| o == p)
    }

    fn nothing_exists(_: &Path) -> bool {
        false
    }

    #[test]
    fn config_flag_space_form_wins() {
        let p = resolve_config_path(
            &args(&["--config", "custom.toml"]),
            Some("env.toml".into()),
            None,
            nothing_exists,
        );
        assert_eq!(p, PathBuf::from("custom.toml"));
    }

    #[test]
    fn config_flag_equals_form() {
        let p = resolve_config_path(&args(&["--config=eq.toml"]), None, None, nothing_exists);
        assert_eq!(p, PathBuf::from("eq.toml"));
    }

    #[test]
    fn env_var_used_when_no_flag() {
        let p = resolve_config_path(
            &args(&["--other", "x"]),
            Some("env.toml".into()),
            None,
            nothing_exists,
        );
        assert_eq!(p, PathBuf::from("env.toml"));
    }

    #[test]
    fn default_when_neither() {
        let p = resolve_config_path(&args(&[]), None, None, nothing_exists);
        assert_eq!(p, PathBuf::from("./repo-explorer.toml"));
    }

    #[test]
    fn version_long_flag_detected() {
        assert!(has_flag(&args(&["--version"]), &["--version", "-V"]));
    }

    #[test]
    fn version_short_flag_detected() {
        assert!(has_flag(&args(&["-V"]), &["--version", "-V"]));
    }

    #[test]
    fn version_flag_detected_among_others() {
        assert!(has_flag(
            &args(&["--config", "x.toml", "--version"]),
            &["--version", "-V"]
        ));
    }

    #[test]
    fn no_version_flag() {
        assert!(!has_flag(
            &args(&["--config", "x.toml"]),
            &["--version", "-V"]
        ));
        assert!(!has_flag(&args(&[]), &["--version", "-V"]));
    }

    #[test]
    fn help_flag_detected() {
        assert!(has_flag(&args(&["--help"]), &["--help", "-h"]));
        assert!(has_flag(&args(&["-h"]), &["--help", "-h"]));
        assert!(!has_flag(&args(&["--config", "x.toml"]), &["--help", "-h"]));
    }

    #[test]
    fn config_value_is_never_read_as_a_subcommand() {
        // `--config setup` names a config file, not the `setup` subcommand.
        let stripped = args_without_config_value(&args(&["--config", "setup"]));
        assert_eq!(stripped, args(&["--config"]));
        assert!(!setup::wants_setup(&stripped));
        // Same for a file named `test` after the `config` token.
        let stripped = args_without_config_value(&args(&["config", "--config", "test"]));
        assert!(!wants_config_test(&stripped));
        // A real subcommand still survives the strip.
        let stripped = args_without_config_value(&args(&["--config", "c.toml", "setup"]));
        assert!(setup::wants_setup(&stripped));
    }

    #[test]
    fn cli_arg_precedence_over_env() {
        let p = resolve_config_path(
            &args(&["--config", "cli.toml"]),
            Some("env.toml".into()),
            None,
            nothing_exists,
        );
        assert_eq!(p, PathBuf::from("cli.toml"));
    }

    #[test]
    fn xdg_default_used_when_it_exists() {
        let xdg = PathBuf::from("/x/.config/repo-explorer/repo-explorer.toml");
        let p = resolve_config_path(
            &args(&[]),
            None,
            Some(xdg.clone()),
            only(&["/x/.config/repo-explorer/repo-explorer.toml"]),
        );
        assert_eq!(p, xdg);
    }

    #[test]
    fn xdg_default_wins_over_an_existing_cwd_config() {
        let xdg = PathBuf::from("/x/.config/repo-explorer/repo-explorer.toml");
        let p = resolve_config_path(
            &args(&[]),
            None,
            Some(xdg.clone()),
            only(&[
                "/x/.config/repo-explorer/repo-explorer.toml",
                "./repo-explorer.toml",
            ]),
        );
        assert_eq!(p, xdg);
    }

    #[test]
    fn existing_cwd_config_is_reachable_when_xdg_default_is_absent() {
        // Regression guard: the launch-directory fallback must not be dead
        // just because a per-user config dir resolves.
        let xdg = PathBuf::from("/x/.config/repo-explorer/repo-explorer.toml");
        let p = resolve_config_path(&args(&[]), None, Some(xdg), only(&["./repo-explorer.toml"]));
        assert_eq!(p, PathBuf::from("./repo-explorer.toml"));
    }

    #[test]
    fn xdg_default_is_the_target_when_no_config_exists_anywhere() {
        let xdg = PathBuf::from("/x/.config/repo-explorer/repo-explorer.toml");
        let p = resolve_config_path(&args(&[]), None, Some(xdg.clone()), nothing_exists);
        assert_eq!(p, xdg, "the wizard must be pointed at the per-user path");
    }

    #[test]
    fn cwd_fallback_when_no_xdg() {
        let p = resolve_config_path(&args(&[]), None, None, nothing_exists);
        assert_eq!(p, PathBuf::from("./repo-explorer.toml"));
    }

    #[test]
    fn env_var_beats_xdg_default() {
        let xdg = PathBuf::from("/x/.config/repo-explorer/repo-explorer.toml");
        let p = resolve_config_path(
            &args(&[]),
            Some("env.toml".into()),
            Some(xdg),
            only(&["/x/.config/repo-explorer/repo-explorer.toml"]),
        );
        assert_eq!(p, PathBuf::from("env.toml"));
    }

    #[test]
    fn wants_config_test_truth_table() {
        assert!(wants_config_test(&["--config-test".to_string()]));
        assert!(wants_config_test(&[
            "config".to_string(),
            "test".to_string()
        ]));
        assert!(wants_config_test(&[
            "--config".to_string(),
            "c.toml".to_string(),
            "config".to_string(),
            "test".to_string()
        ]));
        assert!(!wants_config_test(&["config".to_string()]));
        assert!(!wants_config_test(&["test".to_string()]));
        assert!(!wants_config_test(&[]));
        assert!(!wants_config_test(&[
            "test".to_string(),
            "config".to_string()
        ]));
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
