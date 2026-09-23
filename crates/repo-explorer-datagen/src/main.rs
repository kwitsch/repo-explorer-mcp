//! Thin binary: parse the CLI and dispatch into the library.

use repo_explorer_datagen::cli::{Command, USAGE, parse_cli};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    init_tracing();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = match parse_cli(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e.message);
            return ExitCode::from(2);
        }
    };
    let code = match command {
        Command::Help => {
            println!("{USAGE}");
            0
        }
        Command::Fetch(a) => repo_explorer_datagen::fetch::run_fetch(
            &a.corpus,
            &a.checkouts,
            Path::new("eval/repos.toml"),
        ),
        Command::Stats(a) => repo_explorer_datagen::stats::run_stats(&a.data),
        Command::Generate(a) => {
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            rt.block_on(repo_explorer_datagen::generate::run_generate(&a))
        }
    };
    ExitCode::from(code)
}

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}
