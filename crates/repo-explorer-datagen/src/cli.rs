//! Hand-rolled `--flag value` CLI parsing (no clap). A usage error becomes a
//! `UsageError` the binary prints to stderr before exiting 2.

use std::path::PathBuf;

pub const USAGE: &str = "\
repo-explorer-datagen — dev-only Laya judge training-data generator

USAGE:
  repo-explorer-datagen fetch    --corpus <file> --checkouts <dir>
  repo-explorer-datagen generate --corpus <file> --checkouts <dir> --out <dir> --memory-command <path>
                                 [--memory-arg <arg>]... [--eval-repos <file>]
                                 [--max-queries-per-repo <n>] [--max-negatives-per-query <n>]
                                 [--seed <u64>] [--only <repo-name>]...
  repo-explorer-datagen stats    --data <dir>
  repo-explorer-datagen --help | -h
";

#[derive(Debug)]
pub struct UsageError {
    pub message: String,
}

fn usage(reason: &str) -> UsageError {
    UsageError {
        message: format!("error: {reason}\n\n{USAGE}"),
    }
}

pub enum Command {
    Help,
    Fetch(FetchArgs),
    Generate(GenerateArgs),
    Stats(StatsArgs),
}

pub struct FetchArgs {
    pub corpus: PathBuf,
    pub checkouts: PathBuf,
}

pub struct GenerateArgs {
    pub corpus: PathBuf,
    pub checkouts: PathBuf,
    pub out: PathBuf,
    pub memory_command: String,
    pub memory_args: Vec<String>,
    pub eval_repos: PathBuf,
    pub max_queries_per_repo: usize,
    pub max_negatives_per_query: usize,
    pub seed: u64,
    pub only: Vec<String>,
}

pub struct StatsArgs {
    pub data: PathBuf,
}

pub fn parse_cli(args: &[String]) -> Result<Command, UsageError> {
    let Some((sub, rest)) = args.split_first() else {
        return Err(usage("missing subcommand"));
    };
    match sub.as_str() {
        "--help" | "-h" => Ok(Command::Help),
        "fetch" => parse_fetch(rest),
        "generate" => parse_generate(rest),
        "stats" => parse_stats(rest),
        other => Err(usage(&format!("unknown subcommand `{other}`"))),
    }
}

fn flag_pairs(args: &[String]) -> Result<Vec<(&str, &str)>, UsageError> {
    let mut pairs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        if !flag.starts_with("--") {
            return Err(usage(&format!("unexpected argument `{flag}`")));
        }
        let Some(value) = args.get(i + 1) else {
            return Err(usage(&format!("flag `{flag}` needs a value")));
        };
        pairs.push((flag.as_str(), value.as_str()));
        i += 2;
    }
    Ok(pairs)
}

fn parse_fetch(args: &[String]) -> Result<Command, UsageError> {
    let (mut corpus, mut checkouts) = (None, None);
    for (flag, value) in flag_pairs(args)? {
        match flag {
            "--corpus" => corpus = Some(PathBuf::from(value)),
            "--checkouts" => checkouts = Some(PathBuf::from(value)),
            other => return Err(usage(&format!("unknown flag `{other}` for fetch"))),
        }
    }
    Ok(Command::Fetch(FetchArgs {
        corpus: corpus.ok_or_else(|| usage("fetch requires --corpus"))?,
        checkouts: checkouts.ok_or_else(|| usage("fetch requires --checkouts"))?,
    }))
}

fn parse_generate(args: &[String]) -> Result<Command, UsageError> {
    let (mut corpus, mut checkouts, mut out, mut memory_command) = (None, None, None, None);
    let mut memory_args: Vec<String> = Vec::new();
    let mut eval_repos = None;
    let (mut max_q, mut max_neg, mut seed) = (400usize, 4usize, 20260923u64);
    let mut only: Vec<String> = Vec::new();
    for (flag, value) in flag_pairs(args)? {
        match flag {
            "--corpus" => corpus = Some(PathBuf::from(value)),
            "--checkouts" => checkouts = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--memory-command" => memory_command = Some(value.to_string()),
            "--memory-arg" => memory_args.push(value.to_string()),
            "--eval-repos" => eval_repos = Some(PathBuf::from(value)),
            "--max-queries-per-repo" => {
                max_q = value
                    .parse()
                    .map_err(|_| usage("--max-queries-per-repo must be an integer"))?
            }
            "--max-negatives-per-query" => {
                max_neg = value
                    .parse()
                    .map_err(|_| usage("--max-negatives-per-query must be an integer"))?
            }
            "--seed" => seed = value.parse().map_err(|_| usage("--seed must be a u64"))?,
            "--only" => only.push(value.to_string()),
            other => return Err(usage(&format!("unknown flag `{other}` for generate"))),
        }
    }
    if memory_args.is_empty() {
        memory_args.push("--stdio".to_string());
    }
    Ok(Command::Generate(GenerateArgs {
        corpus: corpus.ok_or_else(|| usage("generate requires --corpus"))?,
        checkouts: checkouts.ok_or_else(|| usage("generate requires --checkouts"))?,
        out: out.ok_or_else(|| usage("generate requires --out"))?,
        memory_command: memory_command
            .ok_or_else(|| usage("generate requires --memory-command"))?,
        memory_args,
        eval_repos: eval_repos.unwrap_or_else(|| PathBuf::from("eval/repos.toml")),
        max_queries_per_repo: max_q,
        max_negatives_per_query: max_neg,
        seed,
        only,
    }))
}

fn parse_stats(args: &[String]) -> Result<Command, UsageError> {
    let mut data = None;
    for (flag, value) in flag_pairs(args)? {
        match flag {
            "--data" => data = Some(PathBuf::from(value)),
            other => return Err(usage(&format!("unknown flag `{other}` for stats"))),
        }
    }
    Ok(Command::Stats(StatsArgs {
        data: data.ok_or_else(|| usage("stats requires --data"))?,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn help_parses() {
        assert!(matches!(parse_cli(&v(&["--help"])), Ok(Command::Help)));
        assert!(matches!(parse_cli(&v(&["-h"])), Ok(Command::Help)));
    }

    #[test]
    fn fetch_requires_corpus_and_checkouts() {
        assert!(parse_cli(&v(&["fetch"])).is_err());
        assert!(parse_cli(&v(&["fetch", "--corpus", "c.toml"])).is_err());
        assert!(matches!(
            parse_cli(&v(&["fetch", "--corpus", "c.toml", "--checkouts", "co"])),
            Ok(Command::Fetch(_))
        ));
    }

    #[test]
    fn generate_applies_defaults() {
        let cmd = parse_cli(&v(&[
            "generate",
            "--corpus",
            "c.toml",
            "--checkouts",
            "co",
            "--out",
            "o",
            "--memory-command",
            "cbm",
        ]))
        .unwrap();
        let Command::Generate(g) = cmd else {
            panic!("expected generate")
        };
        assert_eq!(g.memory_args, vec!["--stdio".to_string()]);
        assert_eq!(g.eval_repos, std::path::PathBuf::from("eval/repos.toml"));
        assert_eq!(g.max_queries_per_repo, 400);
        assert_eq!(g.max_negatives_per_query, 4);
        assert_eq!(g.seed, 20260923);
        assert!(g.only.is_empty());
    }

    #[test]
    fn generate_collects_repeatables_and_overrides() {
        let cmd = parse_cli(&v(&[
            "generate",
            "--corpus",
            "c",
            "--checkouts",
            "co",
            "--out",
            "o",
            "--memory-command",
            "cbm",
            "--memory-arg",
            "--stdio",
            "--memory-arg",
            "--verbose",
            "--only",
            "chi",
            "--only",
            "zod",
            "--seed",
            "5",
            "--max-queries-per-repo",
            "10",
        ]))
        .unwrap();
        let Command::Generate(g) = cmd else {
            panic!("expected generate")
        };
        assert_eq!(
            g.memory_args,
            vec!["--stdio".to_string(), "--verbose".to_string()]
        );
        assert_eq!(g.only, vec!["chi".to_string(), "zod".to_string()]);
        assert_eq!(g.seed, 5);
        assert_eq!(g.max_queries_per_repo, 10);
    }

    #[test]
    fn usage_errors() {
        assert!(parse_cli(&v(&["frobnicate"])).is_err());
        assert!(parse_cli(&v(&["stats"])).is_err());
        assert!(parse_cli(&v(&["stats", "--nope", "x"])).is_err());
        assert!(parse_cli(&v(&["fetch", "--corpus"])).is_err()); // dangling flag
        assert!(matches!(
            parse_cli(&v(&["stats", "--data", "d"])),
            Ok(Command::Stats(_))
        ));
    }
}
