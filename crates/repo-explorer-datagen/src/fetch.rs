//! The `fetch` subcommand: clone/fetch/checkout each corpus repo at its pinned
//! rev and verify a root license file. Git runs via `std::process::Command`
//! with inherited stderr. No LLM, no network beyond git.

use crate::corpus::{check_eval_exclusion, load_corpus, load_eval_urls, validate_corpus};
use std::path::Path;
use std::process::Command;

/// Run `fetch`. Returns the process exit code (0 ok, 1 invalid corpus or >=1
/// repo failed).
pub fn run_fetch(corpus_path: &Path, checkouts: &Path, eval_repos: &Path) -> u8 {
    let corpus = match load_corpus(corpus_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if let Err(e) = validate_corpus(&corpus) {
        eprintln!("{e}");
        return 1;
    }
    let eval_urls = match load_eval_urls(eval_repos) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if let Err(e) = check_eval_exclusion(&corpus, &eval_urls) {
        eprintln!("{e}");
        return 1;
    }

    let mut fetched: Vec<String> = Vec::new();
    let mut failed: Vec<(String, String)> = Vec::new();
    for repo in &corpus.repos {
        let dir = checkouts.join(&repo.name);
        match fetch_one(&repo.url, &repo.rev, &dir) {
            Ok(()) => fetched.push(repo.name.clone()),
            Err(e) => failed.push((repo.name.clone(), e)),
        }
    }

    let summary = serde_json::json!({
        "fetched": fetched,
        "failed": failed.iter().map(|(n, e)| serde_json::json!({"name": n, "error": e})).collect::<Vec<_>>(),
    });
    println!("{summary}");
    if failed.is_empty() { 0 } else { 1 }
}

fn fetch_one(url: &str, rev: &str, dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        run_git(
            &[
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                &dir.to_string_lossy(),
            ],
            None,
        )?;
    }
    run_git(&["fetch", "--filter=blob:none", "origin", rev], Some(dir))?;
    run_git(&["checkout", "--detach", "--force", rev], Some(dir))?;
    let head = git_output(&["rev-parse", "HEAD"], dir)?;
    if head.trim() != rev {
        return Err(format!(
            "HEAD is {} after checkout, expected {rev}",
            head.trim()
        ));
    }
    if !has_license_file(dir) {
        return Err("no root-level license file found".to_string());
    }
    Ok(())
}

fn run_git(args: &[&str], dir: Option<&Path>) -> Result<(), String> {
    let mut cmd = Command::new("git");
    if let Some(d) = dir {
        cmd.arg("-C").arg(d);
    }
    cmd.args(args);
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn git: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("git {} exited with {status}", args.join(" ")))
    }
}

fn git_output(args: &[&str], dir: &Path) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|e| format!("failed to spawn git: {e}"))?;
    if !out.status.success() {
        return Err(format!("git {} failed", args.join(" ")));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn has_license_file(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        let stem = name.split('.').next().unwrap_or(&name);
        if matches!(stem, "license" | "licence" | "copying" | "unlicense")
            || name.starts_with("license-")
        {
            return true;
        }
    }
    false
}
