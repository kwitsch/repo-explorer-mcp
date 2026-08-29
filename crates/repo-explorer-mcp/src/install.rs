//! `--install` / `--uninstall` CLI modes.
//!
//! One-shot, non-interactive self-integration with Claude Code. `--install`
//! registers this binary as a user-scoped MCP server (via the real `claude`
//! CLI, never by hand-editing `~/.claude.json`) and writes an `explore`
//! subagent to `<home>/.claude/agents/explore.md`; `--uninstall` reverses
//! exactly those two changes, idempotently. Dispatched before config
//! resolution, so neither flag ever loads or creates `repo-explorer.toml`.
//! stdout carries only the final JSON report; all other diagnostics go to
//! stderr. Install checked before uninstall, so `--install` wins if both are
//! passed.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// The MCP server registration name and the root of the agent's
/// `tools: mcp__<name>` scoping string — one source of truth so the two can
/// never drift.
const SERVER_NAME: &str = "repo-explorer-mcp";

/// True when `--install` is present among the (config-value-stripped) args.
pub fn wants_install(args: &[String]) -> bool {
    crate::has_flag(args, &["--install"])
}

/// True when `--uninstall` is present among the (config-value-stripped) args.
pub fn wants_uninstall(args: &[String]) -> bool {
    crate::has_flag(args, &["--uninstall"])
}

/// Locate the `claude` CLI on PATH (honoring PATHEXT on Windows). `None`
/// means Claude Code is treated as not installed.
fn claude_binary() -> Option<PathBuf> {
    which::which("claude").ok()
}

/// `<home>/.claude/agents/explore.md`, or an error when no home dir resolves.
/// The `Option<PathBuf>` seam lets tests compute the path without a real HOME.
fn explore_agent_path_in(home: Option<PathBuf>) -> Result<PathBuf> {
    let home = home.ok_or_else(|| {
        anyhow!("no home directory available to place the explore agent (is HOME set?)")
    })?;
    Ok(home.join(".claude").join("agents").join("explore.md"))
}

/// The real agent path, keyed off `dirs::home_dir()`. Claude Code's dir is
/// always literally `<home>/.claude` (not XDG-derived), so `home_dir()` — not
/// `config_dir()`/`data_dir()` — is correct here.
fn explore_agent_path() -> Result<PathBuf> {
    explore_agent_path_in(dirs::home_dir())
}

/// The full markdown+frontmatter document written to the agent file. The
/// `tools:` line is derived from `SERVER_NAME`, so it always matches the
/// registered server name.
fn explore_agent_markdown() -> String {
    format!(
        r#"---
name: explore
description: Locate code and answer "where/how is X implemented" questions about the current repository by delegating to the repo-explorer-mcp explore_repository tool. Use for fast, read-only codebase exploration.
tools: mcp__{SERVER_NAME}
model: haiku
---

You are a focused repository-exploration agent. Use the `explore_repository`
tool from the repo-explorer-mcp MCP server to answer questions about where and
how functionality is implemented in the current repository.

Prefer a single well-scoped `explore_repository` call over guessing. Report the
concrete file paths and symbols it returns, with a short explanation of how they
fit together. Do not modify files.
"#
    )
}

/// argv for `claude mcp add repo-explorer-mcp --scope user -- <exe>`. The `--`
/// guard separates the stdio command from `claude`'s own flags.
fn mcp_add_args(exe: &Path) -> Vec<String> {
    vec![
        "mcp".to_string(),
        "add".to_string(),
        SERVER_NAME.to_string(),
        "--scope".to_string(),
        "user".to_string(),
        "--".to_string(),
        exe.display().to_string(),
    ]
}

/// argv for `claude mcp remove repo-explorer-mcp --scope user`.
fn mcp_remove_args() -> Vec<String> {
    vec![
        "mcp".to_string(),
        "remove".to_string(),
        SERVER_NAME.to_string(),
        "--scope".to_string(),
        "user".to_string(),
    ]
}

/// Invoke `claude` with `args`, bounded by [`crate::update::SUBPROCESS_TIMEOUT`]
/// so a hung CLI cannot block the flag forever. Errors on spawn failure or
/// timeout; otherwise yields the `Output` for the caller to interpret.
fn run_claude(claude: &Path, args: &[String]) -> Result<std::process::Output> {
    let mut command = std::process::Command::new(claude);
    command.args(args);
    crate::update::run_with_timeout(command, crate::update::SUBPROCESS_TIMEOUT).ok_or_else(|| {
        anyhow!(
            "`claude {}` failed to run or did not exit within {:?}",
            args.join(" "),
            crate::update::SUBPROCESS_TIMEOUT
        )
    })
}

/// Trimmed combination of a subprocess's stdout and stderr, for surfacing the
/// `claude` CLI's own diagnostics verbatim in a step's `detail`.
fn combined_output(output: &std::process::Output) -> String {
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text.trim().to_string()
}

#[derive(serde::Serialize)]
struct InstallReport {
    status: &'static str,
    claude_code_detected: bool,
    /// Top-level diagnostic when no per-step detail applies — notably the
    /// claude-not-found `--install` short-circuit, where `steps` is empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    steps: Vec<StepReport>,
}

#[derive(serde::Serialize)]
struct StepReport {
    name: &'static str,
    action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

/// `--install`: register the user-scope MCP server and write the explore
/// agent. Fails fast (empty `steps`, non-zero exit) when `claude` is absent.
pub fn run_install() -> ExitCode {
    let Some(claude) = claude_binary() else {
        let report = InstallReport {
            status: "error",
            claude_code_detected: false,
            message: Some("Claude Code (`claude`) not found on PATH; install it first".to_string()),
            steps: Vec::new(),
        };
        crate::print_report(&report, "install");
        return ExitCode::FAILURE;
    };

    let steps = vec![install_mcp_server(&claude), install_agent_file()];

    let had_error = steps.iter().any(|s| s.action == "error");
    let report = InstallReport {
        status: if had_error { "error" } else { "ok" },
        claude_code_detected: true,
        message: None,
        steps,
    };
    crate::print_report(&report, "install");
    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Idempotent MCP registration: remove any prior entry (ignoring its status),
/// then add. The registered command is `std::env::current_exe()`.
fn install_mcp_server(claude: &Path) -> StepReport {
    let exe =
        match std::env::current_exe().context("failed to resolve the running executable's path") {
            Ok(p) => p,
            Err(e) => {
                return StepReport {
                    name: "mcp-server",
                    action: "error",
                    detail: Some(format!("{e:#}")),
                };
            }
        };

    // Remove-then-add: a missing prior entry is fine, so its exit status is
    // ignored.
    let _ = run_claude(claude, &mcp_remove_args());

    match run_claude(claude, &mcp_add_args(&exe)) {
        Ok(output) if output.status.success() => StepReport {
            name: "mcp-server",
            action: "installed",
            detail: Some(exe.display().to_string()),
        },
        Ok(output) => StepReport {
            name: "mcp-server",
            action: "error",
            detail: Some(combined_output(&output)),
        },
        Err(e) => StepReport {
            name: "mcp-server",
            action: "error",
            detail: Some(format!("{e:#}")),
        },
    }
}

/// Write (or overwrite) the explore agent file, creating `agents/` if needed.
fn install_agent_file() -> StepReport {
    let path = match explore_agent_path() {
        Ok(p) => p,
        Err(e) => {
            return StepReport {
                name: "agent-file",
                action: "error",
                detail: Some(format!("{e:#}")),
            };
        }
    };

    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return StepReport {
            name: "agent-file",
            action: "error",
            detail: Some(format!(
                "failed to create agent directory {}: {e}",
                parent.display()
            )),
        };
    }

    match std::fs::write(&path, explore_agent_markdown()) {
        Ok(()) => StepReport {
            name: "agent-file",
            action: "installed",
            detail: Some(path.display().to_string()),
        },
        Err(e) => StepReport {
            name: "agent-file",
            action: "error",
            detail: Some(format!("failed to write {}: {e}", path.display())),
        },
    }
}

/// `--uninstall`: reverse the two install steps. Tolerates an absent `claude`
/// (the MCP step is skipped) and still deletes the agent file.
pub fn run_uninstall() -> ExitCode {
    let claude = claude_binary();
    let steps = vec![
        uninstall_mcp_server(claude.as_deref()),
        uninstall_agent_file(),
    ];

    let had_error = steps.iter().any(|s| s.action == "error");
    let report = InstallReport {
        status: if had_error { "error" } else { "ok" },
        claude_code_detected: claude.is_some(),
        message: None,
        steps,
    };
    crate::print_report(&report, "uninstall");
    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Remove the user-scope MCP entry. A non-zero `claude mcp remove` (already
/// absent) or a spawn/timeout error is `skipped`, never `error` — teardown
/// never hard-fails here.
fn uninstall_mcp_server(claude: Option<&Path>) -> StepReport {
    let Some(claude) = claude else {
        return StepReport {
            name: "mcp-server",
            action: "skipped",
            detail: Some("claude not found on PATH; nothing to remove".to_string()),
        };
    };

    match run_claude(claude, &mcp_remove_args()) {
        Ok(output) if output.status.success() => StepReport {
            name: "mcp-server",
            action: "removed",
            detail: None,
        },
        Ok(output) => StepReport {
            name: "mcp-server",
            action: "skipped",
            detail: Some(combined_output(&output)),
        },
        Err(e) => StepReport {
            name: "mcp-server",
            action: "skipped",
            detail: Some(format!("{e:#}")),
        },
    }
}

/// Delete the explore agent file. Already-absent (`NotFound`) is `skipped`;
/// any other IO error is `error`.
fn uninstall_agent_file() -> StepReport {
    let path = match explore_agent_path() {
        Ok(p) => p,
        Err(e) => {
            return StepReport {
                name: "agent-file",
                action: "error",
                detail: Some(format!("{e:#}")),
            };
        }
    };

    match std::fs::remove_file(&path) {
        Ok(()) => StepReport {
            name: "agent-file",
            action: "removed",
            detail: Some(path.display().to_string()),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => StepReport {
            name: "agent-file",
            action: "skipped",
            detail: Some(format!("already absent: {}", path.display())),
        },
        Err(e) => StepReport {
            name: "agent-file",
            action: "error",
            detail: Some(format!("failed to remove {}: {e}", path.display())),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn wants_install_truth_table() {
        assert!(wants_install(&args(&["--install"])));
        assert!(wants_install(&args(&["--config", "x.toml", "--install"])));
        assert!(!wants_install(&args(&["--config"])));
        assert!(!wants_install(&args(&[])));
        assert!(!wants_install(&args(&["--uninstall"])));
    }

    #[test]
    fn wants_uninstall_truth_table() {
        assert!(wants_uninstall(&args(&["--uninstall"])));
        assert!(wants_uninstall(&args(&[
            "--config",
            "x.toml",
            "--uninstall"
        ])));
        assert!(!wants_uninstall(&args(&["--config"])));
        assert!(!wants_uninstall(&args(&[])));
        assert!(!wants_uninstall(&args(&["--install"])));
    }

    #[test]
    fn explore_agent_path_composes_under_home() {
        let p = explore_agent_path_in(Some(PathBuf::from("/home/u"))).unwrap();
        assert_eq!(
            p,
            PathBuf::from("/home/u")
                .join(".claude")
                .join("agents")
                .join("explore.md")
        );
    }

    #[test]
    fn explore_agent_path_errors_without_home() {
        assert!(explore_agent_path_in(None).is_err());
    }

    #[test]
    fn explore_agent_markdown_has_required_frontmatter() {
        let md = explore_agent_markdown();
        assert!(md.contains("name: explore"));
        assert!(md.contains("model: haiku"));
        assert!(md.contains("tools: mcp__repo-explorer-mcp"));
        // Exactly one frontmatter fence delimiter pair (open + close).
        assert_eq!(md.matches("---").count(), 2);
        assert!(md.starts_with("---\n"));
    }

    #[test]
    fn mcp_add_args_exact_argv() {
        let exe = PathBuf::from("/opt/bin/repo-explorer-mcp");
        assert_eq!(
            mcp_add_args(&exe),
            args(&[
                "mcp",
                "add",
                "repo-explorer-mcp",
                "--scope",
                "user",
                "--",
                "/opt/bin/repo-explorer-mcp"
            ])
        );
    }

    #[test]
    fn mcp_remove_args_exact_argv() {
        assert_eq!(
            mcp_remove_args(),
            args(&["mcp", "remove", "repo-explorer-mcp", "--scope", "user"])
        );
    }
}
