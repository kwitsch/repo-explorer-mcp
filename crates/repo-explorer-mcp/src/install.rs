//! `--install` / `--uninstall` CLI modes.
//!
//! One-shot, non-interactive self-integration with Claude Code. `--install`
//! registers this binary as a user-scoped MCP server (via the real `claude`
//! CLI, never by hand-editing `~/.claude.json`) and writes an `explore`
//! subagent to `<home>/.claude/agents/explore.md`; `--uninstall` reverses
//! those two changes idempotently, but only deletes the agent file if its
//! contents still match what install wrote. Dispatched before config
//! resolution, so neither flag ever loads or creates `repo-explorer.toml`.
//! stdout carries only the final JSON report; all other diagnostics go to
//! stderr. Install checked before uninstall, so `--install` wins if both are
//! passed.

use anyhow::{Context, Result, anyhow};
use std::ffi::OsString;
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
/// guard separates the stdio command from `claude`'s own flags. `OsString`
/// (not `String`) so a non-UTF-8 executable path is passed through verbatim
/// instead of being lossily mangled to `U+FFFD`.
fn mcp_add_args(exe: &Path) -> Vec<OsString> {
    vec![
        OsString::from("mcp"),
        OsString::from("add"),
        OsString::from(SERVER_NAME),
        OsString::from("--scope"),
        OsString::from("user"),
        OsString::from("--"),
        exe.as_os_str().to_os_string(),
    ]
}

/// argv for `claude mcp remove repo-explorer-mcp --scope user`.
fn mcp_remove_args() -> Vec<OsString> {
    vec![
        OsString::from("mcp"),
        OsString::from("remove"),
        OsString::from(SERVER_NAME),
        OsString::from("--scope"),
        OsString::from("user"),
    ]
}

/// Invoke `claude` with `args`, bounded by [`crate::update::SUBPROCESS_TIMEOUT`]
/// so a hung CLI cannot block the flag forever. Errors on spawn failure or
/// timeout; otherwise yields the `Output` for the caller to interpret.
fn run_claude(claude: &Path, args: &[OsString]) -> Result<std::process::Output> {
    let mut command = std::process::Command::new(claude);
    command.args(args);
    crate::update::run_with_timeout(command, crate::update::SUBPROCESS_TIMEOUT).ok_or_else(|| {
        anyhow!(
            "`claude {}` failed to run or did not exit within {:?}",
            args.iter()
                .map(|a| a.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" "),
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

/// Build a `StepReport` named `name` from a `run_claude` result: a clean exit
/// maps to `success_action` (with `success_detail`); a non-zero exit or a
/// spawn/timeout failure both map to `failure_action`, with the CLI's own
/// combined output (or the error) surfaced verbatim in `detail`. Shared tail
/// of `install_mcp_server`/`uninstall_mcp_server`, which differ only in these
/// action strings and whether a failure is reported as `"error"` or
/// `"skipped"`.
fn claude_step_result(
    result: Result<std::process::Output>,
    name: &'static str,
    success_action: &'static str,
    success_detail: Option<String>,
    failure_action: &'static str,
) -> StepReport {
    match result {
        Ok(output) if output.status.success() => StepReport {
            name,
            action: success_action,
            detail: success_detail,
        },
        Ok(output) => StepReport {
            name,
            action: failure_action,
            detail: Some(combined_output(&output)),
        },
        Err(e) => StepReport {
            name,
            action: failure_action,
            detail: Some(format!("{e:#}")),
        },
    }
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

    claude_step_result(
        run_claude(claude, &mcp_add_args(&exe)),
        "mcp-server",
        "installed",
        Some(exe.display().to_string()),
        "error",
    )
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

    if let Err(e) = crate::ensure_parent_dir(&path) {
        return StepReport {
            name: "agent-file",
            action: "error",
            detail: Some(format!(
                "failed to create agent directory {}: {e}",
                path.parent().unwrap_or(&path).display()
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
/// (the MCP step is skipped) and only deletes the agent file if its contents
/// still match what install wrote.
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

    claude_step_result(
        run_claude(claude, &mcp_remove_args()),
        "mcp-server",
        "removed",
        None,
        "skipped",
    )
}

/// Delete the explore agent file, but only if its contents still match what
/// `install_agent_file` wrote. Already-absent (`NotFound`) is `skipped`, as is
/// a file whose contents were replaced or hand-edited after install (so a
/// user's own agent at this conventional path is never silently destroyed).
/// Any other IO error is `error`.
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
    uninstall_agent_file_at(&path)
}

/// The path-parameterized core of [`uninstall_agent_file`], split out so
/// tests can exercise the content-match guard against a temp file instead of
/// the real `<home>/.claude/agents/explore.md`.
fn uninstall_agent_file_at(path: &Path) -> StepReport {
    match std::fs::read_to_string(path) {
        Ok(contents) if contents == explore_agent_markdown() => match std::fs::remove_file(path) {
            Ok(()) => StepReport {
                name: "agent-file",
                action: "removed",
                detail: Some(path.display().to_string()),
            },
            Err(e) => StepReport {
                name: "agent-file",
                action: "error",
                detail: Some(format!("failed to remove {}: {e}", path.display())),
            },
        },
        Ok(_) => StepReport {
            name: "agent-file",
            action: "skipped",
            detail: Some(format!(
                "{} was modified after install; leaving it in place",
                path.display()
            )),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => StepReport {
            name: "agent-file",
            action: "skipped",
            detail: Some(format!("already absent: {}", path.display())),
        },
        Err(e) => StepReport {
            name: "agent-file",
            action: "error",
            detail: Some(format!("failed to read {}: {e}", path.display())),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn os_args(v: &[&str]) -> Vec<OsString> {
        v.iter().map(OsString::from).collect()
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

    /// A path under the OS temp dir unique to this test invocation, so
    /// parallel test runs never collide.
    fn unique_temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "repo-explorer-mcp-test-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn uninstall_agent_file_at_removes_matching_content() {
        let path = unique_temp_path("uninstall-match");
        std::fs::write(&path, explore_agent_markdown()).unwrap();
        let report = uninstall_agent_file_at(&path);
        assert_eq!(report.action, "removed");
        assert!(!path.exists());
    }

    #[test]
    fn uninstall_agent_file_at_skips_modified_content() {
        let path = unique_temp_path("uninstall-modified");
        std::fs::write(&path, "not what install wrote").unwrap();
        let report = uninstall_agent_file_at(&path);
        assert_eq!(report.action, "skipped");
        // The user's own file at this path must survive.
        assert!(path.exists());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn uninstall_agent_file_at_skips_already_absent() {
        let path = unique_temp_path("uninstall-absent");
        let report = uninstall_agent_file_at(&path);
        assert_eq!(report.action, "skipped");
    }

    #[test]
    fn mcp_add_args_exact_argv() {
        let exe = PathBuf::from("/opt/bin/repo-explorer-mcp");
        assert_eq!(
            mcp_add_args(&exe),
            os_args(&[
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
    #[cfg(unix)]
    fn mcp_add_args_preserves_non_utf8_exe_path() {
        use std::os::unix::ffi::OsStrExt;

        let exe = PathBuf::from(std::ffi::OsStr::from_bytes(b"/opt/bin/repo\xFFexplorer"));
        let got = mcp_add_args(&exe);
        assert_eq!(got.last().unwrap(), exe.as_os_str());
    }

    #[test]
    fn mcp_remove_args_exact_argv() {
        assert_eq!(
            mcp_remove_args(),
            os_args(&["mcp", "remove", "repo-explorer-mcp", "--scope", "user"])
        );
    }
}
