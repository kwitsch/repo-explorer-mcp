//! Subprocess spawning with a hard timeout. Uses `tokio::process` with
//! `kill_on_drop(true)`, so a timed-out child is reaped when the (dropped)
//! `wait_with_output` future releases it — no orphaned process.

use repo_explorer_core::search::SearchError;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

pub(crate) struct SpawnSpec {
    pub backend: &'static str,
    pub program: PathBuf,
    pub args: Vec<String>,
    pub cwd: PathBuf,
    pub timeout: Duration,
}

/// True when `stdout` is exactly one line and that line is `rg --json`'s
/// summary trailer (`{"data":{...},"type":"summary"}`). `rg --json` emits
/// this trailer even on a total failure -- e.g. the target path doesn't
/// exist -- where it performed zero searches, so a lone trailer line is
/// vacuous, not evidence of a real (if partial) result. A genuine partial
/// walk (a match found alongside an unrelated permission-denied path
/// elsewhere in the tree) always has at least one `begin`/`match`/`end`
/// event ahead of the trailer, so it never matches this check.
fn is_vacuous_rg_summary_only(stdout: &str) -> bool {
    let mut lines = stdout.lines().filter(|line| !line.trim().is_empty());
    matches!(lines.next(), Some(line) if line.contains("\"type\":\"summary\""))
        && lines.next().is_none()
}

/// Spawn the process, capture stdout, and enforce the timeout.
///
/// Exit `0` and exit `1` (rg/rtk "no matches") are both success; exit `1`
/// simply yields empty output. Exit `2` with non-empty stdout is also
/// success: rg/rtk use it for a partial-error walk (e.g. a match found
/// alongside an unrelated permission-denied path elsewhere in the tree),
/// and stdout already holds the complete, valid result in that case --
/// unless that stdout is itself only `rg --json`'s vacuous summary trailer
/// (see `is_vacuous_rg_summary_only`), which means no search actually ran.
/// Any other exit code (a bare `2`, `>= 3`, or a signal) is a real failure.
/// A spawn failure or timeout maps to the matching `SearchError`.
pub(crate) async fn run(spec: &SpawnSpec) -> Result<String, SearchError> {
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args)
        .current_dir(&spec.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let child = cmd.spawn().map_err(|e| SearchError::BackendFailed {
        backend: spec.backend,
        message: format!("failed to spawn: {e}"),
    })?;

    // `timeout_seconds = 0` is the explicit "no timeout" opt-out: await the
    // child directly rather than firing an instantaneous timeout.
    let wait = child.wait_with_output();
    let output = if spec.timeout.is_zero() {
        wait.await
    } else {
        // Timed out: dropping the future here drops the child, and
        // kill_on_drop(true) reaps it — no orphan is left behind.
        tokio::time::timeout(spec.timeout, wait)
            .await
            .map_err(|_| SearchError::Timeout {
                backend: spec.backend,
                seconds: spec.timeout.as_secs(),
            })?
    }
    .map_err(|e| SearchError::BackendFailed {
        backend: spec.backend,
        message: format!("process error: {e}"),
    })?;

    let stdout = String::from_utf8(output.stdout)
        .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).into_owned());

    match output.status.code() {
        Some(0) | Some(1) => Ok(stdout),
        Some(2) if !stdout.is_empty() && !is_vacuous_rg_summary_only(&stdout) => Ok(stdout),
        other => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let code_str = other
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".to_string());
            Err(SearchError::BackendFailed {
                backend: spec.backend,
                message: format!("exit {code_str}: {}", stderr.trim()),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vacuous_summary_only_stdout_is_detected() {
        // Captured verbatim from `rg --json -H -S -- needle nonexistent_dir`,
        // which exits 2 having performed zero searches.
        let stdout = "{\"data\":{\"elapsed_total\":{\"human\":\"0.000889s\",\"nanos\":889018,\"secs\":0},\"stats\":{\"bytes_printed\":0,\"bytes_searched\":0,\"elapsed\":{\"human\":\"0.000000s\",\"nanos\":0,\"secs\":0},\"matched_lines\":0,\"matches\":0,\"searches\":0,\"searches_with_match\":0}},\"type\":\"summary\"}\n";
        assert!(is_vacuous_rg_summary_only(stdout));
    }

    #[test]
    fn summary_preceded_by_real_events_is_not_vacuous() {
        // A genuine partial-error walk: a real match plus the trailing
        // summary must not be mistaken for a vacuous, zero-search stdout.
        let stdout = concat!(
            "{\"type\":\"begin\",\"data\":{\"path\":{\"text\":\"a.rs\"}}}\n",
            "{\"type\":\"match\",\"data\":{\"path\":{\"text\":\"a.rs\"},\"lines\":{\"text\":\"needle\\n\"},\"line_number\":1}}\n",
            "{\"type\":\"end\",\"data\":{\"path\":{\"text\":\"a.rs\"}}}\n",
            "{\"data\":{},\"type\":\"summary\"}\n",
        );
        assert!(!is_vacuous_rg_summary_only(stdout));
    }

    #[test]
    fn empty_stdout_is_not_vacuous_summary() {
        // Distinct case (already caught by the `!stdout.is_empty()` guard at
        // the call site) but the helper itself must not misclassify it.
        assert!(!is_vacuous_rg_summary_only(""));
    }

    #[test]
    fn non_summary_single_line_is_not_vacuous() {
        // An rtk raw match line never carries a `"type":"summary"` marker.
        assert!(!is_vacuous_rg_summary_only("src/x.rs:5:hello\n"));
    }
}
