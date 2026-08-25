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

/// Spawn the process, capture stdout, and enforce the timeout.
///
/// Exit `0` and exit `1` (rg/rtk "no matches") are both success; exit `1`
/// simply yields empty output. Any other exit code (>= 2, or a signal) is a
/// real failure. A spawn failure or timeout maps to the matching `SearchError`.
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

    match output.status.code() {
        Some(0) | Some(1) => Ok(String::from_utf8(output.stdout)
            .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).into_owned())),
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
