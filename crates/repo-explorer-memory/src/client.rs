//! `rmcp` client plumbing: connect to `codebase-memory-mcp` over stdio, call a
//! tool, decode its result, and derive the project name from a repo root. All
//! `rmcp`/tool failures are mapped to `repo_explorer_core::memory::MemoryError`
//! here, so the rest of the crate — and all of core — stays `rmcp`-free at the
//! type level.

use repo_explorer_core::config::CodebaseMemoryConfig;
use repo_explorer_core::memory::MemoryError;
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde_json::{Map, Value};
use std::path::Path;

/// A connected `rmcp` client to `codebase-memory-mcp`.
#[derive(Debug)]
pub(crate) struct MemoryClient {
    service: RunningService<RoleClient, ()>,
}

impl MemoryClient {
    /// Connect over the configured transport. Stdio (`command`) only in Stage 2;
    /// a `endpoint` config yields `MemoryError::UnsupportedTransport`.
    /// Config validation guarantees exactly one of `command`/`endpoint` is set.
    pub(crate) async fn connect(config: &CodebaseMemoryConfig) -> Result<Self, MemoryError> {
        if config.command.is_some() && config.endpoint.is_some() {
            return Err(MemoryError::Transport(
                "codebase_memory config sets both `command` and `endpoint`; \
                 exactly one is required (Config::validate should have rejected this)"
                    .to_string(),
            ));
        }
        match &config.command {
            Some(cmd) => {
                let mut command = tokio::process::Command::new(cmd);
                command.args(&config.args);
                let transport = TokioChildProcess::new(command)
                    .map_err(|e| MemoryError::Transport(format!("failed to spawn `{cmd}`: {e}")))?;
                let service = ().serve(transport).await.map_err(|e| {
                    MemoryError::Transport(format!("failed to initialize MCP client: {e}"))
                })?;
                Ok(Self { service })
            }
            None => Err(MemoryError::UnsupportedTransport(
                "network endpoint".to_string(),
            )),
        }
    }

    /// Invoke a tool by name with the given JSON arguments. Maps transport
    /// failures to `Transport` and `is_error == Some(true)` to `ToolFailed`
    /// (message pulled from the result's text content).
    pub(crate) async fn call(
        &self,
        tool: &'static str,
        args: Map<String, Value>,
    ) -> Result<CallToolResult, MemoryError> {
        let params = CallToolRequestParams::new(tool).with_arguments(args);
        let result = self
            .service
            .peer()
            .call_tool(params)
            .await
            .map_err(|e| MemoryError::Transport(format!("tool `{tool}` call failed: {e}")))?;
        if result.is_error == Some(true) {
            return Err(MemoryError::ToolFailed {
                tool,
                message: text_of(&result),
            });
        }
        Ok(result)
    }

    /// Best-effort graceful shutdown of the child service.
    pub(crate) async fn close(&mut self) {
        let _ = self.service.close().await;
    }
}

/// Concatenate all text content blocks of a result into a single string.
fn text_of(result: &CallToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| c.as_text().map(|t| t.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Decode a successful tool result into a JSON value: prefer `structured_content`
/// (typed JSON); otherwise parse the concatenated text blocks as JSON; a
/// non-JSON text response is returned as `Value::String` so per-tool mappers
/// can parse plain-text table formats (`search_code` answers in one).
pub(crate) fn decode_result(
    _tool: &'static str,
    result: CallToolResult,
) -> Result<Value, MemoryError> {
    if let Some(sc) = result.structured_content {
        return Ok(sc);
    }
    let text = text_of(&result);
    if text.trim().is_empty() {
        return Ok(Value::Null);
    }
    Ok(serde_json::from_str(&text).unwrap_or(Value::String(text)))
}

/// Derive the project name from the repo root's final path component (matching
/// `index_repository`'s documented default of the directory name). Root or
/// non-UTF-8 paths error as `InvalidInput` — a bad `repo_root` is not a
/// transport/connection failure, so callers must not treat it as retryable.
///
/// Canonicalizes `repo_root` first — same normalization `run_index` applies
/// before calling `index_repository` — so a relative value like `.` resolves
/// to its real directory name instead of erroring out immediately.
pub(crate) async fn project_name(repo_root: &Path) -> Result<String, MemoryError> {
    let abs = canonicalize_repo_root(repo_root).await;
    project_name_from_abs(repo_root, &abs)
}

/// Derive the project name from an already-canonicalized path, without
/// canonicalizing again. Shared by `project_name` and by call sites (like
/// `ensure_fresh_index`) that need both the project name and the
/// canonicalized path itself and must not `canonicalize` twice for one
/// logical resolution.
pub(crate) fn project_name_from_abs(repo_root: &Path, abs: &Path) -> Result<String, MemoryError> {
    abs.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            MemoryError::InvalidInput(format!(
                "cannot derive project name from repo_root `{}`",
                repo_root.display()
            ))
        })
}

/// Canonicalize `repo_root` off the async runtime thread (the blocking
/// `std::fs::canonicalize` syscall runs via `spawn_blocking`), falling back
/// to the original path unchanged if canonicalization fails for any reason
/// (missing path, non-UTF8 quirks, or the blocking task itself failing).
pub(crate) async fn canonicalize_repo_root(repo_root: &Path) -> std::path::PathBuf {
    let owned = repo_root.to_path_buf();
    tokio::task::spawn_blocking(move || std::fs::canonicalize(&owned).unwrap_or(owned))
        .await
        .unwrap_or_else(|_| repo_root.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::config::{CodebaseMemoryConfig, default_staleness_seconds};
    use std::path::Path;

    fn cfg_endpoint() -> CodebaseMemoryConfig {
        CodebaseMemoryConfig {
            command: None,
            args: vec![],
            endpoint: Some("http://localhost:9999".to_string()),
            staleness_seconds: default_staleness_seconds(),
        }
    }

    #[tokio::test]
    async fn endpoint_transport_is_unsupported() {
        let err = MemoryClient::connect(&cfg_endpoint()).await.unwrap_err();
        assert_eq!(
            err,
            MemoryError::UnsupportedTransport("network endpoint".to_string())
        );
    }

    #[tokio::test]
    async fn project_name_from_directory() {
        assert_eq!(
            project_name(Path::new("/home/user/my-repo")).await.unwrap(),
            "my-repo".to_string()
        );
    }

    #[tokio::test]
    async fn project_name_root_path_errors() {
        let err = project_name(Path::new("/")).await.unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));
    }
}
