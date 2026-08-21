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
/// (typed JSON); otherwise parse the concatenated text blocks as JSON. Any
/// failure becomes `MemoryError::Decode`.
pub(crate) fn decode_result(
    tool: &'static str,
    result: &CallToolResult,
) -> Result<Value, MemoryError> {
    if let Some(sc) = &result.structured_content {
        return Ok(sc.clone());
    }
    let text = text_of(result);
    if text.trim().is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&text).map_err(|e| MemoryError::Decode {
        tool,
        message: format!("response was neither structured nor valid JSON text: {e}"),
    })
}

/// Derive the project name from the repo root's final path component (matching
/// `index_repository`'s documented default of the directory name). Root or
/// non-UTF-8 paths error as `Transport`.
pub(crate) fn project_name(repo_root: &Path) -> Result<String, MemoryError> {
    repo_root
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            MemoryError::Transport(format!(
                "cannot derive project name from repo_root `{}`",
                repo_root.display()
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use repo_explorer_core::config::CodebaseMemoryConfig;
    use std::path::Path;

    fn cfg_endpoint() -> CodebaseMemoryConfig {
        CodebaseMemoryConfig {
            command: None,
            args: vec![],
            endpoint: Some("http://localhost:9999".to_string()),
            staleness_seconds: 3600,
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

    #[test]
    fn project_name_from_directory() {
        assert_eq!(
            project_name(Path::new("/home/user/my-repo")).unwrap(),
            "my-repo".to_string()
        );
    }

    #[test]
    fn project_name_root_path_errors() {
        let err = project_name(Path::new("/")).unwrap_err();
        assert!(matches!(err, MemoryError::Transport(_)));
    }
}
