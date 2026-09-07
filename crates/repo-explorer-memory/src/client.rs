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
use sha2::{Digest, Sha256};
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
/// (typed JSON); otherwise parse the concatenated text blocks as JSON; a
/// non-JSON text response is returned as `Value::String` so per-tool mappers
/// can parse plain-text table formats (`search_code` answers in one).
pub(crate) fn decode_result(result: CallToolResult) -> Result<Value, MemoryError> {
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
    let base = abs.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        MemoryError::InvalidInput(format!(
            "cannot derive project name from repo_root `{}`",
            repo_root.display()
        ))
    })?;
    // Hash the CANONICAL abs path (not the raw repo_root): `.` and the
    // absolute spelling of one repo must resolve to one stable upstream
    // project, the opposite of the raw-path in-process cache keys. 8 hex chars
    // (32 bits) is plenty to keep basename collisions across many repos
    // astronomically unlikely; it is not a security boundary (a collision only
    // costs a shared index), so a truncated prefix is fine.
    let hash = hex::encode(Sha256::digest(abs.to_string_lossy().as_bytes()));
    Ok(format!("{base}-{}", &hash[..8]))
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
        // The name is now `{basename}-{8 hex chars of Sha256 over the
        // canonical abs path}` so two different repos that share a basename
        // never collide upstream. `/home/user/my-repo` does not exist, so
        // canonicalize falls back to the raw path and the hash is taken over
        // that exact string.
        let name = project_name(Path::new("/home/user/my-repo")).await.unwrap();
        assert!(
            name.starts_with("my-repo-"),
            "name must keep the human-readable basename prefix: {name}"
        );
        let suffix = name.strip_prefix("my-repo-").unwrap();
        assert_eq!(suffix.len(), 8, "hash suffix must be 8 hex chars: {name}");
        assert!(
            suffix.chars().all(|c| c.is_ascii_hexdigit()),
            "hash suffix must be hex: {name}"
        );
    }

    #[tokio::test]
    async fn project_name_disambiguates_shared_basename_across_parents() {
        // Two distinct repositories that share a directory basename must
        // derive DIFFERENT project names (the whole point of the hash suffix).
        // Neither path exists, so canonicalize falls back to the raw path and
        // the hash is taken over the distinct full paths.
        let a = project_name(Path::new("/home/a/my-repo")).await.unwrap();
        let b = project_name(Path::new("/home/b/my-repo")).await.unwrap();
        assert!(a.starts_with("my-repo-") && b.starts_with("my-repo-"));
        assert_ne!(
            a, b,
            "same basename under different parents must not collide"
        );
    }

    #[tokio::test]
    async fn project_name_root_path_errors() {
        let err = project_name(Path::new("/")).await.unwrap_err();
        assert!(matches!(err, MemoryError::InvalidInput(_)));
    }
}
