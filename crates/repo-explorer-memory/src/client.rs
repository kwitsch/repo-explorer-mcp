//! `rmcp` client plumbing: connect to `codebase-memory-mcp` over stdio, call a
//! tool, decode its result, and resolve a repo root to the upstream project
//! name. All `rmcp`/tool failures are mapped to
//! `repo_explorer_core::memory::MemoryError` here, so the rest of the crate —
//! and all of core — stays `rmcp`-free at the type level.

use repo_explorer_core::config::CodebaseMemoryConfig;
use repo_explorer_core::memory::MemoryError;
use rmcp::ServiceExt;
use rmcp::model::{CallToolRequestParams, CallToolResult};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use serde_json::{Map, Value};
use std::path::Path;

/// `list_projects` page size. CBM's default is 50; one page covers every
/// realistic per-user daemon, the loop below follows `has_more` regardless.
const LIST_PROJECTS_PAGE: u64 = 200;
/// Hard stop for the paging loop so a misbehaving server that always reports
/// `has_more` cannot spin it forever.
const LIST_PROJECTS_MAX_ROWS: usize = 10_000;

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

    /// Resolve the already-canonicalized repo root `abs` to the name of the
    /// upstream project indexed from it, by paging `list_projects` and
    /// matching `root_path`. `codebase-memory-mcp` names projects itself
    /// (from the path it was given at `index_repository` time — e.g.
    /// `/home/k/repos/x` → `home-k-repos-x`, with its own normalization
    /// rules) and shares one index across every client, so deriving a name
    /// locally would either miss the existing index or, if also passed as
    /// `name`, build a duplicate one. `Ok(None)` means no project has this
    /// root: the caller must index first and read the assigned name from
    /// that response.
    pub(crate) async fn find_project_by_root(
        &self,
        abs: &Path,
    ) -> Result<Option<String>, MemoryError> {
        let mut rows = Vec::new();
        let mut offset = 0u64;
        loop {
            let mut args = Map::new();
            args.insert("format".to_string(), Value::String("json".to_string()));
            args.insert(
                "limit".to_string(),
                Value::Number(LIST_PROJECTS_PAGE.into()),
            );
            args.insert("offset".to_string(), Value::Number(offset.into()));
            let json = decode_result(self.call("list_projects", args).await?)?;
            let page = project_rows(&json);
            let returned = page.len() as u64;
            rows.extend(page);
            let has_more = json
                .get("has_more")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !has_more || returned == 0 || rows.len() >= LIST_PROJECTS_MAX_ROWS {
                break;
            }
            offset = json
                .get("next_offset")
                .and_then(Value::as_u64)
                .unwrap_or(offset + returned);
        }
        let abs = abs.to_path_buf();
        // `project_matching_root` may `canonicalize` every listed root — a
        // blocking syscall per row, kept off the runtime thread.
        tokio::task::spawn_blocking(move || project_matching_root(rows, &abs))
            .await
            .map_err(|e| MemoryError::Transport(format!("list_projects match task failed: {e}")))
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

/// The `(name, root_path)` pairs of a `list_projects{format:"json"}` page
/// (`{"projects":[{"name":..,"root_path":..},..],"has_more":..}`); rows
/// missing either field are skipped.
pub(crate) fn project_rows(json: &Value) -> Vec<(String, String)> {
    json.get("projects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|p| {
            Some((
                p.get("name")?.as_str()?.to_string(),
                p.get("root_path")?.as_str()?.to_string(),
            ))
        })
        .collect()
}

/// The name of the row whose `root_path` is `abs`: an exact path comparison
/// first (`Path` equality already ignores a trailing separator), then, for
/// roots recorded under another spelling (a symlinked checkout, `..`
/// segments), the row whose root canonicalizes to `abs`. Roots that no longer
/// exist on disk (deleted worktrees) simply fail to canonicalize and never
/// match.
pub(crate) fn project_matching_root(rows: Vec<(String, String)>, abs: &Path) -> Option<String> {
    if let Some((name, _)) = rows.iter().find(|(_, root)| Path::new(root) == abs) {
        return Some(name.clone());
    }
    rows.into_iter()
        .find(|(_, root)| std::fs::canonicalize(root).is_ok_and(|p| p == abs))
        .map(|(name, _)| name)
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

    /// Real CBM 0.11.0 `list_projects{format:"json"}` page: `branch` is
    /// optional per row and must not matter.
    #[test]
    fn project_rows_reads_real_list_projects_page() {
        let json = serde_json::json!({
            "projects": [
                {"name": "home-k-.config-repo-explorer", "root_path": "/home/k/.config/repo-explorer"},
                {"name": "home-k-repos-repo-explorer-mcp", "root_path": "/home/k/repos/repo-explorer-mcp", "branch": "main"},
                {"name": "broken-row-without-root"}
            ],
            "total": 119, "offset": 0, "limit": 200, "returned": 3, "has_more": false
        });
        assert_eq!(
            project_rows(&json),
            vec![
                (
                    "home-k-.config-repo-explorer".to_string(),
                    "/home/k/.config/repo-explorer".to_string()
                ),
                (
                    "home-k-repos-repo-explorer-mcp".to_string(),
                    "/home/k/repos/repo-explorer-mcp".to_string()
                ),
            ]
        );
        assert!(project_rows(&Value::String("not json".into())).is_empty());
    }

    #[test]
    fn project_matching_root_matches_exact_root_only() {
        let rows = vec![
            ("a".to_string(), "/nonexistent/repos/x".to_string()),
            (
                "a-worktree".to_string(),
                "/nonexistent/repos/x/.claude/worktrees/w".to_string(),
            ),
            ("b".to_string(), "/nonexistent/repos/x-other/".to_string()),
        ];
        assert_eq!(
            project_matching_root(rows.clone(), Path::new("/nonexistent/repos/x")),
            Some("a".to_string())
        );
        // A trailing separator in the recorded root is not a different root.
        assert_eq!(
            project_matching_root(rows.clone(), Path::new("/nonexistent/repos/x-other")),
            Some("b".to_string())
        );
        // A prefix/suffix of the root (parent repo, nested worktree) never matches.
        assert_eq!(
            project_matching_root(rows, Path::new("/nonexistent/repos")),
            None
        );
    }

    #[test]
    fn project_matching_root_falls_back_to_canonical_spelling() {
        // The recorded root is a `..`-spelled alias of the canonical `abs`
        // (only an existing path canonicalizes, so use the temp dir).
        let tmp = std::env::temp_dir();
        let abs = std::fs::canonicalize(&tmp).unwrap();
        let alias = tmp.join("..").join(tmp.file_name().unwrap());
        let rows = vec![("t".to_string(), alias.to_string_lossy().into_owned())];
        assert_eq!(project_matching_root(rows, &abs), Some("t".to_string()));
    }
}
