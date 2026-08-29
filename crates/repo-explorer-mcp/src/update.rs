//! `--update` CLI mode.
//!
//! Instead of booting the MCP server, checks `repo-explorer-mcp` itself, its
//! `which`-resolved runtime dependency (`rg`/ripgrep), and its two managed
//! install-if-absent copies (`rtk`, `codebase-memory-mcp`) against their GitHub
//! releases, installing any newer version found. Never runs alongside the main
//! exploration logic.

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// Upper bound on how long a network request or a `--version` subprocess
/// check may run before it's treated as failed, so a stalled connection or a
/// hung/misbehaving binary can't make `--update` block forever.
pub(crate) const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(10);
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// GitHub owner/repo for `repo-explorer-mcp` itself.
const SELF_OWNER: &str = "kwitsch";
const SELF_REPO: &str = "repo-explorer-mcp";

/// A binary this project shells out to at runtime, checked/updated alongside
/// itself. `command` is both the display name and the executable looked up
/// via `which`.
struct DependencyBinary {
    command: &'static str,
    owner: &'static str,
    repo: &'static str,
}

const DEPENDENCY_BINARIES: &[DependencyBinary] = &[DependencyBinary {
    command: "rg",
    owner: "BurntSushi",
    repo: "ripgrep",
}];

/// The shared per-user directory repo-explorer installs its managed binaries
/// into: `~/.local/bin` on Linux (`dirs::executable_dir()`), and
/// `%LOCALAPPDATA%\repo-explorer-mcp` on Windows
/// (`dirs::data_local_dir().join("repo-explorer-mcp")`) — the same directory
/// the npx installer's `installDir()` places the main binary in on each
/// platform. Errors when no such directory is resolvable (e.g. no HOME /
/// LOCALAPPDATA).
fn managed_bin_dir() -> Result<PathBuf> {
    if let Some(dir) = dirs::executable_dir() {
        // Linux: $XDG_BIN_HOME or $HOME/.local/bin. (None on Windows/macOS.)
        Ok(dir)
    } else {
        // Windows: %LOCALAPPDATA%\repo-explorer-mcp (matches installDir()).
        let base = dirs::data_local_dir().ok_or_else(|| {
            anyhow!(
                "no local data directory available to place managed binaries (is LOCALAPPDATA set?)"
            )
        })?;
        Ok(base.join("repo-explorer-mcp"))
    }
}

/// Pure, testable path composer: the managed dir *is* the final directory, so
/// the binary sits directly inside it.
fn binary_path_in(bin_dir: &Path, file: &str) -> PathBuf {
    bin_dir.join(file)
}

/// File name of the managed `codebase-memory-mcp` binary, with the platform's
/// executable suffix on Windows.
fn memory_binary_file_name() -> &'static str {
    if cfg!(windows) {
        "codebase-memory-mcp.exe"
    } else {
        "codebase-memory-mcp"
    }
}

/// File name of the managed `rtk` binary, with the platform's executable suffix
/// on Windows.
fn rtk_binary_file_name() -> &'static str {
    if cfg!(windows) { "rtk.exe" } else { "rtk" }
}

/// Absolute path of the repo-explorer-managed `codebase-memory-mcp` binary in
/// the shared managed bin dir (`~/.local/bin` on Linux,
/// `%LOCALAPPDATA%\repo-explorer-mcp` on Windows). Errors when no managed dir is
/// resolvable (e.g. no HOME). Never consults PATH.
pub(crate) fn dedicated_memory_binary_path() -> Result<PathBuf> {
    Ok(binary_path_in(
        &managed_bin_dir()?,
        memory_binary_file_name(),
    ))
}

/// Absolute path of the repo-explorer-managed `rtk` binary in the shared managed
/// bin dir, alongside `codebase-memory-mcp`. Errors when no managed dir is
/// resolvable. Never consults PATH.
pub(crate) fn dedicated_rtk_binary_path() -> Result<PathBuf> {
    Ok(binary_path_in(&managed_bin_dir()?, rtk_binary_file_name()))
}

/// Install-if-absent / update-if-stale the private `codebase-memory-mcp`
/// copy at [`dedicated_memory_binary_path`], via the shared
/// [`check_and_install`] pipeline. Unlike the `which`-resolved dependency
/// binaries, a *missing* binary is installed (not "skipped"):
/// repo-explorer-mcp owns this copy outright and never falls back to a
/// PATH/global install — see `check_and_install`'s `install_if_missing`.
async fn provision_or_update_memory_binary(client: &reqwest::Client) -> ComponentReport {
    let name = "codebase-memory-mcp".to_string();

    let path = match dedicated_memory_binary_path() {
        Ok(p) => p,
        Err(e) => {
            return ComponentReport {
                name,
                current_version: None,
                latest_version: None,
                action: "error",
                detail: Some(e.to_string()),
            };
        }
    };

    if let Err(e) = crate::ensure_parent_dir(&path) {
        return ComponentReport {
            name,
            current_version: None,
            latest_version: None,
            action: "error",
            detail: Some(format!(
                "failed to create binary directory {}: {e}",
                path.parent().unwrap_or(&path).display()
            )),
        };
    }

    let path_exists = path.exists();
    let current = if path_exists {
        read_installed_version_blocking(path.clone()).await
    } else {
        None
    };

    // Only a genuinely absent file should trigger a fresh install. A file
    // that exists but whose version couldn't be probed (a transient
    // `--version` timeout or format mismatch) must be skipped like any other
    // tracked dependency, never silently redownloaded/overwritten.
    let install_if_missing = !path_exists;

    check_and_install(
        client,
        name,
        ReleaseSource {
            owner: "DeusData",
            repo: "codebase-memory-mcp",
            command: "codebase-memory-mcp",
        },
        current,
        InstallTarget::Path(&path),
        install_if_missing,
    )
    .await
}

/// Install-if-absent / update-if-stale the managed `rtk` copy at
/// [`dedicated_rtk_binary_path`], via the shared [`check_and_install`] pipeline.
/// Like the `codebase-memory-mcp` copy and unlike the `which`-resolved `rg`
/// dependency, a *missing* binary is installed (not "skipped"):
/// repo-explorer-mcp owns this copy and search is mandatory, so it never falls
/// back to a PATH/global install — see `check_and_install`'s `install_if_missing`.
async fn provision_or_update_rtk_binary(client: &reqwest::Client) -> ComponentReport {
    let name = "rtk".to_string();

    let path = match dedicated_rtk_binary_path() {
        Ok(p) => p,
        Err(e) => {
            return ComponentReport {
                name,
                current_version: None,
                latest_version: None,
                action: "error",
                detail: Some(e.to_string()),
            };
        }
    };

    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return ComponentReport {
            name,
            current_version: None,
            latest_version: None,
            action: "error",
            detail: Some(format!(
                "failed to create binary directory {}: {e}",
                parent.display()
            )),
        };
    }

    let current = if path.exists() {
        read_installed_version_blocking(path.clone()).await
    } else {
        None
    };

    check_and_install(
        client,
        name,
        ReleaseSource {
            owner: "rtk-ai",
            repo: "rtk",
            command: "rtk",
        },
        current,
        InstallTarget::Path(&path),
        true,
    )
    .await
}

/// True when `--update` is present among the raw CLI args.
pub fn wants_update(args: &[String]) -> bool {
    crate::has_flag(args, &["--update"])
}

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    assets: Vec<Asset>,
}

#[derive(Debug, Clone, Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

#[derive(serde::Serialize)]
struct UpdateReport {
    status: &'static str,
    components: Vec<ComponentReport>,
}

#[derive(serde::Serialize)]
struct ComponentReport {
    name: String,
    current_version: Option<String>,
    latest_version: Option<String>,
    action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

/// Run the update flow: check + install `repo-explorer-mcp` itself and each
/// dependency binary concurrently. Prints a structured JSON report to stdout
/// (stdout is otherwise reserved for the MCP protocol stream, but no MCP
/// session exists in this mode) and returns non-zero if any component failed.
pub async fn run_update() -> ExitCode {
    let client = match build_http_client() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("repo-explorer-mcp: failed to build HTTP client for update check: {e:#}");
            return ExitCode::FAILURE;
        }
    };

    let mut handles = Vec::with_capacity(3 + DEPENDENCY_BINARIES.len());
    let self_client = client.clone();
    handles.push((
        SELF_REPO,
        tokio::spawn(async move { update_self(&self_client).await }),
    ));
    for dep in DEPENDENCY_BINARIES {
        let client = client.clone();
        handles.push((
            dep.command,
            tokio::spawn(async move { update_dependency(&client, dep).await }),
        ));
    }

    let rtk_client = client.clone();
    handles.push((
        "rtk",
        tokio::spawn(async move { provision_or_update_rtk_binary(&rtk_client).await }),
    ));

    let memory_client = client.clone();
    handles.push((
        "codebase-memory-mcp",
        tokio::spawn(async move { provision_or_update_memory_binary(&memory_client).await }),
    ));

    let mut components = Vec::with_capacity(handles.len());
    for (name, handle) in handles {
        components.push(match handle.await {
            Ok(report) => report,
            Err(e) => ComponentReport {
                name: name.to_string(),
                current_version: None,
                latest_version: None,
                action: "error",
                detail: Some(format!("update task panicked: {e}")),
            },
        });
    }

    let had_error = components.iter().any(|c| c.action == "error");
    let report = UpdateReport {
        status: if had_error { "error" } else { "ok" },
        components,
    };
    crate::print_report(&report, "update");

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("repo-explorer-mcp-updater")
        .timeout(HTTP_TIMEOUT)
        .build()
        .context("failed to construct reqwest client")
}

async fn update_self(client: &reqwest::Client) -> ComponentReport {
    let current = semver::Version::parse(env!("CARGO_PKG_VERSION"))
        .expect("CARGO_PKG_VERSION is validated as SemVer by cargo at build time");

    check_and_install(
        client,
        SELF_REPO.to_string(),
        ReleaseSource {
            owner: SELF_OWNER,
            repo: SELF_REPO,
            command: SELF_REPO,
        },
        Some(current),
        InstallTarget::SelfExe,
        false,
    )
    .await
}

async fn update_dependency(client: &reqwest::Client, dep: &DependencyBinary) -> ComponentReport {
    let name = dep.command.to_string();

    let path = match which::which(dep.command) {
        Ok(p) => p,
        Err(_) => {
            return ComponentReport {
                name,
                current_version: None,
                latest_version: None,
                action: "skipped",
                detail: Some("binary not found on PATH; install it first".to_string()),
            };
        }
    };

    let current = read_installed_version_blocking(path.clone()).await;

    check_and_install(
        client,
        name,
        ReleaseSource {
            owner: dep.owner,
            repo: dep.repo,
            command: dep.command,
        },
        current,
        InstallTarget::Path(&path),
        false,
    )
    .await
}

/// Shared fetch-release -> parse-tag -> compare-versions -> pick-asset ->
/// install -> [`ComponentReport`] sequence used by `update_self`,
/// `update_dependency`, and `provision_or_update_memory_binary`, once each
/// has arrived at its own `current` version (or `None`, if it couldn't be
/// determined but a release lookup is still worth doing to report the latest
/// version).
///
/// `install_if_missing` controls what a `None` `current` means: for the
/// `which`-resolved PATH dependency (`rg`) it means "unmanaged installation,
/// don't touch it" (`action: "skipped"`); for the managed `rtk` and
/// `codebase-memory-mcp` copies — which repo-explorer-mcp owns outright and
/// never falls back to a PATH/global install — it means "not installed yet" and
/// triggers a fresh install (`action: "installed"`) instead.
async fn check_and_install(
    client: &reqwest::Client,
    name: String,
    source: ReleaseSource<'_>,
    current: Option<semver::Version>,
    target: InstallTarget<'_>,
    install_if_missing: bool,
) -> ComponentReport {
    let (release, latest) = match fetch_latest_release(client, source.owner, source.repo)
        .await
        .and_then(|r| parse_tag_version(&r.tag_name).map(|v| (r, v)))
    {
        Ok(pair) => pair,
        Err(e) => {
            return ComponentReport {
                name,
                current_version: current.as_ref().map(|v| v.to_string()),
                latest_version: None,
                action: "error",
                detail: Some(e.to_string()),
            };
        }
    };

    let current = match current {
        Some(current) => current,
        None if !install_if_missing => {
            return ComponentReport {
                name,
                current_version: None,
                latest_version: Some(latest.to_string()),
                action: "skipped",
                detail: Some(
                    "could not determine the installed version; skipping to avoid overwriting \
                     an unmanaged installation"
                        .to_string(),
                ),
            };
        }
        None => {
            return install_release(
                client,
                name,
                None,
                &release,
                &latest,
                source.command,
                target,
            )
            .await;
        }
    };

    let current_version = Some(current.to_string());
    let latest_version = Some(latest.to_string());

    if latest <= current {
        return ComponentReport {
            name,
            current_version,
            latest_version,
            action: "up-to-date",
            detail: None,
        };
    }

    install_release(
        client,
        name,
        Some(current),
        &release,
        &latest,
        source.command,
        target,
    )
    .await
}

/// The GitHub release to check and the executable name to extract from its
/// assets, grouped so [`check_and_install`] takes one argument instead of
/// three for this trio.
struct ReleaseSource<'a> {
    owner: &'a str,
    repo: &'a str,
    command: &'a str,
}

/// Pick the platform asset out of `release` and install it to `target`,
/// reporting `action: "installed"` when `current` was `None` (nothing to
/// replace) or `"updated"` otherwise. Shared tail of [`check_and_install`]'s
/// two install-triggering branches (stale, and missing-but-owned).
async fn install_release(
    client: &reqwest::Client,
    name: String,
    current: Option<semver::Version>,
    release: &Release,
    latest: &semver::Version,
    command: &str,
    target: InstallTarget<'_>,
) -> ComponentReport {
    let current_version = current.as_ref().map(|v| v.to_string());
    let latest_version = Some(latest.to_string());

    let Some(asset) = pick_asset(&release.assets) else {
        return ComponentReport {
            name,
            current_version,
            latest_version,
            action: "error",
            detail: Some(format!(
                "no release asset matched this platform ({})",
                current_os_keyword()
            )),
        };
    };

    match install_from_asset(client, &release.assets, asset, command, target).await {
        Ok(note) => ComponentReport {
            name,
            current_version,
            latest_version,
            action: if current.is_some() {
                "updated"
            } else {
                "installed"
            },
            detail: note.map(str::to_string),
        },
        Err(e) => ComponentReport {
            name,
            current_version,
            latest_version,
            action: "error",
            detail: Some(e.to_string()),
        },
    }
}

/// Spawn `<path> --version`, bounded by [`SUBPROCESS_TIMEOUT`]. Shared by
/// [`read_installed_version`] (best-effort version-string extraction) and
/// [`verify_executable`] (hard exit-status check) so the invocation itself is
/// defined once.
fn run_version_probe(path: &Path) -> Option<std::process::Output> {
    let mut command = std::process::Command::new(path);
    command.arg("--version");
    run_with_timeout(command, SUBPROCESS_TIMEOUT)
}

/// Run `<path> --version` (bounded by [`SUBPROCESS_TIMEOUT`]) and pull the
/// first semver-looking substring out of its output (checked on stdout, then
/// stderr, since CLIs disagree on which stream `--version` writes to).
fn read_installed_version(path: &Path) -> Option<semver::Version> {
    let output = run_version_probe(path)?;
    extract_semver(&String::from_utf8_lossy(&output.stdout))
        .or_else(|| extract_semver(&String::from_utf8_lossy(&output.stderr)))
}

/// [`read_installed_version`] off the async runtime's worker thread: it
/// blocks the calling thread for up to [`SUBPROCESS_TIMEOUT`] via
/// [`run_with_timeout`]'s synchronous polling loop, and `run_update` now
/// checks every component concurrently, so leaving it on a worker thread
/// could starve other tasks on a runtime with few of them.
async fn read_installed_version_blocking(path: std::path::PathBuf) -> Option<semver::Version> {
    tokio::task::spawn_blocking(move || read_installed_version(&path))
        .await
        .unwrap_or(None)
}

/// Find the first substring made of digits and `.` that parses as semver,
/// preferring a match on the first line — a tool's own version conventionally
/// leads its `--version` banner, ahead of any bundled library versions it
/// might also print — and falling back to the rest of the text otherwise.
/// A bare `MAJOR.MINOR` is accepted too, treated as `MAJOR.MINOR.0`, and a
/// run of 4+ components (e.g. a VCS-revision-suffixed `1.2.3.4`) is read as
/// its leading `MAJOR.MINOR.PATCH`.
fn extract_semver(text: &str) -> Option<semver::Version> {
    let first_line = text.lines().next().unwrap_or("");
    scan_for_semver(first_line).or_else(|| scan_for_semver(text))
}

fn scan_for_semver(text: &str) -> Option<semver::Version> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            let candidate = &text[start..i];
            if let Ok(v) = semver::Version::parse(candidate) {
                return Some(v);
            }
            let dot_count = candidate.matches('.').count();
            if dot_count == 1
                && let Ok(v) = semver::Version::parse(&format!("{candidate}.0"))
            {
                return Some(v);
            }
            if dot_count >= 3 {
                let leading = candidate.split('.').take(3).collect::<Vec<_>>().join(".");
                if let Ok(v) = semver::Version::parse(&leading) {
                    return Some(v);
                }
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Run `command` to completion, killing it and returning `None` if it hasn't
/// exited within `timeout`. Assumes small output (a `--version` banner is at
/// most a few lines) — output isn't drained until the process exits, so a
/// process that blocks on a full stdout/stderr pipe before exiting would
/// still hang until the timeout.
pub(crate) fn run_with_timeout(
    mut command: std::process::Command,
    timeout: Duration,
) -> Option<std::process::Output> {
    use std::io::Read;
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    let mut child = command.spawn().ok()?;

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => return None,
        }
    };

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    if let Some(mut out) = child.stdout.take() {
        let _ = out.read_to_end(&mut stdout);
    }
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_end(&mut stderr);
    }
    Some(std::process::Output {
        status,
        stdout,
        stderr,
    })
}

fn parse_tag_version(tag: &str) -> Result<semver::Version> {
    semver::Version::parse(tag.trim_start_matches('v'))
        .with_context(|| format!("release tag `{tag}` is not a valid semver version"))
}

async fn fetch_latest_release(
    client: &reqwest::Client,
    owner: &str,
    repo: &str,
) -> Result<Release> {
    let url = format!("https://api.github.com/repos/{owner}/{repo}/releases/latest");
    let response = client
        .get(&url)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .with_context(|| format!("failed to reach the GitHub releases API for {owner}/{repo}"))?
        .error_for_status()
        .with_context(|| format!("GitHub releases API returned an error for {owner}/{repo}"))?;
    response
        .json::<Release>()
        .await
        .with_context(|| format!("failed to parse the GitHub release response for {owner}/{repo}"))
}

fn current_os_keyword() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    }
}

/// The archive formats a release asset may be packaged in. A single source
/// of truth for "is this name an archive, and if so which kind" — shared by
/// `pick_asset` (which assets are even eligible) and `extract_binary` (how to
/// unpack the one that was picked), so a newly-supported format only needs
/// to be added here once.
enum ArchiveFormat {
    TarGz,
    Zip,
}

/// Expects `name` already lowercased by the caller, so a name lowercased
/// once up front (e.g. by `pick_asset`, per asset) isn't lowercased again here.
fn archive_format(name: &str) -> Option<ArchiveFormat> {
    if name.ends_with(".tar.gz") || name.ends_with(".tgz") {
        Some(ArchiveFormat::TarGz)
    } else if name.ends_with(".zip") {
        Some(ArchiveFormat::Zip)
    } else {
        None
    }
}

/// Expects `name` already lowercased by the caller (see [`archive_format`]).
fn is_archive(name: &str) -> bool {
    archive_format(name).is_some()
}

/// Release-asset architecture keywords for the host this binary is actually
/// running on (`std::env::consts::ARCH`), not a hardcoded guess — a build
/// running on e.g. `aarch64` must never match an `x86_64` asset.
fn arch_keywords() -> Vec<&'static str> {
    match std::env::consts::ARCH {
        "x86_64" => vec!["x86_64", "amd64"],
        "aarch64" => vec!["aarch64", "arm64"],
        other => vec![other],
    }
}

/// Pick the release asset matching the current OS/arch. Restricted to known
/// archive extensions (never a bare/unknown-format asset such as `.mcpb`,
/// `.txt`, or `.json`, which would otherwise be installed verbatim as if it
/// were the binary) and skips UI-bundle variants (`-ui-`). Among the
/// remaining candidates, prefers a non-`portable` build, then (on a tie)
/// the MSVC toolchain over GNU — MSVC is what Rust's default Windows
/// toolchain and scoop/winget/choco `rg` installs produce, so it's a safer
/// default than depending on GitHub's incidental asset-list order when a
/// release publishes both `-pc-windows-gnu` and `-pc-windows-msvc` archives.
fn pick_asset(assets: &[Asset]) -> Option<&Asset> {
    let os = current_os_keyword();
    let arch_keywords = arch_keywords();
    // Only Windows has an MSVC-vs-GNU choice to break; on every other OS
    // this must never penalize `-gnu` (e.g. Linux's gnu vs musl builds).
    let windows = cfg!(target_os = "windows");
    assets
        .iter()
        .filter_map(|a| {
            let name = a.name.to_lowercase();
            let matches = name.contains(os)
                && arch_keywords.iter().any(|k| name.contains(k))
                && is_archive(&name)
                && !name.contains("-ui-");
            matches.then_some((a, name))
        })
        .min_by_key(|(_, name)| (name.contains("portable"), windows && name.contains("-gnu")))
        .map(|(a, _)| a)
}

/// Find `<asset>.sha256`, the sidecar checksum naming convention this
/// project's own release workflow uses (best-effort for third-party repos).
fn find_checksum_asset<'a>(assets: &'a [Asset], target: &Asset) -> Option<&'a Asset> {
    let expected = format!("{}.sha256", target.name);
    assets.iter().find(|a| a.name == expected)
}

enum InstallTarget<'a> {
    SelfExe,
    Path(&'a Path),
}

/// Download `asset`, verify it against a `.sha256` sidecar when present,
/// extract the `command` binary out of it if it's an archive, and install it.
/// Returns a warning note (e.g. "integrity not verified") on success.
async fn install_from_asset(
    client: &reqwest::Client,
    assets: &[Asset],
    asset: &Asset,
    command: &str,
    target: InstallTarget<'_>,
) -> Result<Option<&'static str>> {
    let data = download(client, &asset.browser_download_url).await?;

    let note = if let Some(checksum_asset) = find_checksum_asset(assets, asset) {
        let checksum_bytes = download(client, &checksum_asset.browser_download_url).await?;
        let checksum_text = std::str::from_utf8(&checksum_bytes)
            .context("checksum sidecar file is not valid UTF-8")?;
        verify_sha256(&data, checksum_text, &asset.name)?;
        None
    } else {
        Some("no checksum sidecar asset found; integrity not verified")
    };

    let binary = extract_binary(&asset.name, &data, command)?;
    // install_self/install_dependency_binary block the calling thread (temp
    // file I/O, a `--version` verification subprocess bounded by
    // SUBPROCESS_TIMEOUT, then a rename/self-replace); run_update now checks
    // every component concurrently via tokio::spawn, so keep this off the
    // async worker threads.
    match target {
        InstallTarget::SelfExe => {
            tokio::task::spawn_blocking(move || install_self(&binary))
                .await
                .context("install task panicked")??;
        }
        InstallTarget::Path(p) => {
            let dest = p.to_path_buf();
            tokio::task::spawn_blocking(move || install_dependency_binary(&dest, &binary))
                .await
                .context("install task panicked")??;
        }
    }
    Ok(note)
}

async fn download(client: &reqwest::Client, url: &str) -> Result<bytes::Bytes> {
    client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to download {url}"))?
        .error_for_status()
        .with_context(|| format!("download failed for {url}"))?
        .bytes()
        .await
        .with_context(|| format!("failed to read response body for {url}"))
}

fn verify_sha256(data: &[u8], checksum_text: &str, asset_name: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let expected = checksum_text
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("checksum file for {asset_name} is empty"))?;
    let mut hasher = Sha256::new();
    hasher.update(data);
    let actual = hex::encode(hasher.finalize());
    if !expected.eq_ignore_ascii_case(&actual) {
        return Err(anyhow!(
            "checksum mismatch for {asset_name}: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

/// True when `entry_name`'s base filename is `command` (optionally with a
/// `.exe` suffix), ignoring any archive-internal directory prefix.
fn matches_binary_name(entry_name: &str, command: &str) -> bool {
    let base = entry_name
        .rsplit(['/', '\\'])
        .next()
        .expect("rsplit always yields at least one item");
    base == command || base.strip_suffix(".exe") == Some(command)
}

fn extract_binary(asset_name: &str, data: &[u8], command: &str) -> Result<Vec<u8>> {
    match archive_format(&asset_name.to_lowercase()) {
        Some(ArchiveFormat::TarGz) => extract_from_tar_gz(data, command),
        Some(ArchiveFormat::Zip) => extract_from_zip(data, command),
        // Not archived — the whole payload is the binary itself.
        None => Ok(data.to_vec()),
    }
}

fn extract_from_tar_gz(data: &[u8], command: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let decoder = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries().context("failed to read tar.gz archive")? {
        let mut entry = entry.context("failed to read a tar.gz entry")?;
        let path_buf = entry.path().context("failed to read a tar.gz entry path")?;
        let path = path_buf.to_string_lossy();
        if matches_binary_name(&path, command) {
            let mut buf = Vec::new();
            entry
                .read_to_end(&mut buf)
                .context("failed to read the binary out of the tar.gz archive")?;
            return Ok(buf);
        }
    }
    Err(anyhow!("no `{command}` entry found in the tar.gz archive"))
}

fn extract_from_zip(data: &[u8], command: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let reader = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(reader).context("failed to read zip archive")?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .context("failed to read a zip archive entry")?;
        let name = file.name();
        if matches_binary_name(name, command) {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)
                .context("failed to read the binary out of the zip archive")?;
            return Ok(buf);
        }
    }
    Err(anyhow!("no `{command}` entry found in the zip archive"))
}

/// Write `data` to a fresh, executable temporary file in `dir`. Kept separate
/// from the final install step so the file can be verified to actually run
/// before anything already installed is touched.
fn write_temp_executable(dir: &Path, prefix: &str, data: &[u8]) -> Result<std::path::PathBuf> {
    let tmp = dir.join(format!(".{prefix}.update-tmp-{}", std::process::id()));
    std::fs::write(&tmp, data)
        .with_context(|| format!("failed to write temporary file at {}", tmp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))
            .with_context(|| format!("failed to set executable permission on {}", tmp.display()))?;
    }
    Ok(tmp)
}

/// Confirm the downloaded/extracted file is actually the right kind of
/// executable (not a bundle, README, or otherwise-mismatched asset) before it
/// replaces anything already installed: run `<path> --version` and require a
/// clean exit.
fn verify_executable(path: &Path) -> Result<()> {
    let output = run_version_probe(path).with_context(|| {
        format!(
            "downloaded update at {} failed to execute (or didn't exit within {SUBPROCESS_TIMEOUT:?})",
            path.display()
        )
    })?;
    if !output.status.success() {
        return Err(anyhow!(
            "downloaded update at {} exited with {} on `--version`",
            path.display(),
            output.status
        ));
    }
    Ok(())
}

/// Atomically replace the currently running executable, after confirming the
/// downloaded file actually runs.
fn install_self(data: &[u8]) -> Result<()> {
    let tmp = write_temp_executable(&std::env::temp_dir(), "repo-explorer-mcp-update", data)?;
    let result = verify_executable(&tmp).and_then(|()| {
        self_replace::self_replace(&tmp)
            .context("failed to install the downloaded update over the running executable")
    });
    let _ = std::fs::remove_file(&tmp);
    result
}

/// Atomically replace a dependency binary in place, next to its existing
/// path, after confirming the downloaded file actually runs.
fn install_dependency_binary(dest: &Path, data: &[u8]) -> Result<()> {
    let dir = dest.parent().ok_or_else(|| {
        anyhow!(
            "dependency binary path `{}` has no parent directory",
            dest.display()
        )
    })?;
    let file_name = dest.file_name().and_then(|n| n.to_str()).unwrap_or("bin");
    let tmp = write_temp_executable(dir, file_name, data)?;
    let result = verify_executable(&tmp).and_then(|()| {
        std::fs::rename(&tmp, dest)
            .with_context(|| format!("failed to install update at {}", dest.display()))
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wants_update_detects_flag() {
        assert!(wants_update(&["--update".to_string()]));
        assert!(wants_update(&[
            "--config".to_string(),
            "x.toml".to_string(),
            "--update".to_string()
        ]));
        assert!(!wants_update(&["--config".to_string()]));
        assert!(!wants_update(&[]));
    }

    #[test]
    fn extract_semver_from_typical_version_output() {
        assert_eq!(
            extract_semver("ripgrep 14.1.0").unwrap(),
            semver::Version::parse("14.1.0").unwrap()
        );
        assert_eq!(
            extract_semver("rtk version 0.3.2 (build abc123)").unwrap(),
            semver::Version::parse("0.3.2").unwrap()
        );
        assert_eq!(
            extract_semver("codebase-memory-mcp v1.2.0").unwrap(),
            semver::Version::parse("1.2.0").unwrap()
        );
    }

    #[test]
    fn extract_semver_prefers_first_line_over_a_later_bundled_version() {
        // Regression: a multi-line `--version` banner that mentions a bundled
        // library's version on a later line must not have that version
        // mistaken for the tool's own.
        assert_eq!(
            extract_semver("ripgrep 14.1.1\nPCRE2 version: 10.42 2022-12-11").unwrap(),
            semver::Version::parse("14.1.1").unwrap()
        );
    }

    #[test]
    fn extract_semver_accepts_two_component_version() {
        assert_eq!(
            extract_semver("tool 2.5").unwrap(),
            semver::Version::parse("2.5.0").unwrap()
        );
    }

    #[test]
    fn extract_semver_none_when_absent() {
        assert!(extract_semver("no version here").is_none());
    }

    #[test]
    fn extract_semver_reads_leading_triple_from_four_component_version() {
        // Regression: a 4+ component version-like run (e.g. a VCS-revision
        // suffix) must not be skipped in favor of a later, unrelated number.
        assert_eq!(
            extract_semver("tool 1.2.3.4 (rev 5.6.7)").unwrap(),
            semver::Version::parse("1.2.3").unwrap()
        );
    }

    #[test]
    fn parse_tag_version_strips_v_prefix() {
        assert_eq!(
            parse_tag_version("v1.2.3").unwrap(),
            semver::Version::parse("1.2.3").unwrap()
        );
        assert_eq!(
            parse_tag_version("1.2.3").unwrap(),
            semver::Version::parse("1.2.3").unwrap()
        );
        assert!(parse_tag_version("not-a-version").is_err());
    }

    fn asset(name: &str) -> Asset {
        Asset {
            name: name.to_string(),
            browser_download_url: format!("https://example.invalid/{name}"),
        }
    }

    #[test]
    fn arch_keywords_never_match_a_foreign_architecture() {
        // Regression: keywords must be derived from the actual host arch, not
        // hardcoded to x86_64 — a foreign-arch asset must never match.
        let keywords = arch_keywords();
        let foreign = if std::env::consts::ARCH == "aarch64" {
            "x86_64"
        } else {
            "aarch64"
        };
        assert!(!keywords.contains(&foreign));
    }

    #[test]
    fn pick_asset_matches_current_platform() {
        let assets = vec![
            asset("tool-1.0.0-x86_64-unknown-linux-gnu.tar.gz"),
            asset("tool-1.0.0-x86_64-unknown-linux-gnu.tar.gz.sha256"),
            asset("tool-1.0.0-x86_64-pc-windows-msvc.zip"),
            asset("tool-1.0.0-x86_64-pc-windows-msvc.zip.sha256"),
        ];
        let picked = pick_asset(&assets).expect("an asset should match this platform");
        assert!(is_archive(&picked.name.to_lowercase()));
        assert!(picked.name.to_lowercase().contains(current_os_keyword()));
    }

    #[test]
    fn pick_asset_skips_non_archive_and_ui_bundle_assets() {
        // Regression: a `.mcpb` bundle (or a UI-only archive) must never be
        // picked and installed as if it were the plain CLI binary archive.
        let os = current_os_keyword();
        let assets = vec![
            asset(&format!("tool-{os}-amd64.mcpb")),
            asset(&format!("tool-ui-{os}-amd64.tar.gz")),
            asset(&format!("tool-{os}-amd64-portable.tar.gz")),
            asset(&format!("tool-{os}-amd64.tar.gz")),
        ];
        let picked = pick_asset(&assets).expect("a proper archive should match");
        assert_eq!(picked.name, format!("tool-{os}-amd64.tar.gz"));
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn pick_asset_prefers_msvc_over_gnu_on_a_tie() {
        // Regression: when a release publishes both toolchain variants for
        // Windows (e.g. ripgrep's `-pc-windows-gnu` and `-pc-windows-msvc`
        // zips), the pick must be deterministic and favor msvc, not whichever
        // happens to come first in GitHub's asset list. Built from the
        // current host's own arch keyword so the test matches on any runner.
        let arch = arch_keywords()[0];
        let gnu = format!("tool-15.2.0-{arch}-windows-gnu.zip");
        let msvc = format!("tool-15.2.0-{arch}-windows-msvc.zip");

        let assets = vec![asset(&gnu), asset(&msvc)];
        let picked = pick_asset(&assets).expect("an asset should match this platform");
        assert_eq!(picked.name, msvc);

        // Order-independence: the same pick regardless of list order.
        let assets_reversed = vec![asset(&msvc), asset(&gnu)];
        let picked_reversed =
            pick_asset(&assets_reversed).expect("an asset should match this platform");
        assert_eq!(picked_reversed.name, msvc);
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn pick_asset_does_not_deprioritize_gnu_off_windows() {
        // Regression: the MSVC-over-GNU tiebreak is Windows-only. Off
        // Windows it must never penalize `-gnu` -- e.g. ripgrep's Linux
        // release publishes both `-unknown-linux-gnu` and
        // `-unknown-linux-musl` archives, and the old unscoped tiebreak
        // silently installed musl every time regardless of list order.
        let os = current_os_keyword();
        let arch = arch_keywords()[0];
        let gnu = format!("tool-15.2.0-{arch}-{os}-gnu.tar.gz");
        let musl = format!("tool-15.2.0-{arch}-{os}-musl.tar.gz");

        let assets = vec![asset(&gnu), asset(&musl)];
        let picked = pick_asset(&assets).expect("an asset should match this platform");
        assert_eq!(
            picked.name, gnu,
            "gnu must not be deprioritized off Windows"
        );
    }

    #[test]
    fn pick_asset_none_when_no_match() {
        let assets = vec![asset("tool-1.0.0-aarch64-apple-darwin.tar.gz")];
        assert!(pick_asset(&assets).is_none());
    }

    #[test]
    fn find_checksum_asset_matches_sidecar_convention() {
        let target = asset("tool-1.0.0-x86_64-unknown-linux-gnu.tar.gz");
        let assets = vec![
            target.clone(),
            asset("tool-1.0.0-x86_64-unknown-linux-gnu.tar.gz.sha256"),
        ];
        let found = find_checksum_asset(&assets, &target).expect("sidecar should be found");
        assert_eq!(
            found.name,
            "tool-1.0.0-x86_64-unknown-linux-gnu.tar.gz.sha256"
        );
    }

    #[test]
    fn find_checksum_asset_none_when_absent() {
        let target = asset("tool-1.0.0-x86_64-unknown-linux-gnu.tar.gz");
        let assets = vec![target.clone()];
        assert!(find_checksum_asset(&assets, &target).is_none());
    }

    #[test]
    fn matches_binary_name_handles_prefixes_and_exe_suffix() {
        assert!(matches_binary_name("rg", "rg"));
        assert!(matches_binary_name("rg-14.1.0/rg", "rg"));
        assert!(matches_binary_name("rg-14.1.0\\rg.exe", "rg"));
        assert!(!matches_binary_name("rg-14.1.0/README.md", "rg"));
    }

    #[test]
    fn verify_sha256_accepts_matching_checksum() {
        let data = b"hello world";
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let real = hex::encode(hasher.finalize());
        assert!(verify_sha256(data, &real, "asset").is_ok());
    }

    #[test]
    fn verify_sha256_rejects_mismatch() {
        let data = b"hello world";
        let wrong = "0".repeat(64);
        assert!(verify_sha256(data, &wrong, "asset").is_err());
    }

    #[test]
    fn extract_binary_returns_raw_bytes_when_not_archived() {
        let data = b"plain-binary-bytes".to_vec();
        let result = extract_binary("codebase-memory-mcp", &data, "codebase-memory-mcp").unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn binary_path_in_joins_file_onto_bin_dir() {
        let dir = Path::new("/some/bin");
        assert_eq!(
            binary_path_in(dir, "codebase-memory-mcp"),
            dir.join("codebase-memory-mcp")
        );
        assert_eq!(binary_path_in(dir, "rtk"), dir.join("rtk"));
        let mem = if cfg!(windows) {
            "codebase-memory-mcp.exe"
        } else {
            "codebase-memory-mcp"
        };
        let rtk = if cfg!(windows) { "rtk.exe" } else { "rtk" };
        assert_eq!(memory_binary_file_name(), mem);
        assert_eq!(rtk_binary_file_name(), rtk);
    }

    #[test]
    fn managed_binaries_removed_from_dependency_binaries() {
        // Regression: rtk and codebase-memory-mcp are provisioned install-if-
        // absent (`provision_or_update_*`), never via the which-based
        // DEPENDENCY_BINARIES loop; only `rg` stays a PATH-resolved dependency.
        assert!(
            DEPENDENCY_BINARIES
                .iter()
                .all(|d| d.command != "codebase-memory-mcp"),
            "codebase-memory-mcp must not be a which-resolved dependency"
        );
        assert!(
            DEPENDENCY_BINARIES.iter().all(|d| d.command != "rtk"),
            "rtk must not be a which-resolved dependency"
        );
        assert!(DEPENDENCY_BINARIES.iter().any(|d| d.command == "rg"));
    }

    #[test]
    fn managed_rtk_and_memory_share_a_parent_dir() {
        let rtk = dedicated_rtk_binary_path().expect("rtk path resolves in the test env");
        let mem = dedicated_memory_binary_path().expect("memory path resolves in the test env");
        assert_eq!(
            rtk.parent(),
            mem.parent(),
            "both managed binaries must live in the same shared bin dir"
        );
        assert_eq!(
            rtk.file_name().unwrap(),
            std::ffi::OsStr::new(rtk_binary_file_name())
        );
        assert_eq!(
            mem.file_name().unwrap(),
            std::ffi::OsStr::new(memory_binary_file_name())
        );
    }
}
