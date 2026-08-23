//! `--update` CLI mode.
//!
//! Instead of booting the MCP server, checks `repo-explorer-mcp` itself and
//! its runtime dependency binaries (`rtk`, `rg`/ripgrep, `codebase-memory-mcp`)
//! against their GitHub releases, and installs any newer version found.
//! Never runs alongside the main exploration logic.

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::path::Path;
use std::process::ExitCode;

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

const DEPENDENCY_BINARIES: &[DependencyBinary] = &[
    DependencyBinary {
        command: "rtk",
        owner: "rtk-ai",
        repo: "rtk",
    },
    DependencyBinary {
        command: "rg",
        owner: "BurntSushi",
        repo: "ripgrep",
    },
    DependencyBinary {
        command: "codebase-memory-mcp",
        owner: "DeusData",
        repo: "codebase-memory-mcp",
    },
];

/// True when `--update` is present among the raw CLI args.
pub fn wants_update(args: &[String]) -> bool {
    args.iter().any(|a| a == "--update")
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

struct Outcome {
    name: String,
    current_version: Option<semver::Version>,
    latest_version: Option<semver::Version>,
    action: &'static str,
    detail: Option<String>,
}

impl Outcome {
    fn into_report(self) -> ComponentReport {
        ComponentReport {
            name: self.name,
            current_version: self.current_version.map(|v| v.to_string()),
            latest_version: self.latest_version.map(|v| v.to_string()),
            action: self.action,
            detail: self.detail,
        }
    }
}

/// Run the update flow: check + install `repo-explorer-mcp` itself, then each
/// dependency binary. Prints a structured JSON report to stdout (stdout is
/// otherwise reserved for the MCP protocol stream, but no MCP session exists
/// in this mode) and returns non-zero if any component failed.
pub async fn run_update() -> ExitCode {
    let client = match build_http_client() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("repo-explorer-mcp: failed to build HTTP client for update check: {e:#}");
            return ExitCode::FAILURE;
        }
    };

    let mut outcomes = vec![update_self(&client).await];
    for dep in DEPENDENCY_BINARIES {
        outcomes.push(update_dependency(&client, dep).await);
    }

    let had_error = outcomes.iter().any(|o| o.action == "error");
    let report = UpdateReport {
        status: if had_error { "error" } else { "ok" },
        components: outcomes.into_iter().map(Outcome::into_report).collect(),
    };
    match serde_json::to_string_pretty(&report) {
        Ok(s) => println!("{s}"),
        Err(e) => eprintln!("repo-explorer-mcp: failed to serialize update report: {e}"),
    }

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn build_http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent("repo-explorer-mcp-updater")
        .build()
        .context("failed to construct reqwest client")
}

async fn update_self(client: &reqwest::Client) -> Outcome {
    let name = SELF_REPO.to_string();
    let current = match semver::Version::parse(env!("CARGO_PKG_VERSION")) {
        Ok(v) => v,
        Err(e) => {
            return Outcome {
                name,
                current_version: None,
                latest_version: None,
                action: "error",
                detail: Some(format!("own package version is not valid semver: {e}")),
            };
        }
    };

    let release = match fetch_latest_release(client, SELF_OWNER, SELF_REPO).await {
        Ok(r) => r,
        Err(e) => {
            return Outcome {
                name,
                current_version: Some(current),
                latest_version: None,
                action: "error",
                detail: Some(e.to_string()),
            };
        }
    };

    let latest = match parse_tag_version(&release.tag_name) {
        Ok(v) => v,
        Err(e) => {
            return Outcome {
                name,
                current_version: Some(current),
                latest_version: None,
                action: "error",
                detail: Some(e.to_string()),
            };
        }
    };

    if latest <= current {
        return Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "up-to-date",
            detail: None,
        };
    }

    let Some(asset) = pick_asset(&release.assets) else {
        return Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "error",
            detail: Some(format!(
                "no release asset matched this platform ({})",
                current_os_keyword()
            )),
        };
    };

    match install_from_asset(
        client,
        &release.assets,
        asset,
        SELF_REPO,
        InstallTarget::SelfExe,
    )
    .await
    {
        Ok(note) => Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "updated",
            detail: note.map(str::to_string),
        },
        Err(e) => Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "error",
            detail: Some(e.to_string()),
        },
    }
}

async fn update_dependency(client: &reqwest::Client, dep: &DependencyBinary) -> Outcome {
    let name = dep.command.to_string();

    let path = match which::which(dep.command) {
        Ok(p) => p,
        Err(_) => {
            return Outcome {
                name,
                current_version: None,
                latest_version: None,
                action: "skipped",
                detail: Some("binary not found on PATH; install it first".to_string()),
            };
        }
    };

    let current_version = read_installed_version(&path);

    let release = match fetch_latest_release(client, dep.owner, dep.repo).await {
        Ok(r) => r,
        Err(e) => {
            return Outcome {
                name,
                current_version,
                latest_version: None,
                action: "error",
                detail: Some(e.to_string()),
            };
        }
    };

    let latest = match parse_tag_version(&release.tag_name) {
        Ok(v) => v,
        Err(e) => {
            return Outcome {
                name,
                current_version,
                latest_version: None,
                action: "error",
                detail: Some(e.to_string()),
            };
        }
    };

    let Some(current) = current_version.clone() else {
        return Outcome {
            name,
            current_version: None,
            latest_version: Some(latest),
            action: "skipped",
            detail: Some(
                "could not determine the installed version; skipping to avoid overwriting an \
                 unmanaged installation"
                    .to_string(),
            ),
        };
    };

    if latest <= current {
        return Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "up-to-date",
            detail: None,
        };
    }

    let Some(asset) = pick_asset(&release.assets) else {
        return Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "error",
            detail: Some(format!(
                "no release asset matched this platform ({})",
                current_os_keyword()
            )),
        };
    };

    match install_from_asset(
        client,
        &release.assets,
        asset,
        dep.command,
        InstallTarget::Path(&path),
    )
    .await
    {
        Ok(note) => Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "updated",
            detail: note.map(str::to_string),
        },
        Err(e) => Outcome {
            name,
            current_version: Some(current),
            latest_version: Some(latest),
            action: "error",
            detail: Some(e.to_string()),
        },
    }
}

/// Run `<path> --version` and pull the first semver-looking substring out of
/// its output (checked on stdout, then stderr, since CLIs disagree on which
/// stream `--version` writes to).
fn read_installed_version(path: &Path) -> Option<semver::Version> {
    let output = std::process::Command::new(path)
        .arg("--version")
        .output()
        .ok()?;
    extract_semver(&String::from_utf8_lossy(&output.stdout))
        .or_else(|| extract_semver(&String::from_utf8_lossy(&output.stderr)))
}

/// Find the first substring made of digits and `.` that parses as semver.
/// A bare `MAJOR.MINOR` is accepted too, treated as `MAJOR.MINOR.0`.
fn extract_semver(text: &str) -> Option<semver::Version> {
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
            if candidate.matches('.').count() == 1
                && let Ok(v) = semver::Version::parse(&format!("{candidate}.0"))
            {
                return Some(v);
            }
        } else {
            i += 1;
        }
    }
    None
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

const ARCH_KEYWORDS: &[&str] = &["x86_64", "amd64"];
const ARCHIVE_EXTENSIONS: &[&str] = &[".tar.gz", ".tgz", ".zip"];

fn is_archive(name: &str) -> bool {
    ARCHIVE_EXTENSIONS.iter().any(|ext| name.ends_with(ext))
}

/// Pick the release asset matching the current OS/arch. Restricted to known
/// archive extensions (never a bare/unknown-format asset such as `.mcpb`,
/// `.txt`, or `.json`, which would otherwise be installed verbatim as if it
/// were the binary) and skips UI-bundle variants (`-ui-`), preferring a
/// non-`portable` build when both are offered for the same platform.
fn pick_asset(assets: &[Asset]) -> Option<&Asset> {
    let os = current_os_keyword();
    let mut candidates: Vec<&Asset> = assets
        .iter()
        .filter(|a| {
            let name = a.name.to_lowercase();
            name.contains(os)
                && ARCH_KEYWORDS.iter().any(|k| name.contains(k))
                && is_archive(&name)
                && !name.contains("-ui-")
        })
        .collect();
    candidates.sort_by_key(|a| a.name.to_lowercase().contains("portable"));
    candidates.into_iter().next()
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
        let checksum_text = String::from_utf8(checksum_bytes)
            .context("checksum sidecar file is not valid UTF-8")?;
        verify_sha256(&data, &checksum_text, &asset.name)?;
        None
    } else {
        Some("no checksum sidecar asset found; integrity not verified")
    };

    let binary = extract_binary(&asset.name, &data, command)?;
    match target {
        InstallTarget::SelfExe => install_self(&binary)?,
        InstallTarget::Path(p) => install_dependency_binary(p, &binary)?,
    }
    Ok(note)
}

async fn download(client: &reqwest::Client, url: &str) -> Result<Vec<u8>> {
    let bytes = client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to download {url}"))?
        .error_for_status()
        .with_context(|| format!("download failed for {url}"))?
        .bytes()
        .await
        .with_context(|| format!("failed to read response body for {url}"))?;
    Ok(bytes.to_vec())
}

fn verify_sha256(data: &[u8], checksum_text: &str, asset_name: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    let expected = checksum_text
        .split_whitespace()
        .next()
        .ok_or_else(|| anyhow!("checksum file for {asset_name} is empty"))?;
    let mut hasher = Sha256::new();
    hasher.update(data);
    let actual = hex_encode(&hasher.finalize());
    if !expected.eq_ignore_ascii_case(&actual) {
        return Err(anyhow!(
            "checksum mismatch for {asset_name}: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// True when `entry_name`'s base filename is `command` (optionally with a
/// `.exe` suffix), ignoring any archive-internal directory prefix.
fn matches_binary_name(entry_name: &str, command: &str) -> bool {
    let base = entry_name.rsplit('/').next().unwrap_or(entry_name);
    let base = base.rsplit('\\').next().unwrap_or(base);
    base == command || base == format!("{command}.exe")
}

fn extract_binary(asset_name: &str, data: &[u8], command: &str) -> Result<Vec<u8>> {
    let lower = asset_name.to_lowercase();
    if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        extract_from_tar_gz(data, command)
    } else if lower.ends_with(".zip") {
        extract_from_zip(data, command)
    } else {
        // Not archived — the whole payload is the binary itself.
        Ok(data.to_vec())
    }
}

fn extract_from_tar_gz(data: &[u8], command: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let decoder = flate2::read::GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    for entry in archive.entries().context("failed to read tar.gz archive")? {
        let mut entry = entry.context("failed to read a tar.gz entry")?;
        let path = entry
            .path()
            .context("failed to read a tar.gz entry path")?
            .to_string_lossy()
            .into_owned();
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
        let name = file.name().to_string();
        if matches_binary_name(&name, command) {
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
    let output = std::process::Command::new(path)
        .arg("--version")
        .output()
        .with_context(|| format!("downloaded update at {} failed to execute", path.display()))?;
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
    let tmp = write_temp_executable(
        &std::env::temp_dir(),
        &format!("repo-explorer-mcp-update-{}", std::process::id()),
        data,
    )?;
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
        let real = hex_encode(&hasher.finalize());
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
}
