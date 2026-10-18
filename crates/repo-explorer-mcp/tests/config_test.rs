use std::path::Path;
use std::process::{Command, Output};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_repo-explorer-mcp")
}

/// One temp config per test; `name` keeps parallel tests from colliding.
fn write_config(name: &str, body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "rex_judge_cfg_{}_{}.toml",
        std::process::id(),
        name
    ));
    std::fs::write(&path, body).unwrap();
    path
}

/// `--config <path> config test --json`, with a dummy `GOOGLE_API_KEY` so the
/// committed eval configs (which name it) validate.
fn config_test(path: &Path) -> Output {
    Command::new(bin())
        .args([
            "--config",
            path.to_str().unwrap(),
            "config",
            "test",
            "--json",
        ])
        .env("GOOGLE_API_KEY", "dummy")
        .output()
        .unwrap()
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

const MEMORY: &str = "[codebase_memory]\ncommand = \"codebase-memory-mcp\"\nargs = [\"--stdio\"]\n";

#[test]
fn llm_free_legacy_mode_alias_config_is_valid() {
    let cfg = write_config(
        "alias",
        &format!("{MEMORY}[judge]\nmode = \"laya\"\n[agent]\nfallback = \"off\"\n"),
    );
    let out = config_test(&cfg);
    std::fs::remove_file(&cfg).ok();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "expected success: {}", combined(&out));
    assert!(stdout.contains("\"status\": \"valid\""), "stdout: {stdout}");
    assert!(combined(&out).contains("deprecated"), "{}", combined(&out));
}

#[test]
fn llm_free_on_offline_config_is_valid() {
    let cfg = write_config(
        "on",
        &format!("{MEMORY}[judge]\nmode = \"on\"\n[agent]\nfallback = \"off\"\n"),
    );
    let out = config_test(&cfg);
    std::fs::remove_file(&cfg).ok();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "expected success: {}", combined(&out));
    assert!(stdout.contains("\"status\": \"valid\""), "stdout: {stdout}");
    assert!(!combined(&out).contains("deprecated"), "{}", combined(&out));
}

#[test]
fn bad_select_threshold_reports_toml_path_and_fails() {
    let cfg = write_config(
        "select_threshold",
        &format!(
            "{MEMORY}[judge]\nmode = \"on\"\nselect_threshold = 0\n[agent]\nfallback = \"off\"\n"
        ),
    );
    let out = config_test(&cfg);
    std::fs::remove_file(&cfg).ok();
    assert!(!out.status.success());
    let all = combined(&out);
    assert!(
        all.contains("\"toml_path\": \"judge.select_threshold\""),
        "combined: {all}"
    );
}

#[test]
fn bad_escalate_below_reports_toml_path_and_fails() {
    let cfg = write_config(
        "escalate_below",
        &format!(
            "{MEMORY}[judge]\nmode = \"on\"\nescalate_below = 300\n[agent]\nfallback = \"off\"\n"
        ),
    );
    let out = config_test(&cfg);
    std::fs::remove_file(&cfg).ok();
    assert!(!out.status.success());
    let all = combined(&out);
    assert!(
        all.contains("\"toml_path\": \"judge.escalate_below\""),
        "combined: {all}"
    );
}

#[test]
fn eval_judge_configs_validate() {
    for name in ["judge-on.toml", "judge-shadow.toml", "judge-offline.toml"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../eval/config")
            .join(name);
        assert!(path.exists(), "{} is missing", path.display());
        let out = config_test(&path);
        let all = combined(&out);
        assert!(out.status.success(), "{name}: {all}");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("\"status\": \"valid\""),
            "{name}: {all}"
        );
        assert!(
            !all.contains("deprecated") && !all.contains("unrecognized"),
            "{name} must not use legacy keys or the alias: {all}"
        );
    }
}
