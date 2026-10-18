use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_repo-explorer-mcp")
}

fn write_config(body: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "rex_judge_cfg_{}_{}.toml",
        std::process::id(),
        body.len()
    ));
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn llm_free_laya_offline_config_is_valid() {
    let cfg = write_config(
        "[codebase_memory]\ncommand = \"codebase-memory-mcp\"\nargs = [\"--stdio\"]\n[judge]\nmode = \"laya\"\n[agent]\nfallback = \"off\"\n",
    );
    let out = Command::new(bin())
        .args([
            "--config",
            cfg.to_str().unwrap(),
            "config",
            "test",
            "--json",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(&cfg).ok();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "expected success, stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("\"status\": \"valid\""), "stdout: {stdout}");
}

#[test]
fn bad_select_threshold_reports_toml_path_and_fails() {
    let cfg = write_config(
        "[codebase_memory]\ncommand = \"codebase-memory-mcp\"\nargs = [\"--stdio\"]\n[judge]\nmode = \"laya\"\nselect_threshold = 0\n[agent]\nfallback = \"off\"\n",
    );
    let out = Command::new(bin())
        .args([
            "--config",
            cfg.to_str().unwrap(),
            "config",
            "test",
            "--json",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(&cfg).ok();
    assert!(!out.status.success());
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("\"toml_path\": \"judge.select_threshold\""),
        "combined: {combined}"
    );
}
