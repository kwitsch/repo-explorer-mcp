//! Ignored parity + latency tests against a real checkpoint. Require
//! `REPO_EXPLORER_LAYA_CHECKPOINT` pointing at a checkpoint dir that also holds
//! `golden.jsonl` (produced by `judge-serve/export_golden.py`).
#![cfg(feature = "candle")]

use repo_explorer_core::config::JudgeDevice;
use repo_explorer_judge::candle::LoadedModel;
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn ckpt_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("REPO_EXPLORER_LAYA_CHECKPOINT").expect("set REPO_EXPLORER_LAYA_CHECKPOINT"),
    )
}

struct Golden {
    checkpoint_sha256: String,
    rows: Vec<Row>,
}

struct Row {
    state: String,
    input_ids: Vec<u32>,
    markers: Vec<usize>,
    logits: [f32; 2],
    p_relevant: f64,
}

fn read_golden(dir: &std::path::Path) -> Golden {
    let f = std::fs::File::open(dir.join("golden.jsonl")).expect("golden.jsonl");
    let mut lines = BufReader::new(f).lines();
    let header: serde_json::Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    let checkpoint_sha256 = header["header"]["checkpoint_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let mut rows = Vec::new();
    for line in lines {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let logits = v["logits"].as_array().unwrap();
        rows.push(Row {
            state: v["state"].as_str().unwrap().to_string(),
            input_ids: v["input_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect(),
            markers: v["markers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect(),
            logits: [
                logits[0].as_f64().unwrap() as f32,
                logits[1].as_f64().unwrap() as f32,
            ],
            p_relevant: v["p_relevant"].as_f64().unwrap(),
        });
    }
    Golden {
        checkpoint_sha256,
        rows,
    }
}

fn checkpoint_sha256(dir: &std::path::Path) -> String {
    let bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
    let mut h = Sha256::new();
    h.update(&bytes);
    hex::encode(h.finalize())
}

#[test]
#[ignore]
fn parity_token_ids() {
    let dir = ckpt_dir();
    let golden = read_golden(&dir);
    assert_eq!(
        golden.checkpoint_sha256,
        checkpoint_sha256(&dir),
        "checkpoint changed: regenerate golden.jsonl"
    );
    let model = LoadedModel::load(&dir, JudgeDevice::Cpu).unwrap();
    for (i, row) in golden.rows.iter().enumerate() {
        let enc = model.encode(&row.state).unwrap();
        assert_eq!(enc.ids, row.input_ids, "input_ids mismatch at row {i}");
        assert_eq!(enc.markers, row.markers, "markers mismatch at row {i}");
    }
}

#[test]
#[ignore]
fn parity_logits_and_decisions() {
    let dir = ckpt_dir();
    let golden = read_golden(&dir);
    let model = LoadedModel::load(&dir, JudgeDevice::Cpu).unwrap();
    let threshold = model.select_threshold();
    let encoded: Vec<_> = golden
        .rows
        .iter()
        .map(|r| model.encode(&r.state).unwrap())
        .collect();
    let raw = model.raw_logits(&encoded).unwrap();
    let mut in_band = 0;
    for (i, (row, got)) in golden.rows.iter().zip(raw.iter()).enumerate() {
        assert!(
            (row.logits[0] - got[0]).abs() <= 1e-3,
            "logit0 row {i}: {} vs {}",
            row.logits[0],
            got[0]
        );
        assert!(
            (row.logits[1] - got[1]).abs() <= 1e-3,
            "logit1 row {i}: {} vs {}",
            row.logits[1],
            got[1]
        );
        let p = repo_explorer_judge::candle::calib::softmax2(*got, model.temperature())[0] as f64;
        assert!(
            (p - row.p_relevant).abs() <= 1e-3,
            "p_relevant row {i}: {p} vs {}",
            row.p_relevant
        );
        let boundary = threshold as f64 * 10.0;
        let golden_permille = (row.p_relevant * 1000.0).round();
        if (golden_permille - boundary).abs() < 2.0 {
            in_band += 1;
            continue;
        }
        let our_permille = (p * 1000.0).round();
        assert_eq!(
            our_permille >= boundary,
            golden_permille >= boundary,
            "decision flip at row {i}"
        );
    }
    println!("{{\"rows\":{},\"in_band\":{in_band}}}", golden.rows.len());
}

#[test]
#[ignore]
fn bench_judge_latency() {
    let dir = ckpt_dir();
    let golden = read_golden(&dir);
    let model = LoadedModel::load(&dir, JudgeDevice::Cpu).unwrap();
    let states: Vec<String> = golden
        .rows
        .iter()
        .take(12)
        .map(|r| r.state.clone())
        .collect();
    for _ in 0..3 {
        let _ = model.judge_states(&states).unwrap();
    }
    let mut ms: Vec<f64> = Vec::new();
    for _ in 0..20 {
        let t = std::time::Instant::now();
        let _ = model.judge_states(&states).unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f64| ms[((ms.len() as f64 - 1.0) * q).round() as usize];
    println!(
        "{{\"p50_ms\":{:.2},\"p95_ms\":{:.2},\"device\":\"cpu\"}}",
        p(0.5),
        p(0.95)
    );
}
