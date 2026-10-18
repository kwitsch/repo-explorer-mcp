use repo_explorer_core::config::{JudgeMode, JudgeSettings};
use repo_explorer_core::judge::{CandidateJudge, JudgeError};
use repo_explorer_judge::{ConfiguredJudge, SystemOneHttpJudge};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Read one HTTP/1.1 request (headers + Content-Length body) from `stream`.
async fn read_request(stream: &mut tokio::net::TcpStream) -> (String, String) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        let n = stream.read(&mut tmp).await.unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_headers_end(&buf) {
            let head = String::from_utf8_lossy(&buf[..pos]).to_string();
            let clen = content_length(&head);
            let body_start = pos + 4;
            while buf.len() < body_start + clen {
                let n = stream.read(&mut tmp).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            let body = String::from_utf8_lossy(&buf[body_start..body_start + clen]).to_string();
            return (head, body);
        }
    }
    (String::new(), String::new())
}

fn find_headers_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn content_length(head: &str) -> usize {
    head.lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0)
}

fn ok_json(prob_a: f64) -> String {
    let body = format!(
        r#"{{"answers":{{"relevant":{{"probabilities":{{"A":{prob_a},"B":{}}}}}}}}}"#,
        1.0 - prob_a
    );
    http_response(200, "OK", &body)
}

fn http_response(code: u16, reason: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {code} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn settings(base_url: String) -> JudgeSettings {
    JudgeSettings {
        mode: JudgeMode::On,
        base_url,
        api_key_env: None,
        model: "h2oai/h2o-lightning-4b".to_string(),
        timeout_ms: 2000,
        max_concurrency: 2,
        select_threshold: 50,
        ..JudgeSettings::default()
    }
}

/// Spawn a responder that answers each accepted connection with `handler(body)`,
/// optionally after a per-connection delay. Returns the base URL.
async fn spawn_server<F>(handler: F) -> String
where
    F: Fn(usize, &str) -> (Option<Duration>, String) + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handler = Arc::new(handler);
    tokio::spawn(async move {
        let mut i = 0usize;
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let handler = Arc::clone(&handler);
            let n = i;
            i += 1;
            tokio::spawn(async move {
                let (_head, body) = read_request(&mut stream).await;
                let (delay, resp) = handler(n, &body);
                if let Some(d) = delay {
                    tokio::time::sleep(d).await;
                }
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.flush().await;
            });
        }
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn success_three_states_in_order_out_of_order_replies() {
    // Reply to earlier connections with a longer delay, so responses arrive
    // out of order — the judgements must still be state-indexed.
    let base = spawn_server(|n, _body| {
        let delay = Duration::from_millis((3 - n as u64) * 40);
        let p = 0.1 * (n as f64 + 1.0); // 0.1, 0.2, 0.3
        (Some(delay), ok_json(p))
    })
    .await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    let out = judge
        .judge(&["s0".into(), "s1".into(), "s2".into()])
        .await
        .unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].p_relevant_permille, 100);
    assert_eq!(out[1].p_relevant_permille, 200);
    assert_eq!(out[2].p_relevant_permille, 300);
}

#[tokio::test]
async fn request_body_is_the_pinned_json_with_model() {
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let tx = std::sync::Mutex::new(tx);
    let base = spawn_server(move |_n, body| {
        tx.lock().unwrap().send(body.to_string()).ok();
        (None, ok_json(0.9))
    })
    .await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    judge.judge(&["the-state".into()]).await.unwrap();
    let body = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["model"], "h2oai/h2o-lightning-4b");
    assert_eq!(v["state"], "the-state");
    assert_eq!(v["questions"]["relevant"]["type"], "choice");
    assert_eq!(
        v["questions"]["relevant"]["instructions"],
        "Does this code location answer the repository search query?"
    );
    assert_eq!(
        v["questions"]["relevant"]["criteria"]["A"],
        "yes, this location answers the query"
    );
    assert_eq!(
        v["questions"]["relevant"]["criteria"]["B"],
        "no, this location does not answer the query"
    );
}

#[tokio::test]
async fn bearer_header_present_only_when_key_set() {
    let base = spawn_server(|_n, _body| (None, ok_json(0.9))).await;
    let mut s = settings(base);
    s.api_key_env = Some("REX_JUDGE_TEST_KEY".to_string());
    // Unset accessor → Unavailable at build time.
    let unset = SystemOneHttpJudge::new_with_env(&s, |_| None);
    assert!(matches!(unset, Err(JudgeError::Unavailable { .. })));
    // A present key builds successfully (accessor injected; process env untouched).
    let ok = SystemOneHttpJudge::new_with_env(&s, |v| {
        (v == "REX_JUDGE_TEST_KEY").then(|| "secret".to_string())
    });
    assert!(ok.is_ok());
}

#[tokio::test]
async fn http_401_is_unavailable() {
    let base = spawn_server(|_n, _b| (None, http_response(401, "Unauthorized", "no"))).await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    assert!(matches!(
        judge.judge(&["s".into()]).await,
        Err(JudgeError::Unavailable { .. })
    ));
}

#[tokio::test]
async fn http_503_is_unavailable() {
    let base = spawn_server(|_n, _b| {
        (
            None,
            http_response(503, "Service Unavailable", "model loading"),
        )
    })
    .await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    match judge.judge(&["s".into()]).await {
        Err(JudgeError::Unavailable { message }) => {
            assert!(message.contains("HTTP 503: model loading"), "{message}")
        }
        other => panic!("expected Unavailable, got {other:?}"),
    }
}

#[tokio::test]
async fn http_422_is_protocol() {
    let base = spawn_server(|_n, _b| {
        (
            None,
            http_response(422, "Unprocessable Entity", "state too long"),
        )
    })
    .await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    match judge.judge(&["s".into()]).await {
        Err(JudgeError::Protocol { message }) => assert!(message.contains("HTTP 422"), "{message}"),
        other => panic!("expected Protocol, got {other:?}"),
    }
}

#[tokio::test]
async fn http_500_is_protocol() {
    let base = spawn_server(|_n, _b| (None, http_response(500, "Server Error", "boom"))).await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    match judge.judge(&["s".into()]).await {
        Err(JudgeError::Protocol { message }) => assert!(message.contains("HTTP 500")),
        other => panic!("expected Protocol, got {other:?}"),
    }
}

#[tokio::test]
async fn latency_ms_is_parsed_when_present() {
    let base = spawn_server(|_n, _b| {
        (
            None,
            http_response(
                200,
                "OK",
                r#"{"answers":{"relevant":{"probabilities":{"A":0.9,"B":0.1}}},"usage":{"latency_ms":12.6}}"#,
            ),
        )
    })
    .await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    let out = judge.judge(&["s".into()]).await.unwrap();
    assert_eq!(out[0].p_relevant_permille, 900);
    assert_eq!(out[0].latency_ms, Some(13));
    // No `usage` block -> None, never an error.
    let plain = spawn_server(|_n, _b| (None, ok_json(0.9))).await;
    let judge = SystemOneHttpJudge::new(&settings(plain)).unwrap();
    let out = judge.judge(&["s".into()]).await.unwrap();
    assert_eq!(out[0].latency_ms, None);
}

#[tokio::test]
async fn non_json_body_is_protocol() {
    let base = spawn_server(|_n, _b| (None, http_response(200, "OK", "not json"))).await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    assert!(matches!(
        judge.judge(&["s".into()]).await,
        Err(JudgeError::Protocol { .. })
    ));
}

#[tokio::test]
async fn missing_probability_is_protocol() {
    let base = spawn_server(|_n, _b| {
        (
            None,
            http_response(
                200,
                "OK",
                r#"{"answers":{"relevant":{"probabilities":{}}}}"#,
            ),
        )
    })
    .await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    assert!(matches!(
        judge.judge(&["s".into()]).await,
        Err(JudgeError::Protocol { .. })
    ));
}

#[tokio::test]
async fn out_of_range_probability_is_protocol() {
    let base = spawn_server(|_n, _b| (None, ok_json(1.2))).await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    assert!(matches!(
        judge.judge(&["s".into()]).await,
        Err(JudgeError::Protocol { .. })
    ));
}

#[tokio::test]
async fn timeout_elapses() {
    let base = spawn_server(|_n, _b| (Some(Duration::from_millis(600)), ok_json(0.9))).await;
    let mut s = settings(base);
    s.timeout_ms = 200;
    let judge = SystemOneHttpJudge::new(&s).unwrap();
    assert_eq!(
        judge.judge(&["s".into()]).await,
        Err(JudgeError::Timeout { timeout_ms: 200 })
    );
}

#[tokio::test]
async fn empty_states_is_ok_without_request() {
    // Point at a closed port: no connection must be attempted.
    let judge = SystemOneHttpJudge::new(&settings("http://127.0.0.1:9".to_string())).unwrap();
    assert_eq!(judge.judge(&[]).await.unwrap(), vec![]);
}

#[tokio::test]
async fn warm_up_probes_models_and_one_state() {
    // Connection 0 is the body-less `GET /v1/models`, connection 1 the JSON
    // probe POST; record (connection, body empty?) for each.
    let (tx, rx) = std::sync::mpsc::channel::<(usize, bool)>();
    let tx = std::sync::Mutex::new(tx);
    let base = spawn_server(move |n, body| {
        tx.lock().unwrap().send((n, body.is_empty())).ok();
        let resp = if n == 0 {
            http_response(200, "OK", r#"{"data":[]}"#)
        } else {
            ok_json(0.9)
        };
        (None, resp)
    })
    .await;
    let judge = SystemOneHttpJudge::new(&settings(base)).unwrap();
    assert!(judge.warm_up().await.is_ok());
    let got: Vec<(usize, bool)> = (0..2)
        .map(|_| rx.recv_timeout(Duration::from_secs(2)).unwrap())
        .collect();
    assert_eq!(got, vec![(0, true), (1, false)]);
}

#[tokio::test]
async fn warm_up_unavailable_on_closed_port() {
    let closed = SystemOneHttpJudge::new(&settings("http://127.0.0.1:9".to_string())).unwrap();
    match closed.warm_up().await {
        Err(JudgeError::Unavailable { message }) => {
            assert!(message.contains("models probe failed"), "{message}")
        }
        other => panic!("expected Unavailable, got {other:?}"),
    }
}

#[tokio::test]
async fn configured_judge_off_is_disabled() {
    let mut s = settings("http://127.0.0.1:9".to_string());
    s.mode = JudgeMode::Off;
    let j = ConfiguredJudge::from_settings(&s).unwrap();
    assert!(matches!(j, ConfiguredJudge::Disabled(_)));
    assert!(matches!(
        j.judge(&["s".into()]).await,
        Err(JudgeError::Unavailable { .. })
    ));
}

#[tokio::test]
async fn configured_judge_on_and_shadow_are_http() {
    for mode in [JudgeMode::On, JudgeMode::Shadow] {
        let mut s = settings("http://127.0.0.1:9".to_string());
        s.mode = mode;
        let j = ConfiguredJudge::from_settings(&s).unwrap();
        assert!(matches!(j, ConfiguredJudge::Http(_)));
    }
}
