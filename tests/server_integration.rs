use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use hetero_infer::{
    create_router, create_router_with_engine, test_utils::AlwaysFailExecutor, EngineConfig,
    ExecutionBatch, ExecutionError, ExecutionOutput, GPUExecutorTrait, InferenceEngine, Scheduler,
    ServingConfig, SimpleTokenizer,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn create_test_config() -> EngineConfig {
    EngineConfig {
        max_total_tokens: 512,
        serving: ServingConfig {
            model_name: "test-model".to_string(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Assert the `usage` block is present, counts a non-empty prompt,
/// and reports `total_tokens == prompt_tokens + completion_tokens`.
fn assert_usage_consistent(json: &Value) {
    let prompt = json["usage"]["prompt_tokens"]
        .as_u64()
        .expect("usage.prompt_tokens must be an integer");
    let completion = json["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens must be an integer");
    let total = json["usage"]["total_tokens"]
        .as_u64()
        .expect("usage.total_tokens must be an integer");
    assert!(prompt > 0, "prompt_tokens should be non-zero");
    assert_eq!(
        total,
        prompt + completion,
        "total_tokens must equal prompt_tokens + completion_tokens"
    );
}

#[tokio::test]
async fn test_health_and_ready_endpoints() {
    let app = create_router(create_test_config()).unwrap();

    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);

    let ready = app
        .oneshot(
            Request::builder()
                .uri("/readyz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ready.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_endpoint_exposes_prometheus_counters() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("hetero_requests_total"));
    assert!(body.contains("hetero_inflight_requests"));
}

#[tokio::test]
async fn test_completions_returns_openai_shape() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["model"], "test-model");
    assert!(json["choices"][0]["text"].is_string());
    assert_usage_consistent(&json);
}

#[tokio::test]
async fn test_chat_completions_returns_assistant_message() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [
                            {"role": "user", "content": "say hi"}
                        ],
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert!(json["choices"][0]["message"]["content"].is_string());
    assert_usage_consistent(&json);
}

#[tokio::test]
async fn test_completions_stream_returns_done_event() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 2,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap(),
        "text/event-stream"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("data: [DONE]"));
    // 流必须以带 finish_reason 的终止 chunk 结束，且其 usage 字段完整
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "stream must end with a stop chunk, got: {body}"
    );
}

#[tokio::test]
async fn test_completions_rejects_invalid_sampling_params() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_completions_rejects_empty_prompt() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_rejects_empty_messages() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [],
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_chat_completions_rejects_invalid_role() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "messages": [{"role": "robot", "content": "hi"}],
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_completions_rejects_unknown_model() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "no-such-model",
                        "prompt": "hello",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_unknown_route_returns_json_envelope() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(Request::builder().uri("/nope").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn test_completions_rejects_malformed_json_with_envelope() {
    let app = create_router(create_test_config()).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from("this is not json"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"]["message"].is_string(),
        "rejection must use the JSON error envelope"
    );
}

#[tokio::test]
async fn test_concurrent_requests_both_complete() {
    // 两个并发请求都必须完成：引擎循环应在两步之间接收第二个请求，
    // 而不是把它锁在第一个请求的完整生成之后。
    let app = create_router(create_test_config()).unwrap();

    let make_request = || {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "test-model",
                    "prompt": "hello world",
                    "max_tokens": 3
                })
                .to_string(),
            ))
            .unwrap()
    };

    let (first, second) = tokio::join!(
        app.clone().oneshot(make_request()),
        app.clone().oneshot(make_request()),
    );

    for response in [first.unwrap(), second.unwrap()] {
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert!(json["choices"][0]["text"].is_string());
        assert_usage_consistent(&json);
    }
}

#[tokio::test]
async fn test_completions_maps_execution_failure_to_500() {
    // 注入永远失败的执行器：生成失败必须映射为 500 + internal_error 信封，
    // 而不是被字符串化吞掉。
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(AlwaysFailExecutor),
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "hello world",
                        "max_tokens": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "internal_error");
    assert!(!json["error"]["message"].as_str().unwrap().is_empty());
}

/// 每次执行都阻塞一小段时间的执行器，用于制造可复现的过载窗口。
struct SlowExecutor;

impl GPUExecutorTrait for SlowExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
        std::thread::sleep(std::time::Duration::from_millis(50));
        Ok(ExecutionOutput {
            next_tokens: vec![100; batch.seq_ids.len()],
            seq_ids: batch.seq_ids.clone(),
        })
    }
}

#[tokio::test]
async fn test_completions_returns_429_when_overloaded() {
    // max_num_seqs=1 + 慢执行器：第一个请求独占序列槽位期间，
    // 第二个请求必须收到 429（而非 500），并带 Retry-After。
    let config = EngineConfig {
        max_num_seqs: 1,
        serving: ServingConfig {
            model_name: "test-model".to_string(),
            ..Default::default()
        },
        ..Default::default()
    };
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::new()),
        Scheduler::new(config.clone()),
        Box::new(SlowExecutor),
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let make_request = || {
        Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                json!({
                    "model": "test-model",
                    "prompt": "hello world",
                    "max_tokens": 5
                })
                .to_string(),
            ))
            .unwrap()
    };

    let (first, second) = tokio::join!(
        app.clone().oneshot(make_request()),
        app.clone().oneshot(make_request()),
    );

    let responses = [first.unwrap(), second.unwrap()];
    let statuses: Vec<StatusCode> = responses.iter().map(|r| r.status()).collect();
    assert!(
        statuses.contains(&StatusCode::OK),
        "one request should succeed, got {statuses:?}"
    );
    assert!(
        statuses.contains(&StatusCode::TOO_MANY_REQUESTS),
        "the other should be rejected as overloaded, got {statuses:?}"
    );

    for response in responses {
        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            assert!(
                response.headers().contains_key("retry-after"),
                "429 must carry Retry-After"
            );
        }
    }
}

/// 统计被执行的序列数并人为放慢每步，用于证明：
/// 客户端断开后引擎不再为它继续消耗算力。
struct CountingExecutor {
    executed_sequences: Arc<AtomicU64>,
    step_delay: Duration,
}

impl GPUExecutorTrait for CountingExecutor {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError> {
        std::thread::sleep(self.step_delay);
        self.executed_sequences
            .fetch_add(batch.num_sequences() as u64, Ordering::SeqCst);
        Ok(ExecutionOutput {
            next_tokens: vec![100; batch.num_sequences()],
            seq_ids: batch.seq_ids.clone(),
        })
    }
}

#[tokio::test]
async fn test_client_disconnect_cancels_generation() {
    let executed = Arc::new(AtomicU64::new(0));
    let config = create_test_config();
    let engine = InferenceEngine::with_components(
        config.clone(),
        Box::new(SimpleTokenizer::without_special_tokens()),
        Scheduler::new(config.clone()),
        Box::new(CountingExecutor {
            executed_sequences: executed.clone(),
            step_delay: Duration::from_millis(5),
        }),
    )
    .unwrap();
    let app = create_router_with_engine(config, engine).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "prompt": "disconnect me",
                        "max_tokens": 500,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 读到第一个 token 片段后立刻丢弃 body，模拟客户端断连
    let mut body = response.into_body();
    let mut saw_chunk = false;
    while let Some(Ok(frame)) = body.frame().await {
        if let Ok(data) = frame.into_data() {
            if data.windows(15).any(|w| w == b"text_completion") {
                saw_chunk = true;
                break;
            }
        }
    }
    assert!(saw_chunk, "stream should produce at least one token chunk");
    drop(body);

    // 断连后执行计数必须停下来：间隔采样两次，不再增长。
    // （若无取消机制，5ms/步的引擎会在两次采样间推进上百步。）
    tokio::time::sleep(Duration::from_millis(300)).await;
    let first_sample = executed.load(Ordering::SeqCst);
    assert!(
        first_sample < 500,
        "request should be cancelled long before max_tokens"
    );
    tokio::time::sleep(Duration::from_millis(500)).await;
    let second_sample = executed.load(Ordering::SeqCst);
    assert_eq!(
        first_sample, second_sample,
        "engine must stop executing a request after its client disconnects"
    );
}
