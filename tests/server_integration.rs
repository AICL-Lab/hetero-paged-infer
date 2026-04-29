use axum::body::{to_bytes, Body};
use axum::http::{Method, Request, StatusCode};
use hetero_infer::{
    create_router, CommandBridgeConfig, EngineConfig, ServingBackendConfig, ServingBackendKind,
    ServingConfig,
};
use serde_json::{json, Value};
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
}

#[tokio::test]
async fn test_command_bridge_backend_uses_prompt_env() {
    let config = EngineConfig {
        serving: ServingConfig {
            model_name: "bridge-model".to_string(),
            backend: ServingBackendConfig {
                kind: ServingBackendKind::CommandBridge,
                command: Some(CommandBridgeConfig {
                    program: "/bin/sh".to_string(),
                    args: vec![
                        "-c".to_string(),
                        "printf 'bridge:%s' \"$HETERO_PROMPT\"".to_string(),
                    ],
                }),
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let app = create_router(config).unwrap();

    let response = app
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "bridge-model",
                        "prompt": "hello bridge",
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
    assert_eq!(json["choices"][0]["text"], "bridge:hello bridge");
}
