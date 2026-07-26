//! HTTP 服务层
//!
//! 提供 OpenAI 兼容的最小服务接口、健康检查与指标暴露。

use crate::config::EngineConfig;
use crate::error::{EngineError, SchedulerError};
use crate::types::GenerationParams;
use crate::InferenceEngine;
use async_stream::stream;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;

const STREAM_CHUNK_CHARS: usize = 32;

#[derive(Default)]
struct ServerMetrics {
    requests_total: AtomicU64,
    errors_total: AtomicU64,
    inflight_requests: AtomicU64,
    streaming_requests_total: AtomicU64,
}

impl ServerMetrics {
    fn render(&self) -> String {
        format!(
            "# TYPE hetero_requests_total counter\nhetero_requests_total {}\n# TYPE hetero_errors_total counter\nhetero_errors_total {}\n# TYPE hetero_inflight_requests gauge\nhetero_inflight_requests {}\n# TYPE hetero_streaming_requests_total counter\nhetero_streaming_requests_total {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.errors_total.load(Ordering::Relaxed),
            self.inflight_requests.load(Ordering::Relaxed),
            self.streaming_requests_total.load(Ordering::Relaxed),
        )
    }
}

#[derive(Clone)]
struct AppState {
    config: EngineConfig,
    engine: Arc<Mutex<InferenceEngine>>,
    metrics: Arc<ServerMetrics>,
    response_counter: Arc<AtomicU64>,
}

impl AppState {
    fn new(config: EngineConfig) -> Result<Self, EngineError> {
        config.validate()?;
        let engine = InferenceEngine::new(config.clone())?;

        Ok(Self {
            config,
            engine: Arc::new(Mutex::new(engine)),
            metrics: Arc::new(ServerMetrics::default()),
            response_counter: Arc::new(AtomicU64::new(1)),
        })
    }

    fn next_id(&self, prefix: &str) -> String {
        let value = self.response_counter.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{value}")
    }

    async fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationResult, String> {
        let mut engine = self.engine.lock().await;
        let request_id = engine
            .submit_request(prompt, params)
            .map_err(|e| e.to_string())?;
        let completed = engine.run();
        let result = completed
            .into_iter()
            .find(|item| item.request_id == request_id)
            .ok_or_else(|| {
                EngineError::Scheduler(SchedulerError::RequestNotFound(request_id)).to_string()
            })?;

        if result.success {
            let text = result.output_text;
            let completion_tokens = estimate_tokens(&text);
            Ok(GenerationResult {
                text,
                prompt_tokens: estimate_tokens(prompt),
                completion_tokens,
            })
        } else {
            Err(result
                .error
                .unwrap_or_else(|| "generation failed".to_string()))
        }
    }
}

#[derive(Debug)]
pub struct GenerationResult {
    /// 生成的文本
    pub text: String,
    /// prompt token 数
    pub prompt_tokens: usize,
    /// 生成 token 数
    pub completion_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: u32,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatCompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatCompletionChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorMessage,
}

#[derive(Serialize)]
struct ErrorMessage {
    message: String,
}

struct PreparedGenerationRequest {
    model: String,
    prompt: String,
    params: GenerationParams,
    stream: bool,
}

/// 根据配置创建 router
pub fn create_router(config: EngineConfig) -> Result<Router, EngineError> {
    let state = Arc::new(AppState::new(config)?);

    Ok(Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state))
}

async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn readyz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ready" }))
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        state.metrics.render(),
    )
        .into_response()
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    state
        .metrics
        .inflight_requests
        .fetch_add(1, Ordering::Relaxed);

    let prepared = match prepare_completion_request(&state, request) {
        Ok(prepared) => prepared,
        Err(message) => {
            state
                .metrics
                .inflight_requests
                .fetch_sub(1, Ordering::Relaxed);
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return error_response(StatusCode::BAD_REQUEST, &message);
        }
    };

    let result = state.generate(&prepared.prompt, prepared.params).await;
    state
        .metrics
        .inflight_requests
        .fetch_sub(1, Ordering::Relaxed);

    match result {
        Ok(generated) if prepared.stream => {
            state
                .metrics
                .streaming_requests_total
                .fetch_add(1, Ordering::Relaxed);
            completion_stream_response(&state, &prepared.model, generated).into_response()
        }
        Ok(generated) => Json(CompletionResponse {
            id: state.next_id("cmpl"),
            object: "text_completion",
            created: unix_timestamp(),
            model: prepared.model,
            choices: vec![CompletionChoice {
                text: generated.text.clone(),
                index: 0,
                finish_reason: "stop",
            }],
            usage: usage(&generated),
        })
        .into_response(),
        Err(message) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &message)
        }
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    state
        .metrics
        .inflight_requests
        .fetch_add(1, Ordering::Relaxed);

    let prepared = match prepare_chat_request(&state, request) {
        Ok(prepared) => prepared,
        Err(message) => {
            state
                .metrics
                .inflight_requests
                .fetch_sub(1, Ordering::Relaxed);
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return error_response(StatusCode::BAD_REQUEST, &message);
        }
    };

    let result = state.generate(&prepared.prompt, prepared.params).await;
    state
        .metrics
        .inflight_requests
        .fetch_sub(1, Ordering::Relaxed);

    match result {
        Ok(generated) if prepared.stream => {
            state
                .metrics
                .streaming_requests_total
                .fetch_add(1, Ordering::Relaxed);
            chat_stream_response(&state, &prepared.model, generated).into_response()
        }
        Ok(generated) => Json(ChatCompletionResponse {
            id: state.next_id("chatcmpl"),
            object: "chat.completion",
            created: unix_timestamp(),
            model: prepared.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated.text.clone(),
                },
                finish_reason: "stop",
            }],
            usage: usage(&generated),
        })
        .into_response(),
        Err(message) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            error_response(StatusCode::INTERNAL_SERVER_ERROR, &message)
        }
    }
}

fn completion_stream_response(
    state: &Arc<AppState>,
    model: &str,
    generated: GenerationResult,
) -> Response {
    let id = state.next_id("cmpl");
    let created = unix_timestamp();
    let model = model.to_string();
    let chunks = text_chunks(&generated.text);
    let usage = usage(&generated);
    Sse::new(stream! {
        for chunk in chunks {
            let payload = serde_json::json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"text": chunk, "index": 0, "finish_reason": serde_json::Value::Null}],
                "usage": usage,
            });
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    })
    .keep_alive(KeepAlive::default())
    .into_response()
}

fn chat_stream_response(
    state: &Arc<AppState>,
    model: &str,
    generated: GenerationResult,
) -> Response {
    let id = state.next_id("chatcmpl");
    let created = unix_timestamp();
    let model = model.to_string();
    let chunks = text_chunks(&generated.text);
    let usage = usage(&generated);
    Sse::new(stream! {
        for chunk in chunks {
            let payload = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": chunk}, "finish_reason": serde_json::Value::Null}],
                "usage": usage,
            });
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]"));
    })
    .keep_alive(KeepAlive::default())
    .into_response()
}

fn prepare_completion_request(
    state: &AppState,
    request: CompletionRequest,
) -> Result<PreparedGenerationRequest, String> {
    let prompt = validate_prompt(request.prompt)?;
    Ok(PreparedGenerationRequest {
        model: request
            .model
            .unwrap_or_else(|| state.config.serving.model_name.clone()),
        prompt,
        params: generation_params(request.max_tokens, request.temperature, request.top_p)?,
        stream: request.stream.unwrap_or(false),
    })
}

fn prepare_chat_request(
    state: &AppState,
    request: ChatCompletionRequest,
) -> Result<PreparedGenerationRequest, String> {
    if request.messages.is_empty() {
        return Err("messages must not be empty".to_string());
    }

    let prompt = validate_prompt(
        request
            .messages
            .iter()
            .map(|message| format!("{}: {}", message.role, message.content))
            .collect::<Vec<_>>()
            .join("\n"),
    )?;

    Ok(PreparedGenerationRequest {
        model: request
            .model
            .unwrap_or_else(|| state.config.serving.model_name.clone()),
        prompt,
        params: generation_params(request.max_tokens, request.temperature, request.top_p)?,
        stream: request.stream.unwrap_or(false),
    })
}

fn generation_params(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
) -> Result<GenerationParams, String> {
    let params = GenerationParams {
        max_tokens: max_tokens.unwrap_or(16),
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
    };
    params.validate().map_err(|err| err.to_string())?;
    Ok(params)
}

fn validate_prompt(prompt: String) -> Result<String, String> {
    if prompt.trim().is_empty() {
        Err("prompt must not be empty".to_string())
    } else {
        Ok(prompt)
    }
}

fn usage(result: &GenerationResult) -> Usage {
    Usage {
        prompt_tokens: result.prompt_tokens,
        completion_tokens: result.completion_tokens,
        total_tokens: result.prompt_tokens + result.completion_tokens,
    }
}

/// 简单的 token 估算函数
///
/// TODO: 使用实际 tokenizer 进行精确估算
/// 当前实现仅为粗略估计，适用于 Mock 执行器场景
fn estimate_tokens(text: &str) -> usize {
    let split_count = text.split_whitespace().count();
    if split_count == 0 {
        text.chars().count()
    } else {
        split_count
    }
}

fn text_chunks(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_len = 0;

    for ch in text.chars() {
        current.push(ch);
        current_len += 1;

        if current_len == STREAM_CHUNK_CHARS {
            chunks.push(std::mem::take(&mut current));
            current_len = 0;
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(ErrorEnvelope {
            error: ErrorMessage {
                message: message.to_string(),
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::text_chunks;

    #[test]
    fn test_text_chunks_keeps_empty_text_as_single_chunk() {
        assert_eq!(text_chunks(""), vec![String::new()]);
    }

    #[test]
    fn test_text_chunks_groups_output_into_bounded_chunks() {
        let text = "a".repeat(65);

        let chunks = text_chunks(&text);

        assert_eq!(
            chunks,
            vec!["a".repeat(32), "a".repeat(32), "a".to_string()]
        );
    }
}
