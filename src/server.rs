//! HTTP 服务层
//!
//! 提供 OpenAI 兼容的最小服务接口、健康检查与指标暴露。
//!
//! # 并发模型
//!
//! [`InferenceEngine`] 由唯一的后台任务（"引擎循环"）独占持有，HTTP handler
//! 不持锁、也不在 async 运行时上执行同步推理：
//!
//! - handler 通过 mpsc 通道提交请求，并等待该请求的专属事件流；
//! - 引擎循环在每一步之间清空提交队列，使调度器能看到并发请求、
//!   组成真正的 continuous batching 批次；
//! - 流式响应随 token 生成逐片段推送（真实的首 token 延迟）。

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::types::{CompletedRequest, FinishReason, GenerationParams, RequestId};
use crate::InferenceEngine;
use async_stream::stream;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode, Uri};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};

/// 提交队列容量；队列满时 handler 异步等待（背压），不会丢失请求。
const SUBMISSION_QUEUE_CAPACITY: usize = 1024;

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
            "# TYPE paged_requests_total counter\npaged_requests_total {}\n# TYPE paged_errors_total counter\npaged_errors_total {}\n# TYPE paged_inflight_requests gauge\npaged_inflight_requests {}\n# TYPE paged_streaming_requests_total counter\npaged_streaming_requests_total {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.errors_total.load(Ordering::Relaxed),
            self.inflight_requests.load(Ordering::Relaxed),
            self.streaming_requests_total.load(Ordering::Relaxed),
        )
    }
}

/// RAII guard：保证 inflight gauge 在任何退出路径（含提前返回）都会递减。
#[derive(Clone)]
struct InflightGuard {
    metrics: Arc<ServerMetrics>,
}

impl InflightGuard {
    fn new(metrics: &Arc<ServerMetrics>) -> Self {
        metrics.inflight_requests.fetch_add(1, Ordering::Relaxed);
        Self {
            metrics: metrics.clone(),
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.metrics
            .inflight_requests
            .fetch_sub(1, Ordering::Relaxed);
    }
}

/// 提交给引擎循环的请求。
struct Submission {
    prompt: String,
    params: GenerationParams,
    /// 回传准入回执，或准入阶段的错误。
    admit: oneshot::Sender<Result<Admission, EngineError>>,
    /// 该请求的事件流（token 片段 + 终态）。
    events: mpsc::UnboundedSender<RequestEvent>,
}

/// 引擎循环接受请求后返回的回执。
struct Admission {
    request_id: RequestId,
    /// 用真实 tokenizer 统计的 prompt token 数（用于精确的 usage 报告）。
    prompt_tokens: usize,
}

/// 引擎循环推送给等待 handler 的每请求事件。
enum RequestEvent {
    /// 新生成的文本片段（每生成一个 token 推送一次）。
    Chunk(String),
    /// 请求到达终态（成功或失败）。
    Done(CompletedRequest),
}

#[derive(Clone)]
struct AppState {
    config: EngineConfig,
    submit_tx: mpsc::Sender<Submission>,
    metrics: Arc<ServerMetrics>,
    response_counter: Arc<AtomicU64>,
}

impl AppState {
    fn next_id(&self, prefix: &str) -> String {
        let value = self.response_counter.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{value}")
    }

    /// 向引擎循环提交请求，返回准入回执与其事件流。
    async fn submit(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<(Admission, mpsc::UnboundedReceiver<RequestEvent>), ApiError> {
        let (admit_tx, admit_rx) = oneshot::channel();
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        self.submit_tx
            .send(Submission {
                prompt: prompt.to_string(),
                params,
                admit: admit_tx,
                events: events_tx,
            })
            .await
            .map_err(|_| ApiError::internal("engine loop is not running"))?;
        let admission = admit_rx
            .await
            .map_err(|_| ApiError::internal("engine loop dropped the request"))??;
        Ok((admission, events_rx))
    }

    /// 非流式生成：等待请求到达终态并返回完整结果。
    async fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationResult, ApiError> {
        let (admission, mut events) = self.submit(prompt, params).await?;
        while let Some(event) = events.recv().await {
            if let RequestEvent::Done(completed) = event {
                return generation_result(admission.prompt_tokens, completed);
            }
            // 非流式模式下忽略 token 片段事件
        }
        Err(ApiError::internal(format!(
            "engine loop ended before request {} completed",
            admission.request_id
        )))
    }
}

/// 将终态请求转换为 API 结果；失败的请求映射为 500 错误。
fn generation_result(
    prompt_tokens: usize,
    completed: CompletedRequest,
) -> Result<GenerationResult, ApiError> {
    if completed.success {
        Ok(GenerationResult {
            text: completed.output_text,
            prompt_tokens,
            completion_tokens: completed.output_tokens.len(),
            finish_reason: completed
                .finish_reason
                .unwrap_or(FinishReason::Stop)
                .as_str()
                .to_string(),
        })
    } else {
        Err(ApiError::internal(
            completed
                .error
                .unwrap_or_else(|| "generation failed".to_string()),
        ))
    }
}

/// HTTP API 错误，输出 OpenAI 风格的错误信封。
///
/// 将引擎分层错误映射到恰当的状态码：
/// 验证错误 → 400，过载（内存压力 / 并发上限）→ 429（含 `Retry-After`），
/// 资源不存在 → 404，其余内部错误 → 500。
enum ApiError {
    BadRequest(String),
    NotFound(String),
    Overloaded(String),
    Internal(String),
}

impl ApiError {
    fn internal(message: impl Into<String>) -> Self {
        ApiError::Internal(message.into())
    }
}

impl From<EngineError> for ApiError {
    fn from(err: EngineError) -> Self {
        match err {
            EngineError::EmptyInput
            | EngineError::InputTooLong(_, _)
            | EngineError::TotalLengthTooLong(_, _)
            | EngineError::InvalidMaxTokens(_)
            | EngineError::InvalidTemperature(_)
            | EngineError::InvalidTopP(_)
            | EngineError::TooManyStopSequences(_)
            | EngineError::UnsupportedGenerationMode(_) => ApiError::BadRequest(err.to_string()),
            EngineError::MemoryPressure | EngineError::MaxConcurrentSequencesReached(_) => {
                ApiError::Overloaded(err.to_string())
            }
            _ => ApiError::Internal(err.to_string()),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "invalid_request_error", m),
            ApiError::Overloaded(m) => (StatusCode::TOO_MANY_REQUESTS, "overloaded_error", m),
            ApiError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", m),
        };
        let mut response = Json(serde_json::json!({
            "error": { "message": message, "type": error_type }
        }))
        .into_response();
        *response.status_mut() = status;
        if status == StatusCode::TOO_MANY_REQUESTS {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

/// 服务层内部的生成结果（HTTP 层私有类型，不属于库公共 API）。
#[derive(Debug)]
struct GenerationResult {
    /// 生成的文本
    text: String,
    /// prompt token 数
    prompt_tokens: usize,
    /// 生成 token 数
    completion_tokens: usize,
    /// 生成结束原因（stop / length）
    finish_reason: String,
}

/// `stop` 停止序列：单个字符串或字符串数组（OpenAI 兼容形态）。
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    /// 是否等价于"不停止"（null / 空字符串 / 空数组）。
    fn is_empty(&self) -> bool {
        match self {
            StopSequence::Single(s) => s.is_empty(),
            StopSequence::Multiple(v) => v.is_empty(),
        }
    }

    /// 展开为字符串列表（PINF-105：stop 已支持，交给引擎生成端检测）。
    fn to_vec(&self) -> Vec<String> {
        match self {
            StopSequence::Single(s) => vec![s.clone()],
            StopSequence::Multiple(v) => v.clone(),
        }
    }
}

/// CPU 后端未实现的 OpenAI 兼容参数（PINF-104）。
///
/// 与 PINF-101 的原则一致：不支持的参数在准入阶段显式拒绝（400），
/// 而不是被 serde 静默忽略。仅"非默认值"被拒绝——默认值（例如
/// `frequency_penalty=0`、`n=1`、`stop=null`/`[]`、`echo=false`）语义上
/// 无害，直接放行。其余完全未知的字段（如 `user`）仍被忽略，不影响
/// 生成语义。
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct UnsupportedParams {
    n: Option<u32>,
    seed: Option<i64>,
    logprobs: Option<serde_json::Value>,
    echo: Option<bool>,
    suffix: Option<String>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    best_of: Option<u32>,
    stream_options: Option<serde_json::Value>,
    // chat/completions 特有
    tools: Option<serde_json::Value>,
    tool_choice: Option<serde_json::Value>,
    response_format: Option<serde_json::Value>,
    function_call: Option<serde_json::Value>,
    functions: Option<serde_json::Value>,
}

/// `logprobs` 是否处于"关闭"状态（`false` 或 `0`）。
fn logprobs_is_off(v: &serde_json::Value) -> bool {
    matches!(v, serde_json::Value::Bool(false)) || matches!(v.as_u64(), Some(0))
}

/// 值是否为空数组（等价于"未提供"）。
fn is_empty_array(v: &serde_json::Value) -> bool {
    matches!(v, serde_json::Value::Array(a) if a.is_empty())
}

/// 拒绝未支持参数的非默认值（PINF-104）。
///
/// 返回 400 + `invalid_request_error`，消息带参数名以便客户端定位。
fn reject_unsupported_params(p: &UnsupportedParams) -> Result<(), ApiError> {
    let unsupported = |name: &str| {
        Err(ApiError::BadRequest(format!(
            "parameter '{name}' is not supported by the CPU reference backend"
        )))
    };

    if p.n.is_some_and(|n| n != 1) {
        return unsupported("n");
    }
    if p.seed.is_some() {
        return unsupported("seed");
    }
    if p.logprobs.as_ref().is_some_and(|v| !logprobs_is_off(v)) {
        return unsupported("logprobs");
    }
    if p.echo == Some(true) {
        return unsupported("echo");
    }
    if p.suffix.as_ref().is_some_and(|s| !s.is_empty()) {
        return unsupported("suffix");
    }
    if p.frequency_penalty.is_some_and(|v| v != 0.0) {
        return unsupported("frequency_penalty");
    }
    if p.presence_penalty.is_some_and(|v| v != 0.0) {
        return unsupported("presence_penalty");
    }
    if p.best_of.is_some_and(|v| v != 1) {
        return unsupported("best_of");
    }
    if p.stream_options.is_some() {
        return unsupported("stream_options");
    }
    if p.tools.as_ref().is_some_and(|v| !is_empty_array(v)) {
        return unsupported("tools");
    }
    if p.tool_choice.is_some() {
        return unsupported("tool_choice");
    }
    if p.response_format.is_some() {
        return unsupported("response_format");
    }
    if p.function_call.is_some() {
        return unsupported("function_call");
    }
    if p.functions.as_ref().is_some_and(|v| !is_empty_array(v)) {
        return unsupported("functions");
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    stop: Option<StopSequence>,
    #[serde(flatten)]
    unsupported: UnsupportedParams,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    stop: Option<StopSequence>,
    #[serde(flatten)]
    unsupported: UnsupportedParams,
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
    finish_reason: String,
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
    finish_reason: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

struct PreparedGenerationRequest {
    model: String,
    prompt: String,
    params: GenerationParams,
    stream: bool,
}

/// 根据配置创建 router（构造默认引擎并启动引擎循环）
///
/// 必须在 tokio 运行时内调用（后台引擎循环经由 `tokio::spawn` 启动）。
pub fn create_router(config: EngineConfig) -> Result<Router, EngineError> {
    let engine = InferenceEngine::new(config.clone())?;
    create_router_with_engine(config, engine)
}

/// 使用预先构造的引擎创建 router（用于测试注入自定义执行器）
///
/// 必须在 tokio 运行时内调用。
pub fn create_router_with_engine(
    config: EngineConfig,
    engine: InferenceEngine,
) -> Result<Router, EngineError> {
    config.validate()?;
    let (submit_tx, submit_rx) = mpsc::channel(SUBMISSION_QUEUE_CAPACITY);
    tokio::spawn(engine_loop(engine, submit_rx));

    let state = Arc::new(AppState {
        config,
        submit_tx,
        metrics: Arc::new(ServerMetrics::default()),
        response_counter: Arc::new(AtomicU64::new(1)),
    });

    Ok(Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(not_found)
        .with_state(state))
}

/// 后台引擎循环：引擎的唯一所有者。
///
/// 每一步之间清空提交队列（使调度器能看到并发请求并组批），并把每请求事件
/// 推送给对应 handler。所有提交端被丢弃（服务器关闭）时退出。
async fn engine_loop(mut engine: InferenceEngine, mut submit_rx: mpsc::Receiver<Submission>) {
    // request_id → 事件发送端，用于路由每请求事件
    let mut waiters: HashMap<RequestId, mpsc::UnboundedSender<RequestEvent>> = HashMap::new();

    loop {
        // 空闲时阻塞等待首个提交，避免忙等。
        if !engine.has_pending_work() {
            match submit_rx.recv().await {
                Some(submission) => admit_submission(&mut engine, submission, &mut waiters),
                None => return, // 所有提交端已丢弃
            }
        }

        // 非阻塞地清空队列中所有提交，让调度器看到并发请求。
        while let Ok(submission) = submit_rx.try_recv() {
            admit_submission(&mut engine, submission, &mut waiters);
        }

        match engine.step_events() {
            Ok(events) => {
                let mut disconnected: Vec<RequestId> = Vec::new();
                for (request_id, chunk) in events.chunks {
                    if let Some(tx) = waiters.get(&request_id) {
                        // 发送失败 == 对端已断开：登记取消，释放调度资源
                        if tx.send(RequestEvent::Chunk(chunk)).is_err() {
                            disconnected.push(request_id);
                        }
                    }
                }
                for request_id in disconnected {
                    waiters.remove(&request_id);
                    engine.cancel_request(request_id);
                }
                for completed in events.completed {
                    if let Some(tx) = waiters.remove(&completed.request_id) {
                        let _ = tx.send(RequestEvent::Done(completed));
                    }
                }
            }
            Err(err) => {
                // 可归属的序列已在 step_events 内被标记失败并经 Done 事件上报；
                // 此处仅记录无法归属的步骤级错误。
                log::error!("engine step failed: {err}");
            }
        }

        // 每步让出一次：生成循环本身没有 await 点，单线程 runtime 上
        // 会饿死 handler / SSE 流任务（首 token 永远到不了客户端）；
        // 多线程 runtime 上也避免独占 worker。
        tokio::task::yield_now().await;
    }
}

fn admit_submission(
    engine: &mut InferenceEngine,
    submission: Submission,
    waiters: &mut HashMap<RequestId, mpsc::UnboundedSender<RequestEvent>>,
) {
    let result = engine
        .submit_request(&submission.prompt, submission.params)
        .map(|(request_id, prompt_tokens)| Admission {
            request_id,
            prompt_tokens,
        });
    if let Ok(admission) = &result {
        waiters.insert(admission.request_id, submission.events);
    }
    let _ = submission.admit.send(result);
}

async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

/// 就绪探针：仅当引擎循环仍在运行时返回 ready。
async fn readyz(State(state): State<Arc<AppState>>) -> Response {
    if state.submit_tx.is_closed() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "not_ready" })),
        )
            .into_response()
    } else {
        Json(serde_json::json!({ "status": "ready" })).into_response()
    }
}

async fn not_found(uri: Uri) -> ApiError {
    ApiError::NotFound(format!("unknown route: {}", uri.path()))
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
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let _guard = InflightGuard::new(&state.metrics);
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    let Json(request) = match body {
        Ok(json) => json,
        Err(rejection) => return ApiError::BadRequest(rejection.body_text()).into_response(),
    };

    let prepared = match prepare_completion_request(&state, request) {
        Ok(prepared) => prepared,
        Err(err) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return err.into_response();
        }
    };

    respond_generation(&state, StreamKind::Completion, "cmpl", prepared).await
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let _guard = InflightGuard::new(&state.metrics);
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);

    let Json(request) = match body {
        Ok(json) => json,
        Err(rejection) => return ApiError::BadRequest(rejection.body_text()).into_response(),
    };

    let prepared = match prepare_chat_request(&state, request) {
        Ok(prepared) => prepared,
        Err(err) => {
            state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
            return err.into_response();
        }
    };

    respond_generation(&state, StreamKind::Chat, "chatcmpl", prepared).await
}

/// completions 与 chat_completions 共用的生成/响应流程。
async fn respond_generation(
    state: &Arc<AppState>,
    kind: StreamKind,
    id_prefix: &str,
    prepared: PreparedGenerationRequest,
) -> Response {
    if prepared.stream {
        match state.submit(&prepared.prompt, prepared.params).await {
            Ok((admission, events)) => {
                state
                    .metrics
                    .streaming_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                stream_response(
                    state,
                    kind,
                    id_prefix,
                    &prepared.model,
                    admission.prompt_tokens,
                    events,
                )
            }
            Err(err) => {
                state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                err.into_response()
            }
        }
    } else {
        match state.generate(&prepared.prompt, prepared.params).await {
            Ok(generated) => match kind {
                StreamKind::Completion => {
                    let response_usage = usage(&generated);
                    Json(CompletionResponse {
                        id: state.next_id(id_prefix),
                        object: "text_completion",
                        created: unix_timestamp(),
                        model: prepared.model,
                        choices: vec![CompletionChoice {
                            text: generated.text,
                            index: 0,
                            finish_reason: generated.finish_reason,
                        }],
                        usage: response_usage,
                    })
                    .into_response()
                }
                StreamKind::Chat => {
                    let response_usage = usage(&generated);
                    Json(ChatCompletionResponse {
                        id: state.next_id(id_prefix),
                        object: "chat.completion",
                        created: unix_timestamp(),
                        model: prepared.model,
                        choices: vec![ChatCompletionChoice {
                            index: 0,
                            message: ChatMessage {
                                role: "assistant".to_string(),
                                content: generated.text,
                            },
                            finish_reason: generated.finish_reason,
                        }],
                        usage: response_usage,
                    })
                    .into_response()
                }
            },
            Err(err) => {
                state.metrics.errors_total.fetch_add(1, Ordering::Relaxed);
                err.into_response()
            }
        }
    }
}

/// 流式响应的两种载荷形状。
#[derive(Clone, Copy)]
enum StreamKind {
    Completion,
    Chat,
}

impl StreamKind {
    fn chunk_payload(&self, id: &str, created: u64, model: &str, chunk: &str) -> serde_json::Value {
        match self {
            StreamKind::Completion => serde_json::json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"text": chunk, "index": 0, "finish_reason": serde_json::Value::Null}],
            }),
            StreamKind::Chat => serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": serde_json::Value::Null}],
            }),
        }
    }

    fn final_payload(
        &self,
        id: &str,
        created: u64,
        model: &str,
        usage: &Usage,
        finish_reason: &str,
    ) -> serde_json::Value {
        let mut payload = match self {
            StreamKind::Completion => serde_json::json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"text": "", "index": 0, "finish_reason": finish_reason}],
            }),
            StreamKind::Chat => serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }),
        };
        payload["usage"] = serde_json::json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        });
        payload
    }
}

/// token 级流式响应：随引擎循环推送的事件逐片段发送，
/// 首字节延迟 = 首 token 生成时间（而非完整生成时间）。
fn stream_response(
    state: &Arc<AppState>,
    kind: StreamKind,
    id_prefix: &str,
    model: &str,
    prompt_tokens: usize,
    mut events: mpsc::UnboundedReceiver<RequestEvent>,
) -> Response {
    let id = state.next_id(id_prefix);
    let created = unix_timestamp();
    let model = model.to_string();
    Sse::new(stream! {
        let mut failure: Option<String> = None;
        let mut terminated = false;
        while let Some(event) = events.recv().await {
            match event {
                RequestEvent::Chunk(chunk) => {
                    yield Ok::<Event, Infallible>(Event::default()
                        .data(kind.chunk_payload(&id, created, &model, &chunk).to_string()));
                }
                RequestEvent::Done(completed) => {
                    terminated = true;
                    if completed.success {
                        let usage = Usage {
                            prompt_tokens,
                            completion_tokens: completed.output_tokens.len(),
                            total_tokens: prompt_tokens + completed.output_tokens.len(),
                        };
                        let finish_reason = completed
                            .finish_reason
                            .unwrap_or(FinishReason::Stop)
                            .as_str();
                        yield Ok::<Event, Infallible>(Event::default()
                            .data(kind.final_payload(&id, created, &model, &usage, finish_reason)
                                .to_string()));
                    } else {
                        failure = completed.error;
                    }
                    break;
                }
            }
        }
        if !terminated {
            // 通道关闭但从未收到终态（如引擎循环异常退出）：
            // 明确报错而不是伪装成正常结束。
            let payload = serde_json::json!({
                "error": {
                    "message": "engine loop ended before request completed",
                    "type": "internal_error"
                }
            });
            yield Ok::<Event, Infallible>(Event::default().data(payload.to_string()));
        }
        if let Some(message) = failure {
            let payload = serde_json::json!({
                "error": { "message": message, "type": "internal_error" }
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
) -> Result<PreparedGenerationRequest, ApiError> {
    reject_unsupported_params(&request.unsupported)?;
    let prompt = validate_prompt(request.prompt)?;
    Ok(PreparedGenerationRequest {
        model: resolve_model(state, request.model)?,
        prompt,
        params: generation_params(
            request.max_tokens,
            request.temperature,
            request.top_p,
            stop_sequences(request.stop),
        )?,
        stream: request.stream.unwrap_or(false),
    })
}

fn prepare_chat_request(
    state: &AppState,
    request: ChatCompletionRequest,
) -> Result<PreparedGenerationRequest, ApiError> {
    reject_unsupported_params(&request.unsupported)?;
    if request.messages.is_empty() {
        return Err(ApiError::BadRequest(
            "messages must not be empty".to_string(),
        ));
    }
    for message in &request.messages {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return Err(ApiError::BadRequest(format!(
                "invalid role '{}': expected one of system, user, assistant",
                message.role
            )));
        }
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
        model: resolve_model(state, request.model)?,
        prompt,
        params: generation_params(
            request.max_tokens,
            request.temperature,
            request.top_p,
            stop_sequences(request.stop),
        )?,
        stream: request.stream.unwrap_or(false),
    })
}

/// 将请求的 `stop` 字段展开为引擎使用的停止序列列表。
/// 空序列（null / 空字符串 / 空数组）等价于未提供。
fn stop_sequences(stop: Option<StopSequence>) -> Vec<String> {
    match stop {
        Some(s) if s.is_empty() => Vec::new(),
        Some(s) => s.to_vec(),
        None => Vec::new(),
    }
}

/// 校验请求的 model 字段：缺省时使用配置的模型名，
/// 显式指定但不匹配时返回 404（OpenAI 惯例）。
fn resolve_model(state: &AppState, requested: Option<String>) -> Result<String, ApiError> {
    match requested {
        Some(name) if name == state.config.serving.model_name => Ok(name),
        Some(name) => Err(ApiError::NotFound(format!("model '{name}' not found"))),
        None => Ok(state.config.serving.model_name.clone()),
    }
}

fn generation_params(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Vec<String>,
) -> Result<GenerationParams, ApiError> {
    // 缺省值即 greedy（后端当前唯一支持的生成模式）；显式传入其他
    // 采样参数的请求会在 submit 阶段以 400 拒绝，而非静默降级。
    let params = GenerationParams {
        max_tokens: max_tokens.unwrap_or(16),
        temperature: temperature.unwrap_or(0.0),
        top_p: top_p.unwrap_or(1.0),
        stop,
    };
    params
        .validate()
        .map_err(|err| ApiError::BadRequest(err.to_string()))?;
    Ok(params)
}

fn validate_prompt(prompt: String) -> Result<String, ApiError> {
    if prompt.trim().is_empty() {
        Err(ApiError::BadRequest("prompt must not be empty".to_string()))
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

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
