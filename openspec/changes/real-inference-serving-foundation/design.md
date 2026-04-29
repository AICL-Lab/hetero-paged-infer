# 设计说明：real-inference-serving-foundation

## Context

当前 `InferenceEngine` 适合验证调度、KV Cache 和错误恢复语义，但不适合作为外部系统直接调用的服务入口。第一阶段目标不是把执行层完全重写为真实 CUDA，而是先建立**稳定的服务边界与可插拔运行时**，使项目能：

1. 在本地保留可测试的 mock 推理路径；
2. 挂接真实 tokenizer；
3. 通过桥接方式调用本地真实推理后端；
4. 以 OpenAI 兼容 API 对外暴露。

## Goals

- 在不破坏现有引擎测试的前提下增加真实 tokenizer 支持
- 提供可配置的服务运行时（本地引擎 / 命令桥接）
- 提供 OpenAI 兼容 completions / chat completions 接口
- 提供 SSE streaming、health、readiness、metrics
- 将 HTTP 适配层限制在 `src/server.rs`

## Non-Goals

- 不在本轮实现原生模型加载与 CUDA kernel
- 不在本轮实现 prefix caching / chunked prefill
- 不在本轮实现多节点路由或 KV 外置
- 不在本轮实现完整 OpenAI 全量字段兼容

## Decisions

### 1. Tokenizer 采用“双实现 + 工厂”

`src/tokenizer.rs` 扩展为：

- `SimpleTokenizer`：保留，继续服务现有单元测试和本地 mock 路径
- `HuggingFaceTokenizer`：从 tokenizer JSON 文件加载
- `build_tokenizer(config: &EngineConfig)`：根据配置创建 tokenizer

`TokenizerTrait` 增加 `try_encode` / `try_decode`，由引擎使用错误可返回版本；原有 `encode` / `decode` 保持便捷接口，用于测试和简单调用。

### 2. 服务运行时与核心引擎解耦

新增 `src/server.rs`，将服务执行抽象为两类后端：

- `LocalEngineBackend`：使用现有 `InferenceEngine`
- `CommandBridgeBackend`：通过本地命令桥接真实后端

这样可以保证：

- HTTP 语义不侵入 `engine.rs`
- 本地 mock 链路继续保留
- 后续若引入 `llama.cpp`、`vLLM` bridge 或自研后端，只需替换服务后端实现

### 3. 命令桥接优先于 HTTP 转发桥接

本轮优先选择**命令桥接**而不是远程 HTTP 转发：

- 更适合“单机真实服务基础”的定位
- 无需先假设额外常驻上游服务
- 测试更容易稳定落地

命令桥接约定：

- 服务把 prompt 和生成参数传给外部命令
- 外部命令返回完整文本到 stdout
- streaming 由服务端按片段转换为 SSE 事件

### 4. Streaming 采用“最小 SSE 兼容”

`stream=true` 时，服务返回 `text/event-stream`，并发送：

- 若干 `data: {...}` 增量事件
- 末尾 `data: [DONE]`

本轮 streaming 是“服务层兼容语义”，不要求本地后端天然逐 token 流式返回；对于命令桥接后端，可以先基于完整文本分片输出。

### 5. Metrics 采用进程内原子计数器

为了避免第一阶段引入过重的 observability 依赖，本轮 metrics 采用：

- `Arc<AtomicU64>` 维护请求总数、错误数、进行中请求数、流式请求数
- `/metrics` 输出 Prometheus text format
- tracing 采用 `tracing` + `tracing-subscriber`

这满足最小可观测性目标，也为后续接入更完整指标库保留空间。

## File Changes

### `Cargo.toml`

- 增加 `axum`、`tokio`、`tracing`、`tracing-subscriber`
- 增加 `async-stream`、`futures-util`
- 增加 `tokenizers`
- 增加 `tower` 作为测试辅助

### `src/config.rs`

新增：

- `TokenizerConfig`
- `TokenizerKind`
- `ServingConfig`
- `ServingBackendConfig`
- `ServingBackendKind`
- 对默认值、序列化和验证的支持

### `src/tokenizer.rs`

新增：

- `HuggingFaceTokenizer`
- `TokenizerTrait::try_encode`
- `TokenizerTrait::try_decode`
- `build_tokenizer`

### `src/error.rs`

新增服务/桥接相关错误变体，确保配置、tokenizer 装载和命令桥接错误能显式上抛。

### `src/engine.rs`

修改默认构造流程，允许从配置构建 tokenizer；完成请求时使用可失败的 decode 路径。

### `src/server.rs`

新增：

- OpenAI 请求/响应 DTO
- `create_router`
- `LocalEngineBackend`
- `CommandBridgeBackend`
- SSE streaming
- `health` / `ready` / `metrics`

### `src/main.rs`

新增 server 启动模式，同时保留 CLI 单次请求模式。

### `src/lib.rs`

导出 server 相关公开类型和构建入口。

### `tests/server_integration.rs`

新增服务集成测试，覆盖：

- `/healthz`
- `/readyz`
- `/metrics`
- `/v1/completions`
- `/v1/chat/completions`
- streaming SSE

## Risks

### 1. 现有引擎是同步实现

`InferenceEngine::run` 会串行跑完当前所有请求，因此 `LocalEngineBackend` 第一版会以互斥方式串行处理请求。这是可接受的基础版行为，但不应被误认为最终高并发表现。

### 2. HuggingFace tokenizer 与现有特定 special token 约定可能不完全一致

为避免把模型生态差异扩大成架构问题，本轮仅要求：

- 能成功装载 tokenizer JSON
- 能进行编码/解码
- 配置中仍保留本地 special token 定义用于 mock 路径

### 3. 命令桥接只能拿到完整文本

这意味着服务端 streaming 是“兼容式 streaming”，不是严格逐 token 真流式。该限制会在后续真实执行后端 change 中解决。

## Validation

实现完成后应满足：

1. `openspec validate --all` 通过
2. `cargo test` 通过
3. `cargo fmt --check` 通过
4. `cargo clippy --all-targets -- -D warnings` 通过
5. server 集成测试验证 completions / chat / streaming / health / metrics
