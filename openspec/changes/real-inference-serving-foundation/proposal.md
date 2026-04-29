# real-inference-serving-foundation

## Why

当前项目已经具备 `PagedAttention`、`Continuous Batching` 和可测试的控制面骨架，但仍停留在“库级 mock 原型”阶段：

- 默认 tokenizer 仍是字符级 mock；
- 默认执行路径仍依赖 `MockGPUExecutor`；
- 缺少 OpenAI 兼容 HTTP 服务入口；
- 缺少最小 health / readiness / metrics / tracing 暴露面。

如果继续直接推进 prefix caching、speculative decoding 或 scale-out，项目会长期缺少一个可集成、可演示、可外部调用的稳定服务基础。因此需要先完成“真实推理与服务基础”。

## What Changes

本 change 聚焦于把项目推进到“单机可用的真实服务基础版”，但保持范围可控：

1. 为 `EngineConfig` 增加 tokenizer / serving / backend bridge 配置；
2. 在现有 `SimpleTokenizer` 之外新增可加载 HuggingFace tokenizer JSON 的实现；
3. 保留本地 mock 推理链路，同时新增**命令桥接执行后端**，允许服务层对接本地真实推理进程；
4. 新增 OpenAI 兼容 HTTP API：
   - `POST /v1/completions`
   - `POST /v1/chat/completions`
   - streaming SSE 基础支持
5. 新增最小 health / readiness / metrics 暴露；
6. 保留 CLI 模式，并新增 server 启动入口。

## Non-Goals

本 change 明确不包含以下能力：

- prefix caching
- chunked prefill
- structured outputs
- tool calling
- speculative decoding
- quantization
- LoRA / multi-LoRA
- disaggregated prefill / decode
- 外部 KV 层
- 原生 CUDA 内核集成

## Impact

完成后，项目将具备：

- 可配置的真实 tokenizer 装载能力；
- 可通过命令桥接接入本地真实推理后端；
- 可被标准 OpenAI 客户端调用的 HTTP 服务入口；
- 最小可观测性和服务健康探针；
- 保持现有 mock 测试链路与核心调度实现不被破坏。
