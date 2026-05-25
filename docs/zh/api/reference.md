# API 参考

本页只保留当前真正稳定且有用的入口，不再手写复制整套内部 trait 签名，因为那正是文档过期的主要来源。

## 主要 Rust API 面

| 项目 | 作用 |
| --- | --- |
| `InferenceEngine` | 请求提交与执行的核心引擎 |
| `EngineConfig` | 引擎与 HTTP 服务的运行配置 |
| `GenerationParams` | 采样与生成限制 |
| `create_router` | OpenAI 兼容的 Axum 路由 |
| `CompletedRequest` | 每个请求的最终结果 |

## 典型库用法

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

let mut engine = InferenceEngine::new(EngineConfig::default())?;
let request_id = engine.submit_request(
    "你好，世界！",
    GenerationParams {
        max_tokens: 32,
        temperature: 1.0,
        top_p: 0.9,
    },
)?;

let completed = engine.run();
assert!(completed.iter().any(|item| item.request_id == request_id));
# Ok::<(), hetero_infer::EngineError>(())
```

## HTTP 服务面

`create_router(config)` 当前暴露：

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `POST /v1/completions`
- `POST /v1/chat/completions`

服务端直接运行本地引擎，不再存在桥接型后端。

## 核心扩展缝隙

以下接口仍然存在，主要服务于内部实现、测试或定向替换：

- `TokenizerTrait`
- `SchedulerTrait`
- `KVCacheManagerTrait`
- `GPUExecutorTrait`

如果需要查看精确签名，应以生成的 Rustdoc 为准，而不是依赖手写镜像文档。

## 权威参考

使用 Rustdoc 查看精确定义：

```bash
cargo doc --no-deps
```

高层说明优先参考：

- [核心类型](core-types.md)
- [快速开始](../setup/quickstart.md)
- [架构概览](../architecture/overview.md)
