# API Reference

This page documents the stable entry points that are useful today. It intentionally avoids mirroring every internal trait signature, because that was a major source of stale documentation.

## Primary Rust API surface

| Item | Role |
| --- | --- |
| `InferenceEngine` | Core engine for request submission and execution |
| `EngineConfig` | Runtime configuration for engine and HTTP serving |
| `GenerationParams` | Sampling and generation limits |
| `create_router` | OpenAI-compatible Axum router |
| `CompletedRequest` | Final per-request result |

## Typical library usage

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

let mut engine = InferenceEngine::new(EngineConfig::default())?;
let request_id = engine.submit_request(
    "Hello, world!",
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

## HTTP serving surface

`create_router(config)` exposes:

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `POST /v1/completions`
- `POST /v1/chat/completions`

The server runs the local engine directly. There is no alternate bridge backend.

## Core extension seams

These interfaces still exist for engine internals and targeted replacement work:

- `TokenizerTrait`
- `Scheduler`
- `KVCacheManager`
- `GPUExecutorTrait`

They are useful for tests and experimentation, but the canonical source of truth is generated Rust documentation, not handwritten duplicated signatures.

## Canonical reference

Use Rustdoc for exact item definitions:

```bash
cargo doc --no-deps
```

For higher-level examples, prefer:

- [Core Types](core-types.md)
- [Quick Start](../setup/quickstart.md)
- [Architecture Overview](../architecture/overview.md)
