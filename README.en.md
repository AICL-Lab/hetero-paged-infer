# Hetero-Paged-Infer

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)

[简体中文](README.md) | English

Heterogeneous inference system — A CPU-GPU collaborative inference engine based on PagedAttention and Continuous Batching.

## Features

- **PagedAttention KV Cache** — Paged VRAM management with on-demand block allocation/deallocation, copy-on-write
- **Continuous Batching** — Scheduler with prefill/decode phase management, decode-priority scheduling
- **Memory Pressure Awareness** — Configurable threshold, auto-reject new requests to prevent OOM
- **CUDA Graph Support** — Capture decode phase for accelerated repeated execution
- **Modular Architecture** — Tokenizer / Scheduler / GPU Executor / KV Cache Manager via traits

## Architecture

```
┌─────────────────────────────────────────────┐
│              InferenceEngine                │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  │
│  │Tokenizer │  │ Scheduler │  │   GPU    │  │
│  │          │  │(prefill/  │  │ Executor │  │
│  │          │  │ decode)   │  │          │  │
│  └──────────┘  └─────┬─────┘  └──────────┘  │
│               ┌──────┴──────┐                │
│               │ KV Cache    │                │
│               │ Manager     │                │
│               └─────────────┘                │
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
cargo build
cargo test
cargo run -- --input "Hello, world!" --max-tokens 50
cargo run -- --config config.json --input "Hello"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size` | 16 | Tokens per physical block |
| `max_num_blocks` | 1024 | Maximum physical blocks |
| `max_batch_size` | 32 | Max sequences per schedule |
| `max_model_len` | 2048 | Max context length |
| `memory_threshold` | 0.9 | Memory pressure threshold (0.0-1.0) |

## Engineering Quality

- Selective `pub use` exports, instance-level request IDs
- `EngineMetrics` real-time tracking, property-based testing (proptest)
- Layered error hierarchy, CI (fmt + clippy + test)

## License

MIT
