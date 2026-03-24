# Hetero-Paged-Infer

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)

English | [简体中文](README.zh-CN.md)

Heterogeneous inference system — A CPU-GPU collaborative inference engine based on PagedAttention and Continuous Batching.

## Features

- **PagedAttention-style KV Cache** — Paged block management with on-demand allocation/deallocation
- **Continuous Batching** — Scheduler with prefill/decode phase management and decode-priority scheduling
- **Memory Pressure Awareness** — Configurable threshold, auto-reject new requests to prevent OOM
- **Mock GPU Executor Interface** — Includes execution and CUDA-graph-shaped interfaces for testing and future backend replacement
- **Modular Architecture** — Tokenizer / Scheduler / GPU Executor / KV Cache Manager via traits

## Current Status

This repository currently focuses on scheduler, KV-cache, batching, and engine orchestration correctness.
The GPU executor is still a **mock implementation** used for testing; real CUDA kernels, real pinned memory,
copy-on-write KV sharing, and async CPU/GPU overlap are **not implemented yet**.

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
