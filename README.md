# Hetero-Paged-Infer

<div align="center">

[![CI](https://github.com/AICL-Lab/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/AICL-Lab/hetero-paged-infer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.82%2B-orange?logo=rust)](https://www.rust-lang.org/)

**A Paged-Memory, Continuously-Batched Inference Engine Scaffold with a Mock Compute Backend**

> ⚠️ **Development Status**: This project is in early development (v0.1.0). The default path still uses a Mock GPU executor. The optional `cuda` feature now compiles and exercises a real CUDA kernel path, with host fallback when no CUDA device is available. Full production attention kernels are still future work.

**[English](README.md) | [中文](README.zh.md) | [Documentation](https://aicl-lab.github.io/hetero-paged-infer/)**

</div>

---

## Overview

Hetero-Paged-Infer is an inference engine scaffold for Large Language Models (LLMs) built in Rust. It implements the paged-memory (PagedAttention-style KV cache) and continuous-batching ideas from [vLLM](https://github.com/vllm-project/vllm) with a modular, testable architecture. The compute backend is currently a deterministic mock — real model computation, sampling, and production-grade kernels are future work.

| Feature | Description | Status |
|---------|-------------|:------:|
| **PagedAttention KV Cache** | Block-based memory management; literature context often reports <5% waste | ✅ |
| **Continuous Batching** | Dynamic prefill/decode scheduling | ✅ |
| **Memory Pressure Awareness** | Configurable OOM prevention | ✅ |
| **Modular Architecture** | Trait-based abstractions | ✅ |
| **Comprehensive Testing** | 121+ tests | ✅ |
| **OpenAI-Compatible Server** | `/v1/completions` + `/v1/chat/completions` + SSE | ✅ |
| **CUDA Feature** | Prefers an nvcc-built kernel path and falls back to a host-compatible backend when nvcc is absent | ✅ Experimental |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine (CPU)                          │
├──────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────────────┐  │
│  │ Tokenizer  │  │ Scheduler  │  │      KV Cache Manager          │  │
│  │            │  │            │  │   BlockPool + PageTable        │  │
│  └─────┬──────┘  └─────┬──────┘  └───────────────┬────────────────┘  │
│        │               │                         │                    │
├────────┼───────────────┼─────────────────────────────────────────────┤
│        │        ┌──────▼──────┐                                       │
│        │        │ GPU Executor│  (CUDA / Mock)                        │
│        │        └──────┬──────┘                                       │
│        │        ┌──────▼──────┐                                       │
│        └───────►│  KV Cache   │  (GPU Memory)                         │
│                 └─────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- **Rust 1.82+** (2021 edition)
- **Linux** (Ubuntu 20.04+ recommended) or **macOS**

### Installation

```bash
# Clone the repository
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# Build in release mode
cargo build --release

# Run the test suite (121+ tests)
cargo test
```

Optional nvcc-backed build:

```bash
CC=/usr/bin/gcc-12 \
CXX=/usr/bin/g++-12 \
CUDAHOSTCXX=/usr/bin/g++-12 \
cargo test --all-features
```

If `nvcc` is not available, `cargo test --all-features` now falls back to a host-compatible backend so CI can still build and test the CUDA feature surface.

### CLI Usage

```bash
# Basic usage
./target/release/hetero-infer --input "Hello, world!" --max-tokens 50

# With custom parameters
./target/release/hetero-infer \
  --input "Explain quantum computing" \
  --max-tokens 100 \
  --temperature 0.8 \
  --top-p 0.95

# Start OpenAI-compatible HTTP server
./target/release/hetero-infer --serve
```

### OpenAI-Compatible Server

```bash
# Start server with default address 127.0.0.1:3000
cargo run -- --serve

# Health / readiness / metrics
curl http://127.0.0.1:3000/healthz
curl http://127.0.0.1:3000/readyz
curl http://127.0.0.1:3000/metrics

# Completions
curl http://127.0.0.1:3000/v1/completions \
  -H "content-type: application/json" \
  -d '{"model":"hetero-infer","prompt":"hello","max_tokens":8}'

# Chat completions
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"hetero-infer","messages":[{"role":"user","content":"say hi"}],"max_tokens":8}'
```

### Library Usage

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

// Create engine with default configuration
let mut engine = InferenceEngine::new(EngineConfig::default())?;

// Submit a generation request
let request_id = engine.submit_request(
    "Hello, world!",
    GenerationParams { 
        max_tokens: 100, 
        temperature: 0.8, 
        top_p: 0.95 
    }
)?;

// Run inference and collect results
let results = engine.run();
for result in results {
    println!("Generated: {}", result.output_text);
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--block-size` | 16 | Tokens per physical block |
| `--max-num-blocks` | 1024 | Total physical blocks |
| `--max-batch-size` | 32 | Max sequences per batch |
| `--max-num-seqs` | 256 | Maximum number of sequences |
| `--max-model-len` | 2048 | Maximum model context length |
| `--max-total-tokens` | 4096 | Maximum tokens per batch |
| `--memory-threshold` | 0.9 | Memory pressure threshold (0.0-1.0) |
| `--max-tokens` | 100 | Maximum tokens to generate |
| `--temperature` | 1.0 | Sampling temperature (currently only range-validated; sampling not yet implemented) |
| `--top-p` | 0.9 | Nucleus sampling threshold (currently only range-validated; sampling not yet implemented) |

Config file (`config.json`):

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "max_total_tokens": 4096,
  "memory_threshold": 0.9,
  "max_retry_attempts": 2,
  "tokenizer": {
    "kind": "simple",
    "path": null
  },
  "serving": {
    "host": "127.0.0.1",
    "port": 3000,
    "model_name": "hetero-infer"
  }
}
```

Load: `./hetero-infer --config config.json`

For a HuggingFace tokenizer file:

```json
{
  "tokenizer": {
    "kind": "huggingface",
    "path": "tokenizer.json"
  }
}
```

## Documentation

| Resource | Link |
|----------|------|
| **GitHub Pages** | [https://aicl-lab.github.io/hetero-paged-infer/](https://aicl-lab.github.io/hetero-paged-infer/) |

| **Architecture Guide** | [docs/en/architecture/overview.md](docs/en/architecture/overview.md) |
| **Contributing Guide** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |

### Local Documentation

```bash
# Build and open API documentation
cargo doc --open

# Build documentation site locally
cd docs
npm install
npm run build
```

## Performance

| Approach | Memory Waste | Throughput | Description |
|----------|:------------:|:----------:|-------------|
| Static Allocation | Prior-art pattern: ~40-60% | Prior-art baseline | Pre-allocate max context for each request |
| Dynamic Allocation | Prior-art pattern: ~20-30% | Literature context: +20% | Resize per request but still fragmented |
| **PagedAttention** | **Literature context: <5%** | **Literature context: +50%** | Block-based sharing with copy-on-write |

> Note: Current benchmark figures are still measured with the mock path or derived from architecture-level estimates. The `cuda` feature now validates a minimal real kernel path, but it is not yet a production attention-kernel implementation.

### Why PagedAttention?

Traditional LLM serving allocates contiguous memory blocks for each request's KV cache, leading to significant memory fragmentation and waste. PagedAttention solves this by:

1. **Block-based allocation**: Split KV cache into fixed-size blocks
2. **On-demand paging**: Allocate blocks only when needed
3. **Copy-on-write** (not yet implemented — future direction): Share blocks across sequences for efficient beam search

## Testing

```bash
# Run all tests
cargo test

# Run with coverage
cargo llvm-cov --html

# Run property-based tests
cargo test -- --test-threads=1
```

| Type | Coverage | Description |
|------|:--------:|-------------|
| Unit Tests | Included in 121+ | Core functionality tests |
| Property Tests | Included in 121+ | Invariant verification with proptest |
| Integration Tests | Included in 121+ | End-to-end workflow tests |
| Doc Tests | Included in 121+ | Documentation examples |
| **Overall** | **121+ tests** | Combined automated coverage across the repository |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

```bash
# Run all checks before submitting
cargo test && cargo fmt --check && cargo clippy
```

## Roadmap

- [x] PagedAttention KV Cache
- [x] Continuous Batching Scheduler
- [x] Memory Pressure Awareness
- [x] Property-Based Testing
- [x] OpenAI-Compatible HTTP Server
- [x] HuggingFace Tokenizer Integration
- [x] nvcc-backed CUDA build path
- [ ] Real CUDA Kernels
- [ ] Async CPU/GPU Overlap

## License

MIT License - See [LICENSE](LICENSE).

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention concept and inspiration
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [Criterion](https://github.com/bheisler/criterion.rs) - Statistical benchmarking

---

<p align="center"><b>Made with ❤️ by AICL-Lab</b></p>
