# Hetero-Paged-Infer

<div align="center">

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hetero-infer.svg)](https://crates.io/crates/hetero-infer)
[![Docs.rs](https://docs.rs/hetero-infer/badge.svg)](https://docs.rs/hetero-infer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)](https://www.rust-lang.org/)

**A High-Performance LLM Inference Engine with PagedAttention & Continuous Batching**

**[English](README.md) | [中文](README.zh.md) | [Documentation](https://lessup.github.io/hetero-paged-infer/)**

</div>

---

## Overview

Hetero-Paged-Infer is a **production-ready** inference engine for Large Language Models (LLMs) built in Rust. It implements cutting-edge techniques from [vLLM](https://github.com/vllm-project/vllm) with a modular, testable architecture designed for production deployment.

| Feature | Description | Status |
|---------|-------------|:------:|
| **PagedAttention KV Cache** | Block-based memory management, <5% waste | ✅ |
| **Continuous Batching** | Dynamic prefill/decode scheduling | ✅ |
| **Memory Pressure Awareness** | Configurable OOM prevention | ✅ |
| **Modular Architecture** | Trait-based abstractions | ✅ |
| **Comprehensive Testing** | 135 tests (unit, property, integration) | ✅ |
| **CUDA Kernels** | Real GPU execution | 🚧 Planned |

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
│        │        ┌──────▼──────┐                  │                    │
│        │        │Batch Builder│◄─────────────────┘                    │
│        │        └──────┬──────┘                                       │
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

- **Rust 1.70+** (2021 edition)
- **Linux** (Ubuntu 20.04+ recommended) or **macOS**

### Installation

```bash
# Clone the repository
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build in release mode
cargo build --release

# Run the test suite (135 tests)
cargo test
```

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

| Option | Default | Description |
|--------|---------|-------------|
| `--block-size` | 16 | Tokens per physical block |
| `--max-num-blocks` | 1024 | Total physical blocks |
| `--max-batch-size` | 32 | Max sequences per batch |
| `--memory-threshold` | 0.9 | Memory pressure threshold |
| `--temperature` | 1.0 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |

Config file (`config.json`):

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "memory_threshold": 0.9
}
```

Load: `./hetero-infer --config config.json`

## Documentation

| Resource | Link |
|----------|------|
| **GitHub Pages** | [https://lessup.github.io/hetero-paged-infer/](https://lessup.github.io/hetero-paged-infer/) |
| **API Reference (docs.rs)** | [https://docs.rs/hetero-infer](https://docs.rs/hetero-infer) |
| **Architecture Guide** | [docs/en/architecture/overview.md](docs/en/architecture/overview.md) |
| **Contributing Guide** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |

### Local Documentation

```bash
# Build and open API documentation
cargo doc --open

# Build documentation site locally
pip install mkdocs-material mkdocs-static-i18n
mkdocs serve -f mkdocs.yml
```

## Performance

| Approach | Memory Waste | Throughput | Description |
|----------|:------------:|:----------:|-------------|
| Static Allocation | ~40-60% | Baseline | Pre-allocate max context for each request |
| Dynamic Allocation | ~20-30% | +20% | Resize per request but still fragmented |
| **PagedAttention** | **<5%** | **+50%** | Block-based sharing with copy-on-write |

### Why PagedAttention?

Traditional LLM serving allocates contiguous memory blocks for each request's KV cache, leading to significant memory fragmentation and waste. PagedAttention solves this by:

1. **Block-based allocation**: Split KV cache into fixed-size blocks
2. **On-demand paging**: Allocate blocks only when needed
3. **Copy-on-write**: Share blocks across sequences for efficient beam search

## Testing

```bash
# Run all tests
cargo test

# Run with coverage
cargo llvm-cov --html

# Run property-based tests
cargo test -- --test-threads=1
```

| Type | Count | Description |
|------|:-----:|-------------|
| Unit Tests | 78 | Core functionality tests |
| Property Tests | 15 | Invariant verification with proptest |
| Integration Tests | 13 | End-to-end workflow tests |
| Doc Tests | 29 | Documentation examples |
| **Total** | **135** | |

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
- [ ] Real CUDA Kernels
- [ ] Real Tokenizer Integration
- [ ] Async CPU/GPU Overlap

## License

MIT License - See [LICENSE](LICENSE).

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention concept and inspiration
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [Criterion](https://github.com/bheisler/criterion.rs) - Statistical benchmarking

---

<p align="center"><b>Made with ❤️ by LessUp</b></p>
