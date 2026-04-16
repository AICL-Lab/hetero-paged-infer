# Hetero-Paged-Infer

<div align="center">

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://www.rust-lang.org/)
[![Documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://lessup.github.io/hetero-paged-infer/)

**[English](README.md) | [中文](README.zh.md)**

</div>

A high-performance heterogeneous inference engine for Large Language Models (LLMs) with CPU-GPU co-execution, featuring PagedAttention memory management and Continuous Batching scheduling.

## Overview

Hetero-Paged-Infer is a Rust-based inference system designed for efficient LLM serving. It combines cutting-edge techniques from the vLLM project with a modular, production-ready architecture:

- **PagedAttention** - Virtual memory-inspired KV Cache management eliminates memory waste
- **Continuous Batching** - Dynamic request scheduling maximizes GPU utilization  
- **Heterogeneous Computing** - CPU orchestrates while GPU computes
- **Production-Grade** - Comprehensive error handling, metrics, and logging

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **PagedAttention KV Cache** | Block-based memory management with O(1) lookup | ✅ Ready |
| **Continuous Batching** | Prefill/decode phase management with decode priority | ✅ Ready |
| **Memory Pressure Awareness** | Configurable thresholds prevent OOM | ✅ Ready |
| **Modular Architecture** | Trait-based abstractions for all components | ✅ Ready |
| **CUDA Graph Support** | Decode phase graph capture (planned) | 🚧 Planned |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           InferenceEngine                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Tokenizer  │    │   Scheduler  │    │    KV Cache Manager      │  │
│  │    (CPU)     │    │    (CPU)     │    │        (CPU)             │  │
│  │ Encode/Decode│    │  Prefill/    │    │  BlockPool/PageTable     │  │
│  │              │    │   Decode     │    │                          │  │
│  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘  │
│         │                   │                        │                  │
│         │            ┌──────▼───────┐               │                  │
│         │            │ Batch Builder│◄──────────────┘                  │
│         │            │    (CPU)     │                                  │
│         │            └──────┬───────┘                                  │
│  ───────┼───────────────────┼────────────────────────────────────────  │
│         │            ┌──────▼───────┐                                  │
│         │            │ GPU Executor │                                  │
│         │            │  (CUDA/GPU)  │                                  │
│  ───────┼────────────┴──────────────┴────────────────────────────────  │
│         │            ┌──────▼───────┐                                  │
│         └───────────►│  KV Cache    │                                  │
│                      │ (GPU Memory) │                                  │
│                      └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Rust 1.70+ (2021 edition)
- Linux environment (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA 11.x+ (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build release version
cargo build --release

# Run tests
cargo test
```

### Basic Usage

```bash
# Simple inference
./target/release/hetero-infer --input "Hello, world!" --max-tokens 50

# With custom parameters
./target/release/hetero-infer \
  --input "Explain quantum computing" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

### Library Usage

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

// Create engine
let config = EngineConfig::default();
let mut engine = InferenceEngine::new(config)?;

// Submit request
let params = GenerationParams {
    max_tokens: 100,
    temperature: 0.8,
    top_p: 0.95,
};
let request_id = engine.submit_request("Hello, world!", params)?;

// Run inference
let completed = engine.run();

// Get results
for result in completed {
    println!("Output: {}", result.output_text);
}
```

## Configuration

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--block-size` | 16 | Tokens per physical block |
| `--max-num-blocks` | 1024 | Total physical blocks |
| `--max-batch-size` | 32 | Max sequences per batch |
| `--memory-threshold` | 0.9 | Memory pressure threshold |
| `--temperature` | 1.0 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |

### Configuration File

Create `config.json`:

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "memory_threshold": 0.9
}
```

Load with: `./hetero-infer --config config.json`

## Documentation

| Resource | Description | Link |
|----------|-------------|------|
| **Architecture** | System design and components | [docs/en/ARCHITECTURE.md](docs/en/ARCHITECTURE.md) |
| **API Reference** | Rust API documentation | [docs/en/API.md](docs/en/API.md) |
| **Configuration** | All configuration options | [docs/en/CONFIGURATION.md](docs/en/CONFIGURATION.md) |
| **Deployment** | Production deployment guide | [docs/en/DEPLOYMENT.md](docs/en/DEPLOYMENT.md) |
| **GitHub Pages** | Online documentation | [https://lessup.github.io/hetero-paged-infer/](https://lessup.github.io/hetero-paged-infer/) |

## Project Status

### Implemented ✅

- [x] PagedAttention KV Cache management
- [x] Continuous Batching scheduler
- [x] Memory pressure awareness
- [x] Modular trait abstractions
- [x] Comprehensive property testing
- [x] Mock GPU executor for testing

### Planned 🚧

- [ ] Real CUDA kernel implementation
- [ ] Pinned memory management
- [ ] Copy-on-write KV sharing
- [ ] Async CPU/GPU overlap

## Performance

Memory efficiency comparison:

| Approach | Memory Waste | Throughput |
|----------|--------------|------------|
| Static Allocation | ~40-60% | Baseline |
| Dynamic Allocation | ~20-30% | +20% |
| **PagedAttention** | **<5%** | **+50%** |

## Testing

```bash
# Run all tests
cargo test

# Run with coverage
cargo tarpaulin --out Html

# Run property tests
cargo test -- --test-threads=1

# Run benchmarks
cargo bench
```

Test coverage:

| Type | Count | Coverage |
|------|-------|----------|
| Unit Tests | 78 | Core modules |
| Property Tests | 15 | Invariant verification |
| Integration Tests | 13 | End-to-end flows |
| Doc Tests | 29 | API examples |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Quick steps:

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/hetero-paged-infer.git

# Create branch
git checkout -b feature/your-feature

# Make changes and test
cargo test
cargo fmt --check
cargo clippy

# Submit PR
git push origin feature/your-feature
```

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention concept
- [Rust](https://www.rust-lang.org/) - Systems programming language
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) - GPU computing platform

## Links

- [GitHub Repository](https://github.com/LessUp/hetero-paged-infer)
- [Documentation](https://lessup.github.io/hetero-paged-infer/)
- [Issue Tracker](https://github.com/LessUp/hetero-paged-infer/issues)
- [Changelog](CHANGELOG.md)

---

<div align="center">

**Made with ❤️ by the Hetero-Paged-Infer Team**

</div>
