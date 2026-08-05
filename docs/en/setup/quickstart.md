# Quick Start

## Prerequisites

- Rust 1.82+ (2021 edition)
- Linux environment (Ubuntu 20.04+ recommended)
- NVIDIA GPU + CUDA 11.x+ (optional; not needed for the default CPU backend)

## Installation

```bash
# Clone the repository
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer

# Build release version
cargo build --release

# Run tests
cargo test
```

## Your First Inference

```bash
# Simple inference
./target/release/hetero-infer --input "Hello, world!" --max-tokens 50

# With custom parameters
./target/release/hetero-infer \
  --input "Explain machine learning" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

> **Note:** The compute core is currently a **mock implementation**: the command
> exercises the full scheduling and KV-cache paging pipeline, but the generated
> content consists of **placeholder tokens** (control characters such as CR/TAB,
> not natural language). This is expected behavior.

Real output looks like this (the `Output:` line contains placeholder control characters):

```text
Heterogeneous Inference System
==============================
Configuration:
  Block size: 16
  Max blocks: 1024
  Max batch size: 32
  Max sequences: 256

Input: Hello, world!
Generating up to 50 tokens...

Output: <placeholder tokens>
Tokens generated: 5
```

## Library Usage

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EngineConfig::default();
    let mut engine = InferenceEngine::new(config)?;

    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        top_p: 0.95,
    };

    let (request_id, _prompt_tokens) = engine.submit_request("Hello, world!", params)?;
    println!("Submitted request: {}", request_id);

    let results = engine.run();

    for result in results {
        if result.success {
            println!("Output: {}", result.output_text);
            println!("Tokens: {}", result.output_tokens.len());
        } else {
            println!("Error: {:?}", result.error);
        }
    }

    // Engine metrics (request counters, memory utilization, ...)
    let metrics = engine.get_metrics();
    println!("Memory utilization: {:.2}%", metrics.memory_utilization * 100.0);

    Ok(())
}
```

## HTTP Server

Add `--serve` to start the OpenAI-compatible server (binds `127.0.0.1:3000` by
default, model name `hetero-infer`):

```bash
./target/release/hetero-infer --serve
```

```bash
# In another terminal
curl http://127.0.0.1:3000/healthz
curl http://127.0.0.1:3000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "hetero-infer", "prompt": "Hello, world!", "max_tokens": 20}'
```

The server also exposes `/readyz` (readiness probe), `/metrics` (Prometheus text
format), `/v1/chat/completions`, and token-level SSE streaming; overloaded
requests receive 429 + `Retry-After`. See [Production Deployment](../deployment/production.md).

## Table of Contents

- [Installation Guide](installation) - Detailed setup instructions
- [Configuration](configuration) - All configuration options
- [API Reference](../api/core-types) - Complete API documentation
