---
layout: home

hero:
  name: "Hetero-Paged-Infer"
  text: "High-Performance LLM Inference"
  tagline: PagedAttention + Continuous Batching in Rust
  image:
    src: /images/logo.svg
    alt: Hetero-Paged-Infer Logo
  actions:
    - theme: brand
      text: Get Started
      link: /en/setup/quickstart
    - theme: alt
      text: View on GitHub
      link: https://github.com/LessUp/hetero-paged-infer
    - theme: alt
      text: API Reference
      link: /en/api/core-types

features:
  - icon: 🧠
    title: PagedAttention
    details: Block-based KV Cache management with on-demand allocation. Achieves <5% memory waste compared to 40-60% with static allocation.
  - icon: ⚡
    title: Continuous Batching
    details: Dynamic prefill/decode scheduling with priority awareness. Maximizes GPU utilization while maintaining low latency.
  - icon: 🛡️
    title: Memory Pressure Awareness
    details: Configurable OOM prevention with graceful degradation. Production-ready error handling and monitoring.
  - icon: 🔧
    title: Modular Architecture
    details: Trait-based abstractions for easy customization. Clean separation between CPU scheduler and GPU executor.
  - icon: 🧪
    title: Comprehensive Testing
    details: 121 tests including unit, property-based, and integration tests. Property tests verify critical invariants.
  - icon: 🚀
    title: OpenAI Compatible
    details: Built-in HTTP server with OpenAI-compatible API. Supports streaming via Server-Sent Events (SSE).
---

## Key Metrics

| Metric | Value | Description |
|--------|:-----:|-------------|
| Memory Waste | **<5%** | vs 40-60% with static allocation |
| Throughput Gain | **+50%** | vs static batching |
| Tests Passed | **121+** | Unit, property, integration |
| Unsafe Code | **0** | Full Rust safety guarantees |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build in release mode
cargo build --release

# Run inference
./target/release/hetero-infer --input "Hello, world!" --max-tokens 50
```

## Documentation

- [Quick Start Guide](/en/setup/quickstart) — Get up and running quickly
- [Architecture](/en/architecture/overview) — System design deep dive
- [API Reference](/en/api/core-types) — Complete API documentation
- [Benchmarks](/en/benchmarks/) — Performance comparisons
