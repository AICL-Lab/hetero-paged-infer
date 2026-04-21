---
title: Hetero-Paged-Infer
hide:
  - navigation
  - toc
---

<div align="center">

# Hetero-Paged-Infer

**High-Performance LLM Inference Engine**

*PagedAttention + Continuous Batching*

[Get Started](setup/quickstart.md){ .md-button .md-button--primary }
[GitHub](https://github.com/LessUp/hetero-paged-infer){ .md-button }

</div>

---

## Features

| Feature | Description |
|---------|-------------|
| **PagedAttention** | Block-based KV Cache, <5% memory waste |
| **Continuous Batching** | Dynamic prefill/decode scheduling |
| **Production Ready** | Error handling, metrics, monitoring |
| **Well Tested** | 135 tests (unit, property, integration) |

---

## Quick Start

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build --release
./target/release/hetero-infer --input "Hello, world!" --max-tokens 50
```

---

## Performance

| Method | Memory Waste | Throughput |
|--------|:------------:|:----------:|
| Static Allocation | ~40-60% | Baseline |
| **PagedAttention** | **<5%** | **+50%** |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│            InferenceEngine (CPU)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │Tokenizer │  │Scheduler │  │ KV Cache Mgr │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       └─────────────┼───────────────┘           │
├─────────────────────┼───────────────────────────┤
│               ┌─────▼─────┐                      │
│               │    GPU    │  Executor + Memory   │
│               └───────────┘                      │
└─────────────────────────────────────────────────┘
```

---

## Documentation

- **[Setup](setup/)** - Installation and configuration
- **[Architecture](architecture/)** - System design
- **[API Reference](api/)** - API documentation
- **[Deployment](deployment/)** - Production deployment
- **[Development](development/)** - Contributing
