# Hetero-Paged-Infer — 异构推理引擎

> High-performance heterogeneous inference engine for Large Language Models
>
> 面向大语言模型的高性能异构推理引擎

<div align="center">

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%2B-76b900.svg)](https://developer.nvidia.com/cuda)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions)

[🇺🇸 English Documentation](en/) · [🇨🇳 中文文档](zh/)

</div>

---

## ✨ 核心特性 / Features

| English | 中文 |
|---------|------|
| **PagedAttention** — Memory-efficient KV Cache management with < 7% waste | **PagedAttention** — 内存高效的 KV Cache 管理，浪费 < 7% |
| **Continuous Batching** — Decode-priority scheduling for optimal throughput | **连续批处理** — Decode 优先调度，实现最佳吞吐量 |
| **Heterogeneous Computing** — CPU orchestration + GPU computation | **异构计算** — CPU 协调 + GPU 计算 |
| **Modular Architecture** — Trait-based abstractions for extensibility | **模块化架构** — 基于 Trait 的可扩展抽象 |
| **Production Ready** — Comprehensive error handling & metrics | **生产就绪** — 完善的错误处理与指标监控 |

---

## 🚀 快速开始 / Quick Start

### Installation

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build --release
```

### Inference

```bash
./target/release/hetero-infer \
  --input "What is the meaning of life?" \
  --max-tokens 100 \
  --temperature 0.8
```

### Library Usage

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

let config = EngineConfig::default();
let mut engine = InferenceEngine::new(config)?;

let result = engine.submit_request("Hello, world!", GenerationParams::default())?;
```

---

## 📊 性能表现 / Performance

| Metric | CPU Only | CPU + GPU | Improvement |
|--------|----------|-----------|-------------|
| Throughput | ~500 tok/s | ~7,000 tok/s | **14x** |
| Latency (p50) | ~200ms | ~50ms | **4x** |
| Memory Efficiency | 55% waste | < 7% waste | **8x** |

---

## 🏗️ 架构概览 / Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Client Layer                          │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                  Inference Engine                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │
│  │ Tokenizer  │  │ Scheduler  │  │ KV Cache Manager   │  │
│  └────────────┘  └────────────┘  └────────────────────┘  │
└──────────────────────┬───────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│                   GPU Executor                           │
│              ┌──────────────────────┐                     │
│              │   KV Cache Memory    │                     │
│              └──────────────────────┘                     │
└──────────────────────────────────────────────────────────┘
```

---

## 📖 文档导航 / Documentation

### 环境搭建 / Setup
- [快速入门](en/setup/quickstart.md) / [Quick Start](en/setup/quickstart.md)
- [安装指南](en/setup/installation.md) / [Installation](en/setup/installation.md)
- [配置说明](en/setup/configuration.md) / [Configuration](en/setup/configuration.md)

### 架构设计 / Architecture
- [系统概览](en/architecture/overview.md) / [Overview](en/architecture/overview.md)
- [设计原则](en/architecture/design.md) / [Design](en/architecture/design.md)

### API 参考 / API Reference
- [核心类型](en/api/core-types.md) / [Core Types](en/api/core-types.md)
- [完整参考](en/api/reference.md) / [Full Reference](en/api/reference.md)

### 部署运维 / Deployment
- [Docker 部署](en/deployment/docker.md) / [Docker Guide](en/deployment/docker.md)
- [生产部署](en/deployment/production.md) / [Production Deploy](en/deployment/production.md)

### 开发指南 / Development
- [贡献指南](en/development/contributing.md) / [Contributing](en/development/contributing.md)
- [变更日志](en/changelog/index.md) / [Changelog](en/changelog/index.md)

---

## 🛠️ 项目状态 / Project Status

| 模块 / Module | 状态 / Status |
|:---|:---|
| Tokenizer | ✅ 已完成 |
| Scheduler (Continuous Batching) | ✅ 已完成 |
| KV Cache Manager (PagedAttention) | ✅ 已完成 |
| GPU Executor (CUDA) | ✅ 已完成 |
| Inference Engine | ✅ 已完成 |
| REST API Server | 🔄 开发中 |
| Model Serving | 📋 计划中 |

---

## 📄 许可证 / License

This project is licensed under the [Apache License 2.0](LICENSE).

---

<div align="center">

**[🇺🇸 English Docs →](en/)** · **[🇨🇳 中文文档 →](zh/)**

*Copyright &copy; 2026 LessUp*

</div>
