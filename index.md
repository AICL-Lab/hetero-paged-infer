---
title: Hetero-Paged-Infer
layout: default
---

# Hetero-Paged-Infer

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)

**异构推理系统** — 基于 PagedAttention 和 Continuous Batching 的 CPU-GPU 协同推理引擎。

---

## 目录

- [核心特性](#核心特性)
- [架构](#架构)
- [快速开始](#快速开始)
- [配置参数](#配置参数)
- [API 文档](#api-文档)
- [文档导航](#文档导航)
- [工程质量](#工程质量)
- [当前状态](#当前状态)

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **PagedAttention KV Cache** | 分页式显存管理，按需分配/释放物理块，支持 copy-on-write |
| **Continuous Batching** | 连续批处理调度器，prefill/decode 分阶段管理，decode 优先调度 |
| **内存压力感知** | 可配置的内存阈值，自动拒绝新请求防止 OOM |
| **CUDA Graph 支持** | decode 阶段可捕获 CUDA Graph 加速重复执行 |
| **模块化架构** | Tokenizer / Scheduler / GPU Executor / KV Cache Manager 均通过 trait 抽象 |

---

## 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           InferenceEngine                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Tokenizer  │    │   Scheduler  │    │    KV Cache Manager      │  │
│  │  (编码/解码)  │    │ (prefill/    │    │   (BlockPool/PageTable)  │  │
│  │              │    │   decode)    │    │                          │  │
│  └──────────────┘    └──────┬───────┘    └───────────┬──────────────┘  │
│                             │                        │                  │
│                      ┌──────▼───────┐               │                  │
│                      │ Batch Builder│◄──────────────┘                  │
│                      └──────┬───────┘                                  │
│                             │                                          │
│  ───────────────────────────┼────────────────────────────────────────  │
│                      ┌──────▼───────┐                                  │
│                      │ GPU Executor │                                  │
│                      │  (CUDA/GPU)  │                                  │
│                      └──────┬───────┘                                  │
│                             │                                          │
│                      ┌──────▼───────┐                                  │
│                      │   KV Cache   │                                  │
│                      │ (GPU Memory) │                                  │
│                      └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 推理流程

```
请求输入 → 分词(CPU) → 调度(CPU) → 执行(GPU) → 解码(CPU) → 输出
                              ↓
                        KV Cache 管理器
```

### 状态机

```
Pending → Prefill → Decode → Completed
                  ↘ Failed
```

---

## 快速开始

### 环境要求

- Rust 1.70+ (2021 edition)
- CUDA 11.x+ (可选，用于真实 GPU 执行)

### 构建

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build --release
```

### 运行

```bash
# 基本使用
cargo run --release -- --input "你好，世界！" --max-tokens 50

# 使用配置文件
cargo run --release -- --config config.example.json --input "你好"

# 查看帮助
cargo run --release -- --help
```

### 测试

```bash
# 运行所有测试
cargo test

# 运行文档测试
cargo test --doc

# 生成文档
cargo doc --open
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 16 | 每个物理块容纳的 token 数 |
| `max_num_blocks` | 1024 | 最大物理块数量 |
| `max_batch_size` | 32 | 单次调度最大序列数 |
| `max_num_seqs` | 256 | 系统最大并发序列数 |
| `max_model_len` | 2048 | 模型最大上下文长度 |
| `max_total_tokens` | 4096 | 单批次最大 token 总数 |
| `memory_threshold` | 0.9 | 内存压力阈值 (0.0-1.0) |

---

## API 文档

### 生成文档

```bash
cargo doc --open
```

### 核心类型

| 类型 | 说明 |
|------|------|
| `InferenceEngine` | 推理引擎主编排器 |
| `EngineConfig` | 引擎配置 |
| `GenerationParams` | 生成参数 |
| `Request` | 推理请求 |
| `Sequence` | 活跃序列（含 KV Cache） |
| `Scheduler` | Continuous Batching 调度器 |
| `KVCacheManager` | KV Cache 管理器 |
| `GPUExecutor` | GPU 执行器接口 |
| `Tokenizer` | 分词器接口 |

---

## 文档导航

- **[README](README.html)** - 项目完整说明
- **[CONTRIBUTING](CONTRIBUTING.html)** - 贡献指南
- **[CHANGELOG](CHANGELOG.html)** - 变更日志
- **[设计文档](https://github.com/LessUp/hetero-paged-infer/blob/main/.kiro/specs/heterogeneous-inference-system/design.md)** - 详细设计
- **[需求文档](https://github.com/LessUp/hetero-paged-infer/blob/main/.kiro/specs/heterogeneous-inference-system/requirements.md)** - 需求规格

---

## 工程质量

| 检查项 | 状态 |
|--------|------|
| 选择性导出 | ✅ 避免命名空间污染 |
| 实例级请求 ID | ✅ 避免测试间状态泄漏 |
| 分层错误体系 | ✅ Memory/Config/Validation/Execution/Scheduler → Engine |
| 属性测试 | ✅ proptest 验证不变量 |
| CI | ✅ fmt + clippy + test + doc |
| 安全审计 | ✅ cargo audit |

### 测试覆盖

| 类型 | 数量 |
|------|------|
| 单元测试 | 78 |
| 属性测试 | 15 |
| 集成测试 | 13 |
| 文档测试 | 29 |
| **总计** | **135** |

---

## 当前状态

### 已实现 ✅

- PagedAttention KV Cache 管理
- Continuous Batching 调度器
- 内存压力感知
- 模块化 trait 抽象
- 完整的属性测试
- Mock GPU 执行器

### 未实现 ❌

- 真实 CUDA kernel
- 真实 pinned memory
- Copy-on-write KV 共享
- 异步 CPU/GPU overlap

---

## 链接

- [GitHub 仓库](https://github.com/LessUp/hetero-paged-infer)
- [问题追踪](https://github.com/LessUp/hetero-paged-infer/issues)
- [Pull Requests](https://github.com/LessUp/hetero-paged-infer/pulls)

---

## 许可证

本项目采用 [MIT](https://opensource.org/licenses/MIT) 许可证。

---

*最后更新: 2026-04-16*
