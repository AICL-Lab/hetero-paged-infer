---
layout: default
title: Hetero-Paged-Infer
---

# Hetero-Paged-Infer

异构推理系统 — 基于 PagedAttention 和 Continuous Batching 的 CPU-GPU 协同推理引擎。

## 核心特性

- **PagedAttention KV Cache** — 分页式显存管理，按需分配/释放物理块，支持 copy-on-write
- **Continuous Batching** — 连续批处理调度器，prefill/decode 分阶段管理，decode 优先调度
- **内存压力感知** — 可配置的内存阈值，自动拒绝新请求防止 OOM
- **CUDA Graph 支持** — decode 阶段可捕获 CUDA Graph 加速重复执行
- **模块化架构** — Tokenizer / Scheduler / GPU Executor / KV Cache Manager 均通过 trait 抽象

## 架构

```
┌─────────────────────────────────────────────┐
│              InferenceEngine                │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  │
│  │Tokenizer │  │ Scheduler │  │   GPU    │  │
│  │(encode/  │  │(prefill/  │  │ Executor │  │
│  │ decode)  │  │ decode/   │  │(execute/ │  │
│  │          │  │ complete) │  │ graph)   │  │
│  └──────────┘  └─────┬─────┘  └──────────┘  │
│                      │                       │
│               ┌──────┴──────┐                │
│               │  KV Cache   │                │
│               │  Manager    │                │
│               └─────────────┘                │
└─────────────────────────────────────────────┘
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Rust 2021 edition |
| 错误处理 | thiserror |
| 序列化 | serde / serde_json |
| CLI | clap 4 |
| 日志 | log / env_logger |
| 测试 | proptest (属性测试) |

## 快速开始

```bash
# 构建
cargo build --release

# 运行测试
cargo test

# 运行
cargo run -- --config config.json
```

## 项目结构

```
src/
├── config.rs        # EngineConfig 配置、验证、JSON 序列化
├── error.rs         # 错误类型体系 (thiserror)
├── types.rs         # 核心数据结构 (Request, Sequence, ExecutionBatch)
├── kv_cache.rs      # PagedAttention KV Cache 管理器
├── scheduler.rs     # Continuous Batching 调度器
├── tokenizer.rs     # 字符级 Tokenizer
├── gpu_executor.rs  # GPU 执行器抽象
└── engine.rs        # 推理引擎编排器
```

## 链接

- [GitHub 仓库](https://github.com/LessUp/hetero-paged-infer)
- [README](README.md)
