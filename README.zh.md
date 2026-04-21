# Hetero-Paged-Infer

<div align="center">

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/hetero-infer.svg)](https://crates.io/crates/hetero-infer)
[![Docs.rs](https://docs.rs/hetero-infer/badge.svg)](https://docs.rs/hetero-infer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)](https://www.rust-lang.org/)

**高性能 LLM 推理引擎 - PagedAttention + Continuous Batching**

**[English](README.md) | [中文](README.zh.md) | [文档](https://lessup.github.io/hetero-paged-infer/zh/)**

</div>

---

## 项目概述

Hetero-Paged-Infer 是一个基于 Rust 构建的**生产级** LLM 推理引擎，实现了 [vLLM](https://github.com/vllm-project/vllm) 的核心技术，具有模块化、可测试的架构，专为生产部署而设计。

| 特性 | 说明 | 状态 |
|------|------|:----:|
| **PagedAttention KV Cache** | 基于块的内存管理，浪费 <5% | ✅ |
| **连续批处理** | 动态 prefill/decode 调度 | ✅ |
| **内存压力感知** | 可配置的 OOM 防护 | ✅ |
| **模块化架构** | 基于 Trait 的抽象设计 | ✅ |
| **全面测试** | 135 个测试（单元、属性、集成） | ✅ |
| **CUDA Kernel** | 真实 GPU 执行 | 🚧 规划中 |

## 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine (CPU)                          │
├──────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────────────┐  │
│  │  Tokenizer │  │ Scheduler  │  │      KV Cache Manager          │  │
│  │  分词器    │  │  调度器    │  │   BlockPool + PageTable        │  │
│  └─────┬──────┘  └─────┬──────┘  └───────────────┬────────────────┘  │
│        │               │                         │                    │
│        │        ┌──────▼──────┐                  │                    │
│        │        │Batch Builder│◄─────────────────┘                    │
│        │        │ 批次构建器  │                                       │
├────────┼───────────────┼─────────────────────────────────────────────┤
│        │        ┌──────▼──────┐                                       │
│        │        │ GPU Executor│  (CUDA / Mock)                        │
│        │        │ GPU 执行器  │                                       │
│        │        └──────┬──────┘                                       │
│        │        ┌──────▼──────┐                                       │
│        └───────►│  KV Cache   │  (GPU Memory)                         │
│                 └─────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- **Rust 1.70+** (2021 edition)
- **Linux** (推荐 Ubuntu 20.04+) 或 **macOS**

### 安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# 以 release 模式构建
cargo build --release

# 运行测试套件（135 个测试）
cargo test
```

### 命令行用法

```bash
# 基本用法
./target/release/hetero-infer --input "你好，世界！" --max-tokens 50

# 使用自定义参数
./target/release/hetero-infer \
  --input "解释量子计算" \
  --max-tokens 100 \
  --temperature 0.8 \
  --top-p 0.95
```

### 库用法

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

// 使用默认配置创建引擎
let mut engine = InferenceEngine::new(EngineConfig::default())?;

// 提交生成请求
let request_id = engine.submit_request(
    "你好，世界！",
    GenerationParams { 
        max_tokens: 100, 
        temperature: 0.8, 
        top_p: 0.95 
    }
)?;

// 运行推理并收集结果
let results = engine.run();
for result in results {
    println!("生成结果: {}", result.output_text);
}
```

## 配置参数

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--block-size` | 16 | 每物理块 token 数 |
| `--max-num-blocks` | 1024 | 物理块总数 |
| `--max-batch-size` | 32 | 每批次最大序列数 |
| `--memory-threshold` | 0.9 | 内存压力阈值 |
| `--temperature` | 1.0 | 采样温度 |
| `--top-p` | 0.9 | 核采样阈值 |

配置文件 (`config.json`):

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "memory_threshold": 0.9
}
```

加载：`./hetero-infer --config config.json`

## 文档

| 资源 | 链接 |
|------|------|
| **GitHub Pages** | [https://lessup.github.io/hetero-paged-infer/zh/](https://lessup.github.io/hetero-paged-infer/zh/) |
| **API 参考 (docs.rs)** | [https://docs.rs/hetero-infer](https://docs.rs/hetero-infer) |
| **架构设计** | [docs/zh/architecture/overview.md](docs/zh/architecture/overview.md) |
| **贡献指南** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **更新日志** | [CHANGELOG.md](CHANGELOG.md) |

### 本地文档

```bash
# 构建并打开 API 文档
cargo doc --open

# 本地构建文档站点
pip install mkdocs-material mkdocs-static-i18n
mkdocs serve -f mkdocs.zh.yml
```

## 性能对比

| 方法 | 内存浪费 | 吞吐率 | 说明 |
|------|:--------:|:------:|------|
| 静态分配 | ~40-60% | 基准 | 为每个请求预分配最大上下文 |
| 动态分配 | ~20-30% | +20% | 按请求调整但仍有碎片 |
| **PagedAttention** | **<5%** | **+50%** | 基于块的共享与写时复制 |

### 为什么选择 PagedAttention？

传统 LLM 服务为每个请求的 KV 缓存分配连续内存块，导致严重的内存碎片和浪费。PagedAttention 通过以下方式解决：

1. **块级分配**：将 KV 缓存分割为固定大小的块
2. **按需分页**：仅在需要时分配块
3. **写时复制**：跨序列共享块，实现高效的 beam search

## 测试

```bash
# 运行所有测试
cargo test

# 运行覆盖率测试
cargo llvm-cov --html

# 运行属性测试
cargo test -- --test-threads=1
```

| 类型 | 数量 | 说明 |
|------|:----:|------|
| 单元测试 | 78 | 核心功能测试 |
| 属性测试 | 15 | 使用 proptest 验证不变量 |
| 集成测试 | 13 | 端到端工作流测试 |
| 文档测试 | 29 | 文档示例 |
| **总计** | **135** | |

## 贡献指南

欢迎贡献！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

```bash
# 提交前运行所有检查
cargo test && cargo fmt --check && cargo clippy
```

## 路线图

- [x] PagedAttention KV Cache
- [x] Continuous Batching 调度器
- [x] 内存压力感知
- [x] 属性测试
- [ ] 真实 CUDA Kernel
- [ ] 真实分词器集成
- [ ] 异步 CPU/GPU 重叠

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)。

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention 概念和灵感来源
- [Rust](https://www.rust-lang.org/) - 系统编程语言
- [Criterion](https://github.com/bheisler/criterion.rs) - 统计基准测试

---

<p align="center"><b>由 LessUp 用 ❤️ 构建</b></p>
