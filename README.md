# Hetero-Paged-Infer

<div align="center">

[![CI](https://github.com/AICL-Lab/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/AICL-Lab/hetero-paged-infer/actions/workflows/ci.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.82%2B-orange?logo=rust)](https://www.rust-lang.org/)

**基于 PagedAttention 分页内存与 Continuous Batching 的推理引擎脚手架（计算后端为 mock）**

> ⚠️ **开发状态**：本项目处于早期开发阶段（v0.1.0）。计算后端为 Mock GPU 执行器（确定性占位 token）。真实模型计算、采样与生产级 CUDA kernel 均为后续工作。

**[文档](#文档) | [更新日志](CHANGELOG.md)**

</div>

---

## 项目概述

Hetero-Paged-Infer 是一个基于 Rust 构建的 LLM 推理引擎脚手架，以模块化、可测试的架构实现了 [vLLM](https://github.com/vllm-project/vllm) 的分页内存（PagedAttention 风格 KV Cache）与连续批处理调度。当前计算后端为 mock（确定性占位 token），真实模型计算、采样与生产级 kernel 均为后续工作。

| 特性 | 说明 | 状态 |
|------|------|:----:|
| **PagedAttention KV Cache** | 基于块的内存管理；文献背景中常见 <5% 的浪费水平 | ✅ |
| **连续批处理** | 动态 prefill/decode 调度 | ✅ |
| **内存压力感知** | 可配置的 OOM 防护 | ✅ |
| **模块化架构** | 基于 Trait 的抽象设计 | ✅ |
| **OpenAI 兼容服务器** | `/v1/completions` + `/v1/chat/completions` + SSE | ✅ |
| **全面测试** | 135 个测试 | ✅ |

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
├────────┼───────────────┼─────────────────────────────────────────────┤
│        │        ┌──────▼──────┐                                       │
│        │        │ GPU Executor│  (Mock)                        │
│        │        │ GPU 执行器  │                                       │
│        │        └──────┬──────┘                                       │
│        │        ┌──────▼──────┐                                       │
│        └───────►│  KV Cache   │  (GPU Memory)                         │
│                 └─────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- **Rust 1.82+** (2021 edition)
- **Linux** (推荐 Ubuntu 20.04+) 或 **macOS**

### 安装

```bash
# 克隆仓库
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
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

# 启动 OpenAI 兼容 HTTP 服务
./target/release/hetero-infer --serve
```

### OpenAI 兼容服务

```bash
# 启动服务，默认地址 127.0.0.1:3000
cargo run -- --serve

# 健康检查 / 就绪检查 / 指标
curl http://127.0.0.1:3000/healthz
curl http://127.0.0.1:3000/readyz
curl http://127.0.0.1:3000/metrics

# Completions 接口
curl http://127.0.0.1:3000/v1/completions \
  -H "content-type: application/json" \
  -d '{"model":"hetero-infer","prompt":"你好","max_tokens":8}'

# Chat Completions 接口
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"hetero-infer","messages":[{"role":"user","content":"说你好"}],"max_tokens":8}'
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

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--block-size` | 16 | 每物理块 token 数 |
| `--max-num-blocks` | 1024 | 物理块总数 |
| `--max-batch-size` | 32 | 每批次最大序列数 |
| `--max-num-seqs` | 256 | 最大序列数 |
| `--max-model-len` | 2048 | 最大模型上下文长度 |
| `--max-total-tokens` | 4096 | 每批次最大 token 总数 |
| `--memory-threshold` | 0.9 | 内存压力阈值 (0.0-1.0) |
| `--max-tokens` | 100 | 最大生成 token 数 |
| `--temperature` | 1.0 | 采样温度（当前仅做范围校验，采样尚未实现） |
| `--top-p` | 0.9 | 核采样阈值（当前仅做范围校验，采样尚未实现） |

配置文件 (`config.json`):

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "max_total_tokens": 4096,
  "memory_threshold": 0.9,
  "max_retry_attempts": 2
}
```

加载：`./hetero-infer --config config.json`

## 文档

| 资源 | 链接 |
|------|------|
| **API 文档** | `cargo doc --open` |
| **贡献指南** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **更新日志** | [CHANGELOG.md](CHANGELOG.md) |

## 性能对比

| 方法 | 内存浪费 | 吞吐率 | 说明 |
|------|:--------:|:------:|------|
| 静态分配 | 先验模式：~40-60% | 文献背景：基线 | 为每个请求预分配最大上下文 |
| 动态分配 | 先验模式：~20-30% | 文献背景：+20% | 按请求调整但仍有碎片 |
| **PagedAttention** | **文献背景：<5%** | **文献背景：+50%** | 基于块的共享与写时复制 |

> 说明：当前性能数字仍然主要来自 mock 路径或架构层面的估算，尚未接入真实 CUDA kernel。

### 为什么选择 PagedAttention？

传统 LLM 服务为每个请求的 KV 缓存分配连续内存块，导致严重的内存碎片和浪费。PagedAttention 通过以下方式解决：

1. **块级分配**：将 KV 缓存分割为固定大小的块
2. **按需分页**：仅在需要时分配块
3. **写时复制**（尚未实现，未来方向）：跨序列共享块，实现高效的 beam search

## 测试

```bash
# 运行所有测试
cargo test

# 运行覆盖率测试
cargo llvm-cov --html

# 运行属性测试
cargo test -- --test-threads=1
```

| 类型 | 覆盖范围 | 说明 |
|------|:--------:|------|
| 单元测试 | 纳入 121+ | 核心功能测试 |
| 属性测试 | 纳入 121+ | 使用 proptest 验证不变量 |
| 集成测试 | 纳入 121+ | 端到端工作流测试 |
| 文档测试 | 纳入 121+ | 文档示例 |
| **整体** | **121+ 个测试** | 覆盖仓库中的自动化验证 |

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
- [x] OpenAI 兼容 HTTP 服务
- [x] HuggingFace Tokenizer 集成
- [ ] 真实 CUDA Kernel
- [ ] 异步 CPU/GPU 重叠

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)。

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention 概念和灵感来源
- [Rust](https://www.rust-lang.org/) - 系统编程语言
- [Criterion](https://github.com/bheisler/criterion.rs) - 统计基准测试

---

<p align="center"><b>由 AICL-Lab 用 ❤️ 构建</b></p>
