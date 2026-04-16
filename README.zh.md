# Hetero-Paged-Infer

<div align="center">

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://www.rust-lang.org/)
[![Documentation](https://img.shields.io/badge/docs-github--pages-blue)](https://lessup.github.io/hetero-paged-infer/)

**[English](README.md) | [中文](README.zh.md)**

</div>

高性能异构推理引擎，支持大语言模型（LLM）的 CPU-GPU 协同执行，采用分页式注意力（PagedAttention）内存管理和连续批处理（Continuous Batching）调度。

## 项目概述

Hetero-Paged-Infer 是一个基于 Rust 的推理系统，专为高效的 LLM 服务而设计。它结合了 vLLM 项目的先进技术与模块化、生产就绪的架构：

- **分页式注意力（PagedAttention）** - 类虚拟内存的 KV Cache 管理，消除内存浪费
- **连续批处理（Continuous Batching）** - 动态请求调度，最大化 GPU 利用率
- **异构计算** - CPU 编排，GPU 计算
- **生产级** - 全面的错误处理、指标收集和日志记录

## 核心特性

| 特性 | 说明 | 状态 |
|------|------|------|
| **分页式注意力 KV Cache** | 基于块的内存管理，O(1) 查找 | ✅ 就绪 |
| **连续批处理** | Prefill/Decode 阶段管理，Decode 优先 | ✅ 就绪 |
| **内存压力感知** | 可配置阈值防止 OOM | ✅ 就绪 |
| **模块化架构** | 所有组件基于 Trait 抽象 | ✅ 就绪 |
| **CUDA Graph 支持** | Decode 阶段图捕获（规划中） | 🚧 规划中 |

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           InferenceEngine                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Tokenizer  │    │   Scheduler  │    │    KV Cache Manager      │  │
│  │    (CPU)     │    │    (CPU)     │    │        (CPU)             │  │
│  │  编码/解码    │    │ (Prefill/    │    │  (BlockPool/PageTable)   │  │
│  │              │    │   Decode)    │    │                          │  │
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

## 快速开始

### 环境要求

- Rust 1.70+（2021 edition）
- Linux 环境（推荐 Ubuntu 20.04+）
- NVIDIA GPU + CUDA 11.x+（可选，用于 GPU 加速）

### 安装

```bash
# 克隆仓库
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# 构建发布版本
cargo build --release

# 运行测试
cargo test
```

### 基本用法

```bash
# 简单推理
./target/release/hetero-infer --input "你好，世界！" --max-tokens 50

# 自定义参数
./target/release/hetero-infer \
  --input "解释量子计算" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-p 0.95
```

### 库用法

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

// 创建引擎
let config = EngineConfig::default();
let mut engine = InferenceEngine::new(config)?;

// 提交请求
let params = GenerationParams {
    max_tokens: 100,
    temperature: 0.8,
    top_p: 0.95,
};
let request_id = engine.submit_request("你好，世界！", params)?;

// 运行推理
let completed = engine.run();

// 获取结果
for result in completed {
    println!("输出: {}", result.output_text);
}
```

## 配置参数

### 命令行选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--block-size` | 16 | 每物理块 token 数 |
| `--max-num-blocks` | 1024 | 物理块总数 |
| `--max-batch-size` | 32 | 每批次最大序列数 |
| `--memory-threshold` | 0.9 | 内存压力阈值 |
| `--temperature` | 1.0 | 采样温度 |
| `--top-p` | 0.9 | 核采样阈值 |

### 配置文件

创建 `config.json`：

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "memory_threshold": 0.9
}
```

加载方式：`./hetero-infer --config config.json`

## 文档

| 资源 | 说明 | 链接 |
|------|------|------|
| **架构指南** | 系统设计与组件 | [docs/zh/ARCHITECTURE.md](docs/zh/ARCHITECTURE.md) |
| **API 参考** | Rust API 文档 | [docs/zh/API.md](docs/zh/API.md) |
| **配置指南** | 所有配置选项 | [docs/zh/CONFIGURATION.md](docs/zh/CONFIGURATION.md) |
| **部署指南** | 生产部署说明 | [docs/zh/DEPLOYMENT.md](docs/zh/DEPLOYMENT.md) |
| **GitHub Pages** | 在线文档 | [https://lessup.github.io/hetero-paged-infer/](https://lessup.github.io/hetero-paged-infer/) |

## 项目状态

### 已实现 ✅

- [x] 分页式注意力 KV Cache 管理
- [x] 连续批处理调度器
- [x] 内存压力感知
- [x] 模块化 Trait 抽象
- [x] 全面的属性测试
- [x] 用于测试的模拟 GPU 执行器

### 规划中 🚧

- [ ] 真实 CUDA kernel 实现
- [ ] 固定内存管理
- [ ] Copy-on-write KV 共享
- [ ] 异步 CPU/GPU 重叠

## 性能对比

内存效率对比：

| 方法 | 内存浪费 | 吞吐率 |
|------|----------|--------|
| 静态分配 | ~40-60% | 基准 |
| 动态分配 | ~20-30% | +20% |
| **分页式注意力** | **<5%** | **+50%** |

## 测试

```bash
# 运行所有测试
cargo test

# 运行覆盖率测试
cargo tarpaulin --out Html

# 运行属性测试
cargo test -- --test-threads=1

# 运行基准测试
cargo bench
```

测试覆盖：

| 类型 | 数量 | 覆盖范围 |
|------|------|----------|
| 单元测试 | 78 | 核心模块 |
| 属性测试 | 15 | 不变量验证 |
| 集成测试 | 13 | 端到端流程 |
| 文档测试 | 29 | API 示例 |

## 贡献指南

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

快速步骤：

```bash
# Fork 并克隆
git clone https://github.com/YOUR_USERNAME/hetero-paged-infer.git

# 创建分支
git checkout -b feature/your-feature

# 修改并测试
cargo test
cargo fmt --check
cargo clippy

# 提交 PR
git push origin feature/your-feature
```

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 分页式注意力概念
- [Rust](https://www.rust-lang.org/) - 系统编程语言
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) - GPU 计算平台

## 链接

- [GitHub 仓库](https://github.com/LessUp/hetero-paged-infer)
- [在线文档](https://lessup.github.io/hetero-paged-infer/)
- [问题追踪](https://github.com/LessUp/hetero-paged-infer/issues)
- [变更日志](CHANGELOG.md)

---

<div align="center">

**由 Hetero-Paged-Infer 团队用 ❤️ 制作**

</div>
