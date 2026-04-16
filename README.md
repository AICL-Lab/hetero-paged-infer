# Hetero-Paged-Infer

[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/文档-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/hetero-paged-infer/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

异构推理系统 — 基于 PagedAttention 和 Continuous Batching 的 CPU-GPU 协同推理引擎。

## 目录

- [核心特性](#核心特性)
- [架构](#架构)
- [快速开始](#快速开始)
- [配置参数](#配置参数)
- [API 文档](#api-文档)
- [工程质量](#工程质量)
- [当前状态](#当前状态)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 核心特性

- **PagedAttention KV Cache** — 分页式显存管理，按需分配/释放物理块，支持 copy-on-write
- **Continuous Batching** — 连续批处理调度器，prefill/decode 分阶段管理，decode 优先调度
- **内存压力感知** — 可配置的内存阈值，自动拒绝新请求防止 OOM
- **CUDA Graph 支持** — decode 阶段可捕获 CUDA Graph 加速重复执行
- **模块化架构** — Tokenizer / Scheduler / GPU Executor / KV Cache Manager 均通过 trait 抽象

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
┌─────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│  请求   │──▶│  分词     │──▶│  调度     │──▶│  执行     │──▶│  解码    │
│  输入   │   │  (CPU)    │   │  (CPU)    │   │  (GPU)    │   │  (CPU)   │
└─────────┘   └───────────┘   └───────────┘   └───────────┘   └──────────┘
                                  │               │
                                  │    ┌──────────┘
                                  ▼    ▼
                            ┌───────────────┐
                            │ KV Cache 管理器│
                            │     (CPU)     │
                            └───────────────┘
```

### 状态机

```
                    ┌─────────────┐
                    │   Pending   │  (等待调度)
                    └──────┬──────┘
                           │ schedule()
                           ▼
                    ┌─────────────┐
            ┌───────│   Prefill   │  (处理输入 tokens)
            │       └──────┬──────┘
            │              │ prefill 完成
            │              ▼
            │       ┌─────────────┐
            │       │   Decode    │◄────┐ (生成 tokens)
            │       └──────┬──────┘     │
            │              │            │ 生成下一个 token
            │              ├────────────┘
            │              │ EOS 或 max_tokens
            │              ▼
            │       ┌─────────────┐
            └──────▶│  Completed  │  (完成)
                    └─────────────┘
```

## 快速开始

### 环境要求

- Rust 1.70+ (2021 edition)
- CUDA 11.x+ (可选，用于真实 GPU 执行)

### 构建

```bash
# 克隆仓库
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# 构建
cargo build --release

# 运行测试
cargo test
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

### 示例输出

```
Heterogeneous Inference System
==============================
Configuration:
  Block size: 16
  Max blocks: 1024
  Max batch size: 32
  Max sequences: 256

Input: 你好，世界！
Generating up to 50 tokens...

Output: 你好，世界！这是一个异构推理系统的演示输出...
Tokens generated: 25
```

## 配置参数

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | - | 配置文件路径 |
| `--block-size` | 16 | 每个物理块容纳的 token 数 |
| `--max-num-blocks` | 1024 | 最大物理块数量 |
| `--max-batch-size` | 32 | 单次调度最大序列数 |
| `--max-num-seqs` | 256 | 系统最大并发序列数 |
| `--max-model-len` | 2048 | 模型最大上下文长度 |
| `--max-total-tokens` | 4096 | 单批次最大 token 总数 |
| `--memory-threshold` | 0.9 | 内存压力阈值 (0.0-1.0) |
| `--input` | - | 输入文本 |
| `--max-tokens` | 100 | 最大生成 token 数 |
| `--temperature` | 1.0 | 采样温度 |
| `--top-p` | 0.9 | Top-p 采样参数 |

### 配置文件格式

创建 `config.json`：

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "max_total_tokens": 4096,
  "memory_threshold": 0.9
}
```

详见 [config.example.json](config.example.json)。

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
| `Scheduler` | 连续批处理调度器 |
| `KVCacheManager` | KV Cache 管理器 |
| `GPUExecutor` | GPU 执行器接口 |
| `Tokenizer` | 分词器接口 |

### 使用示例

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

// 创建引擎
let config = EngineConfig::default();
let mut engine = InferenceEngine::new(config)?;

// 提交请求
let params = GenerationParams {
    max_tokens: 100,
    temperature: 1.0,
    top_p: 0.9,
};
let request_id = engine.submit_request("你好", params)?;

// 运行推理
let completed = engine.run();

for result in completed {
    println!("输出: {}", result.output_text);
}
```

## 工程质量

### 代码规范

- **选择性导出** — `lib.rs` 使用精确的 `pub use` 避免命名空间污染
- **实例级请求 ID** — 避免全局静态计数器在测试间泄漏
- **分层错误体系** — `MemoryError` / `ConfigError` / `ValidationError` / `ExecutionError` / `SchedulerError` → `EngineError`
- **代码风格** — `rustfmt.toml` + `.editorconfig`

### 测试覆盖

- **单元测试** — 每个模块独立测试
- **属性测试** — 使用 `proptest` 验证不变量
- **集成测试** — 端到端流程验证

```bash
# 运行所有测试
cargo test

# 运行属性测试
cargo test -- --test-threads=1
```

### CI/CD

- GitHub Actions: `cargo fmt --check` + `cargo clippy` + `cargo test`
- 代码覆盖率报告
- 自动文档部署

## 当前状态

当前仓库主要聚焦于调度器、KV Cache、批处理和引擎编排的正确性。

**已实现：**
- ✅ PagedAttention KV Cache 管理
- ✅ Continuous Batching 调度器
- ✅ 内存压力感知
- ✅ 模块化 trait 抽象
- ✅ 完整的属性测试

**未实现：**
- ❌ 真实 CUDA kernel
- ❌ 真实 pinned memory
- ❌ Copy-on-write KV 共享
- ❌ 异步 CPU/GPU overlap

`GPUExecutor` 目前是 **mock 实现**，用于测试和验证调度逻辑。

## 项目结构

```
src/
├── lib.rs           # 库入口，模块声明与选择性导出
├── main.rs          # CLI 入口 (clap)
├── config.rs        # EngineConfig 配置、验证、JSON 序列化
├── error.rs         # 错误类型体系 (thiserror)
├── types.rs         # 核心数据结构 (Request, Sequence, ExecutionBatch, ...)
├── kv_cache.rs      # PagedAttention KV Cache 管理器
├── scheduler.rs     # Continuous Batching 调度器
├── tokenizer.rs     # 字符级 Tokenizer (测试用)
├── gpu_executor.rs  # GPU 执行器抽象 + Mock 实现
├── engine.rs        # 推理引擎编排器
└── test_utils.rs    # 测试工具函数

tests/
└── integration_tests.rs  # 端到端集成测试

.kiro/specs/heterogeneous-inference-system/
├── design.md        # 设计文档
├── requirements.md  # 需求文档
└── tasks.md         # 任务追踪
```

## 贡献指南

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `thiserror` | 1.0 | 派生错误类型 |
| `serde` | 1.0 | 序列化框架 |
| `serde_json` | 1.0 | JSON 序列化 |
| `clap` | 4.0 | 命令行参数解析 |
| `log` | 0.4 | 日志门面 |
| `env_logger` | 0.10 | 日志实现 |
| `proptest` | 1.4 (dev) | 属性测试 |

## 许可证

[MIT](LICENSE)
