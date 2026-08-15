# Paged-Infer

<div align="center">

[![CI](https://github.com/AICL-Lab/paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/AICL-Lab/paged-infer/actions/workflows/ci.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.82%2B-orange?logo=rust)](https://www.rust-lang.org/)

**基于 PagedAttention 分页内存与 Continuous Batching 的推理引擎脚手架（CPU 参考执行器）**

> ⚠️ **开发状态**：本项目处于早期开发阶段（v0.1.0）。计算后端为 CPU 参考执行器（随机初始化小型 Transformer，确定性输出）。生产级 CUDA kernel 为后续工作。

**[文档](#文档) | [更新日志](CHANGELOG.md)**

</div>

---

## 项目概述

Paged-Infer 是一个基于 Rust 构建的 LLM 推理引擎脚手架，以模块化、可测试的架构实现了 [vLLM](https://github.com/vllm-project/vllm) 的分页内存（PagedAttention 风格 KV Cache）与连续批处理调度。计算后端为 CPU 参考执行器（随机初始化小型 Transformer，确定性输出），生产级 CUDA kernel 为后续工作。

| 特性 | 说明 | 状态 |
|------|------|:----:|
| **PagedAttention KV Cache** | 基于块的内存管理；文献背景中常见 <5% 的浪费水平 | ✅ |
| **连续批处理** | 动态 prefill/decode 调度 | ✅ |
| **内存压力感知** | 可配置的 OOM 防护 | ✅ |
| **模块化架构** | 基于 Trait 的抽象设计 | ✅ |
| **OpenAI 兼容服务器** | `/v1/completions` + `/v1/chat/completions` + SSE | ✅ |
| **自动化验证** | unit、integration、server integration 与 property tests | ✅ |

在五仓学习路径中，本仓库只练习 LLM Serving 控制面；真实模型权重加载与 token 计算属于 `tiny-llm`。整体顺序见 [`cuda-kernel-academy/LEARNING_PATH.md`](https://github.com/AICL-Lab/cuda-kernel-academy/blob/master/LEARNING_PATH.md)。

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
│        │        │ CPUExecutor │  (参考)                        │
│        │        │ GPU 执行器  │                                       │
│        │        └──────┬──────┘                                       │
│        │        ┌──────▼──────┐                                       │
│        └───────►│  KV Cache   │  (GPU Memory)                         │
│                 └─────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 调度器设计讲解

> 面试讲述用。用 2 分钟讲清状态机、准入控制与调度优先级；同时主动说明
> 当前架构练习的边界（无抢占），避免被追问时措手不及。

### 请求生命周期状态机

```
            准入通过
  ┌──────┐  add_request  ┌─────────┐   首批调度   ┌─────────┐
  │ 新请求 ├──────────────►│ Pending │─────────────►│ Prefill │
  └──────┘                └─────────┘              └────┬────┘
       │                                               │ 生成首个 token
       │ 取消/失败（任意阶段）                           ▼
       │                                               ┌─────────┐
       │         ┌─────────────────────────────────────►│ Decode  │
       │         │     每步生成 1 token                 └────┬────┘
       │         │                                          │
       ▼         ▼                                          │ EOS / stop / max_tokens
  ┌─────────────────┐                                       ▼
  │ Failed（释放KV）│                                ┌─────────────┐
  └─────────────────┘                                │  Completed   │
                                                     └─────────────┘
```

- **Pending**：已通过准入、等待首个调度步。仅占一个序列槽位，不占 KV 块。
- **Prefill**：一次性处理整个 prompt，计算首个输出 token，分配 KV 块。
- **Decode**：每步生成一个 token（自回归），KV 逐块增长。
- **Completed / Failed**：终态。请求移出调度器，KV 块全部归还自由池。

### 准入控制（三层）

1. **请求级校验**（`submit_request` 阶段）：
   - 参数合法（`max_tokens`、greedy 采样、`stop` ≤ 4、`logprobs` ≤ 5）
   - 总长度不超 `max_model_len`
   - 并发序列不超 `max_num_seqs`（否则 `MaxConcurrentSequencesReached` → HTTP 429）
2. **批次预算**（每步 `schedule`）：
   - 序列数 ≤ `max_batch_size`
   - 本步总 token ≤ `max_total_tokens`（decode 每序列 1，prefill 按输入长度）
   - 单请求块需求 ≤ `max_num_blocks`（超出直接失败，而非悄悄截断）
3. **内存压力**：KV 池利用率 ≥ `memory_threshold` 时暂停启动新 prefill
   （pending 保留），已解码序列继续推进；`MemoryPressure` → HTTP 429 + `Retry-After`。

### 每步调度优先级

```
1. Decode 序列   —— 优先，降低在途请求的尾延迟
2. Prefill 序列  —— 已开始 prefill 的继续推进
3. Pending 队列  —— 新请求（非内存压力下），先来先服务
```

### 抢占策略与边界

**当前实现没有抢占**（vLLM 式的 swap / preempt-resume 未实现）。遇到内存
压力时的策略是"拒绝新 prefill、保住已解码序列"，而非驱逐旧序列。这是本仓库
（架构练习）的明确边界——面试时主动说明，并解释真实系统如何用
`swap`（KV 换出到 CPU 内存）与 `preempt-resume`（按 sequence group 抢占）
应对长尾负载。

### 资源不变量

- KV 块池恒满足 `used_blocks + free_blocks == total_blocks`
- 任何终止路径（完成 / 取消 / 失败 / 客户端断开）都归还 KV 块，内存利用率
  回到基线 —— 由穷举属性测试覆盖

## 快速开始

### 环境要求

- **Rust 1.82+** (2021 edition)
- **Linux** (推荐 Ubuntu 20.04+) 或 **macOS**

### 安装

```bash
# 克隆仓库
git clone https://github.com/AICL-Lab/paged-infer.git
cd paged-infer

# 以 release 模式构建
cargo build --release

# 运行测试套件
cargo test
```

### 命令行用法

```bash
# 基本用法
./target/release/paged-infer --input "你好，世界！" --max-tokens 50

# 使用自定义参数（当前 CPU 后端仅支持 greedy：--temperature 0.0 --top-p 1.0，
# 其他采样参数会在提交时返回错误，而不是被静默忽略）
./target/release/paged-infer \
  --input "解释量子计算" \
  --max-tokens 100

# 启动 OpenAI 兼容 HTTP 服务
./target/release/paged-infer --serve
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
  -d '{"model":"paged-infer","prompt":"你好","max_tokens":8}'

# Chat Completions 接口
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{"model":"paged-infer","messages":[{"role":"user","content":"说你好"}],"max_tokens":8}'
```

### 库用法

```rust
use paged_infer::{EngineConfig, GenerationParams, InferenceEngine};

// 使用默认配置创建引擎
let mut engine = InferenceEngine::new(EngineConfig::default())?;

// 提交生成请求（greedy 解码，当前后端唯一支持的模式）
let request_id = engine.submit_request(
    "你好，世界！",
    GenerationParams {
        max_tokens: 100,
        ..GenerationParams::default()
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
| `--temperature` | 0.0 | 采样温度；CPU 后端仅支持 0.0（greedy），其他值提交时返回错误 |
| `--top-p` | 1.0 | 核采样阈值；CPU 后端仅支持 1.0，其他值提交时返回错误 |
| `--tokenizer` | 无 | HuggingFace tokenizer.json 路径；设置后引擎改用 HF tokenizer（词表与模型一致，如 Qwen2.5），替代默认的 SimpleTokenizer |

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

加载：`./paged-infer --config config.json`

## 文档

| 资源 | 链接 |
|------|------|
| **API 文档** | `cargo doc --open` |
| **贡献指南** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **更新日志** | [CHANGELOG.md](CHANGELOG.md) |

## 性能边界

当前计算后端为 CPU 参考执行器（随机权重小模型），因此仓库不宣称真实 token 吞吐或 GPU 利用率。现阶段 benchmark 只用于观察调度、KV 分页与服务控制面的相对开销；接入真实 CUDA kernel 后才能建立硬件性能基线。

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
| 单元测试 | 核心模块 | 分页、调度、执行和配置 |
| 属性测试 | 状态不变量 | 资源守恒、队列唯一性和容量上限 |
| 集成测试 | 端到端工作流 | engine 与请求生命周期 |
| Server 集成 | HTTP/SSE | API、取消、健康检查与指标 |

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
- [x] CPU 参考执行器（paged KV cache + transformer 前向）
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
