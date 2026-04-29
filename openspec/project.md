# 项目上下文

> 本文档为 AI 编码助手提供异构推理项目的权威信息。

## 项目概述

**Hetero-Paged-Infer** 是一个用 Rust 编写的高性能 LLM（大语言模型）推理引擎，实现了来自 [vLLM](https://github.com/vllm-project/vllm) 的前沿技术。它采用模块化架构，专为 CPU-GPU 异构计算的生产部署而设计。

| 属性 | 值 |
|-----------|-------|
| **名称** | hetero-infer |
| **版本** | 0.1.0 |
| **语言** | Rust (2021 Edition) |
| **最低 Rust 版本** | 1.70+ |
| **许可证** | MIT |

### 核心功能

| 功能 | 描述 | 状态 |
|---------|-------------|--------|
| **PagedAttention KV Cache** | 基于块的内存管理，浪费率 <5% | ✅ 已实现 |
| **Continuous Batching** | 动态 prefill/decode 调度，decode 优先 | ✅ 已实现 |
| **内存压力感知** | 可配置的 OOM 防止（基于阈值） | ✅ 已实现 |
| **模块化架构** | 基于 trait 的抽象，易于测试 | ✅ 已实现 |
| **OpenAI 兼容服务器** | `/v1/completions` + `/v1/chat/completions` + SSE | ✅ 已实现 |
| **HuggingFace Tokenizer** | 支持 HuggingFace tokenizer JSON 加载 | ✅ 已实现 |
| **命令桥接后端** | 支持外部推理进程集成 | ✅ 已实现 |
| **全面测试** | 121+ 测试（单元、属性、集成、服务） | ✅ 已实现 |
| **CUDA 内核** | 真实 GPU 执行 | 🚧 计划中 |

### 架构图

```
┌──────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine (CPU)                          │
├──────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────────────┐  │
│  │ Tokenizer  │  │ Scheduler  │  │      KV Cache Manager          │  │
│  │            │  │            │  │   BlockPool + PageTable        │  │
│  └─────┬──────┘  └─────┬──────┘  └───────────────┬────────────────┘  │
│        │               │                         │                    │
├────────┼───────────────┼─────────────────────────────────────────────┤
│        │        ┌──────▼──────┐                                       │
│        │        │ GPU Executor│  (MockCUDA - 真实 CUDA 计划中)        │
│        │        └──────┬──────┘                                       │
│        │        ┌──────▼──────┐                                       │
│        └───────►│  KV Cache   │  (GPU 内存抽象)                       │
│                 └─────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## 技术栈

### 核心依赖（来自 Cargo.toml）

| Crate | 用途 |
|-------|---------|
| `serde` / `serde_json` | 配置序列化/反序列化 |
| `clap` | 使用 derive 宏的 CLI 参数解析 |
| `thiserror` | 使用 derive 宏的错误类型定义 |
| `log` / `env_logger` | 日志框架 |

### 开发依赖

| Crate | 用途 |
|-------|---------|
| `proptest` | 属性测试（生成测试） |
| `criterion` | 带 HTML 报告的统计基准测试 |

## 项目结构

### 源代码组织（`src/`）

| 文件 | 行数 | 描述 |
|------|-------|-------------|
| `lib.rs` | 137 | 库入口点，模块导出 |
| `main.rs` | 136 | 使用 `clap` derive 宏的 CLI 入口点 |
| `engine.rs` | 868 | 核心 `InferenceEngine` - 协调所有组件 |
| `scheduler.rs` | 1110 | 带 decode 优先级的 Continuous Batching 调度器 |
| `kv_cache.rs` | 610 | PagedAttention 内存管理器（BlockPool + PageTable） |
| `gpu_executor.rs` | 657 | GPU 执行 trait 和 MockGPUExecutor |
| `tokenizer.rs` | 437 | SimpleTokenizer、HuggingFace Tokenizer 支持 |
| `server.rs` | 501 | OpenAI 兼容 HTTP 服务层 |
| `types.rs` | 831 | 核心数据结构（Request、Sequence 等） |
| `config.rs` | 648 | 带 JSON 序列化的 EngineConfig |
| `error.rs` | 229 | 使用 `thiserror` 的错误类型 |
| `test_utils.rs` | 58 | 共享测试辅助函数 |

### 代码统计

- **~6,200** 行源代码（src/）
- **~780** 行集成测试
- **140+** 总测试用例

## 配置字段

### EngineConfig 字段

| 字段 | 默认值 | 描述 |
|-------|---------|-------------|
| `block_size` | 16 | 每个物理块的 token 数 |
| `max_num_blocks` | 1024 | 总物理块数 |
| `max_batch_size` | 32 | 每批次最大序列数 |
| `max_num_seqs` | 256 | 最大并发序列数 |
| `max_model_len` | 2048 | 最大上下文长度 |
| `max_total_tokens` | 4096 | 每批次最大 token 数 |
| `memory_threshold` | 0.9 | 内存压力阈值（0.0-1.0） |

## 请求状态机

```
Pending → Prefill → Decode → Completed
                    ↘ Failed
```

## 关键数据结构

### 核心类型（来自 `types.rs`）

| 类型 | 描述 |
|------|-------------|
| `RequestId` | `u64` - 请求唯一标识符 |
| `SeqId` | `u64` - 序列唯一标识符 |
| `TokenId` | `u32` - Token ID 类型 |
| `BlockIdx` | `u32` - 物理块索引 |
| `Request` | 带输入/输出 token 的推理请求 |
| `Sequence` | 带 KV Cache 块的活动请求 |
| `GenerationParams` | 采样参数（max_tokens、temperature、top_p） |
| `ExecutionBatch` | GPU 执行批次 |
| `ExecutionOutput` | GPU 执行结果 |
| `MemoryStats` | 内存利用率统计 |

## 核心 Trait

```rust
// 分词器
trait TokenizerTrait: Send + Sync {
    fn encode(&self, text: &str) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId]) -> String;
    fn vocab_size(&self) -> u32;
}

// 调度器
trait SchedulerTrait {
    fn add_request(&mut self, request: Request) -> Result<SeqId, SchedulerError>;
    fn schedule(&mut self) -> SchedulerOutput;
    fn update_sequences(&mut self, outputs: &ExecutionOutput, eos_token_id: TokenId);
}

// GPU 执行器
trait GPUExecutorTrait: Send {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError>;
    fn capture_decode_graph(&mut self, batch_size: u32) -> Result<(), ExecutionError>;
}

// KV Cache
trait KVCacheManagerTrait {
    fn allocate_sequence(&mut self, seq_id: SeqId, num_tokens: u32) -> Result<(), MemoryError>;
    fn free_sequence(&mut self, seq_id: SeqId);
    fn get_memory_stats(&self) -> MemoryStats;
}
```

## 未来路线图

- [ ] 真实 CUDA 内核实现
- [ ] 异步 CPU/GPU 重叠
- [ ] 多 GPU 支持
- [x] HTTP/gRPC API 服务器（OpenAI 兼容）
- [x] HuggingFace Tokenizer 集成
- [ ] 前缀缓存 (Prefix Caching)
- [ ] 推测解码 (Speculative Decoding)
