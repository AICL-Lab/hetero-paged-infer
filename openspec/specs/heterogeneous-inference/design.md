# 异构推理系统设计文档

## Context

本文档描述异构推理微服务的架构设计，利用 CPU-GPU 协同执行实现高效的 LLM 推理。系统实现 PagedAttention 进行内存高效的 KV Cache 管理，以及 Continuous Batching 实现高吞吐量。

## Goals / Non-Goals

### Goals

- 实现 PagedAttention 内存管理，支持高效的 KV Cache 分配和释放
- 实现 Continuous Batching 调度器，最大化 GPU 利用率
- 提供模块化的 trait 抽象，便于测试和扩展
- 支持属性测试验证系统不变量

### Non-Goals

- 实现真实的 CUDA 内核（当前使用 mock 实现）
- 实现真正的锁页内存传输
- 支持 Copy-on-Write KV 共享
- 实现异步 CPU/GPU 重叠

## Decisions

### 1. 架构分工

- **CPU 职责**: 分词、请求调度、KV Cache 页管理、批次准备
- **GPU 职责**: 注意力计算、矩阵操作、token 生成

### 2. 核心创新

1. **PagedAttention** - 虚拟内存启发的块管理
2. **Continuous Batching** - Prefill 和 Decode 阶段交错
3. **CUDA Graphs** - 减少内核启动开销
4. **双缓冲批次准备** - 延迟隐藏

### 3. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           InferenceEngine                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Tokenizer  │    │   Scheduler  │    │    KV Cache Manager      │  │
│  │    (CPU)     │    │    (CPU)     │    │        (CPU)             │  │
│  │ Encode/Decode│    │  Prefill/    │    │  BlockPool/PageTable     │  │
│  │              │    │   Decode     │    │                          │  │
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

### 4. 推理流程

```
┌─────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│ Request │──▶│ Tokenize  │──▶│ Schedule  │──▶│ Execute   │──▶│ Decode   │
│ Input   │   │  (CPU)    │   │  (CPU)    │   │  (GPU)    │   │  (CPU)   │
└─────────┘   └───────────┘   └───────────┘   └───────────┘   └──────────┘
                                  │               │
                                  │    ┌──────────┘
                                  ▼    ▼
                            ┌───────────────┐
                            │ KV Cache Mgr  │
                            │     (CPU)     │
                            └───────────────┘
```

### 5. 状态机

```
                    ┌─────────────┐
                    │   Pending   │  (等待调度)
                    └──────┬──────┘
                           │ schedule()
                           ▼
                    ┌─────────────┐
            ┌───────│   Prefill   │  (处理输入 token)
            │       └──────┬──────┘
            │              │ prefill 完成
            │              ▼
            │       ┌─────────────┐
            │       │   Decode    │◄────┐ (生成 token)
            │       └──────┬──────┘     │
            │              │            │ 生成下一个 token
            │              ├────────────┘
            │              │ EOS 或 max_tokens
            │              ▼
            │       ┌─────────────┐
            └──────▶│  Completed  │  (完成)
                    └─────────────┘
```

## Component Interfaces

### 1. Request

```rust
struct Request {
    id: u64,                    // 唯一请求标识符
    input_tokens: Vec<u32>,     // 输入 token 序列
    output_tokens: Vec<u32>,    // 输出 token 序列
    max_tokens: u32,            // 最大生成 token 数
    temperature: f32,           // 采样温度
    top_p: f32,                 // Top-p 采样参数
    state: RequestState,        // 当前状态
    created_at: Instant,        // 创建时间戳
}

enum RequestState {
    Pending,                    // 等待调度
    Prefill,                    // Prefill 阶段
    Decode,                     // Decode 阶段
    Completed,                  // 已完成
    Failed(String),             // 失败并附带错误
}
```

### 2. Sequence

表示活动请求及其 KV Cache 块的集合：

```rust
struct Sequence {
    seq_id: u64,                        // 唯一序列标识符
    request: Request,                   // 关联请求
    logical_blocks: Vec<LogicalBlock>,  // 逻辑块列表
    num_computed_tokens: u32,           // 已计算 token 数
    num_generated_tokens: u32,          // 已生成 token 数
}

struct LogicalBlock {
    block_idx: u32,                     // 逻辑块索引
    physical_block: Option<PhysicalBlockRef>,  // 物理块引用
}
```

### 3. KVCacheManager Interface

```rust
trait KVCacheManagerTrait {
    /// 为新序列分配块
    fn allocate_sequence(&mut self, seq_id: u64, num_tokens: u32) -> Result<(), MemoryError>;

    /// 为序列增长分配额外块
    fn allocate_block(&mut self, seq_id: u64) -> Result<PhysicalBlockRef, MemoryError>;

    /// 释放序列的所有块
    fn free_sequence(&mut self, seq_id: u64);

    /// 获取 GPU 执行用的块表
    fn get_block_table(&self, seq_id: u64) -> Option<Vec<u32>>;

    /// 查询内存状态
    fn get_memory_stats(&self) -> MemoryStats;

    /// 检查是否可以分配 n 个块
    fn can_allocate(&self, num_blocks: u32) -> bool;
}

struct MemoryStats {
    total_blocks: u32,      // 总块数
    used_blocks: u32,       // 已用块数
    free_blocks: u32,       // 空闲块数
    num_sequences: u32,     // 活动序列数
}
```

### 4. Scheduler Interface

```rust
trait SchedulerTrait {
    /// 将请求添加到待处理队列
    fn add_request(&mut self, request: Request) -> Result<u64, SchedulerError>;

    /// 调度下一批次执行
    fn schedule(&mut self) -> SchedulerOutput;

    /// GPU 执行后更新序列状态
    fn update_sequences(&mut self, outputs: &ExecutionOutput);

    /// 获取已完成的请求
    fn get_completed(&mut self) -> Vec<Request>;

    /// 检查是否有待处理工作
    fn has_pending_work(&self) -> bool;
}

struct SchedulerOutput {
    prefill_sequences: Vec<SequenceRef>,   // Prefill 序列
    decode_sequences: Vec<SequenceRef>,    // Decode 序列
    block_tables: HashMap<u64, Vec<u32>>,  // 块表
    total_tokens: u32,                     // 总 token 数
}
```

### 5. GPUExecutor Interface

```rust
trait GPUExecutorTrait {
    /// 执行批次序列
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;

    /// 为 decode 阶段捕获 CUDA Graph
    fn capture_decode_graph(&mut self, batch_size: u32);

    /// 使用捕获的 CUDA Graph 执行
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}

struct ExecutionBatch {
    input_tokens: Vec<u32>,        // 所有序列的扁平化 token
    positions: Vec<u32>,           // 每个 token 的位置
    seq_lens: Vec<u32>,            // 每序列长度
    block_tables: Vec<Vec<u32>>,   // Paged attention 块表
    is_prefill: Vec<bool>,         // Prefill/Decode 标志
}

struct ExecutionOutput {
    next_tokens: Vec<u32>,         // 每序列的下一个 token
    logits: Option<Vec<f32>>,      // Logits（可选）
}
```

### 6. Tokenizer Interface

```rust
trait TokenizerTrait {
    fn encode(&self, text: &str) -> Vec<u32>;   // 文本 → token
    fn decode(&self, tokens: &[u32]) -> String; // Token → 文本
    fn vocab_size(&self) -> u32;                // 词汇表大小
    fn bos_token_id(&self) -> u32;              // BOS token ID
    fn eos_token_id(&self) -> u32;              // EOS token ID
    fn pad_token_id(&self) -> u32;              // PAD token ID
}
```

### 7. InferenceEngine (主协调器)

```rust
struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn TokenizerTrait>,
    scheduler: Box<dyn SchedulerTrait>,
    kv_cache_manager: Box<dyn KVCacheManagerTrait>,
    gpu_executor: Box<dyn GPUExecutorTrait>,
}

impl InferenceEngine {
    fn submit_request(&mut self, text: &str, params: GenerationParams) -> RequestId;
    fn step(&mut self) -> Vec<CompletedRequest>;  // 执行单步
    fn run(&mut self);                            // 主循环
}

struct EngineConfig {
    block_size: u32,          // 每块 token 数（如 16）
    max_num_blocks: u32,      // 总 KV cache 块数
    max_batch_size: u32,      // 最大批次大小
    max_num_seqs: u32,        // 最大并发序列数
    max_model_len: u32,       // 最大序列长度
}
```

## Data Models

### Physical Block Layout (GPU Memory)

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU KV Cache Memory Pool                      │
├─────────────────────────────────────────────────────────────────┤
│  Block 0    │  Block 1    │  Block 2    │  ...  │  Block N-1   │
│ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │       │ ┌─────────┐  │
│ │ K[0:16] │ │ │ K[0:16] │ │ │ K[0:16] │ │       │ │ K[0:16] │  │
│ │ V[0:16] │ │ │ V[0:16] │ │ │ V[0:16] │ │       │ │ V[0:16] │  │
│ └─────────┘ │ └─────────┘ │ └─────────┘ │       │ └─────────┘  │
└─────────────────────────────────────────────────────────────────┘

每个块存储：
- K cache: [block_size, num_heads, head_dim]
- V cache: [block_size, num_heads, head_dim]
```

### Page Table Structure

```
Sequence 0:                    Sequence 1:
┌──────────────┐              ┌──────────────┐
│ 逻辑 → 物理  │              │ 逻辑 → 物理  │
├──────────────┤              ├──────────────┤
│   0   →   3  │              │   0   →   1  │
│   1   →   7  │              │   1   →   5  │
│   2   →   12 │              │   2   →   9  │
└──────────────┘              └──────────────┘

物理块可以是非连续的，以实现高效的内存利用。
```

## Risks / Trade-offs

### 当前实现限制

1. **GPU 执行器是 Mock 实现**：用于测试和验证调度逻辑，不执行真实 CUDA 内核
2. **无真正的锁页内存**：CPU-GPU 传输未优化
3. **无异步 CPU/GPU 重叠**：批次准备和执行是串行的

### 已知风险

1. **内存碎片**：长时间运行可能导致碎片化，需要定期整理
2. **调度饥饿**：大量 decode 请求可能阻塞 prefill 请求
3. **GPU 故障传播**：单个序列的 GPU 错误可能影响整个批次

## Rollout Plan

### 已完成 ✅

- PagedAttention KV Cache 管理
- Continuous Batching 调度器
- 内存压力感知
- 模块化 trait 抽象
- 全面的属性测试
- 用于测试的 Mock GPU 执行器

### 待实现 ❌

- 真实 CUDA 内核
- 真正的锁页内存
- Copy-on-Write KV 共享
- 异步 CPU/GPU 重叠

## 测试策略

### 单元测试

单元测试关注具体示例和边界情况：

1. **KV Cache Manager**
   - 分配单个序列，验证块映射
   - 分配多个序列，验证隔离
   - 释放序列，验证块返回
   - 边界情况：精确分配到容量

2. **Scheduler**
   - 添加单个请求，验证入队
   - 仅 prefill 请求的批次调度
   - 仅 decode 请求的批次调度
   - 边界情况：空调度器返回空批次

3. **Tokenizer**
   - 编码已知文本，验证预期 token
   - 解码已知 token，验证预期文本
   - 处理空字符串
   - 处理特殊字符

4. **Configuration**
   - 加载有效配置文件
   - 拒绝无效配置值
   - 为缺失字段应用默认值

### 属性测试

属性测试使用 `proptest` crate 验证许多生成输入的通用属性。

每个属性测试必须：
- 最少运行 100 次迭代
- 引用设计文档属性
- 使用标签格式：**Feature: heterogeneous-inference-system, Property N: [属性文本]**

**测试配置：**
```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // 属性测试
}
```

### 集成测试

1. **端到端请求流程**
   - 提交请求，运行至完成
   - 验证生成输出 token
   - 验证完成时释放 KV cache

2. **Continuous Batching**
   - 在交错时间提交多个请求
   - 验证全部正确完成
   - 验证形成混合 prefill/decode 批次

3. **内存压力**
   - 将内存填充到阈值
   - 验证新 prefill 被拒绝
   - 完成部分请求
   - 验证新 prefill 再次被接受
