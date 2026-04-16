# 设计文档：异构推理系统

## 概述

本文档描述一个异构推理微服务的设计，利用 CPU-GPU 协同执行实现高效的 LLM 推理。系统实现 PagedAttention 进行内存高效的 KV cache 管理，以及 Continuous Batching 实现高吞吐量。

### 架构原则

- **CPU 负责**：分词、请求调度、KV Cache 页管理、批次准备
- **GPU 负责**：Attention 计算、矩阵运算、Token 生成

### 核心创新

1. **PagedAttention** - 类虚拟内存的块管理
2. **Continuous Batching** - Prefill 和 Decode 阶段混合
3. **CUDA Graphs** - 减少 kernel 启动开销
4. **双缓冲批次准备** - 延迟隐藏

## 架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           InferenceEngine                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Tokenizer  │    │   Scheduler  │    │    KV Cache Manager      │  │
│  │    (CPU)     │    │    (CPU)     │    │        (CPU)             │  │
│  │  (分词/解码)  │    │ (prefill/    │    │  (BlockPool/PageTable)   │  │
│  │              │    │   decode)    │    │                          │  │
│  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘  │
│         │                   │                        │                  │
│         │            ┌──────▼───────┐               │                  │
│         │            │ Batch Builder│◄──────────────┘                  │
│         │            │    (CPU)     │                                  │
│         │            │  批次构建器   │                                  │
│         │            └──────┬───────┘                                  │
│         │                   │                                          │
│  ───────┼───────────────────┼────────────────────────────────────────  │
│         │            ┌──────▼───────┐                                  │
│         │            │ GPU Executor │                                  │
│         │            │  (CUDA/GPU)  │                                  │
│         │            │  GPU 执行器   │                                  │
│         │            └──────┬───────┘                                  │
│         │                   │                                          │
│         │            ┌──────▼───────┐                                  │
│         │            │  KV Cache    │                                  │
│         │            │ (GPU Memory) │                                  │
│         │            │  GPU 显存     │                                  │
│         │            └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 推理流程

```
┌─────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│  请求   │──▶│   分词    │──▶│   调度    │──▶│   执行    │──▶│   解码   │
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

## 组件与接口

### 1. Request（请求）

```rust
struct Request {
    id: u64,                    // 请求唯一标识符
    input_tokens: Vec<u32>,     // 输入 token 序列
    output_tokens: Vec<u32>,    // 输出 token 序列
    max_tokens: u32,            // 最大生成 token 数
    temperature: f32,           // 采样温度
    top_p: f32,                 // Top-p 采样参数
    state: RequestState,        // 当前状态
    created_at: Instant,        // 创建时间
}

enum RequestState {
    Pending,                    // 等待调度
    Prefill,                    // Prefill 阶段
    Decode,                     // Decode 阶段
    Completed,                  // 已完成
    Failed(String),             // 失败
}
```

### 2. Sequence（序列）

活跃请求及其 KV Cache 块的集合：

```rust
struct Sequence {
    seq_id: u64,                        // 序列唯一标识符
    request: Request,                   // 关联的请求
    logical_blocks: Vec<LogicalBlock>,  // 逻辑块列表
    num_computed_tokens: u32,           // 已计算的 token 数
    num_generated_tokens: u32,          // 已生成的 token 数
}

struct LogicalBlock {
    block_idx: u32,                     // 逻辑块索引
    physical_block: Option<PhysicalBlockRef>,  // 物理块引用
}
```

### 3. KVCacheManager 接口

```rust
trait KVCacheManagerTrait {
    /// 为新序列分配块
    fn allocate_sequence(&mut self, seq_id: u64, num_tokens: u32) -> Result<(), MemoryError>;
    
    /// 序列增长时分配额外块
    fn allocate_block(&mut self, seq_id: u64) -> Result<PhysicalBlockRef, MemoryError>;
    
    /// 释放序列的所有块
    fn free_sequence(&mut self, seq_id: u64);
    
    /// 获取块表用于 GPU 执行
    fn get_block_table(&self, seq_id: u64) -> Option<Vec<u32>>;
    
    /// 查询内存状态
    fn get_memory_stats(&self) -> MemoryStats;
    
    /// 检查是否可分配 n 个块
    fn can_allocate(&self, num_blocks: u32) -> bool;
}

struct MemoryStats {
    total_blocks: u32,      // 总块数
    used_blocks: u32,       // 已用块数
    free_blocks: u32,       // 空闲块数
    num_sequences: u32,     // 活跃序列数
}
```

### 4. Scheduler 接口

```rust
trait SchedulerTrait {
    /// 添加新请求到待处理队列
    fn add_request(&mut self, request: Request) -> Result<u64, SchedulerError>;
    
    /// 调度下一批次用于执行
    fn schedule(&mut self) -> SchedulerOutput;
    
    /// GPU 执行后更新序列状态
    fn update_sequences(&mut self, outputs: &ExecutionOutput);
    
    /// 获取已完成的请求
    fn get_completed(&mut self) -> Vec<Request>;
    
    /// 检查是否有待处理的工作
    fn has_pending_work(&self) -> bool;
}

struct SchedulerOutput {
    prefill_sequences: Vec<SequenceRef>,   // Prefill 序列
    decode_sequences: Vec<SequenceRef>,    // Decode 序列
    block_tables: HashMap<u64, Vec<u32>>,  // 块表
    total_tokens: u32,                     // 总 token 数
}
```

### 5. GPUExecutor 接口

```rust
trait GPUExecutorTrait {
    /// 执行一批序列
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
    
    /// 捕获 decode 阶段的 CUDA Graph
    fn capture_decode_graph(&mut self, batch_size: u32);
    
    /// 使用捕获的 CUDA Graph 执行
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}

struct ExecutionBatch {
    input_tokens: Vec<u32>,        // 所有序列的 token（扁平化）
    positions: Vec<u32>,           // 每个 token 的位置
    seq_lens: Vec<u32>,            // 各序列长度
    block_tables: Vec<Vec<u32>>,   // Paged Attention 块表
    is_prefill: Vec<bool>,         // Prefill/Decode 标志
}

struct ExecutionOutput {
    next_tokens: Vec<u32>,         // 各序列的下一个 token
    logits: Option<Vec<f32>>,      // Logits（可选）
}
```

### 6. Tokenizer 接口

```rust
trait TokenizerTrait {
    fn encode(&self, text: &str) -> Vec<u32>;   // 文本 → token
    fn decode(&self, tokens: &[u32]) -> String; // token → 文本
    fn vocab_size(&self) -> u32;                // 词表大小
    fn bos_token_id(&self) -> u32;              // BOS token ID
    fn eos_token_id(&self) -> u32;              // EOS token ID
    fn pad_token_id(&self) -> u32;              // PAD token ID
}
```

### 7. InferenceEngine（主编排器）

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
    fn step(&mut self) -> Vec<CompletedRequest>;  // 单步执行
    fn run(&mut self);                            // 主循环
}

struct EngineConfig {
    block_size: u32,          // 每块 token 数（如 16）
    max_num_blocks: u32,      // KV cache 总块数
    max_batch_size: u32,      // 最大批次大小
    max_num_seqs: u32,        // 最大并发序列数
    max_model_len: u32,       // 最大序列长度
}
```

## 数据模型

### 物理块布局（GPU 显存）

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

### 页表结构

```
Sequence 0:                    Sequence 1:
┌──────────────┐              ┌──────────────┐
│ Logical → Physical │        │ Logical → Physical │
├──────────────┤              ├──────────────┤
│   0   →   3   │              │   0   →   1   │
│   1   →   7   │              │   1   →   5   │
│   2   →   12  │              │   2   →   9   │
└──────────────┘              └──────────────┘

物理块可以不连续，实现高效的内存利用。
```

### GPU 批次数据布局

```rust
struct GPUBatchData {
    // Pinned host memory 用于快速传输
    input_ids: PinnedBuffer<u32>,      // [total_tokens]
    positions: PinnedBuffer<u32>,       // [total_tokens]
    
    // 序列元数据
    seq_start_locs: PinnedBuffer<u32>,  // [num_seqs + 1] 累积
    seq_lens: PinnedBuffer<u32>,        // [num_seqs]
    
    // 块表（填充到 max_blocks_per_seq）
    block_tables: PinnedBuffer<u32>,    // [num_seqs, max_blocks_per_seq]
    
    // Attention 上下文长度
    context_lens: PinnedBuffer<u32>,    // [num_seqs]
}
```

## 正确性属性

属性是系统在所有有效执行中应保持的特征或行为。

### 属性 1：请求 ID 唯一性

*对于任意* 提交到调度器的请求集合，所有分配的序列 ID 应唯一无重复。

**验证：需求 1.2**

### 属性 2：参数验证正确性

*对于任意* 生成参数 (max_tokens, temperature, top_p)，验证函数返回 true 当且仅当所有参数在有效范围内（max_tokens > 0, 0 < temperature ≤ 2.0, 0 < top_p ≤ 1.0）。

**验证：需求 1.3**

### 属性 3：序列启动时的块分配

*对于任意* 具有 n 个输入 token 的新序列，KV Cache 管理器应分配 ceil(n / block_size) 个逻辑块，每个映射到不同的物理块。

**验证：需求 2.2**

### 属性 4：增长时的块分配

*对于任意* 超出当前块容量的序列，当 token 数跨越块边界时，KV Cache 管理器应恰好分配一个额外的物理块。

**验证：需求 2.3**

### 属性 5：块计数不变量

*对于任意* KV Cache 管理器状态，不变量 `used_blocks + free_blocks == total_blocks` 应成立。此外，序列释放时，所有块应返回空闲池。

**验证：需求 2.4, 2.5**

### 属性 6：调度器队列状态一致性

*对于任意* 调度器中的请求，它应恰好在一个队列中：pending、prefill 或 decode。

**验证：需求 3.1**

### 属性 7：批次大小约束

*对于任意* 调度的批次，序列数不应超过 max_batch_size，token 总数不应超过 max_total_tokens。

**验证：需求 3.5**

### 属性 8：Decode 优先于 Prefill

*对于任意* prefill 和 decode 请求都待处理且批次容量允许的调度决策，所有合格的 decode 请求应在 prefill 请求之前调度。

**验证：需求 3.7**

### 属性 9：Prefill 到 Decode 转换

*对于任意* 完成 prefill 阶段的序列，它应在同一调度周期内立即转换到 decode 状态。

**验证：需求 3.3**

### 属性 10：完成条件

*对于任意* decode 阶段的序列，它当且仅当生成 EOS token 或达到 max_tokens 时转换到完成状态。

**验证：需求 3.4**

### 属性 11：可变序列长度处理

*对于任意* 包含不同长度序列的批次，GPU 执行器应为每个序列独立产生正确的 attention 输出。

**验证：需求 4.2**

### 属性 12：内存统计不变量

*对于任意* KV Cache 管理器状态，报告的内存统计应满足：`total_blocks == used_blocks + free_blocks` 且 `num_sequences == 已分配块的序列数`。

**验证：需求 6.2**

### 属性 13：内存压力响应

*对于任意* 内存利用率超过配置阈值的状态，调度器应拒绝新的 prefill 请求直到内存释放。

**验证：需求 6.3**

### 属性 14：配置验证

*对于任意* 配置输入，验证应拒绝 block_size ≤ 0, max_num_blocks ≤ 0, max_batch_size ≤ 0, 或 max_num_seqs ≤ 0 的配置。

**验证：需求 7.2**

### 属性 15：分词器往返

*对于任意* 有效文本输入，解码编码后的 token 应产生与原始输入等价的文本（考虑规范化）。

**验证：需求 8.4**

## 错误处理

### 内存错误

| 错误条件 | 处理策略 |
|----------|----------|
| 无空闲块 | 返回 `MemoryError::OutOfBlocks`，调度器停止接受新 prefill |
| 块分配失败 | 记录错误，标记请求失败，释放部分分配 |
| GPU 内存分配失败 | 优雅降级，减少 max_num_blocks 并重试 |

### 请求错误

| 错误条件 | 处理策略 |
|----------|----------|
| 无效参数 | 立即返回验证错误，不排队请求 |
| 分词失败 | 返回带详情的错误，不创建序列 |
| 超过最大序列长度 | 根据配置截断或拒绝 |

### 执行错误

| 错误条件 | 处理策略 |
|----------|----------|
| CUDA kernel 错误 | 记录错误，标记受影响序列为失败，继续处理其他 |
| GPU 超时 | 记录警告，重试一次，然后标记为失败 |
| 输出含 NaN/Inf | 检测并标记序列为失败 |

### 恢复策略

```rust
enum RecoveryAction {
    Retry { max_attempts: u32 },   // 重试
    SkipSequence,                  // 跳过序列
    ResetBatch,                    // 重置批次
    Shutdown,                      // 关闭引擎
}

impl InferenceEngine {
    fn handle_error(&mut self, error: ExecutionError) -> RecoveryAction {
        match error {
            ExecutionError::CudaError(_) => RecoveryAction::SkipSequence,
            ExecutionError::MemoryError(_) => RecoveryAction::ResetBatch,
            ExecutionError::FatalError(_) => RecoveryAction::Shutdown,
        }
    }
}
```

## 测试策略

### 单元测试

单元测试关注具体示例和边界情况：

1. **KV Cache 管理器**
   - 分配单个序列，验证块映射
   - 分配多个序列，验证隔离
   - 释放序列，验证块返回
   - 边界：恰好在容量时分配

2. **调度器**
   - 添加单个请求，验证排队
   - 调度仅 prefill 请求的批次
   - 调度仅 decode 请求的批次
   - 边界：空调度器返回空批次

3. **分词器**
   - 编码已知文本，验证预期 token
   - 解码已知 token，验证预期文本
   - 处理空字符串
   - 处理特殊字符

4. **配置**
   - 加载有效配置文件
   - 拒绝无效配置值
   - 对缺失字段应用默认值

### 属性测试

属性测试验证跨多个生成输入的通用属性。使用 `proptest` crate。

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

1. **端到端请求流**
   - 提交请求，运行直到完成
   - 验证输出 token 已生成
   - 验证完成后 KV cache 已释放

2. **连续批处理**
   - 以交错时间提交多个请求
   - 验证全部正确完成
   - 验证形成混合 prefill/decode 批次

3. **内存压力**
   - 填充内存到阈值
   - 验证新 prefill 被拒绝
   - 完成一些请求
   - 验证新 prefill 再次被接受

## 当前实现状态

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

`GPUExecutor` 目前是 **mock 实现**，用于测试和验证调度逻辑。
