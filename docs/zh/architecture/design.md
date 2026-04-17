# 架构指南

## 概述

Hetero-Paged-Infer 实现了异构计算架构，CPU 和 GPU 协同工作，高效完成 LLM 推理任务。系统将控制流（CPU）与计算密集型操作（GPU）分离，最大化资源利用率。

## 设计原则

- **CPU 职责**：分词、请求调度、KV Cache 页管理、批次准备
- **GPU 职责**：注意力计算、矩阵运算、token 生成
- **内存效率**：分页式注意力避免填充和碎片导致的内存浪费
- **吞吐优化**：连续批处理最大化 GPU 利用率

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
│         │            └──────┬───────┘                                  │
│         │                   │                                          │
│         │            ┌──────▼───────┐                                  │
│         │            │  KV Cache    │                                  │
│         │            │ (GPU Memory) │                                  │
│         │            └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 推理引擎（InferenceEngine）

协调所有组件的主编排器：

```rust
pub struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn TokenizerTrait>,
    scheduler: Box<dyn SchedulerTrait>,
    kv_cache_manager: Box<dyn KVCacheManagerTrait>,
    gpu_executor: Box<dyn GPUExecutorTrait>,
}
```

**职责**：
- 请求提交与生命周期管理
- 逐步执行循环
- 错误恢复策略
- 指标收集

### 2. 调度器（Scheduler）

实现带解码优先策略的连续批处理：

**关键特性**：
- 独立的 Prefill 和 Decode 请求队列
- 动态批次构建
- 内存压力感知
- 请求状态转换

**调度算法**：
1. 优先处理 Decode 请求（降低在途请求延迟）
2. 剩余批次容量用于 Prefill 请求
3. 尊重内存阈值和批次大小限制

### 3. KV Cache 管理器（KVCacheManager）

实现分页式注意力内存管理：

**核心概念**：
- **物理块（Physical Block）**：连续的 GPU 内存区域
- **逻辑块（Logical Block）**：通过页表映射的虚拟块
- **块池（Block Pool）**：物理块的空闲列表管理

**内存布局**：
```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU KV Cache 内存池                          │
├─────────────────────────────────────────────────────────────────┤
│  Block 0    │  Block 1    │  Block 2    │  ...  │  Block N-1   │
│ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │       │ ┌─────────┐  │
│ │ K[0:16] │ │ │ K[0:16] │ │ │ K[0:16] │ │       │ │ K[0:16] │  │
│ │ V[0:16] │ │ │ V[0:16] │ │ │ V[0:16] │ │       │ │ V[0:16] │  │
│ └─────────┘ │ └─────────┘ │ └─────────┘ │       │ └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**页表示例**：
```
Sequence 0:                    Sequence 1:
┌─────────────┐               ┌─────────────┐
│ Logical → Physical         │ Logical → Physical        │
├─────────────┤               ├─────────────┤
│   0   →   3  │              │   0   →   1  │
│   1   →   7  │              │   1   →   5  │
│   2   →  12  │              │   2   →   9  │
└─────────────┘               └─────────────┘
```

### 4. GPU 执行器（GPUExecutor）

抽象 GPU 计算（当前为模拟实现）：

**接口**：
```rust
pub trait GPUExecutorTrait {
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
    fn capture_decode_graph(&mut self, batch_size: u32);
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}
```

**特性**（规划中）：
- 分页式注意力 kernel
- Decode 阶段的 CUDA Graph 捕获
- FP16/BF16 计算与 FP32 累加
- 融合操作优化内存带宽

### 5. 分词器（Tokenizer）

基于 trait 的分词接口：

```rust
pub trait TokenizerTrait {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn vocab_size(&self) -> u32;
    fn bos_token_id(&self) -> u32;
    fn eos_token_id(&self) -> u32;
    fn pad_token_id(&self) -> u32;
}
```

## 推理流程

```
┌─────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│  请求   │──▶│   分词    │──▶│   调度    │──▶│   执行    │──▶│   解码   │
│  输入   │   │   (CPU)   │   │   (CPU)   │   │   (GPU)   │   │  (CPU)   │
└─────────┘   └───────────┘   └───────────┘   └───────────┘   └──────────┘
                                  │               │
                                  │    ┌──────────┘
                                  ▼    ▼
                            ┌───────────────┐
                            │ KV Cache Mgr  │
                            │    (CPU)      │
                            └───────────────┘
```

**详细步骤**：

1. **请求提交**
   - 客户端提交文本输入和生成参数
   - 分词器将文本编码为 token ID
   - 请求进入调度器的待处理队列

2. **调度**
   - 调度器选择下一批次的请求
   - Prefill 请求优先于 Decode 请求
   - 强制执行批次大小和 token 限制
   - 执行内存压力检查

3. **KV Cache 分配**
   - 新序列分配逻辑块
   - 物理块通过页表映射
   - 跟踪块分配以生成内存统计

4. **GPU 执行**
   - 准备批次数据（输入 ID、位置、块表）
   - 执行 GPU kernel（Prefill 或 Decode）
   - 生成输出 token

5. **状态更新**
   - 更新序列状态
   - 移除完成的序列
   - 将新 token 解码为文本

## 状态机

```
                    ┌─────────────┐
                    │   Pending   │  （等待调度）
                    └──────┬──────┘
                           │ schedule()
                           ▼
                    ┌─────────────┐
            ┌───────│   Prefill   │  （处理输入 token）
            │       └──────┬──────┘
            │              │ Prefill 完成
            │              ▼
            │       ┌─────────────┐
            │       │   Decode    │◄────┐ （生成 token）
            │       └──────┬──────┘     │
            │              │            │ 生成下一个 token
            │              ├────────────┘
            │              │ EOS 或 max_tokens
            │              ▼
            │       ┌─────────────┐
            └──────▶│  Completed  │  （完成）
                    └─────────────┘
```

**状态转换**：

| 从 | 到 | 触发条件 |
|----|----|----------|
| Pending | Prefill | 被调度首次执行 |
| Prefill | Decode | 所有输入 token 处理完成 |
| Decode | Decode | Token 生成，但未完成 |
| Decode | Completed | 生成 EOS token 或达到 max_tokens |
| 任意 | Failed | 执行期间发生错误 |

## 数据结构

### Request（请求）

```rust
pub struct Request {
    pub id: u64,
    pub input_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub state: RequestState,
}
```

### Sequence（序列）

```rust
pub struct Sequence {
    pub seq_id: u64,
    pub request: Request,
    pub logical_blocks: Vec<LogicalBlock>,
    pub num_computed_tokens: u32,
}
```

### ExecutionBatch（执行批次）

```rust
pub struct ExecutionBatch {
    pub input_tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub seq_lens: Vec<u32>,
    pub block_tables: Vec<Vec<u32>>,
    pub is_prefill: Vec<bool>,
}
```

## 设计属性

架构保持以下正确性属性：

1. **请求 ID 唯一性** - 所有序列 ID 唯一
2. **批次约束** - 序列数 ≤ max_batch_size，token 数 ≤ max_total_tokens
3. **Decode 优先性** - 容量允许时，Decode 请求先于 Prefill 调度
4. **块数不变量** - used_blocks + free_blocks == total_blocks
5. **内存压力响应** - 超过阈值时拒绝新 Prefill 请求

## 当前实现状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 调度器 | ✅ 完成 | 完整的连续批处理与 Decode 优先策略 |
| KV Cache 管理器 | ✅ 完成 | 分页式注意力与块池管理 |
| 推理引擎 | ✅ 完成 | 编排与错误处理 |
| 分词器 | ✅ 完成 | 简单字符级实现 |
| GPU 执行器 | ⚠️ 模拟 | 接口已定义，模拟实现 |
| CUDA Kernel | ❌ 规划 | 真实 kernel 实现待开发 |

---

*API 详情见 [API.md](./API.md)。配置选项见 [CONFIGURATION.md](./CONFIGURATION.md)。*
