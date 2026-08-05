# 核心类型

本页说明各核心类型是什么、用于什么场景。字段清单已逐一对照源码核对；精确签名与完整说明以 Rustdoc 为准：

```bash
cargo doc --no-deps --open
```

常用类型别名（定义于 `types/mod.rs`）：

| 别名 | 定义 | 用途 |
|------|------|------|
| `RequestId` | `u64` | 请求标识符 |
| `SeqId` | `u64` | 序列标识符 |
| `TokenId` | `u32` | Token ID |
| `BlockIdx` | `u32` | 物理块索引 |

## InferenceEngine

推理操作的主要编排器，协调 tokenizer、scheduler、执行流水线与 KV Cache 完成端到端推理。所有字段均为私有，只能通过方法交互；运行时指标经 `get_metrics()` 获取，不存在公共字段。

### 主要方法

```rust
impl InferenceEngine {
    pub fn new(config: EngineConfig) -> Result<Self, EngineError>;

    // 注入自定义组件（用于测试）
    pub fn with_components(
        config: EngineConfig,
        tokenizer: Box<dyn TokenizerTrait>,
        scheduler: Scheduler,
        gpu_executor: Box<dyn GPUExecutorTrait>,
    ) -> Result<Self, EngineError>;

    pub fn submit_request(
        &mut self,
        text: &str,
        params: GenerationParams,
    ) -> Result<RequestId, EngineError>;

    // 执行一步推理，返回本步完成的请求
    pub fn step(&mut self) -> Result<Vec<CompletedRequest>, EngineError>;

    // 执行一步推理并返回细粒度事件（用于 token 级流式响应）
    pub fn step_events(&mut self) -> Result<StepEvents, EngineError>;

    // 运行推理循环，直到所有请求到达终态
    pub fn run(&mut self) -> Vec<CompletedRequest>;

    pub fn has_pending_work(&self) -> bool;
    pub fn memory_utilization(&self) -> f32;
    pub fn config(&self) -> &EngineConfig;
    pub fn get_metrics(&self) -> EngineMetrics;
}
```

### 提交并运行

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

let mut engine = InferenceEngine::new(EngineConfig::default())?;
let request_id = engine.submit_request(
    "你好，世界！",
    GenerationParams {
        max_tokens: 32,
        temperature: 1.0,
        top_p: 1.0,
    },
)?;

let completed = engine.run();
assert!(completed.iter().any(|item| item.request_id == request_id));
# Ok::<(), hetero_infer::EngineError>(())
```

### 手动步进循环

```rust
while engine.has_pending_work() {
    let completed = engine.step()?;
    for result in &completed {
        println!("{}: {}", result.request_id, result.output_text);
    }
}
# Ok::<(), hetero_infer::EngineError>(())
```

需要驱动 token 级流式响应时改用 `step_events()`，它额外报告本步为各请求新生成的文本片段。

## StepEvents

`step_events()` 产生的单步事件，供服务层驱动流式响应。

```rust
pub struct StepEvents {
    /// 本步到达终态（成功或失败）的请求
    pub completed: Vec<CompletedRequest>,
    /// 本步为各请求新生成的文本片段：(request_id, 片段)
    pub chunks: Vec<(RequestId, String)>,
}
```

## EngineConfig

推理引擎的配置。所有字段公开，`Default` 提供一组可用默认值。

| 字段 | 默认值 | 描述 |
|-------|---------|-------------|
| block_size | 16 | 每个 KV Cache 块容纳的 token 数 |
| max_num_blocks | 1024 | 物理块总数（决定 KV Cache 总容量） |
| max_batch_size | 32 | 单次调度最大序列数 |
| max_num_seqs | 256 | 系统最大并发序列数 |
| max_model_len | 2048 | 最大序列长度（输入 + 输出） |
| max_total_tokens | 4096 | 单批次最大 token 总数 |
| memory_threshold | 0.9 | 内存压力阈值，范围 (0.0, 1.0] |
| max_retry_attempts | 2 | GPU 执行超时重试次数 |
| special_tokens | bos=1, eos=2, pad=0, unk=3 | 特殊 token ID |
| tokenizer | Simple | tokenizer 实现类型与路径 |
| serving | 127.0.0.1:3000 | HTTP 服务主机、端口与模型名 |

```rust
let config = EngineConfig {
    max_batch_size: 64,
    max_num_blocks: 2048,
    ..Default::default()
};
```

## GenerationParams

采样与生成限制。作为整体传给 `submit_request()`，并以嵌套字段 `params` 保存在 `Request` 中。

```rust
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}
```

| 字段 | 默认值 | 范围 | 描述 |
|-------|---------|-------|-------------|
| max_tokens | 100 | > 0 | 最大生成 token 数 |
| temperature | 1.0 | [0.0, 2.0] | 采样温度（0.0 表示贪心解码，合法） |
| top_p | 1.0 | (0.0, 1.0] | 核采样阈值 |

> 现状：默认后端为 mock 实现，生成确定性占位 token。`temperature` 与 `top_p` 目前只在提交时校验，尚不影响输出；采样未实现。

## Request 与 RequestState

单个推理请求。由引擎在 `submit_request()` 内部创建，通常无需手动构造。

```rust
pub struct Request {
    pub id: RequestId,
    pub input_tokens: Vec<TokenId>,
    pub output_tokens: Vec<TokenId>,
    pub params: GenerationParams, // 嵌套的生成参数
    pub state: RequestState,
}

pub enum RequestState {
    Pending,
    Prefill,
    Decode,
    Completed,
    Failed(String),
}
```

状态转换：`Pending → Prefill → Decode → Completed`，出错时转入 `Failed`。

## CompletedRequest

请求的最终结果，由 `step()` / `run()` 返回。

```rust
pub struct CompletedRequest {
    pub request_id: RequestId,
    pub input_text: Option<String>, // 输入解码失败时为 None
    pub output_text: String,
    pub output_tokens: Vec<TokenId>,
    pub success: bool,
    pub error: Option<String>, // 失败时的错误信息
}
```

消费方应先检查 `success`，再读取 `output_text`。

## EngineMetrics

运行时统计快照，经 `engine.get_metrics()` 获取（引擎字段私有，不可直接访问）。

```rust
pub struct EngineMetrics {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub total_tokens_generated: u64,
    pub memory_utilization: f32,
    pub active_sequences: u32,
}
```

引擎当前没有延迟/吞吐测量能力：不存在平均延迟或 tok/s 等字段，请勿构建依赖此类指标的监控。

```rust
let metrics = engine.get_metrics();
println!(
    "完成 {}/{} 个请求，生成 {} tokens，内存利用率 {:.2}",
    metrics.completed_requests,
    metrics.total_requests,
    metrics.total_tokens_generated,
    metrics.memory_utilization,
);
# Ok::<(), hetero_infer::EngineError>(())
```

## Sequence 与 SchedulerOutput

调度层内部类型。`Sequence` 是挂载了 KV Cache 块的活跃请求；`SchedulerOutput` 是一次调度产出的批次快照，由执行流水线消费。

```rust
pub struct Sequence {
    pub seq_id: SeqId,
    pub request: Request,
    pub logical_blocks: Vec<LogicalBlock>,
    pub num_computed_tokens: u32,
    pub num_generated_tokens: u32,
}

pub struct SchedulerOutput {
    pub prefill_sequences: Vec<Sequence>,
    pub decode_sequences: Vec<Sequence>,
    pub total_tokens: u32,
}
```

## ExecutionBatch 与 ExecutionOutput

调度器与 GPU 执行器之间的数据契约。

```rust
pub struct ExecutionBatch {
    pub input_tokens: Vec<TokenId>, // 所有序列的 token（扁平化）
    pub positions: Vec<u32>,
    pub seq_lens: Vec<u32>,
    pub block_tables: Vec<Vec<BlockIdx>>, // Paged Attention 块表
    pub is_prefill: Vec<bool>,
    pub seq_ids: Vec<SeqId>,
    pub context_lens: Vec<u32>,
}

pub struct ExecutionOutput {
    pub next_tokens: Vec<TokenId>,
    pub logits: Option<Vec<f32>>,
    pub seq_ids: Vec<SeqId>,
}
```

`logits` 当前恒为 `None`：mock 与 CUDA 后端都直接产出占位 token，不输出 logits。

## 内存类型

```rust
pub struct PhysicalBlockRef {
    pub block_idx: BlockIdx,
}

pub struct LogicalBlock {
    pub block_idx: u32, // 序列内逻辑索引
    pub physical_block: PhysicalBlockRef, // 创建时即完成物理映射
}

pub struct MemoryStats {
    pub total_blocks: u32,
    pub used_blocks: u32,
    pub free_blocks: u32,
    pub num_sequences: u32,
}
```

`MemoryStats::utilization()` 返回 `used_blocks / total_blocks`。

## GPUExecutorTrait

GPU 计算的替换接口，目前只剩一个方法（早期 CUDA graph 相关方法已移除）：

```rust
pub trait GPUExecutorTrait: Send {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError>;
}
```

默认实现 `MockGPUExecutor` 生成确定性占位 token；启用 `cuda` feature 时切换为 `CudaExecutor`（nvcc 编译后端桥接），目前同样产出占位 token，尚未接入真实模型计算。详见[架构概览](../architecture/overview.md)。

## 使用示例

### 完整工作流

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), hetero_infer::EngineError> {
    // 配置
    let config = EngineConfig {
        max_batch_size: 64,
        max_num_blocks: 2048,
        ..Default::default()
    };

    // 创建引擎
    let mut engine = InferenceEngine::new(config)?;

    // 提交多个请求（GenerationParams 是 Copy，可直接多次传递）
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        ..Default::default()
    };

    engine.submit_request("First prompt", params)?;
    engine.submit_request("Second prompt", params)?;
    engine.submit_request("Third prompt", params)?;

    // 运行推理
    let completed = engine.run();

    // 处理结果
    for result in completed {
        if result.success {
            println!("Request {}: {}", result.request_id, result.output_text);
        } else if let Some(err) = &result.error {
            eprintln!("Request {} failed: {}", result.request_id, err);
        }
    }

    // 查看指标
    let metrics = engine.get_metrics();
    println!(
        "提交 {}，完成 {}，生成 {} tokens",
        metrics.total_requests,
        metrics.completed_requests,
        metrics.total_tokens_generated,
    );

    Ok(())
}
```

> 注意：输出文本由 mock 后端的占位 token 解码而来，不是真实模型生成结果。

---

下一篇: [完整 API 参考](reference)
