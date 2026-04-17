# 核心类型

## InferenceEngine

推理操作的主要编排器。

```rust
pub struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn TokenizerTrait>,
    scheduler: Box<dyn SchedulerTrait>,
    kv_cache_manager: Box<dyn KVCacheManagerTrait>,
    gpu_executor: Box<dyn GPUExecutorTrait>,
    metrics: EngineMetrics,
}
```

### 方法

#### `new()`

```rust
pub fn new(config: EngineConfig) -> Result<Self, EngineError>
```

使用给定配置创建一个新的推理引擎。

```rust
let config = EngineConfig::default();
let engine = InferenceEngine::new(config)?;
```

#### `submit_request()`

```rust
pub fn submit_request(
    &mut self,
    text: &str,
    params: GenerationParams
) -> Result<u64, EngineError>
```

提交文本生成请求。

```rust
let params = GenerationParams {
    max_tokens: 100,
    temperature: 0.8,
    top_p: 0.95,
};
let request_id = engine.submit_request("Hello", params)?;
```

#### `run()`

```rust
pub fn run(&mut self) -> Vec<CompletedRequest>
```

运行主推理循环，直到所有请求完成。

```rust
let completed = engine.run();
for result in completed {
    println!("{}", result.output_text);
}
```

#### `step()`

```rust
pub fn step(&mut self) -> Vec<CompletedRequest>
```

执行单次调度和推理步骤。

```rust
while engine.has_pending_work() {
    let completed = engine.step();
    process_results(&completed);
}
```

## EngineConfig

推理引擎的配置。

```rust
pub struct EngineConfig {
    pub block_size: u32,
    pub max_num_blocks: u32,
    pub max_batch_size: u32,
    pub max_num_seqs: u32,
    pub max_model_len: u32,
    pub max_total_tokens: u32,
    pub memory_threshold: f32,
}
```

| 字段 | 默认值 | 描述 |
|-------|---------|-------------|
| block_size | 16 | 每个物理块包含的 token 数 |
| max_num_blocks | 1024 | 物理块总数 |
| max_batch_size | 32 | 每批次最大序列数 |
| max_num_seqs | 256 | 最大并发序列数 |
| max_model_len | 2048 | 最大上下文长度 |
| max_total_tokens | 4096 | 每批次最大 token 数 |
| memory_threshold | 0.9 | 内存压力阈值 |

```rust
impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            max_num_blocks: 1024,
            max_batch_size: 32,
            max_num_seqs: 256,
            max_model_len: 2048,
            max_total_tokens: 4096,
            memory_threshold: 0.9,
        }
    }
}
```

## GenerationParams

文本生成参数。

```rust
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}
```

| 字段 | 默认值 | 范围 | 描述 |
|-------|---------|-------|-------------|
| max_tokens | 100 | 1+ | 最大生成 token 数 |
| temperature | 1.0 | 0.0-2.0 | 采样温度 |
| top_p | 0.9 | 0.0-1.0 | 核采样阈值 |

## Request

表示一个推理请求。

```rust
pub struct Request {
    pub id: u64,
    pub input_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub state: RequestState,
    pub created_at: Instant,
}

pub enum RequestState {
    Pending,
    Prefill,
    Decode,
    Completed,
    Failed(String),
}
```

## Sequence

已分配 KV Cache 的活跃请求。

```rust
pub struct Sequence {
    pub seq_id: u64,
    pub request: Request,
    pub logical_blocks: Vec<LogicalBlock>,
    pub num_computed_tokens: u32,
    pub num_generated_tokens: u32,
}
```

## CompletedRequest

已完成的推理结果。

```rust
pub struct CompletedRequest {
    pub request_id: u64,
    pub input_text: String,
    pub output_text: String,
    pub input_tokens: usize,
    pub generated_tokens: usize,
    pub duration: Duration,
}
```

## EngineMetrics

运行时性能指标。

```rust
pub struct EngineMetrics {
    pub requests_processed: u64,
    pub tokens_generated: u64,
    pub avg_latency_ms: f64,
    pub throughput_tok_per_sec: f64,
}
```

## 使用示例

### 完整工作流

```rust
use hetero_infer::*;
use std::time::Duration;

fn main() -> Result<(), EngineError> {
    // 配置
    let config = EngineConfig {
        max_batch_size: 64,
        max_num_blocks: 2048,
        ..Default::default()
    };

    // 创建引擎
    let mut engine = InferenceEngine::new(config)?;

    // 提交多个请求
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        ..Default::default()
    };

    engine.submit_request("First prompt", params.clone())?;
    engine.submit_request("Second prompt", params.clone())?;
    engine.submit_request("Third prompt", params)?;

    // 运行推理
    let completed = engine.run();

    // 处理结果
    for result in completed {
        println!("Request {}: {}",
            result.request_id,
            result.output_text
        );
    }

    // 查看指标
    let metrics = &engine.metrics;
    println!("Throughput: {:.2} tok/s",
        metrics.throughput_tok_per_sec
    );

    Ok(())
}
```

---

下一篇: [Trait 接口](traits.md)
