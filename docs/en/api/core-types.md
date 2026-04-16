# Core Types

## InferenceEngine

Main orchestrator for inference operations.

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

### Methods

#### `new()`

```rust
pub fn new(config: EngineConfig) -> Result<Self, EngineError>
```

Create a new inference engine with the given configuration.

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

Submit a text generation request.

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

Run the main inference loop until all requests complete.

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

Execute a single scheduling and inference step.

```rust
while engine.has_pending_work() {
    let completed = engine.step();
    process_results(&completed);
}
```

## EngineConfig

Configuration for the inference engine.

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

| Field | Default | Description |
|-------|---------|-------------|
| block_size | 16 | Tokens per physical block |
| max_num_blocks | 1024 | Total physical blocks |
| max_batch_size | 32 | Max sequences per batch |
| max_num_seqs | 256 | Max concurrent sequences |
| max_model_len | 2048 | Max context length |
| max_total_tokens | 4096 | Max tokens per batch |
| memory_threshold | 0.9 | Memory pressure threshold |

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

Parameters for text generation.

```rust
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}
```

| Field | Default | Range | Description |
|-------|---------|-------|-------------|
| max_tokens | 100 | 1+ | Maximum tokens to generate |
| temperature | 1.0 | 0.0-2.0 | Sampling temperature |
| top_p | 0.9 | 0.0-1.0 | Nucleus sampling threshold |

## Request

Represents an inference request.

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

Active request with KV Cache allocation.

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

Result of completed inference.

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

Runtime performance metrics.

```rust
pub struct EngineMetrics {
    pub requests_processed: u64,
    pub tokens_generated: u64,
    pub avg_latency_ms: f64,
    pub throughput_tok_per_sec: f64,
}
```

## Usage Examples

### Complete Workflow

```rust
use hetero_infer::*;
use std::time::Duration;

fn main() -> Result<(), EngineError> {
    // Configure
    let config = EngineConfig {
        max_batch_size: 64,
        max_num_blocks: 2048,
        ..Default::default()
    };
    
    // Create engine
    let mut engine = InferenceEngine::new(config)?;
    
    // Submit multiple requests
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        ..Default::default()
    };
    
    engine.submit_request("First prompt", params.clone())?;
    engine.submit_request("Second prompt", params.clone())?;
    engine.submit_request("Third prompt", params)?;
    
    // Run inference
    let completed = engine.run();
    
    // Process results
    for result in completed {
        println!("Request {}: {}", 
            result.request_id, 
            result.output_text
        );
    }
    
    // Check metrics
    let metrics = &engine.metrics;
    println!("Throughput: {:.2} tok/s", 
        metrics.throughput_tok_per_sec
    );
    
    Ok(())
}
```

---

Next: [Trait Interfaces](traits.md)
