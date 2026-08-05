# Core Types

This page explains what each core type is and when it is used. Field lists are checked against the source one by one; for exact signatures and full documentation, defer to Rustdoc:

```bash
cargo doc --no-deps --open
```

Common type aliases (defined in `types/mod.rs`):

| Alias | Definition | Purpose |
|------|------|------|
| `RequestId` | `u64` | Request identifier |
| `SeqId` | `u64` | Sequence identifier |
| `TokenId` | `u32` | Token ID |
| `BlockIdx` | `u32` | Physical block index |

## InferenceEngine

Main orchestrator for inference operations: coordinates the tokenizer, scheduler, execution pipeline and KV Cache for end-to-end inference. All fields are private; interaction is only through methods. Runtime metrics are obtained via `get_metrics()`; there are no public fields.

### Main methods

```rust
impl InferenceEngine {
    pub fn new(config: EngineConfig) -> Result<Self, EngineError>;

    // Inject custom components (for testing)
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

    // Execute one inference step, returning requests completed in this step
    pub fn step(&mut self) -> Result<Vec<CompletedRequest>, EngineError>;

    // Execute one step and return fine-grained events (for token-level streaming)
    pub fn step_events(&mut self) -> Result<StepEvents, EngineError>;

    // Run the inference loop until all requests reach a terminal state
    pub fn run(&mut self) -> Vec<CompletedRequest>;

    pub fn has_pending_work(&self) -> bool;
    pub fn memory_utilization(&self) -> f32;
    pub fn config(&self) -> &EngineConfig;
    pub fn get_metrics(&self) -> EngineMetrics;
}
```

### Submit and run

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

let mut engine = InferenceEngine::new(EngineConfig::default())?;
let request_id = engine.submit_request(
    "Hello, world!",
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

### Manual step loop

```rust
while engine.has_pending_work() {
    let completed = engine.step()?;
    for result in &completed {
        println!("{}: {}", result.request_id, result.output_text);
    }
}
# Ok::<(), hetero_infer::EngineError>(())
```

To drive token-level streaming responses, use `step_events()` instead; it additionally reports the text chunks newly generated for each request in this step.

## StepEvents

Per-step events produced by `step_events()`, used by the serving layer to drive streaming responses.

```rust
pub struct StepEvents {
    /// Requests that reached a terminal state (success or failure) in this step
    pub completed: Vec<CompletedRequest>,
    /// Text chunks newly generated for each request in this step: (request_id, chunk)
    pub chunks: Vec<(RequestId, String)>,
}
```

## EngineConfig

Configuration for the inference engine. All fields are public; `Default` provides a usable set of defaults.

| Field | Default | Description |
|-------|---------|-------------|
| block_size | 16 | Tokens per KV Cache block |
| max_num_blocks | 1024 | Total physical blocks (determines KV Cache capacity) |
| max_batch_size | 32 | Max sequences per scheduling round |
| max_num_seqs | 256 | Max concurrent sequences in the system |
| max_model_len | 2048 | Max sequence length (input + output) |
| max_total_tokens | 4096 | Max total tokens per batch |
| memory_threshold | 0.9 | Memory pressure threshold, range (0.0, 1.0] |
| max_retry_attempts | 2 | Retries on GPU execution timeout |
| special_tokens | bos=1, eos=2, pad=0, unk=3 | Special token IDs |
| tokenizer | Simple | Tokenizer implementation kind and path |
| serving | 127.0.0.1:3000 | HTTP service host, port and model name |

```rust
let config = EngineConfig {
    max_batch_size: 64,
    max_num_blocks: 2048,
    ..Default::default()
};
```

## GenerationParams

Sampling and generation limits. Passed as a whole to `submit_request()` and stored as the nested field `params` inside `Request`.

```rust
pub struct GenerationParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
}
```

| Field | Default | Range | Description |
|-------|---------|-------|-------------|
| max_tokens | 100 | > 0 | Maximum tokens to generate |
| temperature | 1.0 | [0.0, 2.0] | Sampling temperature (0.0 means greedy decoding, valid) |
| top_p | 1.0 | (0.0, 1.0] | Nucleus sampling threshold |

> Current state: the default backend is a mock implementation that generates deterministic placeholder tokens. `temperature` and `top_p` are currently only validated at submission time and do not yet affect output; sampling is not implemented.

## Request and RequestState

A single inference request. Created internally by the engine in `submit_request()`; usually no need to construct manually.

```rust
pub struct Request {
    pub id: RequestId,
    pub input_tokens: Vec<TokenId>,
    pub output_tokens: Vec<TokenId>,
    pub params: GenerationParams, // Nested generation parameters
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

State transitions: `Pending → Prefill → Decode → Completed`, or `Failed` on error.

## CompletedRequest

Final result of a request, returned by `step()` / `run()`.

```rust
pub struct CompletedRequest {
    pub request_id: RequestId,
    pub input_text: Option<String>, // None if decoding the input fails
    pub output_text: String,
    pub output_tokens: Vec<TokenId>,
    pub success: bool,
    pub error: Option<String>, // Error message on failure
}
```

Consumers should check `success` before reading `output_text`.

## EngineMetrics

Runtime statistics snapshot, obtained via `engine.get_metrics()` (engine fields are private and cannot be accessed directly).

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

The engine currently has no latency/throughput measurement capability: there are no fields such as average latency or tok/s — do not build monitoring that depends on such metrics.

```rust
let metrics = engine.get_metrics();
println!(
    "completed {}/{} requests, {} tokens generated, memory utilization {:.2}",
    metrics.completed_requests,
    metrics.total_requests,
    metrics.total_tokens_generated,
    metrics.memory_utilization,
);
# Ok::<(), hetero_infer::EngineError>(())
```

## Sequence and SchedulerOutput

Scheduler-layer internal types. `Sequence` is an active request with KV Cache blocks attached; `SchedulerOutput` is the batch snapshot produced by one scheduling round, consumed by the execution pipeline.

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

## ExecutionBatch and ExecutionOutput

The data contract between the scheduler and the GPU executor.

```rust
pub struct ExecutionBatch {
    pub input_tokens: Vec<TokenId>, // Tokens of all sequences (flattened)
    pub positions: Vec<u32>,
    pub seq_lens: Vec<u32>,
    pub block_tables: Vec<Vec<BlockIdx>>, // Paged Attention block tables
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

`logits` is currently always `None`: both the mock and CUDA backends produce placeholder tokens directly and do not output logits.

## Memory types

```rust
pub struct PhysicalBlockRef {
    pub block_idx: BlockIdx,
}

pub struct LogicalBlock {
    pub block_idx: u32, // Logical index within the sequence
    pub physical_block: PhysicalBlockRef, // Physically mapped at creation time
}

pub struct MemoryStats {
    pub total_blocks: u32,
    pub used_blocks: u32,
    pub free_blocks: u32,
    pub num_sequences: u32,
}
```

`MemoryStats::utilization()` returns `used_blocks / total_blocks`.

## GPUExecutorTrait

The replacement interface for GPU computation. Only one method remains (the earlier CUDA graph related methods have been removed):

```rust
pub trait GPUExecutorTrait: Send {
    fn execute(&mut self, batch: &ExecutionBatch) -> Result<ExecutionOutput, ExecutionError>;
}
```

The default implementation `MockGPUExecutor` generates deterministic placeholder tokens; enabling the `cuda` feature switches to `CudaExecutor` (a bridge to the nvcc-compiled backend), which currently also produces placeholder tokens and is not yet connected to real model computation. See [Architecture Overview](../architecture/overview.md) for details.

## Usage Examples

### Complete Workflow

```rust
use hetero_infer::{EngineConfig, GenerationParams, InferenceEngine};

fn main() -> Result<(), hetero_infer::EngineError> {
    // Configure
    let config = EngineConfig {
        max_batch_size: 64,
        max_num_blocks: 2048,
        ..Default::default()
    };

    // Create engine
    let mut engine = InferenceEngine::new(config)?;

    // Submit multiple requests (GenerationParams is Copy, so it can be passed directly multiple times)
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.8,
        ..Default::default()
    };

    engine.submit_request("First prompt", params)?;
    engine.submit_request("Second prompt", params)?;
    engine.submit_request("Third prompt", params)?;

    // Run inference
    let completed = engine.run();

    // Process results
    for result in completed {
        if result.success {
            println!("Request {}: {}", result.request_id, result.output_text);
        } else if let Some(err) = &result.error {
            eprintln!("Request {} failed: {}", result.request_id, err);
        }
    }

    // Check metrics
    let metrics = engine.get_metrics();
    println!(
        "submitted {}, completed {}, generated {} tokens",
        metrics.total_requests,
        metrics.completed_requests,
        metrics.total_tokens_generated,
    );

    Ok(())
}
```

> Note: the output text is decoded from placeholder tokens produced by the mock backend, not real model generation results.

---

Next: [Full API Reference](reference)
