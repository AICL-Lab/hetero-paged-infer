# Design Document: Heterogeneous Inference System

## Overview

This document describes the design of a heterogeneous inference microservice that leverages CPU-GPU co-execution for efficient LLM inference. The system implements PagedAttention for memory-efficient KV cache management and Continuous Batching for high throughput.

The architecture follows a pipeline design where:
- **CPU** handles: Tokenization, Request Scheduling, KV Cache page management, Batch preparation
- **GPU** handles: Attention computation, Matrix operations, Token generation

Key innovations:
1. PagedAttention with virtual memory-like block management
2. Continuous Batching mixing Prefill and Decode phases
3. CUDA Graphs for reduced kernel launch overhead
4. Double-buffered batch preparation for latency hiding

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Inference Engine                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Tokenizer  │    │   Scheduler  │    │    KV Cache Manager      │  │
│  │    (CPU)     │    │    (CPU)     │    │        (CPU)             │  │
│  └──────┬───────┘    └──────┬───────┘    └───────────┬──────────────┘  │
│         │                   │                        │                  │
│         │            ┌──────▼───────┐               │                  │
│         │            │ Batch Builder│◄──────────────┘                  │
│         │            │    (CPU)     │                                  │
│         │            └──────┬───────┘                                  │
│         │                   │                                          │
│  ───────┼───────────────────┼──────────────────────────────────────── │
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

### Pipeline Flow

```
┌─────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│ Request │──▶│ Tokenize  │──▶│ Schedule  │──▶│  Execute  │──▶│ Detokenize│
│  Input  │   │   (CPU)   │   │   (CPU)   │   │   (GPU)   │   │   (CPU)   │
└─────────┘   └───────────┘   └───────────┘   └───────────┘   └──────────┘
                                    │               │
                                    │    ┌──────────┘
                                    ▼    ▼
                              ┌───────────────┐
                              │  KV Cache Mgr │
                              │     (CPU)     │
                              └───────────────┘
```

## Components and Interfaces

### 1. Request

```rust
struct Request {
    id: u64,
    input_tokens: Vec<u32>,
    output_tokens: Vec<u32>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    state: RequestState,
    created_at: Instant,
}

enum RequestState {
    Pending,
    Prefill,
    Decode,
    Completed,
    Failed(String),
}
```

### 2. Sequence (Active Request with KV Cache)

```rust
struct Sequence {
    seq_id: u64,
    request: Request,
    logical_blocks: Vec<LogicalBlock>,
    num_computed_tokens: u32,
    num_generated_tokens: u32,
}

struct LogicalBlock {
    block_idx: u32,
    physical_block: Option<PhysicalBlockRef>,
}
```

### 3. KV Cache Manager Interface

```rust
trait KVCacheManager {
    /// Allocate blocks for a new sequence
    fn allocate_sequence(&mut self, seq_id: u64, num_tokens: u32) -> Result<(), MemoryError>;
    
    /// Allocate additional block when sequence grows
    fn allocate_block(&mut self, seq_id: u64) -> Result<PhysicalBlockRef, MemoryError>;
    
    /// Free all blocks for a completed sequence
    fn free_sequence(&mut self, seq_id: u64);
    
    /// Get block table for GPU execution
    fn get_block_table(&self, seq_id: u64) -> &[u32];
    
    /// Query memory status
    fn get_memory_stats(&self) -> MemoryStats;
    
    /// Check if can allocate n blocks
    fn can_allocate(&self, num_blocks: u32) -> bool;
}

struct MemoryStats {
    total_blocks: u32,
    used_blocks: u32,
    free_blocks: u32,
    num_sequences: u32,
}
```

### 4. Scheduler Interface

```rust
trait Scheduler {
    /// Add new request to pending queue
    fn add_request(&mut self, request: Request);
    
    /// Schedule next batch for execution
    fn schedule(&mut self) -> SchedulerOutput;
    
    /// Update sequences after GPU execution
    fn update_sequences(&mut self, outputs: &ExecutionOutput);
    
    /// Get completed requests
    fn get_completed(&mut self) -> Vec<Request>;
    
    /// Check if scheduler has work
    fn has_pending_work(&self) -> bool;
}

struct SchedulerOutput {
    prefill_sequences: Vec<SequenceRef>,
    decode_sequences: Vec<SequenceRef>,
    block_tables: HashMap<u64, Vec<u32>>,
    total_tokens: u32,
}
```

### 5. GPU Executor Interface

```rust
trait GPUExecutor {
    /// Execute a batch of sequences
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
    
    /// Capture CUDA graph for decode phase
    fn capture_decode_graph(&mut self, batch_size: u32);
    
    /// Execute using captured CUDA graph
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}

struct ExecutionBatch {
    /// Token IDs for all sequences (flattened)
    input_tokens: Vec<u32>,
    /// Position IDs for each token
    positions: Vec<u32>,
    /// Sequence lengths for attention masking
    seq_lens: Vec<u32>,
    /// Block tables for paged attention (seq_id -> block indices)
    block_tables: Vec<Vec<u32>>,
    /// Flags indicating prefill vs decode
    is_prefill: Vec<bool>,
}

struct ExecutionOutput {
    /// Next token for each sequence
    next_tokens: Vec<u32>,
    /// Logits if needed for sampling
    logits: Option<Vec<f32>>,
}
```

### 6. Tokenizer Interface

```rust
trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn vocab_size(&self) -> u32;
    fn bos_token_id(&self) -> u32;
    fn eos_token_id(&self) -> u32;
    fn pad_token_id(&self) -> u32;
}
```

### 7. Inference Engine (Main Orchestrator)

```rust
struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn Tokenizer>,
    scheduler: Box<dyn Scheduler>,
    kv_cache_manager: Box<dyn KVCacheManager>,
    gpu_executor: Box<dyn GPUExecutor>,
}

impl InferenceEngine {
    fn submit_request(&mut self, text: &str, params: GenerationParams) -> RequestId;
    fn step(&mut self) -> Vec<CompletedRequest>;
    fn run(&mut self);  // Main loop
}

struct EngineConfig {
    block_size: u32,          // Tokens per block (e.g., 16)
    max_num_blocks: u32,      // Total KV cache blocks
    max_batch_size: u32,      // Max sequences per batch
    max_num_seqs: u32,        // Max concurrent sequences
    max_model_len: u32,       // Max sequence length
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

Each block stores:
- K cache: [block_size, num_heads, head_dim]
- V cache: [block_size, num_heads, head_dim]
```

### Page Table Structure

```
Sequence 0:                    Sequence 1:
┌──────────────┐              ┌──────────────┐
│ Logical → Physical │        │ Logical → Physical │
├──────────────┤              ├──────────────┤
│   0   →   3   │              │   0   →   1   │
│   1   →   7   │              │   1   →   5   │
│   2   →   12  │              │   2   →   9   │
└──────────────┘              └──────────────┘

Physical blocks can be non-contiguous, enabling efficient memory utilization.
```

### Batch Data Layout for GPU

```rust
struct GPUBatchData {
    // Pinned host memory for fast transfer
    input_ids: PinnedBuffer<u32>,      // [total_tokens]
    positions: PinnedBuffer<u32>,       // [total_tokens]
    
    // Sequence metadata
    seq_start_locs: PinnedBuffer<u32>,  // [num_seqs + 1] cumulative
    seq_lens: PinnedBuffer<u32>,        // [num_seqs]
    
    // Block tables (padded to max_blocks_per_seq)
    block_tables: PinnedBuffer<u32>,    // [num_seqs, max_blocks_per_seq]
    
    // Context lengths for attention
    context_lens: PinnedBuffer<u32>,    // [num_seqs]
}
```

### Continuous Batching State Machine

```
                    ┌─────────────┐
                    │   Pending   │
                    └──────┬──────┘
                           │ schedule()
                           ▼
                    ┌─────────────┐
            ┌───────│   Prefill   │
            │       └──────┬──────┘
            │              │ prefill complete
            │              ▼
            │       ┌─────────────┐
            │       │   Decode    │◄────┐
            │       └──────┬──────┘     │
            │              │            │ generate token
            │              ├────────────┘
            │              │ EOS or max_tokens
            │              ▼
            │       ┌─────────────┐
            └──────▶│  Completed  │
                    └─────────────┘
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Request ID Uniqueness

*For any* set of requests submitted to the Scheduler, all assigned sequence IDs shall be unique with no collisions.

**Validates: Requirements 1.2**

### Property 2: Parameter Validation Correctness

*For any* generation parameters (max_tokens, temperature, top_p), the validation function shall return true if and only if all parameters are within their acceptable ranges (max_tokens > 0, 0 < temperature <= 2.0, 0 < top_p <= 1.0).

**Validates: Requirements 1.3**

### Property 3: Block Allocation on Sequence Start

*For any* new sequence with n input tokens, the KV_Cache_Manager shall allocate ceil(n / block_size) logical blocks, each mapped to a distinct physical block.

**Validates: Requirements 2.2**

### Property 4: Block Allocation on Growth

*For any* sequence that grows beyond its current block capacity, the KV_Cache_Manager shall allocate exactly one additional physical block when the token count crosses a block boundary.

**Validates: Requirements 2.3**

### Property 5: Block Count Invariant

*For any* state of the KV_Cache_Manager, the invariant `used_blocks + free_blocks == total_blocks` shall hold. Additionally, when a sequence is freed, all its blocks shall return to the free pool.

**Validates: Requirements 2.4, 2.5**

### Property 6: Scheduler Queue State Consistency

*For any* request in the scheduler, it shall be in exactly one queue: pending (if not yet scheduled), prefill (if in prefill phase), or decode (if in decode phase).

**Validates: Requirements 3.1**

### Property 7: Batch Size Constraints

*For any* scheduled batch, the number of sequences shall not exceed max_batch_size, and the total number of tokens shall not exceed max_total_tokens.

**Validates: Requirements 3.5**

### Property 8: Decode Priority Over Prefill

*For any* scheduling decision where both prefill and decode requests are pending and batch capacity allows, all eligible decode requests shall be scheduled before any prefill requests.

**Validates: Requirements 3.7**

### Property 9: Prefill to Decode Transition

*For any* sequence that completes its prefill phase, it shall immediately transition to decode state in the same scheduling cycle.

**Validates: Requirements 3.3**

### Property 10: Completion Conditions

*For any* sequence in decode phase, it shall transition to completed state if and only if it generates an EOS token or reaches max_tokens.

**Validates: Requirements 3.4**

### Property 11: Variable Sequence Length Handling

*For any* batch containing sequences of different lengths, the GPU_Executor shall produce correct attention outputs for each sequence independently.

**Validates: Requirements 4.2**

### Property 12: Memory Statistics Invariant

*For any* state of the KV_Cache_Manager, the reported memory statistics shall satisfy: `total_blocks == used_blocks + free_blocks` and `num_sequences == count of sequences with allocated blocks`.

**Validates: Requirements 6.2**

### Property 13: Memory Pressure Response

*For any* state where memory utilization exceeds the configured threshold, the Scheduler shall reject new prefill requests until memory is freed.

**Validates: Requirements 6.3**

### Property 14: Configuration Validation

*For any* configuration input, the validation shall reject configurations where block_size <= 0, max_num_blocks <= 0, max_batch_size <= 0, or max_num_seqs <= 0.

**Validates: Requirements 7.2**

### Property 15: Tokenizer Round-Trip

*For any* valid text input, decoding the encoded tokens shall produce text equivalent to the original input (accounting for normalization).

**Validates: Requirements 8.4**

## Error Handling

### Memory Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| No free blocks available | Return `MemoryError::OutOfBlocks`, scheduler stops accepting new prefills |
| Block allocation fails | Log error, mark request as failed, free partial allocations |
| GPU memory allocation fails | Graceful degradation, reduce max_num_blocks and retry |

### Request Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| Invalid parameters | Return validation error immediately, do not queue request |
| Tokenization fails | Return error with details, do not create sequence |
| Max sequence length exceeded | Truncate or reject based on configuration |

### Execution Errors

| Error Condition | Handling Strategy |
|----------------|-------------------|
| CUDA kernel error | Log error, mark affected sequences as failed, continue with others |
| GPU timeout | Log warning, retry once, then mark as failed |
| NaN/Inf in outputs | Detect and mark sequence as failed |

### Recovery Strategies

```rust
enum RecoveryAction {
    Retry { max_attempts: u32 },
    SkipSequence,
    ResetBatch,
    Shutdown,
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

## Testing Strategy

### Unit Tests

Unit tests focus on specific examples and edge cases:

1. **KV Cache Manager**
   - Allocate single sequence, verify block mapping
   - Allocate multiple sequences, verify isolation
   - Free sequence, verify blocks returned
   - Edge case: allocate when exactly at capacity

2. **Scheduler**
   - Add single request, verify queued
   - Schedule batch with only prefill requests
   - Schedule batch with only decode requests
   - Edge case: empty scheduler returns empty batch

3. **Tokenizer**
   - Encode known text, verify expected tokens
   - Decode known tokens, verify expected text
   - Handle empty string
   - Handle special characters

4. **Configuration**
   - Load valid config file
   - Reject invalid config values
   - Apply default values for missing fields

### Property-Based Tests

Property-based tests validate universal properties across many generated inputs. We will use the `proptest` crate for Rust.

Each property test must:
- Run minimum 100 iterations
- Reference the design document property
- Use tag format: **Feature: heterogeneous-inference-system, Property N: [property_text]**

**Test Configuration:**
```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    
    // Property tests here
}
```

**Property Test Implementations:**

1. **Property 1: Request ID Uniqueness**
   - Generate N random requests
   - Submit all to scheduler
   - Assert all sequence IDs are unique

2. **Property 2: Parameter Validation**
   - Generate random (max_tokens, temperature, top_p) tuples
   - Assert validation result matches expected based on ranges

3. **Property 5: Block Count Invariant**
   - Generate sequence of allocate/free operations
   - After each operation, assert used + free == total

4. **Property 7: Batch Size Constraints**
   - Generate random request workload
   - Schedule batches
   - Assert each batch respects size limits

5. **Property 15: Tokenizer Round-Trip**
   - Generate random valid text strings
   - Assert decode(encode(text)) == normalize(text)

### Integration Tests

1. **End-to-End Request Flow**
   - Submit request, run until completion
   - Verify output tokens generated
   - Verify KV cache freed after completion

2. **Continuous Batching**
   - Submit multiple requests with staggered timing
   - Verify all complete correctly
   - Verify mixed prefill/decode batches formed

3. **Memory Pressure**
   - Fill memory to threshold
   - Verify new prefills rejected
   - Complete some requests
   - Verify new prefills accepted again

### Test Data Generators

```rust
// Generate valid generation parameters
fn arb_valid_params() -> impl Strategy<Value = GenerationParams> {
    (1u32..1000, 0.1f32..2.0, 0.1f32..1.0)
        .prop_map(|(max_tokens, temp, top_p)| GenerationParams {
            max_tokens,
            temperature: temp,
            top_p,
        })
}

// Generate sequence of cache operations
fn arb_cache_ops() -> impl Strategy<Value = Vec<CacheOp>> {
    prop::collection::vec(
        prop_oneof![
            any::<u64>().prop_map(CacheOp::Allocate),
            any::<u64>().prop_map(CacheOp::Free),
            (any::<u64>(), 1u32..100).prop_map(|(id, n)| CacheOp::Grow(id, n)),
        ],
        0..50
    )
}

// Generate valid text for tokenizer testing
fn arb_valid_text() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9 .,!?]{1,100}"
}
```
