# RFC-0001: Heterogeneous Inference System Architecture

| Metadata | Value |
|----------|-------|
| **RFC ID** | 0001 |
| **Title** | Heterogeneous Inference System Architecture |
| **Status** | Accepted |
| **Created** | 2026-04-16 |
| **Authors** | Hetero-Paged-Infer Team |

## Abstract

This RFC describes the architecture of a heterogeneous inference microservice leveraging CPU-GPU co-execution for efficient LLM inference. The system implements PagedAttention for memory-efficient KV cache management and Continuous Batching for high throughput.

## Architecture Principles

- **CPU Responsibilities**: Tokenization, request scheduling, KV Cache page management, batch preparation
- **GPU Responsibilities**: Attention computation, matrix operations, token generation

### Core Innovations

1. **PagedAttention** - Virtual memory-inspired block management
2. **Continuous Batching** - Prefill and Decode phase interleaving
3. **CUDA Graphs** - Reduced kernel launch overhead
4. **Double-Buffered Batch Preparation** - Latency hiding

## System Architecture

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

### Inference Flow

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

### State Machine

```
                    ┌─────────────┐
                    │   Pending   │  (awaiting scheduling)
                    └──────┬──────┘
                           │ schedule()
                           ▼
                    ┌─────────────┐
            ┌───────│   Prefill   │  (processing input tokens)
            │       └──────┬──────┘
            │              │ prefill complete
            │              ▼
            │       ┌─────────────┐
            │       │   Decode    │◄────┐ (generating tokens)
            │       └──────┬──────┘     │
            │              │            │ generate next token
            │              ├────────────┘
            │              │ EOS or max_tokens
            │              ▼
            │       ┌─────────────┐
            └──────▶│  Completed  │  (finished)
                    └─────────────┘
```

## Component Interfaces

### 1. Request

```rust
struct Request {
    id: u64,                    // Unique request identifier
    input_tokens: Vec<u32>,     // Input token sequence
    output_tokens: Vec<u32>,    // Output token sequence
    max_tokens: u32,            // Maximum tokens to generate
    temperature: f32,           // Sampling temperature
    top_p: f32,                 // Top-p sampling parameter
    state: RequestState,        // Current state
    created_at: Instant,        // Creation timestamp
}

enum RequestState {
    Pending,                    // Awaiting scheduling
    Prefill,                    // Prefill phase
    Decode,                     // Decode phase
    Completed,                  // Completed
    Failed(String),             // Failed with error
}
```

### 2. Sequence

Collection representing an active request with its KV Cache blocks:

```rust
struct Sequence {
    seq_id: u64,                        // Unique sequence identifier
    request: Request,                   // Associated request
    logical_blocks: Vec<LogicalBlock>,  // List of logical blocks
    num_computed_tokens: u32,           // Number of computed tokens
    num_generated_tokens: u32,          // Number of generated tokens
}

struct LogicalBlock {
    block_idx: u32,                     // Logical block index
    physical_block: Option<PhysicalBlockRef>,  // Physical block reference
}
```

### 3. KVCacheManager Interface

```rust
trait KVCacheManagerTrait {
    /// Allocate blocks for a new sequence
    fn allocate_sequence(&mut self, seq_id: u64, num_tokens: u32) -> Result<(), MemoryError>;

    /// Allocate additional block for sequence growth
    fn allocate_block(&mut self, seq_id: u64) -> Result<PhysicalBlockRef, MemoryError>;

    /// Free all blocks for a sequence
    fn free_sequence(&mut self, seq_id: u64);

    /// Get block table for GPU execution
    fn get_block_table(&self, seq_id: u64) -> Option<Vec<u32>>;

    /// Query memory status
    fn get_memory_stats(&self) -> MemoryStats;

    /// Check if n blocks can be allocated
    fn can_allocate(&self, num_blocks: u32) -> bool;
}

struct MemoryStats {
    total_blocks: u32,      // Total blocks
    used_blocks: u32,       // Used blocks
    free_blocks: u32,       // Free blocks
    num_sequences: u32,     // Active sequences
}
```

### 4. Scheduler Interface

```rust
trait SchedulerTrait {
    /// Add request to pending queue
    fn add_request(&mut self, request: Request) -> Result<u64, SchedulerError>;

    /// Schedule next batch for execution
    fn schedule(&mut self) -> SchedulerOutput;

    /// Update sequence state after GPU execution
    fn update_sequences(&mut self, outputs: &ExecutionOutput);

    /// Get completed requests
    fn get_completed(&mut self) -> Vec<Request>;

    /// Check if there is pending work
    fn has_pending_work(&self) -> bool;
}

struct SchedulerOutput {
    prefill_sequences: Vec<SequenceRef>,   // Prefill sequences
    decode_sequences: Vec<SequenceRef>,    // Decode sequences
    block_tables: HashMap<u64, Vec<u32>>,  // Block tables
    total_tokens: u32,                     // Total tokens
}
```

### 5. GPUExecutor Interface

```rust
trait GPUExecutorTrait {
    /// Execute a batch of sequences
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;

    /// Capture CUDA Graph for decode phase
    fn capture_decode_graph(&mut self, batch_size: u32);

    /// Execute using captured CUDA Graph
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}

struct ExecutionBatch {
    input_tokens: Vec<u32>,        // Flattened tokens for all sequences
    positions: Vec<u32>,           // Position for each token
    seq_lens: Vec<u32>,            // Per-sequence lengths
    block_tables: Vec<Vec<u32>>,   // Paged attention block tables
    is_prefill: Vec<bool>,         // Prefill/Decode flags
}

struct ExecutionOutput {
    next_tokens: Vec<u32>,         // Next token per sequence
    logits: Option<Vec<f32>>,      // Logits (optional)
}
```

### 6. Tokenizer Interface

```rust
trait TokenizerTrait {
    fn encode(&self, text: &str) -> Vec<u32>;   // Text → tokens
    fn decode(&self, tokens: &[u32]) -> String; // Tokens → text
    fn vocab_size(&self) -> u32;                // Vocabulary size
    fn bos_token_id(&self) -> u32;              // BOS token ID
    fn eos_token_id(&self) -> u32;              // EOS token ID
    fn pad_token_id(&self) -> u32;              // PAD token ID
}
```

### 7. InferenceEngine (Main Orchestrator)

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
    fn step(&mut self) -> Vec<CompletedRequest>;  // Execute single step
    fn run(&mut self);                            // Main loop
}

struct EngineConfig {
    block_size: u32,          // Tokens per block (e.g., 16)
    max_num_blocks: u32,      // Total KV cache blocks
    max_batch_size: u32,      // Maximum batch size
    max_num_seqs: u32,        // Maximum concurrent sequences
    max_model_len: u32,       // Maximum sequence length
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

Physical blocks may be non-contiguous for efficient memory utilization.
```

### GPU Batch Data Layout

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

    // Attention context lengths
    context_lens: PinnedBuffer<u32>,    // [num_seqs]
}
```

## Correctness Properties

### Property 1: Request ID Uniqueness

*For any* set of requests submitted to the scheduler, all assigned sequence IDs shall be unique with no duplicates.

**Validates: REQ-1.2**

### Property 2: Parameter Validation Correctness

*For any* generation parameters (max_tokens, temperature, top_p), the validation function returns true if and only if all parameters are within valid ranges (max_tokens > 0, 0 < temperature ≤ 2.0, 0 < top_p ≤ 1.0).

**Validates: REQ-1.3**

### Property 3: Block Allocation on Sequence Start

*For any* new sequence with n input tokens, the KV Cache Manager shall allocate ceil(n / block_size) logical blocks, each mapped to a distinct physical block.

**Validates: REQ-2.2**

### Property 4: Block Allocation on Growth

*For any* sequence exceeding current block capacity, the KV Cache Manager shall allocate exactly one additional physical block when token count crosses a block boundary.

**Validates: REQ-2.3**

### Property 5: Block Count Invariant

*For any* KV Cache Manager state, the invariant `used_blocks + free_blocks == total_blocks` shall hold. Additionally, when sequences are freed, all blocks shall return to the free pool.

**Validates: REQ-2.4, REQ-2.5**

### Property 6: Scheduler Queue State Consistency

*For any* request in the scheduler, it shall exist in exactly one queue: pending, prefill, or decode.

**Validates: REQ-3.1**

### Property 7: Batch Size Constraints

*For any* scheduled batch, the number of sequences shall not exceed max_batch_size, and total tokens shall not exceed max_total_tokens.

**Validates: REQ-3.5**

### Property 8: Decode Priority over Prefill

*For any* scheduling decision with both prefill and decode requests pending and batch capacity available, all eligible decode requests shall be scheduled before prefill requests.

**Validates: REQ-3.7**

### Property 9: Prefill to Decode Transition

*For any* sequence completing prefill phase, it shall immediately transition to decode state within the same scheduling cycle.

**Validates: REQ-3.3**

### Property 10: Completion Condition

*For any* sequence in decode phase, it shall transition to completed state if and only if it generates an EOS token or reaches max_tokens.

**Validates: REQ-3.4**

### Property 11: Variable Sequence Length Handling

*For any* batch containing sequences of varying lengths, the GPU Executor shall independently produce correct attention output for each sequence.

**Validates: REQ-4.2**

### Property 12: Memory Statistics Invariant

*For any* KV Cache Manager state, reported memory statistics shall satisfy: `total_blocks == used_blocks + free_blocks` and `num_sequences == sequences with allocated blocks`.

**Validates: REQ-6.2**

### Property 13: Memory Pressure Response

*For any* state where memory utilization exceeds configured threshold, the scheduler shall reject new prefill requests until memory is freed.

**Validates: REQ-6.3**

### Property 14: Configuration Validation

*For any* configuration input, validation shall reject configurations with block_size ≤ 0, max_num_blocks ≤ 0, max_batch_size ≤ 0, or max_num_seqs ≤ 0.

**Validates: REQ-7.2**

### Property 15: Tokenizer Round-Trip

*For any* valid text input, decoding the encoded tokens shall produce text equivalent to the original (accounting for normalization).

**Validates: REQ-8.4**

## Error Handling

### Memory Errors

| Error Condition | Handling Strategy |
|-----------------|-------------------|
| No free blocks | Return `MemoryError::OutOfBlocks`, scheduler stops accepting new prefill |
| Block allocation fails | Log error, mark request failed, free partial allocation |
| GPU memory allocation failure | Graceful degradation, reduce max_num_blocks and retry |

### Request Errors

| Error Condition | Handling Strategy |
|-----------------|-------------------|
| Invalid parameters | Return validation error immediately, do not queue request |
| Tokenization failure | Return error with details, do not create sequence |
| Exceeds maximum sequence length | Truncate or reject based on configuration |

### Execution Errors

| Error Condition | Handling Strategy |
|-----------------|-------------------|
| CUDA kernel error | Log error, mark affected sequences failed, continue processing others |
| GPU timeout | Log warning, retry once, then mark as failed |
| Output contains NaN/Inf | Detect and mark sequence as failed |

### Recovery Strategies

```rust
enum RecoveryAction {
    Retry { max_attempts: u32 },   // Retry operation
    SkipSequence,                  // Skip affected sequence
    ResetBatch,                    // Reset current batch
    Shutdown,                      // Shut down engine
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

## Test Strategy

### Unit Tests

Unit tests focus on concrete examples and edge cases:

1. **KV Cache Manager**
   - Allocate single sequence, verify block mapping
   - Allocate multiple sequences, verify isolation
   - Free sequence, verify blocks returned
   - Edge case: allocate exactly at capacity

2. **Scheduler**
   - Add single request, verify queuing
   - Schedule batch with prefill-only requests
   - Schedule batch with decode-only requests
   - Edge case: empty scheduler returns empty batch

3. **Tokenizer**
   - Encode known text, verify expected tokens
   - Decode known tokens, verify expected text
   - Handle empty string
   - Handle special characters

4. **Configuration**
   - Load valid configuration file
   - Reject invalid configuration values
   - Apply defaults for missing fields

### Property Tests

Property tests verify universal properties across many generated inputs using the `proptest` crate.

Each property test must:
- Run minimum 100 iterations
- Reference design document properties
- Use label format: **Feature: heterogeneous-inference-system, Property N: [property text]**

**Test Configuration:**
```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // Property tests
}
```

### Integration Tests

1. **End-to-End Request Flow**
   - Submit request, run until completion
   - Verify output tokens generated
   - Verify KV cache freed on completion

2. **Continuous Batching**
   - Submit multiple requests at staggered times
   - Verify all complete correctly
   - Verify formation of mixed prefill/decode batches

3. **Memory Pressure**
   - Fill memory to threshold
   - Verify new prefill is rejected
   - Complete some requests
   - Verify new prefill is accepted again

## Current Implementation Status

### Implemented ✅

- PagedAttention KV Cache management
- Continuous Batching scheduler
- Memory pressure awareness
- Modular trait abstractions
- Comprehensive property testing
- Mock GPU executor for testing

### Not Implemented ❌

- Real CUDA kernels
- True pinned memory
- Copy-on-write KV sharing
- Asynchronous CPU/GPU overlap

The `GPUExecutor` is currently a **mock implementation** for testing and validating scheduling logic.
