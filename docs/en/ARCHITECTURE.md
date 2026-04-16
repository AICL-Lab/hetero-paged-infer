# Architecture Guide

## Overview

Hetero-Paged-Infer implements a heterogeneous computing architecture where CPU and GPU collaborate efficiently for LLM inference. The system separates control flow (CPU) from compute-heavy operations (GPU) to maximize resource utilization.

## Design Principles

- **CPU Responsibilities**: Tokenization, request scheduling, KV Cache page management, batch preparation
- **GPU Responsibilities**: Attention computation, matrix operations, token generation
- **Memory Efficiency**: PagedAttention avoids memory waste from padding and fragmentation
- **Throughput Optimization**: Continuous batching maximizes GPU utilization

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           InferenceEngine                                │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   Tokenizer  │    │   Scheduler  │    │    KV Cache Manager      │  │
│  │    (CPU)     │    │    (CPU)     │    │        (CPU)             │  │
│  │  Encode/Decode│    │(Prefill/    │    │  (BlockPool/PageTable)   │  │
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

## Core Components

### 1. InferenceEngine

The main orchestrator that coordinates all components:

```rust
pub struct InferenceEngine {
    config: EngineConfig,
    tokenizer: Box<dyn TokenizerTrait>,
    scheduler: Box<dyn SchedulerTrait>,
    kv_cache_manager: Box<dyn KVCacheManagerTrait>,
    gpu_executor: Box<dyn GPUExecutorTrait>,
}
```

**Responsibilities**:
- Request submission and lifecycle management
- Step-by-step execution loop
- Error recovery strategies
- Metrics collection

### 2. Scheduler

Implements Continuous Batching with decode-priority scheduling:

**Key Features**:
- Separate queues for prefill and decode requests
- Dynamic batch formation
- Memory pressure awareness
- Request state transitions

**Scheduling Algorithm**:
1. Prioritize decode requests (lower latency for in-flight requests)
2. Fill remaining batch capacity with prefill requests
3. Respect memory thresholds and batch size limits

### 3. KV Cache Manager

Implements PagedAttention memory management:

**Core Concepts**:
- **Physical Blocks**: Contiguous GPU memory regions
- **Logical Blocks**: Virtual blocks mapped via page table
- **Block Pool**: Free list management for physical blocks

**Memory Layout**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU KV Cache Memory Pool                      │
├─────────────────────────────────────────────────────────────────┤
│  Block 0    │  Block 1    │  Block 2    │  ...  │  Block N-1   │
│ ┌─────────┐ │ ┌─────────┐ │ ┌─────────┐ │       │ ┌─────────┐  │
│ │ K[0:16] │ │ │ K[0:16] │ │ │ K[0:16] │ │       │ │ K[0:16] │  │
│ │ V[0:16] │ │ │ V[0:16] │ │ │ V[0:16] │ │       │ │ V[0:16] │  │
│ └─────────┘ │ └─────────┘ │ └─────────┘ │ │ └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Page Table Example**:
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

### 4. GPU Executor

Abstracts GPU computation (currently mock implementation):

**Interface**:
```rust
pub trait GPUExecutorTrait {
    fn execute(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
    fn capture_decode_graph(&mut self, batch_size: u32);
    fn execute_graph(&mut self, batch: &ExecutionBatch) -> ExecutionOutput;
}
```

**Features** (planned):
- PagedAttention kernel
- CUDA Graph capture for decode phase
- FP16/BF16 computation with FP32 accumulation
- Fused operations for memory bandwidth optimization

### 5. Tokenizer

Trait-based tokenization interface:

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

## Inference Flow

```
┌─────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│ Request │──▶│ Tokenize  │──▶│ Schedule  │──▶│  Execute  │──▶│  Decode  │
│  Input  │   │   (CPU)   │   │   (CPU)   │   │   (GPU)   │   │  (CPU)   │
└─────────┘   └───────────┘   └───────────┘   └───────────┘   └──────────┘
                                  │               │
                                  │    ┌──────────┘
                                  ▼    ▼
                            ┌───────────────┐
                            │ KV Cache Mgr  │
                            │    (CPU)      │
                            └───────────────┘
```

**Detailed Steps**:

1. **Request Submission**
   - Client submits text input with generation parameters
   - Tokenizer encodes text to token IDs
   - Request queued in scheduler's pending queue

2. **Scheduling**
   - Scheduler selects requests for next batch
   - Decode requests prioritized over prefill
   - Batch size and token limits enforced
   - Memory pressure checks performed

3. **KV Cache Allocation**
   - New sequences allocate logical blocks
   - Physical blocks mapped via page table
   - Block allocations tracked for memory stats

4. **GPU Execution**
   - Batch data prepared (input IDs, positions, block tables)
   - GPU kernel executed (prefill or decode)
   - Output tokens generated

5. **State Updates**
   - Sequence states updated
   - Completed sequences removed
   - New tokens decoded to text

## State Machine

```
                    ┌─────────────┐
                    │   Pending   │  (Waiting for scheduling)
                    └──────┬──────┘
                           │ schedule()
                           ▼
                    ┌─────────────┐
            ┌───────│   Prefill   │  (Processing input tokens)
            │       └──────┬──────┘
            │              │ Prefill complete
            │              ▼
            │       ┌─────────────┐
            │       │   Decode    │◄────┐ (Generating tokens)
            │       └──────┬──────┘     │
            │              │            │ Generate next token
            │              ├────────────┘
            │              │ EOS or max_tokens
            │              ▼
            │       ┌─────────────┐
            └──────▶│  Completed  │  (Done)
                    └─────────────┘
```

**State Transitions**:

| From | To | Trigger |
|------|-----|---------|
| Pending | Prefill | Scheduled for first execution |
| Prefill | Decode | All input tokens processed |
| Decode | Decode | Token generated, not finished |
| Decode | Completed | EOS token or max_tokens reached |
| Any | Failed | Error during execution |

## Data Structures

### Request

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

### Sequence

```rust
pub struct Sequence {
    pub seq_id: u64,
    pub request: Request,
    pub logical_blocks: Vec<LogicalBlock>,
    pub num_computed_tokens: u32,
}
```

### ExecutionBatch

```rust
pub struct ExecutionBatch {
    pub input_tokens: Vec<u32>,
    pub positions: Vec<u32>,
    pub seq_lens: Vec<u32>,
    pub block_tables: Vec<Vec<u32>>,
    pub is_prefill: Vec<bool>,
}
```

## Design Properties

The architecture maintains these correctness properties:

1. **Request ID Uniqueness** - All sequence IDs are unique
2. **Batch Constraints** - Sequence count ≤ max_batch_size, tokens ≤ max_total_tokens
3. **Decode Priority** - Decode requests scheduled before prefill when capacity allows
4. **Block Count Invariant** - used_blocks + free_blocks == total_blocks
5. **Memory Pressure Response** - New prefill requests rejected when memory threshold exceeded

## Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| Scheduler | ✅ Complete | Full continuous batching with decode priority |
| KV Cache Manager | ✅ Complete | PagedAttention with block pool |
| Inference Engine | ✅ Complete | Orchestration with error handling |
| Tokenizer | ✅ Complete | Simple character-level implementation |
| GPU Executor | ⚠️ Mock | Interface defined, mock implementation |
| CUDA Kernels | ❌ Planned | Real kernel implementation pending |

---

*For API details, see [API.md](./API.md). For configuration options, see [CONFIGURATION.md](./CONFIGURATION.md).*
