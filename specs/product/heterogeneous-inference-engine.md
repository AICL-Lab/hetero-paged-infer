# Product Requirements: Heterogeneous Inference Engine

## Introduction

This document defines the product requirements for the Heterogeneous Inference Engine—a high-performance LLM inference microservice leveraging CPU-GPU co-execution. The system implements PagedAttention for efficient KV Cache management and Continuous Batching for optimal throughput.

## Glossary

| Term | Description |
|------|-------------|
| **Inference Engine** | Core system orchestrating CPU and GPU resources for LLM inference |
| **Scheduler** | CPU component responsible for request scheduling and batch organization |
| **KV Cache Manager** | CPU component managing KV Cache memory via paged allocation |
| **GPU Executor** | Component executing CUDA/Triton kernels for attention and matrix operations |
| **Tokenizer** | CPU component for text-to-token and token-to-text conversion |
| **Request** | Individual inference request with input tokens and generation parameters |
| **Batch** | Collection of requests processed together on the GPU |
| **Prefill Phase** | Initial phase processing all input tokens to populate KV cache |
| **Decode Phase** | Autoregressive generation phase producing one token per step |
| **Physical Block** | Contiguous region in GPU memory storing KV cache for a fixed number of tokens |
| **Logical Block** | Virtual block mapped to physical blocks via page table |
| **Page Table** | Per-sequence mapping structure from logical blocks to physical blocks |

## Requirements

---

### REQ-1: Request Management

**User Story:** As a client, I want to submit inference requests to the system and receive generated text responses.

#### Acceptance Criteria

1. WHEN a client submits a request with input text and generation parameters, THEN the inference engine shall tokenize the input and create a pending request entry
2. WHEN a request is created, THEN the scheduler SHALL assign a unique sequence ID and initialize request state
3. WHEN generation parameters include max_tokens, temperature, and top_p, THEN the inference engine SHALL validate parameters are within acceptable ranges
4. IF a request contains invalid parameters, THEN the inference engine SHALL return a descriptive error without processing the request
5. WHEN request completes generation, THEN the inference engine SHALL decode the output and return the response to the client

---

### REQ-2: PagedAttention KV Cache Management

**User Story:** As a system operator, I want efficient GPU memory utilization to serve more concurrent requests.

#### Acceptance Criteria

1. The KV Cache Manager SHALL partition GPU memory into fixed-size physical blocks (e.g., 16 tokens per block)
2. WHEN a new sequence starts, THEN the KV Cache Manager SHALL allocate logical blocks and map them to available physical blocks via page table
3. WHEN a sequence generates tokens exceeding current block capacity, THEN the KV Cache Manager SHALL allocate additional physical blocks on demand
4. WHEN a sequence completes, THEN the KV Cache Manager SHALL release all physical blocks back to the free pool
5. The KV Cache Manager SHALL maintain a free block list and track block usage per sequence
6. WHEN no physical blocks are available, THEN the KV Cache Manager SHALL signal memory pressure to the scheduler
7. For all sequences, the KV Cache Manager SHALL provide O(1) lookup from logical block index to physical block pointer

---

### REQ-3: Continuous Batching Scheduler

**User Story:** As a system operator, I want to maximize GPU utilization through continuous batching for high throughput.

#### Acceptance Criteria

1. The scheduler SHALL maintain independent queues for prefill and decode requests
2. WHEN scheduling a batch, THEN the scheduler SHALL combine prefill and decode requests into a single GPU execution
3. WHEN prefill requests complete, THEN the scheduler SHALL immediately transition them to decode phase without waiting for batch completion
4. WHEN a decode request generates an EOS token or reaches max_tokens, THEN the scheduler SHALL mark it complete and remove from active set
5. The scheduler SHALL enforce maximum batch size and maximum total token constraints
6. WHEN a new request arrives, THEN the scheduler SHALL insert it into the next available batch slot (continuous insertion)
7. The scheduler SHALL prioritize decode requests over prefill requests to minimize latency of in-progress requests

---

### REQ-4: GPU Kernel Execution

**User Story:** As a developer, I want optimized GPU kernels for attention computation so inference is fast.

#### Acceptance Criteria

1. The GPU Executor SHALL implement paged attention kernel reading KV cache indirectly via block table
2. WHEN executing attention, THEN the GPU Executor SHALL support variable sequence lengths within a batch
3. The GPU Executor SHALL implement fused operations to minimize GPU memory bandwidth
4. WHEN a batch contains mixed prefill and decode requests, THEN the GPU Executor SHALL handle different attention modes appropriately
5. The GPU Executor SHALL use CUDA Graphs to reduce kernel launch overhead in decode phase
6. The GPU Executor SHALL support FP16/BF16 computation with FP32 accumulation for numerical stability

---

### REQ-5: CPU-GPU Pipeline Coordination

**User Story:** As a system architect, I want efficient CPU-GPU coordination so neither becomes a bottleneck.

#### Acceptance Criteria

1. The inference engine SHALL use asynchronous CUDA streams to overlap CPU and GPU work
2. WHEN CPU prepares the next batch, THEN the GPU Executor SHALL concurrently execute the current batch
3. The inference engine SHALL use pinned (page-locked) host memory for CPU-GPU data transfer
4. WHEN transmitting batch metadata, THEN the inference engine SHALL send only block table updates to minimize transfer size
5. The inference engine SHALL implement double-buffered batch preparation to hide latency
6. IF GPU execution stalls, THEN the inference engine SHALL log a warning and continue processing

---

### REQ-6: Memory Pool Management

**User Story:** As a system operator, I want predictable memory usage so the system remains stable under load.

#### Acceptance Criteria

1. ON system startup, the KV Cache Manager SHALL preallocate a configurable percentage of GPU memory for KV cache blocks
2. The KV Cache Manager SHALL track memory statistics including total blocks, used blocks, and fragmentation rate
3. WHEN memory utilization exceeds threshold, THEN the scheduler SHALL stop accepting new prefill requests
4. The inference engine SHALL expose memory usage metrics for monitoring
5. IF memory allocation fails, THEN the inference engine SHALL gracefully reject new requests rather than crash

---

### REQ-7: Configuration & Initialization

**User Story:** As a deployer, I want configurable system parameters to tune for different hardware and workloads.

#### Acceptance Criteria

1. The inference engine SHALL accept configuration for: block_size, max_num_blocks, max_batch_size, max_num_seqs
2. WHEN loading configuration, THEN the inference engine SHALL validate all parameters and report errors
3. The inference engine SHALL support configuration via file or command-line arguments
4. ON initialization, the inference engine SHALL log system configuration and detected GPU capabilities
5. The inference engine SHALL detect and report available GPU memory and compute capability

---

### REQ-8: Tokenization

**User Story:** As a client, I want accurate text tokenization so inputs are processed correctly.

#### Acceptance Criteria

1. The tokenizer SHALL encode input text to token IDs using a configurable vocabulary
2. The tokenizer SHALL accurately decode token IDs back to text
3. WHEN encoding, the tokenizer SHALL handle special tokens (BOS, EOS, PAD) correctly
4. For all valid text input, decode(encode(text)) SHALL produce equivalent text (round-trip property)
5. The tokenizer SHALL run on CPU to avoid GPU memory overhead

---

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-1 | ✅ Complete | Request management implemented |
| REQ-2 | ✅ Complete | KV Cache management implemented |
| REQ-3 | ✅ Complete | Scheduler implemented |
| REQ-4 | ⚠️ Mock | GPU kernel is mock implementation |
| REQ-5 | ⚠️ Mock | Async overlap not implemented |
| REQ-6 | ✅ Complete | Memory pool management implemented |
| REQ-7 | ✅ Complete | Configuration system implemented |
| REQ-8 | ✅ Complete | Tokenizer implemented (simple implementation) |

---

## Test Coverage Summary

| Test Type | Count | Description |
|-----------|-------|-------------|
| Unit Tests | 78 | Each module tested independently |
| Property Tests | 15 | Invariant verification using proptest |
| Integration Tests | 13 | End-to-end flow validation |
| Doc Tests | 29 | API example validation |
