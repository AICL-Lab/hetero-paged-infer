# Requirements Document

## Introduction

This document defines the requirements for a Heterogeneous Inference Microservice - a high-performance inference engine that leverages CPU-GPU co-execution for Large Language Model (LLM) inference. The system implements PagedAttention for efficient KV Cache management and Continuous Batching for optimal throughput.

## Glossary

- **Inference_Engine**: The core system that orchestrates CPU and GPU resources for LLM inference
- **Scheduler**: CPU component responsible for request scheduling and batch organization
- **KV_Cache_Manager**: CPU component managing Key-Value cache memory using paged allocation
- **GPU_Executor**: GPU component executing CUDA/Triton kernels for attention and matrix operations
- **Tokenizer**: CPU component for text-to-token and token-to-text conversion
- **Request**: A single inference request containing input tokens and generation parameters
- **Batch**: A collection of requests processed together on GPU
- **Prefill_Phase**: Initial phase where all input tokens are processed to populate KV cache
- **Decode_Phase**: Autoregressive generation phase producing one token per step
- **Physical_Block**: A contiguous GPU memory region holding KV cache for fixed number of tokens
- **Logical_Block**: Virtual block mapped to physical blocks via page table
- **Page_Table**: Mapping structure from logical blocks to physical blocks per sequence

## Requirements

### Requirement 1: Request Management

**User Story:** As a client, I want to submit inference requests to the system, so that I can get generated text responses.

#### Acceptance Criteria

1. WHEN a client submits a request with input text and generation parameters THEN THE Inference_Engine SHALL tokenize the input and create a pending request entry
2. WHEN a request is created THEN THE Scheduler SHALL assign a unique sequence ID and initialize request state
3. WHEN generation parameters include max_tokens, temperature, and top_p THEN THE Inference_Engine SHALL validate parameters are within acceptable ranges
4. IF a request contains invalid parameters THEN THE Inference_Engine SHALL return a descriptive error without processing
5. WHEN a request completes generation THEN THE Inference_Engine SHALL detokenize output and return the response to client

### Requirement 2: KV Cache Management with PagedAttention

**User Story:** As a system operator, I want efficient GPU memory utilization for KV cache, so that I can serve more concurrent requests.

#### Acceptance Criteria

1. THE KV_Cache_Manager SHALL divide GPU memory into fixed-size physical blocks (e.g., 16 tokens per block)
2. WHEN a new sequence starts THEN THE KV_Cache_Manager SHALL allocate logical blocks and map them to available physical blocks via page table
3. WHEN a sequence generates tokens beyond current block capacity THEN THE KV_Cache_Manager SHALL allocate additional physical blocks on demand
4. WHEN a sequence completes THEN THE KV_Cache_Manager SHALL release all physical blocks back to free pool
5. THE KV_Cache_Manager SHALL maintain a free block list and track block usage per sequence
6. WHEN no free physical blocks are available THEN THE KV_Cache_Manager SHALL signal memory pressure to Scheduler
7. FOR ALL sequences, THE KV_Cache_Manager SHALL provide O(1) lookup from logical block index to physical block pointer

### Requirement 3: Continuous Batching Scheduler

**User Story:** As a system operator, I want to maximize GPU utilization through continuous batching, so that I can achieve high throughput.

#### Acceptance Criteria

1. THE Scheduler SHALL maintain separate queues for prefill requests and decode requests
2. WHEN scheduling a batch THEN THE Scheduler SHALL combine prefill and decode requests into a single GPU execution
3. WHEN a prefill request completes THEN THE Scheduler SHALL immediately transition it to decode phase without waiting for batch completion
4. WHEN a decode request generates EOS token or reaches max_tokens THEN THE Scheduler SHALL mark it complete and remove from active set
5. THE Scheduler SHALL respect maximum batch size and maximum total tokens constraints
6. WHEN new requests arrive THEN THE Scheduler SHALL insert them into next available batch slot (continuous insertion)
7. THE Scheduler SHALL prioritize decode requests over prefill requests to minimize latency for in-flight requests

### Requirement 4: GPU Kernel Execution

**User Story:** As a developer, I want optimized GPU kernels for attention computation, so that inference is fast.

#### Acceptance Criteria

1. THE GPU_Executor SHALL implement paged attention kernel that reads KV cache via block table indirection
2. WHEN executing attention THEN THE GPU_Executor SHALL support variable sequence lengths within a batch
3. THE GPU_Executor SHALL implement fused operations to minimize GPU memory bandwidth
4. WHEN a batch contains mixed prefill and decode requests THEN THE GPU_Executor SHALL handle different attention patterns appropriately
5. THE GPU_Executor SHALL use CUDA Graphs to reduce kernel launch overhead for decode phase
6. THE GPU_Executor SHALL support FP16/BF16 computation with FP32 accumulation for numerical stability

### Requirement 5: CPU-GPU Pipeline Coordination

**User Story:** As a system architect, I want efficient CPU-GPU coordination, so that neither processor becomes a bottleneck.

#### Acceptance Criteria

1. THE Inference_Engine SHALL use asynchronous CUDA streams for overlapping CPU and GPU work
2. WHEN CPU prepares next batch THEN THE GPU_Executor SHALL be executing current batch concurrently
3. THE Inference_Engine SHALL use pinned (page-locked) host memory for CPU-GPU data transfers
4. WHEN transferring batch metadata THEN THE Inference_Engine SHALL minimize transfer size by sending only block table updates
5. THE Inference_Engine SHALL implement double buffering for batch preparation to hide latency
6. IF GPU execution stalls THEN THE Inference_Engine SHALL log warning and continue processing

### Requirement 6: Memory Pool Management

**User Story:** As a system operator, I want predictable memory usage, so that the system runs stably under load.

#### Acceptance Criteria

1. WHEN the system starts THEN THE KV_Cache_Manager SHALL pre-allocate a configurable percentage of GPU memory for KV cache blocks
2. THE KV_Cache_Manager SHALL track memory statistics including total blocks, used blocks, and fragmentation ratio
3. WHEN memory utilization exceeds threshold THEN THE Scheduler SHALL stop accepting new prefill requests
4. THE Inference_Engine SHALL provide memory usage metrics for monitoring
5. IF memory allocation fails THEN THE Inference_Engine SHALL gracefully reject new requests rather than crash

### Requirement 7: Configuration and Initialization

**User Story:** As a deployer, I want configurable system parameters, so that I can tune for different hardware and workloads.

#### Acceptance Criteria

1. THE Inference_Engine SHALL accept configuration for: block_size, max_num_blocks, max_batch_size, max_num_seqs
2. WHEN configuration is loaded THEN THE Inference_Engine SHALL validate all parameters and report errors
3. THE Inference_Engine SHALL support configuration via file or command-line arguments
4. WHEN initialized THEN THE Inference_Engine SHALL log system configuration and detected GPU capabilities
5. THE Inference_Engine SHALL detect and report available GPU memory and compute capability

### Requirement 8: Tokenization

**User Story:** As a client, I want accurate text tokenization, so that my inputs are correctly processed.

#### Acceptance Criteria

1. THE Tokenizer SHALL encode input text to token IDs using a configurable vocabulary
2. THE Tokenizer SHALL decode token IDs back to text accurately
3. WHEN encoding THEN THE Tokenizer SHALL handle special tokens (BOS, EOS, PAD) correctly
4. FOR ALL valid text inputs, encoding then decoding SHALL produce equivalent text (round-trip property)
5. THE Tokenizer SHALL operate on CPU to avoid GPU memory overhead
