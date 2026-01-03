# Implementation Plan: Heterogeneous Inference System

## Overview

This plan implements a heterogeneous inference microservice in Rust with CPU-GPU co-execution. The implementation follows a bottom-up approach: core data structures → KV Cache Manager → Scheduler → GPU Executor → Inference Engine integration.

## Tasks

- [x] 1. Project setup and core data structures
  - [x] 1.1 Initialize Rust project with Cargo
    - Create `Cargo.toml` with dependencies: `cuda-runtime-sys`, `proptest`, `thiserror`, `tokio`
    - Set up project structure: `src/lib.rs`, `src/main.rs`
    - _Requirements: 7.1, 7.3_

  - [x] 1.2 Implement core types and enums
    - Implement `RequestState` enum (Pending, Prefill, Decode, Completed, Failed)
    - Implement `Request` struct with id, tokens, parameters, state
    - Implement `GenerationParams` struct with max_tokens, temperature, top_p
    - Implement `Sequence` struct with seq_id, request, logical_blocks
    - _Requirements: 1.1, 1.2_

  - [x] 1.3 Implement configuration types
    - Implement `EngineConfig` struct with block_size, max_num_blocks, max_batch_size, max_num_seqs
    - Implement configuration validation logic
    - Implement config loading from file and CLI args
    - _Requirements: 7.1, 7.2_

  - [x] 1.4 Write property test for configuration validation
    - **Property 14: Configuration Validation**
    - **Validates: Requirements 7.2**

  - [x] 1.5 Write property test for parameter validation
    - **Property 2: Parameter Validation Correctness**
    - **Validates: Requirements 1.3**

- [x] 2. Checkpoint - Core types complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 3. KV Cache Manager implementation
  - [x] 3.1 Implement PhysicalBlock and memory pool
    - Implement `PhysicalBlock` struct representing GPU memory region
    - Implement `BlockPool` with free list management
    - Implement block allocation and deallocation
    - _Requirements: 2.1, 2.5_

  - [x] 3.2 Implement LogicalBlock and page table
    - Implement `LogicalBlock` struct with physical block mapping
    - Implement `PageTable` for logical-to-physical mapping
    - Implement O(1) lookup from logical index to physical pointer
    - _Requirements: 2.2, 2.7_

  - [x] 3.3 Implement KVCacheManager trait and struct
    - Implement `allocate_sequence()` for new sequences
    - Implement `allocate_block()` for sequence growth
    - Implement `free_sequence()` for cleanup
    - Implement `get_block_table()` for GPU execution
    - Implement `get_memory_stats()` for monitoring
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 3.4 Write property test for block count invariant
    - **Property 5: Block Count Invariant**
    - **Validates: Requirements 2.4, 2.5**

  - [x] 3.5 Write property test for block allocation on sequence start
    - **Property 3: Block Allocation on Sequence Start**
    - **Validates: Requirements 2.2**

  - [x] 3.6 Write property test for block allocation on growth
    - **Property 4: Block Allocation on Growth**
    - **Validates: Requirements 2.3**

  - [x] 3.7 Write property test for memory statistics invariant
    - **Property 12: Memory Statistics Invariant**
    - **Validates: Requirements 6.2**

- [x] 4. Checkpoint - KV Cache Manager complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Scheduler implementation
  - [x] 5.1 Implement request queues
    - Implement pending queue for new requests
    - Implement prefill queue for sequences in prefill phase
    - Implement decode queue for sequences in decode phase
    - Implement completed list for finished requests
    - _Requirements: 3.1_

  - [x] 5.2 Implement Scheduler trait and core scheduling logic
    - Implement `add_request()` to queue new requests
    - Implement `schedule()` to create batches mixing prefill and decode
    - Implement decode priority over prefill
    - Implement batch size and token constraints
    - _Requirements: 3.2, 3.5, 3.6, 3.7_

  - [x] 5.3 Implement sequence state transitions
    - Implement prefill-to-decode transition after prefill completes
    - Implement completion detection (EOS token or max_tokens)
    - Implement `update_sequences()` after GPU execution
    - Implement `get_completed()` to retrieve finished requests
    - _Requirements: 3.3, 3.4_

  - [x] 5.4 Implement memory pressure handling
    - Integrate with KVCacheManager for memory status
    - Stop accepting new prefills when memory exceeds threshold
    - Resume accepting when memory freed
    - _Requirements: 6.3_

  - [x] 5.5 Write property test for request ID uniqueness
    - **Property 1: Request ID Uniqueness**
    - **Validates: Requirements 1.2**

  - [x] 5.6 Write property test for scheduler queue state consistency
    - **Property 6: Scheduler Queue State Consistency**
    - **Validates: Requirements 3.1**

  - [x] 5.7 Write property test for batch size constraints
    - **Property 7: Batch Size Constraints**
    - **Validates: Requirements 3.5**

  - [x] 5.8 Write property test for decode priority
    - **Property 8: Decode Priority Over Prefill**
    - **Validates: Requirements 3.7**

  - [x] 5.9 Write property test for prefill to decode transition
    - **Property 9: Prefill to Decode Transition**
    - **Validates: Requirements 3.3**

  - [x] 5.10 Write property test for completion conditions
    - **Property 10: Completion Conditions**
    - **Validates: Requirements 3.4**

  - [x] 5.11 Write property test for memory pressure response
    - **Property 13: Memory Pressure Response**
    - **Validates: Requirements 6.3**

- [x] 6. Checkpoint - Scheduler complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Tokenizer implementation
  - [x] 7.1 Implement Tokenizer trait and basic implementation
    - Implement `encode()` for text to token IDs
    - Implement `decode()` for token IDs to text
    - Implement special token handling (BOS, EOS, PAD)
    - Use simple vocabulary for testing (can integrate real tokenizer later)
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 7.2 Write property test for tokenizer round-trip
    - **Property 15: Tokenizer Round-Trip**
    - **Validates: Requirements 8.4**

- [x] 8. Checkpoint - Tokenizer complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. GPU Executor implementation
  - [x] 9.1 Implement GPU memory management
    - Implement pinned host memory buffers for CPU-GPU transfer
    - Implement GPU memory allocation for KV cache blocks
    - Implement async CUDA stream management
    - _Requirements: 5.1, 5.3_

  - [x] 9.2 Implement ExecutionBatch and GPUBatchData
    - Implement `ExecutionBatch` struct for batch metadata
    - Implement `GPUBatchData` with pinned buffers
    - Implement batch data preparation from scheduler output
    - _Requirements: 4.2_

  - [x] 9.3 Implement paged attention kernel interface
    - Define CUDA kernel interface for paged attention
    - Implement block table indirection for KV cache access
    - Support variable sequence lengths in batch
    - Handle mixed prefill/decode attention patterns
    - _Requirements: 4.1, 4.2, 4.4_

  - [x] 9.4 Implement GPUExecutor trait and struct
    - Implement `execute()` for batch execution
    - Implement `capture_decode_graph()` for CUDA graph capture
    - Implement `execute_graph()` for graph-based execution
    - _Requirements: 4.5_

  - [x] 9.5 Write property test for variable sequence length handling
    - **Property 11: Variable Sequence Length Handling**
    - **Validates: Requirements 4.2**

- [x] 10. Checkpoint - GPU Executor complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Inference Engine integration
  - [x] 11.1 Implement InferenceEngine struct
    - Wire together Tokenizer, Scheduler, KVCacheManager, GPUExecutor
    - Implement `submit_request()` for request submission
    - Implement `step()` for single iteration
    - _Requirements: 1.1, 1.5_

  - [x] 11.2 Implement main inference loop
    - Implement `run()` main loop
    - Implement double buffering for batch preparation
    - Implement async CPU-GPU overlap
    - _Requirements: 5.2, 5.5_

  - [x] 11.3 Implement error handling and recovery
    - Implement error types (MemoryError, ExecutionError, etc.)
    - Implement recovery strategies (retry, skip, reset)
    - Implement graceful degradation under memory pressure
    - _Requirements: 1.4, 2.6, 6.5_

  - [x] 11.4 Implement monitoring and metrics
    - Implement memory usage metrics
    - Implement throughput metrics
    - Implement logging for configuration and GPU capabilities
    - _Requirements: 6.4, 7.4, 7.5_

- [x] 12. Checkpoint - Integration complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Integration tests
  - [x] 13.1 Write end-to-end request flow test
    - Submit request, run until completion
    - Verify output tokens generated
    - Verify KV cache freed after completion
    - _Requirements: 1.1, 1.5, 2.4_

  - [x] 13.2 Write continuous batching integration test
    - Submit multiple requests with staggered timing
    - Verify all complete correctly
    - Verify mixed prefill/decode batches formed
    - _Requirements: 3.2, 3.3, 3.4_

  - [x] 13.3 Write memory pressure integration test
    - Fill memory to threshold
    - Verify new prefills rejected
    - Complete some requests
    - Verify new prefills accepted again
    - _Requirements: 6.3, 6.5_

- [x] 14. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks including tests are required for comprehensive coverage
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- GPU kernel implementation (9.3) may require CUDA toolkit installation
- For initial development, GPU executor can be mocked to test CPU components
