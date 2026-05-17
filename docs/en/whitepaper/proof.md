# Proof and Limits

## What is already implemented

The current repository already implements the control-plane shape of an inference engine. There is a concrete `InferenceEngine`, a `Scheduler` that separates pending/prefill/decode work, a `KVCacheManager` with paged block allocation, a batch execution pipeline, configuration validation, and integration tests that exercise mixed request flows and memory-pressure handling.

The codebase also exposes future production seams explicitly instead of hiding them. `GPUExecutorTrait` defines the executor boundary, decode-graph capture is modeled in the interface, and the engine is structured so the mock executor can be replaced without rewriting the scheduler or KV cache story.

## What is mocked or simulated

The executor is currently a `MockGPUExecutor`, not a real CUDA backend. It simulates execution, validates batch constraints, and generates deterministic tokens for tests. CUDA graph capture is represented as interface behavior and mock state, not as actual captured GPU work.

This means any claim that depends on kernel quality, memory bandwidth, fused attention kernels, or GPU occupancy must be treated as future work. The architecture is real; the production GPU path is not complete yet.

## Which claims are measured

The strongest measured claims in the current project are correctness and structural behavior claims:

- requests can move through pending → prefill → decode → completed,
- continuous batching accepts new work while earlier requests are already decoding,
- memory pressure gates admission and the engine keeps running instead of crashing,
- KV cache resources are allocated and released through explicit block accounting,
- configuration and request validation reject invalid inputs early.

These are supported by the existing unit and integration tests in the Rust codebase, plus the successful documentation build that wires the whitepaper and architecture pages into the site.

## Which claims are estimated or inherited from reference literature

Claims about the *general* advantages of paged KV caches and continuous batching come from established reference systems and papers, not from this repository alone. For example, lower fragmentation from page-based KV allocation and better accelerator utilization from dynamic batching are standard arguments in the serving literature.

Hetero-Paged-Infer inherits those design motivations, but readers should not confuse inherited architectural rationale with repository-specific benchmark proof. Until a real CUDA backend and measurement harness are in place, the project should present those benefits as well-supported expectations rather than as final, locally proven performance numbers.

## What remains to be built before production credibility

Several milestones still separate the current codebase from production-grade credibility:

1. a real CUDA executor with actual attention and decode kernels,
2. end-to-end benchmarking on representative hardware and models,
3. serving/runtime hardening beyond the current architectural skeleton,
4. stronger observability around latency, memory, and failure behavior,
5. proof that the documented scheduling and memory decisions still hold under real GPU execution.

In short: the project already proves that the architecture is shaped seriously, but it does not yet prove that the GPU implementation is production-ready. That distinction is intentional and should remain explicit throughout the whitepaper.
