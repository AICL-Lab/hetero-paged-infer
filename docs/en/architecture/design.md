# Design Principles

## Problem

A modern inference engine has to coordinate request lifecycle, memory management, and execution boundaries at the same time. If those concerns are mixed together, the code quickly becomes hard to reason about: schedulers start owning memory policy, executors leak serving assumptions, and it becomes difficult to tell which claims are about real implementation versus future optimization.

Hetero-Paged-Infer therefore treats architecture as the first problem to solve. The question is not only “how do we generate the next token,” but “how do we keep the engine understandable when paged KV cache management, continuous batching, and future GPU specialization all need to evolve together?”

## Design choice

The project uses a control-plane/data-plane split. The Rust engine owns request validation, tokenization, scheduling, KV cache bookkeeping, and batch assembly on the control side. Execution itself sits behind `GPUExecutorTrait`, so the engine can talk to a mock executor today and a CUDA backend later without rewriting the outer orchestration.

This choice makes the project production-shaped even before the final GPU path exists. `InferenceEngine` is the service-facing boundary, `Scheduler` owns admission and phase transitions, `KVCacheManager` owns paged memory metadata, and the execution pipeline is the seam where hardware-specific work plugs in.

## Trade-off

The benefit of this structure is clarity. Components have explicit responsibilities, tests can target scheduling and memory invariants independently, and future CUDA work has a defined place to land. It also keeps the documentation honest: the engine can describe current behavior precisely without pretending the mock executor is already a production kernel stack.

The cost is that abstraction alone does not create performance. A trait boundary and a clean engine loop are useful only if the eventual backend can honor the same assumptions under real hardware pressure. Until that happens, the architecture is stronger than the final throughput story—and this project deliberately states that fact instead of hiding it.

## Current implementation status

Today, the structural pieces are real: engine construction, request submission, scheduler state transitions, paged KV allocation, and test coverage for mixed execution scenarios are implemented in Rust. The executor boundary is also real, but the default backend is still `MockGPUExecutor`.

So the current design page should be read as an argument about system shape that is already encoded in the repository, not as proof that the CUDA production path is finished. For the decision-level deep dives, continue with [PagedAttention](./paged-attention) and [Continuous Batching](./continuous-batching).
