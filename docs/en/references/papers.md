# Papers

This page is a reading guide rather than a raw bibliography. Each entry explains how a paper maps onto this repository.

## Efficient Memory Management for Large Language Model Serving with PagedAttention

**Authors:** Woosuk Kwon et al.
**Venue:** SOSP 2023
**Links:** [Paper](https://arxiv.org/abs/2309.06180) · [Reference implementation](https://github.com/vllm-project/vllm)

- **Why it matters to this repo:** It provides the vocabulary and systems argument for block-based KV-cache allocation.
- **Which subsystem it influenced:** `KVCacheManager`, memory-management docs, and the benchmark framing for memory efficiency.
- **What was adopted vs deliberately not adopted:** Adopted the page/block mental model, on-demand growth, and sharing-oriented reasoning; deliberately not adopted the paper's production-performance numbers as local benchmark claims.

## Orca: A Distributed Serving System for Transformer-Based Generative Language Models

**Authors:** Gyeongmin Yu et al.
**Venue:** OSDI 2022
**Links:** [Paper](https://arxiv.org/abs/2205.11000)

- **Why it matters to this repo:** Orca is the clearest background source for iteration-level scheduling and continuous batching.
- **Which subsystem it influenced:** `Scheduler`, mixed prefill/decode flow, and latency/throughput methodology language.
- **What was adopted vs deliberately not adopted:** Adopted the scheduling intuition and request-mixing model; deliberately not adopted Orca's distributed serving scope as a claim that this repository already matches.

## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors:** Tri Dao et al.
**Venue:** NeurIPS 2022
**Links:** [Paper](https://arxiv.org/abs/2205.14135) · [Code](https://github.com/Dao-AILab/flash-attention)

- **Why it matters to this repo:** It explains why attention kernels should be discussed in terms of memory traffic, not just FLOPs.
- **Which subsystem it influenced:** Future GPU-executor direction and the way the docs discuss attention-kernel bottlenecks.
- **What was adopted vs deliberately not adopted:** Adopted the IO-aware optimization target as a design reference; deliberately not adopted the paper's speedup numbers as if they had already been reproduced here.

## FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**Authors:** Tri Dao
**Venue:** ICLR 2024
**Links:** [Paper](https://arxiv.org/abs/2307.08691) · [Code](https://github.com/Dao-AILab/flash-attention)

- **Why it matters to this repo:** It sharpens the performance target for any future real GPU backend.
- **Which subsystem it influenced:** Executor interface planning and comparison language around production gaps.
- **What was adopted vs deliberately not adopted:** Adopted the idea that work partitioning matters at the kernel layer; deliberately not adopted any claim that this repository already implements FlashAttention-class kernels.

## How to use this page

Read these papers together with the [benchmark methodology](/en/benchmarks/methodology). If a documentation claim depends on one of these papers, it should be labeled as inherited rather than measured.
