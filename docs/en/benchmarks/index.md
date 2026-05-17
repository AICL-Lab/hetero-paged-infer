# Benchmarks

This section is a proof system for performance claims. It tells readers what the repository measures today, what is only simulated through the mock executor, and what is inherited from external papers or reference projects.

::: warning Claim discipline
This page separates three categories:
1. Measured in this repository
2. Derived from mock-executor simulation
3. Inherited from external literature or reference projects
:::

## Evidence ladder

| Category | What it means here | Current examples |
|----------|--------------------|------------------|
| Measured in this repository | Reproducible output from `cargo bench` or runtime/integration tests | Engine creation, scheduler steps, batch-processing loops, KV-cache allocation/freeing |
| Derived from mock-executor simulation | Behavior observed while the mock executor stands in for real GPU kernels | Decode-first scheduling, mixed prefill/decode batch construction, graph-capture API shape |
| Inherited from literature or reference projects | Claims borrowed from papers or mature systems and explicitly labeled as such | PagedAttention memory-fragmentation reductions, production CUDA-kernel throughput expectations |

## What is proven today

- The repository contains Criterion benchmarks in `benches/engine_benchmark.rs`.
- Integration tests in `tests/integration_tests.rs` exercise request completion, memory tracking, and continuous batching.
- The default engine still runs on `MockGPUExecutor`, so these pages do **not** claim production GPU throughput or latency numbers yet.

## How to read the benchmark pages

1. Start with [Methodology](/en/benchmarks/methodology) to understand acceptable evidence.
2. Read each benchmark page by category instead of treating every number as equally strong.
3. Follow links into [Comparison](/en/comparison/) and [References](/en/references/) when a claim depends on prior art rather than repository measurements.

## Benchmark map

- [Methodology](/en/benchmarks/methodology) — Sources of truth and reproduction rules
- [Memory Efficiency](/en/benchmarks/memory-efficiency) — What memory behavior is measured versus inferred
- [Throughput](/en/benchmarks/throughput) — What scheduler throughput evidence exists today
- [Latency](/en/benchmarks/latency) — What latency claims are currently justified
