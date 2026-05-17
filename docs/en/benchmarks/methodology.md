# Benchmark Methodology

This page defines the proof rules for benchmark claims in the documentation.

::: warning Claim discipline
This page separates three categories:
1. Measured in this repository
2. Derived from mock-executor simulation
3. Inherited from external literature or reference projects
:::

## Sources of truth

1. `cargo bench` output from `benches/engine_benchmark.rs`
2. Engine/runtime tests in `tests/integration_tests.rs`
3. Architecture-derived estimates that are explicitly labeled as estimates

## How claims are classified

### 1. Measured in this repository

Use this label only when a reader can rerun repository benchmarks or tests and observe the same class of result.

### 2. Derived from mock-executor simulation

Use this label when behavior depends on `MockGPUExecutor`, scheduler simulation, or architecture walkthroughs rather than real CUDA execution.

### 3. Inherited from external literature or reference projects

Use this label when a statement comes from papers such as PagedAttention, Orca, FlashAttention, or from mature systems such as vLLM and TensorRT-LLM.

## Reproduction path

```bash
cargo bench
cargo test --test integration_tests
```

Run those commands from the repository root. If a benchmark page cannot be tied back to one of these commands or to a clearly labeled estimate, it should not be presented as proof.

## What is intentionally not claimed yet

- No real CUDA-kernel throughput tables
- No production TTFT/ITL report
- No apples-to-apples cross-engine shootout executed in this repository

## Reading order

- [Benchmarks overview](/en/benchmarks/)
- [Memory Efficiency](/en/benchmarks/memory-efficiency)
- [Throughput](/en/benchmarks/throughput)
- [Latency](/en/benchmarks/latency)
