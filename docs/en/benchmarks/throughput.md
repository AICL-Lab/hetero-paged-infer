# Throughput

This page explains what throughput evidence exists today and where readers should treat statements as simulations or inherited expectations.

::: warning Claim discipline
This page separates three categories:
1. Measured in this repository
2. Derived from mock-executor simulation
3. Inherited from external literature or reference projects
:::

## Measured in this repository

### What is actually benchmarked

- `cargo bench` measures `engine_step`, `batch_processing/{1,4,8,16,32}`, and scheduler operations in `benches/engine_benchmark.rs`.
- Those benchmarks run through repository code paths and are useful for comparing scheduling and orchestration overhead inside this codebase.

### What measured throughput does **not** mean yet

- These benchmarks do not represent production GPU tokens/sec.
- The default engine uses `MockGPUExecutor`, which validates batch shape and emits deterministic tokens instead of running CUDA kernels.
- Any absolute tokens/sec figure on this page would therefore overstate what the repository has proved.

## Derived from mock-executor simulation

- `tests/integration_tests.rs` includes a continuous-batching test that submits a second request while the first is already decoding.
- The scheduler implementation favors decode work first and fills remaining capacity with prefill work, so mixed-batch behavior is observable in simulation.
- This supports the claim that the scheduler is organized for utilization-aware batching, but it is still a simulation result until real kernels are attached.

## Inherited from external literature or reference projects

- The broader claim that continuous batching improves throughput versus static batching comes from systems such as Orca and vLLM.
- Those papers justify *why* the design is promising, but they do not automatically establish the exact uplift of this repository.

## What readers should take away

| Statement | Category | Safe reading |
|-----------|----------|--------------|
| Batch-processing code paths scale across multiple configured batch sizes | Measured in this repository | The software stack is benchmarked across several batch sizes |
| Mixed prefill/decode scheduling should improve utilization over naive static execution | Derived from mock-executor simulation | The scheduler shape is promising, but hardware proof is pending |
| Large production throughput gains are possible with continuous batching | Inherited from literature | Treat as prior art, not as a completed local benchmark |

## Related

- [Benchmark Methodology](/en/benchmarks/methodology)
- [Latency](/en/benchmarks/latency)
- [Continuous Batching Architecture](/en/architecture/continuous-batching)
- [Projects Reading Guide](/en/references/projects)
