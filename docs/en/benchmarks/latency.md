# Latency

Latency is where claim discipline matters most: the repository has scheduler logic and tests, but not yet a production-grade wall-clock latency report.

::: warning Claim discipline
This page separates three categories:
1. Measured in this repository
2. Derived from mock-executor simulation
3. Inherited from external literature or reference projects
:::

## Measured in this repository

- Integration tests verify request completion, mixed prefill/decode execution, and memory-pressure-related control flow.
- The repository currently does **not** publish direct wall-clock TTFT or ITL benchmark tables from real GPU execution.
- `cargo bench` covers orchestration overhead, but those measurements are not the same as user-visible serving latency.

## Derived from mock-executor simulation

- The scheduler is decode-first, which is the repository's main latency-preserving policy for in-flight requests.
- Memory-threshold handling is designed to pause new prefills before the system reaches an OOM state.
- The GPU-executor trait exposes CUDA-graph capture hooks, but the current mock executor reuses the normal execution path, so no local latency reduction from graphs is proved yet.

## Inherited from external literature or reference projects

- Chunked prefill, CUDA graphs, and other low-latency serving techniques are supported by external systems literature.
- They explain why this architecture is shaped the way it is, but they should not be read as measured TTFT/ITL numbers for this repository.

## Current proof status

| Question | Current answer |
|----------|----------------|
| Does the repository expose latency-oriented scheduling policies? | Yes, in code and integration tests |
| Does the repository publish production TTFT/ITL measurements? | No |
| Can readers treat external low-latency numbers as local benchmark results? | No |

## What to evaluate this page for

- Whether the latency-control mechanisms are present and documented.
- Whether the repository clearly marks unmeasured GPU claims as future work.
- Whether the transition from mock execution to real kernels will have a clear place to plug in new measurements.

## Related

- [Benchmark Methodology](/en/benchmarks/methodology)
- [Throughput](/en/benchmarks/throughput)
- [Memory Management](/en/architecture/memory-management)
- [Papers Reading Guide](/en/references/papers)
