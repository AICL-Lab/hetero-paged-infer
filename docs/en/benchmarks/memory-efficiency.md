# Memory Efficiency

This page focuses on what the repository can currently prove about KV-cache behavior, not on borrowed headline numbers alone.

::: warning Claim discipline
This page separates three categories:
1. Measured in this repository
2. Derived from mock-executor simulation
3. Inherited from external literature or reference projects
:::

## Measured in this repository

### Verified by benchmarks and tests

- `cargo bench` measures `allocate_sequence` and `allocate_block` behavior in `benches/engine_benchmark.rs`.
- `tests/integration_tests.rs` checks that memory utilization rises after allocation and drops after request completion.
- End-to-end request-flow tests verify that KV-cache state is released once work finishes.

### What that proves

| Claim | Evidence type | Current proof |
|-------|---------------|---------------|
| KV-cache allocation and growth paths exist and are benchmarked | Measured in this repository | Criterion benchmarks cover allocation-heavy code paths |
| Memory utilization is observable from the engine | Measured in this repository | Integration tests assert utilization changes with load |
| Completed requests release their working state | Measured in this repository | End-to-end tests expect low post-run utilization |

## Derived from mock-executor simulation

- The repository's block-based allocator and page-table model imply that waste is bounded by the last partially filled block of each live sequence.
- Copy-on-write sharing and reference counting support the design argument for efficient branching workloads.
- These are architecture-backed conclusions, but they are still **derived** until real GPU-memory traces are collected.

## Inherited from external literature or reference projects

- The familiar "40-60% waste with naive allocation" and "<5% waste with PagedAttention" framing comes from the vLLM/PagedAttention paper and surrounding literature, not from repository-local measurements.
- Those figures remain useful for orientation, but they should be read as inherited context rather than as a completed benchmark result for this codebase.

## What this page proves today

- The memory-management design is implemented and exercised.
- The repository has evidence for allocator behavior and cleanup behavior.
- It does **not** yet publish real VRAM captures or hardware-level fragmentation studies.

## Related

- [Benchmark Methodology](/en/benchmarks/methodology)
- [PagedAttention Architecture](/en/architecture/paged-attention)
- [Memory Management](/en/architecture/memory-management)
- [Papers Reading Guide](/en/references/papers)
