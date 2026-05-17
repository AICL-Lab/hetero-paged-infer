# Comparison with Other Inference Engines

This page is not a checkbox matrix. It is a positioning argument about what Hetero-Paged-Infer is strong at today, where it is still incomplete, and why its implementation language changes the maintenance trade-off.

## Where this project competes well

- **Architecture transparency**: the scheduler, KV-cache manager, executor trait, and API surface are small enough to inspect directly.
- **Testability**: key behaviors are covered by unit tests, property tests, integration tests, and Criterion microbenchmarks.
- **Documentation discipline**: the whitepaper, benchmark methodology, and reference guides explicitly separate measured claims from inherited ones.
- **Systems-learning value**: for readers studying PagedAttention-style serving designs, this repository is easier to audit than large production stacks.

## Where it is behind vLLM / TensorRT-LLM

- **Real kernels**: the default runtime still uses `MockGPUExecutor`, so production GPU throughput and latency are not yet established.
- **Production features**: multi-GPU execution, quantization depth, prefix caching, speculative decoding, and hardware-specific tuning are still behind mature engines.
- **Operational maturity**: vLLM and TensorRT-LLM have broader deployment stories, larger user communities, and more battle-tested kernels.
- **Benchmark strength**: this repository documents local proof and design intent; it does not yet provide a fair head-to-head shootout against those systems.

## Why Rust changes the maintenance story

- Rust makes ownership, mutability, and concurrency constraints explicit in code that would otherwise rely on review discipline.
- Trait boundaries keep scheduler, KV-cache, tokenizer, and executor concerns separable, which lowers the cost of replacing the current mock executor with a real backend.
- The trade-off is ecosystem maturity: Python/C++ serving stacks have more production integrations today, while Rust offers a stronger long-term correctness story for systems code.

## What readers should evaluate this project for

Evaluate Hetero-Paged-Infer if you care about:

- learning and auditing LLM-serving architecture,
- experimenting with scheduler/KV-cache design under strong type boundaries,
- extending a Rust-based inference core with a future real executor,
- understanding how to document evidence safely instead of overselling benchmarks.

Do **not** evaluate it today as if it were already a drop-in replacement for vLLM or TensorRT-LLM in production throughput competitions. Read the [benchmark methodology](/en/benchmarks/methodology) and [reference guides](/en/references/) together with this page.
