# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- 仓库由 `hetero-paged-infer` 更名为 `paged-infer`：原名中的 "hetero"（异构）与当前纯 CPU 参考后端的实际状态不符。crate 更名为 `paged-infer`/`paged_infer`，指标名同步更名为 `paged_*`。旧仓库地址自动重定向。

### Added

- Engine event API: `InferenceEngine::step_events()` / `StepEvents` (per-step completions plus per-token text events), `create_router_with_engine()` for injecting custom executors into the HTTP layer (used by tests), and `submit_request()` returning `(request_id, prompt_tokens)` from a single tokenization pass.
- Server: true token-level SSE streaming, 429 overload rejection with `Retry-After`, graceful shutdown (SIGTERM / Ctrl+C), a real `/readyz` probe, `model` validation (404 on mismatch), chat role whitelist, and a JSON error envelope with a `type` field on every error path (including malformed bodies and unknown routes).
- Tests: GPU-timeout retry-exhaustion path, server concurrency / 500 / 429 / 404 coverage, and strengthened assertions replacing tautological ones; CI now runs the test suite with `--all-features` and a `cargo bench -- --test` smoke run.
- `test-utils` cargo feature: test helpers are now compiled only under `cfg(test)` or this feature, keeping them out of release builds.
- `InferenceEngine::cancel_request()` / `Scheduler::cancel_by_request_id()`: cancel in-flight requests (any stage) and free their KV blocks.
- `--host` / `--port` CLI overrides for serve mode (work with or without `--config`).
- Integration test proving the engine stops executing a request after its client disconnects.

### Removed

- Command bridge serving path and its related config, tests, and exported API surface.
- Local AI-tool residue in `.gitignore` (`.claude/`, `CLAUDE.local.md`, `.omc/`).
- GitHub Pages sections for whitepaper, benchmarks, comparison, references, and speculative advanced configuration.
- Repository-scoped AI workflow scaffolding: `CLAUDE.md` and the full legacy spec tree.
- Changelog mirrors under GitHub Pages (`docs/en/changelog/`, `docs/zh/changelog/`).
- Unused docs dependency `vitepress-plugin-llms`.
- Unused direct Rust dependencies `tracing` and `tracing-subscriber`.
- Dead code: CUDA Graph trait methods (never invoked by the engine), `Request::created_at`, `Scheduler::get_sequence_mut`, `PageTable::get_physical`, `RequestState::is_active`/`is_terminal`, the never-occurring unmapped `LogicalBlock` state, and the misleading copy-on-write comment on `PhysicalBlock::ref_count`.
- Fake after-the-fact "streaming" (chunking finished text into 32-char slices), replaced by true token-level streaming.
- Dead fields `Sequence::num_computed_tokens` and `ExecutionOutput::logits` (never read).
- `LogicalBlock` wrapper type: `PageTable` / `Sequence` now hold `Vec<PhysicalBlockRef>` directly.
- Unused `KVCacheManager::has_sequence` and `Scheduler::num_prefill_sequences`.
- `InferenceEngine::count_prompt_tokens` (folded into `submit_request`'s return value, eliminating per-request double tokenization).
- Redundant `config.validate()` calls in `main` / `create_router` (constructors already validate).
- `usize_from_u32` helper (inlined).

### Changed

- Server concurrency model: the engine is now owned by a single background loop; handlers submit via channels and await per-request events. Continuous batching now works under concurrent HTTP load, and the async runtime is no longer blocked (previously a global mutex serialized every request behind a full `engine.run()`).
- Scheduler determinism: active sequences use `BTreeMap` (FCFS by submission order) instead of `HashMap`, making scheduling fair and reproducible.
- `temperature == 0.0` (greedy decoding) is now accepted; the valid range is `[0.0, 2.0]`.
- `build_execution_batch` fails fast on an inconsistent decode sequence instead of silently skipping it (which would have left the request stuck forever).
- Usage reporting uses the real tokenizer: `prompt_tokens` / `completion_tokens` are exact counts, not whitespace estimates.
- Overflow-safe token arithmetic in the scheduler (`saturating_add`) and an always-safe `Sequence::context_len`.
- Benchmarks rebuilt with `iter_batched` so they no longer panic or degrade into measuring idle/error paths; filler benches removed.
- Crate metadata, CLI about text, README, and docs now describe the project as a paged-memory + continuous-batching scaffold with a mock compute backend; unsupported claims ("CPU-GPU co-execution", fabricated throughput charts, fictional preemption API) were removed.
- The engine loop now yields once per step; previously it had no await point during active generation.
- Client disconnect now cancels the in-flight request instead of generating to `max_tokens` for nobody; the SSE stream reports an error event rather than a bare `[DONE]` when the event channel closes without a terminal event.
- `InferenceEngine::run()` drains completions buffered before the loop (e.g. cancelled requests).
- A too-large prefill no longer head-of-line-blocks smaller prefills (`continue` instead of `break`).
- `step_events` attributes generated tokens via an O(1) map instead of a per-token linear scan.
- `ExecutionError::CudaError` renamed to `ExecutionError::BackendError` (the mock backend is not CUDA).
- CLI config built from `EngineConfig::default()` + overrides instead of duplicated default literals.
- Docs and examples updated for the new `submit_request` signature, removed fields, and `LogicalBlock` removal.

### Fixed

- Benchmark suite crashed during warmup (`bench_request_submission` exhausted `max_num_seqs`); the whole suite is now runnable and smoke-tested in CI.
- Mock executor divide-by-zero panic on an empty vocabulary (now an error, consistent with the CUDA backend).
- Double-free and zero `block_size` misuse are now caught by debug assertions with clear messages.
- Engine loop had no await point during active generation, starving handlers / SSE streams on single-threaded runtimes (first token never reached the client until the batch finished) and pinning a worker on multi-threaded runtimes; now yields each step.

## [0.1.0] - 2026-04-16

### Added

- Bilingual project docs (README, docs site, API references, deployment and development guides).
- Core Rust inference engine modules for paged KV cache, scheduler, tokenizer, and execution pipeline.
- OpenAI-compatible serving endpoints (`/v1/completions`, `/v1/chat/completions`) with health/readiness/metrics endpoints.
- Mock GPU executor and broad automated test coverage across unit/property/integration/doc tests.

### Changed

- 2026-03-13: CPU-safe CI fixes and clippy-driven code cleanup (`div_ceil`, `HashMap::entry` usage).
- 2026-03-10: GitHub Actions workflow standardization for permissions, concurrency, and docs pipeline reliability.

## Historical Notes

Legacy spec archive records were removed during repository simplification.
Durable project history is now condensed in this changelog and GitHub Releases.

[0.1.0]: https://github.com/AICL-Lab/paged-infer/releases/tag/v0.1.0
