# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A real `cuda` feature implementation with `build.rs`, an nvcc-compiled backend library, Rust FFI wiring, a minimal CUDA kernel path, a no-nvcc host-compatible fallback build for CI, and feature-gated CUDA executor tests.

### Removed

- Command bridge serving path and its related config, tests, and exported API surface.
- Local AI-tool residue in `.gitignore` (`.claude/`, `CLAUDE.local.md`, `.omc/`).
- GitHub Pages sections for whitepaper, benchmarks, comparison, references, and speculative advanced configuration.
- Repository-scoped AI workflow scaffolding: `CLAUDE.md` and the full `openspec/` tree.
- Changelog mirrors under GitHub Pages (`docs/en/changelog/`, `docs/zh/changelog/`).
- Unused docs dependency `vitepress-plugin-llms`.
- Unused direct Rust dependencies `tracing` and `tracing-subscriber`.

### Changed

- Serving now runs a single local-engine path instead of maintaining alternate backend abstractions.
- Configuration and API docs were rewritten to document only the capabilities that currently exist in the repository.
- Contributor workflow simplified to direct code/test/docs maintenance in `CONTRIBUTING.md`.
- Docs landing pages simplified to focus on core engine capabilities and practical navigation.
- Root `CHANGELOG.md` is now the single project changelog source.

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

Legacy OpenSpec archive records were removed during repository simplification.
Durable project history is now condensed in this changelog and GitHub Releases.

[0.1.0]: https://github.com/LessUp/hetero-paged-infer/releases/tag/v0.1.0
