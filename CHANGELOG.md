# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- GitHub Issue templates (bug report, feature request)
- Pull request template with checklist
- Dependabot configuration for dependency updates

### Changed

- Migrated changelog archives to `openspec/archive/`
- Removed redundant `RELEASE_NOTES.md` (consolidated into this file)
- Removed duplicate `changelog/*.en.md` files
- Removed redundant `docs/*/README.md` files
- Unified test count to 122 across all documentation
- Cleaned up `.qwen/` directory (unused AI tool config)

## [0.1.0] - 2026-04-16

### Added

#### Documentation System
- Complete bilingual README (English + Chinese)
- CONTRIBUTING.md contribution guidelines
- CHANGELOG.md changelog
- config.example.json example configuration
- Full rustdoc documentation for all public APIs
- Comprehensive docs/ directory with:
  - English documentation (5 files)
  - Chinese documentation (5 files)

#### PagedAttention KV Cache
- Block-based memory management with on-demand allocation/deallocation
- BlockPool with FIFO free list management
- PageTable for logical to physical block mapping
- Memory pressure detection and handling

#### Continuous Batching Scheduler
- Prefill/decode phase separation
- Decode-priority scheduling strategy
- Dynamic batch formation
- Request state machine management

#### Inference Engine
- InferenceEngine main orchestrator
- Error recovery strategies (retry/skip/reset/shutdown)
- EngineMetrics metrics collection
- Step-by-step and continuous execution modes

#### Modular Architecture
- TokenizerTrait tokenizer interface
- SchedulerTrait scheduler interface
- GPUExecutorTrait GPU executor interface
- KVCacheManagerTrait KV Cache manager interface

#### Testing
- 78 unit tests covering all modules
- 15 property tests using proptest
- 13 integration tests for end-to-end flows
- 29 documentation tests

#### Mock Implementations
- MockGPUExecutor for testing
- SimpleTokenizer character-level tokenizer

### Changed

- 2026-04-16: Complete documentation overhaul with bilingual support
- 2026-03-13: Fixed clippy warnings, optimized `div_ceil` and `HashMap::entry` usage
- 2026-03-10: Unified GitHub Actions workflow configuration

---

## Detailed Changelog

See the [openspec/archive/](./openspec/archive/) directory for historical change logs.

- [2026-04-16_documentation-overhaul.md](./openspec/archive/2026-04-16_documentation-overhaul.md) - Documentation overhaul
- [2026-03-13_workflow-cpu-safe-ci.md](./openspec/archive/2026-03-13_workflow-cpu-safe-ci.md) - CI fixes
- [2026-03-10_workflow-deep-standardization.md](./openspec/archive/2026-03-10_workflow-deep-standardization.md) - Workflow standardization

---

[0.1.0]: https://github.com/LessUp/hetero-paged-infer/releases/tag/v0.1.0
