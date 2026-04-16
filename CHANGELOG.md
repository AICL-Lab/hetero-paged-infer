# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive bilingual documentation (English/Chinese)
- GitHub Pages documentation site
- CI/CD workflow optimizations

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

## Release Notes

### v0.1.0 - Initial Release

**Release Date**: 2026-04-16

#### Overview

First stable release of Hetero-Paged-Infer, a heterogeneous inference system for Large Language Models. This release provides a production-ready foundation with PagedAttention memory management and Continuous Batching scheduling.

#### Key Features

1. **KV Cache Management**
   - PagedAttention block-based memory management
   - Dynamic block allocation and deallocation
   - Memory-efficient design with <5% waste

2. **Scheduler**
   - Continuous batching with prefill/decode phases
   - Decode-priority scheduling for lower latency
   - Memory pressure awareness

3. **Inference Engine**
   - Request submission and execution
   - Error recovery mechanisms
   - Metrics collection and monitoring

4. **Testing**
   - Comprehensive unit test coverage
   - Property tests for invariant verification
   - Integration tests for end-to-end validation

5. **Documentation**
   - Bilingual documentation (English/Chinese)
   - API documentation (rustdoc)
   - Architecture and deployment guides

#### Known Limitations

- GPU Executor is currently a mock implementation
- Real CUDA kernels not yet implemented
- Async CPU/GPU overlap planned for future releases

#### Installation

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build --release
```

#### Quick Start

```bash
./target/release/hetero-infer --input "Hello, world!" --max-tokens 50
```

---

## Detailed Changelog

See the [changelog/](./changelog/) directory for detailed change logs.

- [2026-04-16_documentation-overhaul.md](./changelog/2026-04-16_documentation-overhaul.md) - Documentation overhaul
- [2026-03-13_workflow-cpu-safe-ci.md](./changelog/2026-03-13_workflow-cpu-safe-ci.md) - CI fixes
- [2026-03-10_workflow-deep-standardization.md](./changelog/2026-03-10_workflow-deep-standardization.md) - Workflow standardization

---

[0.1.0]: https://github.com/LessUp/hetero-paged-infer/releases/tag/v0.1.0
