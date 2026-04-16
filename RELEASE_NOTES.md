## Hetero-Paged-Infer v0.1.0 Release Notes

### 🎉 First Stable Release

We're excited to announce the first stable release of **Hetero-Paged-Infer** - a high-performance heterogeneous inference engine for Large Language Models!

---

### 📋 Overview

Hetero-Paged-Infer is a Rust-based inference system featuring:

- **PagedAttention** memory management - reduces KV Cache waste to <5%
- **Continuous Batching** scheduling - maximizes GPU utilization
- **Modular Architecture** - trait-based design for flexibility
- **Production Ready** - comprehensive error handling and monitoring

---

### ✨ Key Features

#### Core Engine
- ✅ PagedAttention KV Cache management
- ✅ Continuous Batching with decode priority
- ✅ Memory pressure awareness and admission control
- ✅ Request state machine (Pending → Prefill → Decode → Completed)

#### Architecture
- ✅ Modular trait-based design
- ✅ Pluggable tokenizer interface
- ✅ Configurable scheduler strategies
- ✅ Mock GPU executor for testing

#### Testing & Quality
- ✅ 78 unit tests
- ✅ 15 property tests (proptest)
- ✅ 13 integration tests
- ✅ 100% rustdoc coverage

#### Documentation
- ✅ Bilingual documentation (English/Chinese)
- ✅ Comprehensive API reference
- ✅ Architecture and deployment guides
- ✅ GitHub Pages site

---

### 🚀 Quick Start

```bash
# Clone and build
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build --release

# Run inference
./target/release/hetero-infer \
  --input "Explain machine learning" \
  --max-tokens 100
```

---

### 📦 What's Included

- Full source code with MIT license
- Comprehensive test suite
- Bilingual documentation
- Example configurations
- CI/CD workflows

---

### ⚠️ Known Limitations

This release provides a solid foundation with mock GPU execution:

- GPU Executor is a mock implementation (real CUDA kernels planned)
- Pinned memory management not yet implemented
- Async CPU/GPU overlap planned for v0.2.0

---

### 📚 Documentation

- **English Docs**: [docs/en/](./docs/en/)
- **中文文档**: [docs/zh/](./docs/zh/)
- **GitHub Pages**: https://lessup.github.io/hetero-paged-infer/
- **API Docs**: Run `cargo doc --open`

---

### 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

---

### 📄 License

MIT License - See [LICENSE](./LICENSE) for details.

---

**Full Changelog**: https://github.com/LessUp/hetero-paged-infer/commits/v0.1.0

---

---

## Hetero-Paged-Infer v0.1.0 发布说明

### 🎉 首个稳定版本

我们很高兴宣布 **Hetero-Paged-Infer** 的首个稳定版本发布 - 一个高性能异构推理引擎！

---

### 📋 项目概述

Hetero-Paged-Infer 是一个基于 Rust 的推理系统，具有以下特性：

- **分页式注意力（PagedAttention）** 内存管理 - 将 KV Cache 浪费降至 <5%
- **连续批处理（Continuous Batching）** 调度 - 最大化 GPU 利用率
- **模块化架构** - 基于 Trait 的灵活设计
- **生产就绪** - 全面的错误处理和监控

---

### ✨ 核心特性

#### 核心引擎
- ✅ 分页式注意力 KV Cache 管理
- ✅ 带解码优先的连续批处理
- ✅ 内存压力感知和准入控制
- ✅ 请求状态机（等待 → 预填充 → 解码 → 完成）

#### 架构
- ✅ 模块化基于 Trait 的设计
- ✅ 可插拔分词器接口
- ✅ 可配置调度器策略
- ✅ 用于测试的模拟 GPU 执行器

#### 测试与质量
- ✅ 78 个单元测试
- ✅ 15 个属性测试（proptest）
- ✅ 13 个集成测试
- ✅ 100% rustdoc 覆盖率

#### 文档
- ✅ 双语文档（中英）
- ✅ 全面的 API 参考
- ✅ 架构和部署指南
- ✅ GitHub Pages 站点

---

### 🚀 快速开始

```bash
# 克隆并构建
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build --release

# 运行推理
./target/release/hetero-infer \
  --input "解释机器学习" \
  --max-tokens 100
```

---

### 📦 包含内容

- 完整源代码（MIT 许可证）
- 全面的测试套件
- 双语文档
- 示例配置
- CI/CD 工作流

---

### ⚠️ 已知限制

本版本提供坚实的基础，使用模拟 GPU 执行：

- GPU 执行器是模拟实现（计划实现真实 CUDA kernel）
- 固定内存管理尚未实现
- 异步 CPU/GPU 重叠计划 v0.2.0 实现

---

### 📚 文档

- **English Docs**: [docs/en/](./docs/en/)
- **中文文档**: [docs/zh/](./docs/zh/)
- **GitHub Pages**: https://lessup.github.io/hetero-paged-infer/
- **API 文档**: 运行 `cargo doc --open`

---

### 🤝 贡献

欢迎贡献！请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解指南。

---

### 📄 许可证

MIT 许可证 - 详情见 [LICENSE](./LICENSE)。

---

**完整变更日志**: https://github.com/LessUp/hetero-paged-infer/commits/v0.1.0
