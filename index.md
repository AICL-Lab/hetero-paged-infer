# Hetero-Paged-Infer

<div align="center">

**High-Performance LLM Inference Engine**

**高性能 LLM 推理引擎**

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/LessUp/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/hetero-paged-infer/actions)
[![Docs.rs](https://docs.rs/hetero-infer/badge.svg)](https://docs.rs/hetero-infer)
[![Crates.io](https://img.shields.io/crates/v/hetero-infer.svg)](https://crates.io/crates/hetero-infer)

[![PagedAttention](https://img.shields.io/badge/PagedAttention-<5%25%20waste-green)]()
[![Continuous Batching](https://img.shields.io/badge/Continuous%20Batching-+50%25%20throughput-green)]()
[![Tests](https://img.shields.io/badge/Tests-135%20passed-brightgreen)]()

</div>

---

**PagedAttention + Continuous Batching** for efficient LLM inference

**分页式注意力 + 连续批处理** 实现高效 LLM 推理

---

## ✨ Key Features | 核心特性

<div align="center">

| Feature | Description | 特性 | 说明 |
|:-------:|:------------|:----:|:-----|
| 🧠 **PagedAttention** | <5% Memory Waste | **分页式注意力** | 内存浪费 <5% |
| ⚡ **Continuous Batching** | +50% Throughput | **连续批处理** | 吞吐率 +50% |
| 🧪 **Well Tested** | 135 Tests (Unit, Property, Integration) | **全面测试** | 135 个测试 |
| 🏭 **Production Ready** | Error handling, metrics, monitoring | **生产就绪** | 错误处理、监控 |
| 🔧 **Modular** | Trait-based abstractions | **模块化** | 基于 Trait 抽象 |

</div>

---

## 🚀 Quick Start | 快速开始

```bash
# Clone | 克隆
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# Build | 构建
cargo build --release

# Run | 运行
./target/release/hetero-infer --input "Hello, world!" --max-tokens 50

# Test | 测试
cargo test
```

---

## 🏗️ Architecture | 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine (CPU)                          │
├──────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────────────┐  │
│  │ Tokenizer  │  │ Scheduler  │  │      KV Cache Manager          │  │
│  │            │  │            │  │   BlockPool + PageTable        │  │
│  └─────┬──────┘  └─────┬──────┘  └───────────────┬────────────────┘  │
│        │               │                         │                    │
│        │        ┌──────▼──────┐                  │                    │
│        │        │Batch Builder│◄─────────────────┘                    │
│        │        └──────┬──────┘                                       │
├────────┼───────────────┼─────────────────────────────────────────────┤
│        │        ┌──────▼──────┐                                       │
│        │        │ GPU Executor│  (CUDA / Mock)                        │
│        │        └──────┬──────┘                                       │
│        │        ┌──────▼──────┐                                       │
│        └───────►│  KV Cache   │  (GPU Memory)                         │
│                 └─────────────┘                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Documentation | 文档

<div align="center">

| 🇺🇸 [English Documentation](en/) | 🇨🇳 [中文文档](zh/) |
|:-------------------------------:|:------------------:|

| 📖 [API Reference (docs.rs)](https://docs.rs/hetero-infer) | 📝 [Changelog](https://github.com/LessUp/hetero-paged-infer/blob/main/CHANGELOG.md) |
|:----------------------------------------------------------:|:----------------------------------------------------------------------------------:|

</div>

---

## 📊 Performance | 性能

| Method | Memory Waste | Throughput |
|--------|:------------:|:----------:|
| Static Allocation | ~40-60% | Baseline |
| Dynamic Allocation | ~20-30% | +20% |
| **PagedAttention** | **<5%** | **+50%** |

---

## 🔗 Links | 链接

- [GitHub Repository](https://github.com/LessUp/hetero-paged-infer)
- [Crates.io](https://crates.io/crates/hetero-infer)
- [Issue Tracker](https://github.com/LessUp/hetero-paged-infer/issues)
- [Contributing Guide](https://github.com/LessUp/hetero-paged-infer/blob/main/CONTRIBUTING.md)

---

## 📜 License | 许可证

MIT License - See [LICENSE](https://github.com/LessUp/hetero-paged-infer/blob/main/LICENSE)

---

<p align="center">
<b>Made with ❤️ by <a href="https://github.com/LessUp">LessUp</a></b>
</p>
