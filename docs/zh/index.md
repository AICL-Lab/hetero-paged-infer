---
layout: home

hero:
  name: "Hetero-Paged-Infer"
  text: "高性能 LLM 推理引擎"
  tagline: PagedAttention + Continuous Batching · Rust 实现
  image:
    src: /images/logo.svg
    alt: Hetero-Paged-Infer Logo
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/setup/quickstart
    - theme: alt
      text: GitHub 仓库
      link: https://github.com/LessUp/hetero-paged-infer
    - theme: alt
      text: API 参考
      link: /zh/api/core-types

features:
  - icon: 🧠
    title: PagedAttention
    details: 基于块的 KV Cache 管理，按需分配。内存浪费 <5%，相比静态分配的 40-60% 大幅降低。
  - icon: ⚡
    title: 连续批处理
    details: 动态 prefill/decode 调度，优先级感知。最大化 GPU 利用率，同时保持低延迟。
  - icon: 🛡️
    title: 内存压力感知
    details: 可配置的 OOM 防护，优雅降级。生产级错误处理和监控。
  - icon: 🔧
    title: 模块化架构
    details: 基于 Trait 的抽象设计，易于定制。CPU 调度器与 GPU 执行器清晰分离。
  - icon: 🧪
    title: 全面测试覆盖
    details: 121 个测试，包括单元测试、属性测试和集成测试。属性测试验证关键不变量。
  - icon: 🚀
    title: OpenAI 兼容
    details: 内置 HTTP 服务器，OpenAI API 兼容。支持 Server-Sent Events (SSE) 流式响应。
---

## 关键指标

| 指标 | 数值 | 说明 |
|------|:----:|------|
| 内存浪费 | **<5%** | 相比静态分配的 40-60% |
| 吞吐提升 | **+50%** | 相比静态批处理 |
| 测试通过 | **121+** | 单元、属性、集成测试 |
| Unsafe 代码 | **0** | 完整 Rust 安全保证 |

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer

# 编译发布版本
cargo build --release

# 运行推理
./target/release/hetero-infer --input "你好，世界！" --max-tokens 50
```

## 文档导航

- [快速开始](/zh/setup/quickstart) — 快速上手指南
- [架构设计](/zh/architecture/overview) — 系统设计深入
- [API 参考](/zh/api/core-types) — 完整 API 文档
- [性能基准](/zh/benchmarks/) — 性能对比分析
