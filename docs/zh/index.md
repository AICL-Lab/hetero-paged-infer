---
title: Hetero-Paged-Infer
hide:
  - navigation
  - toc
---

<div align="center">

# Hetero-Paged-Infer

**高性能 LLM 推理引擎**

*PagedAttention + Continuous Batching*

[快速开始](setup/quickstart.md){ .md-button .md-button--primary }
[GitHub](https://github.com/LessUp/hetero-paged-infer){ .md-button }

</div>

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **分页式注意力** | 基于块的 KV Cache，内存浪费 <5% |
| **连续批处理** | 动态 prefill/decode 调度 |
| **生产就绪** | 错误处理、指标监控 |
| **全面测试** | 135 个测试（单元、属性、集成） |

---

## 快速开始

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build --release
./target/release/hetero-infer --input "你好，世界！" --max-tokens 50
```

---

## 性能表现

| 方法 | 内存浪费 | 吞吐率 |
|------|:--------:|:------:|
| 静态分配 | ~40-60% | 基准 |
| **PagedAttention** | **<5%** | **+50%** |

---

## 系统架构

```
┌─────────────────────────────────────────────────┐
│            InferenceEngine (CPU)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  分词器  │  │  调度器  │  │ KV Cache管理 │   │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
│       └─────────────┼───────────────┘           │
├─────────────────────┼───────────────────────────┤
│               ┌─────▼─────┐                      │
│               │    GPU    │  执行器 + 内存       │
│               └───────────┘                      │
└─────────────────────────────────────────────────┘
```

---

## 文档导航

- **[环境搭建](setup/)** - 安装与配置
- **[架构设计](architecture/)** - 系统设计
- **[API 参考](api/)** - API 文档
- **[部署运维](deployment/)** - 生产部署
- **[开发指南](development/)** - 贡献指南
