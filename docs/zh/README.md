# Hetero-Paged-Infer 文档中心

欢迎使用 Hetero-Paged-Infer 文档。本目录包含全面的指南，帮助您理解、配置和部署异构推理系统。

## 文档结构

| 文档 | 说明 |
|------|------|
| [README.md](./README.md) | 本文档 - 文档概览 |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 系统架构与设计原则 |
| [API.md](./API.md) | API 参考与使用示例 |
| [CONFIGURATION.md](./CONFIGURATION.md) | 配置选项与性能调优 |
| [DEPLOYMENT.md](./DEPLOYMENT.md) | 部署与运维指南 |

## 快速链接

- **GitHub 仓库**: [LessUp/hetero-paged-infer](https://github.com/LessUp/hetero-paged-infer)
- **English Docs**: [../en/README.md](../en/README.md)
- **主 README**: [../../README.md](../../README.md)

## 项目概述

Hetero-Paged-Infer 是一个高性能大语言模型（LLM）推理引擎，利用 CPU-GPU 异构计算实现高效推理。它实现了：

- **分页式注意力（PagedAttention）** - 高效的 KV Cache 内存管理
- **连续批处理（Continuous Batching）** - 实现最优吞吐率
- **模块化架构** - 基于 trait 的抽象设计
- **生产就绪** - 完善的错误处理和监控机制

## 快速开始

1. 阅读 [架构指南](./ARCHITECTURE.md) 了解系统设计
2. 查看 [配置指南](./CONFIGURATION.md) 了解配置选项
3. 参考 [API 指南](./API.md) 获取集成示例
4. 遵循 [部署指南](./DEPLOYMENT.md) 进行生产部署

## 支持

- [GitHub Issues](https://github.com/LessUp/hetero-paged-infer/issues)
- [贡献指南](../../CONTRIBUTING.md)

---

*最后更新: 2026-04-16*
