---
title: Hetero-Paged-Infer
---

<FlagshipHero
  eyebrow="Rust LLM Serving Engine"
  title="聚焦核心路径的 Rust 推理引擎"
  summary="Hetero-Paged-Infer 聚焦可落地的核心能力：分页式 KV Cache、连续批处理调度，以及 OpenAI 兼容 HTTP 接口。"
  primary-text="查看架构"
  primary-href="/zh/architecture/overview"
  secondary-text="快速开始"
  secondary-href="/zh/setup/quickstart"
/>

<ProofStrip :items="[
  { label: '语言', value: 'Rust', detail: '以内存安全和清晰接口组织系统代码。' },
  { label: '内存', value: '分页 KV Cache', detail: '基于块的分配与内存账本。' },
  { label: '调度', value: '连续批处理', detail: 'Prefill/Decode 流程与 Decode 优先行为。' },
  { label: '服务', value: 'OpenAI 兼容', detail: '提供 completions/chat 与运行探针接口。' },
]" />

## 文档导航

<SectionGrid :cards="[
  { title: '架构', summary: '引擎结构、调度模型与内存管理设计。', href: '/zh/architecture/overview' },
  { title: '安装配置', summary: '安装、配置与本地使用。', href: '/zh/setup/quickstart' },
  { title: 'API', summary: '核心类型与 HTTP API 参考。', href: '/zh/api/' },
  { title: '部署', summary: 'Docker 与生产部署说明。', href: '/zh/deployment/' },
  { title: '开发', summary: '贡献流程与验证命令。', href: '/zh/development/contributing' },
]" />

## 当前边界

- 仓库已提供可本地运行的引擎、调度器、KV Cache 管理器与 HTTP 服务层。
- 文档站仅保留当前实现所需的说明，移除了历史性、展示性和白皮书式内容。
- 真实 CUDA Kernel 仍是后续工作，不被当作当前能力描述。
