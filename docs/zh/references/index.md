# 参考文献

本页面列出了影响 Hetero-Paged-Infer 的学术论文和开源项目。

## 论文

LLM 推理优化的核心研究论文：

- [论文引用](/zh/references/papers) — 学术出版物
- [项目引用](/zh/references/projects) — 开源实现

## 为什么这些文献很重要

Hetero-Paged-Infer 实现了以下研究的技术：

| 技术 | 论文 | 影响 |
|------|------|------|
| PagedAttention | Kwon et al., SOSP 2023 | <5% 内存浪费 |
| 连续批处理 | Yu et al., OSDI 2022 | +50% 吞吐量 |
| FlashAttention | Dao et al., NeurIPS 2022 | 2-4x 注意力加速 |