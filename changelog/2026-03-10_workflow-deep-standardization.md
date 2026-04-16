---
title: Workflow 深度标准化
date: 2026-03-10
categories: [维护]
tags: [ci, github-actions]
---

## 变更内容

- CI workflow 统一 `permissions: contents: read` 与 `concurrency` 配置
- Pages workflow 补充 `actions/configure-pages@v5` 步骤
- Pages workflow 添加 `paths` 触发过滤，减少无效构建

## 背景

全仓库第二轮 GitHub Actions 深度标准化：统一命名、权限、并发、路径过滤与缓存策略。

## 影响范围

| 文件 | 变更类型 |
|------|----------|
| `.github/workflows/ci.yml` | 权限、并发配置 |
| `.github/workflows/pages.yml` | 新增步骤、路径过滤 |

## 相关链接

- [GitHub Actions 最佳实践](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
