---
title: Workflow 深度标准化
date: 2026-03-10
categories: [维护]
tags: [ci, github-actions]
version: patch
---

## 摘要

对 GitHub Actions 工作流进行全面标准化，实施 CI/CD 管道可靠性和可维护性的最佳实践。

## 变更内容

### CI 工作流增强
- 统一 `permissions: contents: read` 以加强安全
- 添加 `concurrency` 配置防止冗余构建
- 实现路径过滤减少不必要的 CI 触发

### Pages 工作流改进
- 集成 `actions/configure-pages@v5` 实现正确的 Jekyll 设置
- 添加 `paths` 触发器过滤文档变更
- 优化稀疏检出加快构建速度

## 背景

GitHub Actions 第二轮深度标准化：统一命名、权限、并发、路径过滤和缓存策略，确保 CI/CD 操作的一致性、安全性和效率。

## 影响分析

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `.github/workflows/ci.yml` | 权限与并发 | 安全和效率改进 |
| `.github/workflows/pages.yml` | 构建优化 | 路径过滤和缓存 |

## 测试

- ✅ 所有工作流语法通过 `actionlint` 验证
- ✅ CI 管道通过新的并发设置
- ✅ 功能分支 Pages 部署验证通过

## 参考

- [GitHub Actions 最佳实践](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [GitHub Actions 安全加固](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)

---

*维护者：DevOps 团队*
