---
title: Workflow CPU-Safe CI 调整
date: 2026-03-13
categories: [维护, Bug修复]
tags: [ci, rust, clippy]
version: patch
---

## 摘要

修复三个阻止 CPU-Safe CI 在 GitHub Hosted Runner 上通过的 Clippy 警告，确保无需 CUDA 依赖的可靠自动测试。

## 变更内容

### 代码质量修复
1. **`div_ceil` API 迁移**
   - 将手动向上取整除法替换为 `div_ceil()` 方法
   - 影响：`config.rs`, `kv_cache.rs`, `tokenizer.rs`

2. **`HashMap::entry` 优化**
   - 简化 entry API 使用提升可读性
   - 保持完全一致的语义行为

### CI 配置
- 保留核心检查：`cargo fmt`, `clippy`, `build`, `test`
- 禁用 CUDA 特性测试（需要不可用的 GPU 环境）
- 实现稳定的纯 CPU CI 管道

## 背景

仓库已将 CUDA 特性测试排除在 GitHub Hosted Runner 之外，但核心 Rust 检查因可修复的 Clippy 问题而失败。本次调整解决这些问题，恢复可靠的 CI 反馈。

## 技术详情

### 修复 1：向上取整除法
```rust
// 修复前
(num_tokens + block_size - 1) / block_size

// 修复后
num_tokens.div_ceil(block_size)
```

### 修复 2：HashMap Entry 模式
```rust
// 优化 entry API 使用以提升清晰度
// 同时保持完全相同的行为
```

## 影响分析

| 文件 | 变更类型 | 影响 |
|------|----------|------|
| `src/config.rs` | API 迁移 | 提升代码质量 |
| `src/kv_cache.rs` | API 迁移 | 提升代码质量 |
| `src/tokenizer.rs` | API 迁移 | 提升代码质量 |

## 测试

- ✅ `cargo fmt --check` 通过
- ✅ `cargo clippy -- -D warnings` 通过
- ✅ `cargo test` 所有测试通过
- ✅ CI 管道在 `ubuntu-latest` 上绿灯

## 迁移指南

用户无需操作。这是内部代码质量改进，无 API 变更。

---

*维护者：Rust 团队*
