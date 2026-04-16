---
title: Workflow CPU-safe CI 调整
date: 2026-03-13
categories: [维护]
tags: [ci, rust, clippy]
---

## 变更内容

- 修复 Rust CPU CI 在 `clippy -D warnings` 下暴露的三个问题
  - `div_ceil` 写法优化
  - `HashMap::entry` 写法优化
- 保留主线 `cargo fmt`、`clippy`、`build` 与 `test` 检查
- 继续停用需要 CUDA 环境的 feature 测试
- 使 GitHub Hosted Runner 上的 CPU-safe CI 能够真实反映仓库状态

## 背景

该仓库已经将 CUDA feature 测试排除在 GitHub Hosted Runner 之外，但主线 Rust 检查仍因可直接修复的 clippy 问题失败。本次调整补齐这些问题，使常规 CI 能恢复通过。

## 修复详情

### 问题 1: `div_ceil` 使用

```rust
// 修复前
(num_tokens + block_size - 1) / block_size

// 修复后
num_tokens.div_ceil(block_size)
```

### 问题 2: `HashMap::entry` 写法

```rust
// 修复前
if let Entry::Vacant(e) = map.entry(key) {
    e.insert(value);
}

// 修复后
if let Entry::Vacant(e) = char_to_id.entry(c) {
    e.insert(next_id);
    id_to_char.insert(next_id, c);
    next_id += 1;
}
```

## 影响范围

| 文件 | 变更类型 |
|------|----------|
| `src/config.rs` | API 使用优化 |
| `src/kv_cache.rs` | API 使用优化 |
| `src/tokenizer.rs` | API 使用优化 |

## 测试结果

- ✅ `cargo fmt --check` 通过
- ✅ `cargo clippy -- -D warnings` 通过
- ✅ `cargo test` 全部通过
