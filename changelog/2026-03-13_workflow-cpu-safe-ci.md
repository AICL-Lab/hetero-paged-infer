# Workflow CPU-safe CI 调整

日期：2026-03-13

## 变更内容

- 修复 Rust CPU CI 在 `clippy -D warnings` 下暴露的三个问题，包含 `div_ceil` 与 `HashMap::entry` 写法
- 保留主线 `cargo fmt`、`clippy`、`build` 与 `test` 检查，仅继续停用需要 CUDA 环境的 feature 测试
- 使 GitHub Hosted Runner 上的 CPU-safe CI 能够真实反映仓库状态，而不是被无关的 lint 失败阻塞

## 背景

该仓库已经将 CUDA feature 测试排除在 GitHub Hosted Runner 之外，但主线 Rust 检查仍因可直接修复的 clippy 问题失败。本次调整补齐这些问题，使常规 CI 能恢复通过。
