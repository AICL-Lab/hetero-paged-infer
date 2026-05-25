# 贡献指南

本页与仓库根目录 [`CONTRIBUTING.md`](https://github.com/AICL-Lab/hetero-paged-infer/blob/master/CONTRIBUTING.md) 保持一致。

## 快速开始

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
```

## 验证命令

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --verbose
cargo test --doc --verbose
cargo doc --no-deps
cargo bench --no-run
cd docs && npm run build
```

## 贡献流程

1. 创建聚焦的功能分支。
2. 实现变更，并在行为变化时补充测试。
3. 运行验证命令。
4. 若涉及项目可见变化，同步更新文档与根目录 `CHANGELOG.md`。
5. 提交 Pull Request。
