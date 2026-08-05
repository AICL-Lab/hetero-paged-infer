# 贡献指南

感谢参与 Hetero-Paged-Infer。

## 开发环境

```bash
git clone https://github.com/AICL-Lab/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
```

## 提交前检查

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo doc --no-deps
```

## 工作流

1. 做一个聚焦的改动
2. 行为变更时同步更新测试
3. 跑通上面的检查命令
4. 项目级行为变更时更新 `CHANGELOG.md`

## 提交信息格式

使用 conventional commits：

```text
<type>(<scope>): <description>
```

类型：`feat`、`fix`、`docs`、`style`、`refactor`、`test`、`chore`。

## 许可证

提交即表示你同意以 MIT 许可证发布你的贡献。
