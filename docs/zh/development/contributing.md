# 贡献指南

感谢您对 Hetero-Paged-Infer 贡献的关注！

## 开发环境搭建

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
cargo test
```

## 代码风格

```bash
# 格式化代码
cargo fmt

# 运行 linter
cargo clippy --all-targets -- -D warnings

# 检查格式化
cargo fmt --check
```

## 测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_engine_creation

# 运行属性测试
cargo test -- --test-threads=1
```

## 提交更改

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

类型: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 许可证

参与贡献即表示您同意您的贡献将根据 MIT 许可证进行授权。
