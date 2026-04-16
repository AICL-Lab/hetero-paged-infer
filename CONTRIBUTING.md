# 贡献指南

感谢您有兴趣为 Hetero-Paged-Infer 做贡献！

## 开发环境设置

### 环境要求

- Rust 1.70+ (2021 edition)
- Git

### 克隆与构建

```bash
git clone https://github.com/LessUp/hetero-paged-infer.git
cd hetero-paged-infer
cargo build
```

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试
cargo test test_engine_creation

# 运行属性测试
cargo test -- --test-threads=1
```

## 代码风格

### 格式化

使用 `rustfmt` 保持代码风格一致：

```bash
cargo fmt --check
```

### Lint

使用 `clippy` 进行静态检查：

```bash
cargo clippy --all-targets -- -D warnings
```

### 文档注释

所有公共 API 必须有文档注释：

```rust
/// 简短描述
///
/// 详细说明。
///
/// # 参数
///
/// * `param1` - 参数说明
///
/// # 返回
///
/// 返回值说明。
///
/// # 示例
///
/// ```rust
/// use my_crate::my_function;
/// let result = my_function();
/// ```
pub fn my_function() -> i32 {
    42
}
```

## 提交代码

### 提交信息格式

使用约定式提交格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

类型：
- `feat` - 新功能
- `fix` - Bug 修复
- `docs` - 文档更新
- `style` - 代码格式（不影响功能）
- `refactor` - 重构
- `test` - 测试相关
- `chore` - 构建/工具相关

示例：

```
feat(scheduler): 添加 decode 优先调度

实现 decode 请求优先于 prefill 请求的调度策略，
以降低正在处理请求的延迟。

Closes #123
```

### Pull Request 流程

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/my-feature`)
3. 提交更改 (`git commit -m 'feat: 添加某功能'`)
4. 推送到分支 (`git push origin feature/my-feature`)
5. 创建 Pull Request

### PR 检查清单

- [ ] 代码通过 `cargo fmt --check`
- [ ] 代码通过 `cargo clippy`
- [ ] 所有测试通过 `cargo test`
- [ ] 新功能有对应的测试
- [ ] 公共 API 有文档注释
- [ ] 更新相关文档

## 测试要求

### 单元测试

每个模块应有单元测试覆盖核心功能：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        assert_eq!(my_function(), expected);
    }
}
```

### 属性测试

使用 `proptest` 进行属性测试：

```rust
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_my_property(input in 0..100) {
            // 测试属性
        }
    }
}
```

## 项目结构

```
src/
├── lib.rs           # 库入口
├── main.rs          # CLI 入口
├── config.rs        # 配置
├── engine.rs        # 推理引擎
├── error.rs         # 错误类型
├── types.rs         # 核心类型
├── kv_cache.rs      # KV Cache 管理
├── scheduler.rs     # 调度器
├── tokenizer.rs     # 分词器
└── gpu_executor.rs  # GPU 执行器

tests/
└── integration_tests.rs  # 集成测试
```

## 获取帮助

如有问题，可以：
- 在 GitHub 上开 Issue
- 查看现有代码和测试作为参考

## 许可证

本项目采用 MIT 许可证。提交代码即表示同意以相同许可发布。
