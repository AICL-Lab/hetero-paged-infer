# OpenSpec AI 助手指令

> 本文档为 AI 编码助手提供异构推理项目的开发指南。

## 项目概述

**Hetero-Paged-Infer** 是一个用 Rust 编写的高性能 LLM 推理引擎，实现 PagedAttention 和 Continuous Batching 技术。

**项目语言**: 代码注释和文档主要使用**中文**。

## OpenSpec 工作流

本项目使用 OpenSpec 进行规格驱动开发。

### 提出变更

使用 `/opsx:propose "<想法>"` 创建新的变更提案。

```bash
# 示例
/opsx:propose "添加流式响应支持"
```

这将创建：
- `openspec/changes/<名称>/proposal.md` - 变更提案
- `openspec/changes/<名称>/specs/` - 规格增量
- `openspec/changes/<名称>/design.md` - 技术方案
- `openspec/changes/<名称>/tasks.md` - 实施清单

### 实施变更

使用 `/opsx:apply` 执行提案中的任务。

### 归档变更

使用 `/opsx:archive` 将已完成的变更移至归档。

## 构建与测试命令

### 基本命令

```bash
# 构建项目
cargo build --release

# 运行所有测试
cargo test

# 运行特定测试
cargo test test_engine_creation

# 运行属性测试（建议单线程）
cargo test -- --test-threads=1

# 仅运行文档测试
cargo test --doc

# 格式化代码
cargo fmt

# 检查格式
cargo fmt --check

# 运行 linter
cargo clippy --all-targets -- -D warnings

# 生成文档
cargo doc --no-deps --open

# 运行基准测试
cargo bench
```

### CLI 使用

```bash
# 构建并运行
./target/release/hetero-infer --input "你好，世界！" --max-tokens 50

# 使用配置文件
./target/release/hetero-infer --config config.json

# 自定义参数
./target/release/hetero-infer \
  --input "解释量子计算" \
  --max-tokens 100 \
  --temperature 0.8 \
  --top-p 0.95 \
  --block-size 16 \
  --max-num-blocks 1024
```

## 代码风格指南

### Rust 格式配置（rustfmt.toml）

```toml
edition = "2021"
max_width = 100
tab_spaces = 4
use_field_init_shorthand = true
use_try_shorthand = true
```

### 注释风格

- **所有注释使用中文** - 与现有代码保持一致
- 公共 API 文档使用 `///`
- 模块级文档使用 `//!`
- 文档注释中包含 `# Examples` 部分

示例：
```rust
/// 推理引擎
///
/// 主编排器，协调所有组件实现端到端推理。
///
/// # 组件
///
/// - **Tokenizer** - 文本与 token 之间的转换
/// - **Scheduler** - 请求调度和批次管理
/// - **GPU Executor** - GPU 计算执行
/// - **KV Cache Manager** - KV Cache 内存管理
///
/// # 示例
///
/// ```rust
/// use hetero_infer::{EngineConfig, InferenceEngine};
///
/// let config = EngineConfig::default();
/// let engine = InferenceEngine::new(config)?;
/// # Ok::<(), hetero_infer::EngineError>(())
/// ```
pub struct InferenceEngine {
    // ...
}
```

### 命名规范

| 项目 | 规范 | 示例 |
|------|------------|---------|
| 类型（结构体、枚举、trait） | `PascalCase` | `InferenceEngine`, `RequestState` |
| 函数、变量、模块 | `snake_case` | `submit_request`, `block_size` |
| 常量 | `SCREAMING_SNAKE_CASE` | `MAX_BATCH_SIZE` |
| 类型别名 | `PascalCase` | `RequestId`, `TokenId` |

### 错误处理模式

使用 `thiserror` 进行错误类型派生：

```rust
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum MemoryError {
    #[error("物理块耗尽：没有可用的空闲块")]
    OutOfBlocks,

    #[error("序列不存在: {0}")]
    SequenceNotFound(u64),
}
```

## 测试策略

### 测试组织

1. **内联单元测试** - 在每个模块底部的 `#[cfg(test)] mod tests { }` 中
2. **内联属性测试** - 使用 `proptest!` 在 `#[cfg(test)] mod property_tests { }` 中
3. **集成测试** - 在 `tests/integration_tests.rs` 中

### 属性测试格式

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// **Feature: heterogeneous-inference-system, Property N: Description**
        /// *For any* condition, the invariant shall hold.
        /// **Validates: Requirements X.Y**
        #[test]
        fn prop_property_name(
            param in 0u32..100,
        ) {
            // 属性验证
            prop_assert!(condition);
        }
    }
}
```

### 测试覆盖率要求

| 测试类型 | 数量 | 目标 |
|-----------|-------|--------|
| 单元测试 | 78 | 核心模块功能 |
| 属性测试 | 22 | 不变量验证 |
| 集成测试 | 13 | 端到端工作流 |
| 文档测试 | 29 | API 示例 |

## 引用格式

- 需求: `REQ-N` → OpenSpec `### Requirement: [名称]`（中文）
- 属性: `PROP-N` → 包含在 design.md 中
- 场景: 使用 GIVEN/WHEN/THEN 关键字，内容用中文
- **注意**: 结构关键字必须用英文以通过 OpenSpec 验证

## 提交消息格式

使用规范提交：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

类型：
- `feat` - 新功能
- `fix` - Bug 修复
- `docs` - 文档更新
- `style` - 代码格式
- `refactor` - 重构
- `test` - 测试相关
- `chore` - 构建/工具

示例：
```
feat(scheduler): 添加 decode 优先调度

实现调度中 decode 请求优先于 prefill 请求，
以减少进行中请求的延迟。

Refs: REQ-3.7
Closes #123
```

## CI/CD 流水线

### 提交前检查清单

提交前确保：
- [ ] `cargo fmt --check` 通过
- [ ] `cargo clippy --all-targets -- -D warnings` 通过
- [ ] `cargo test` 通过
- [ ] 新功能有测试
- [ ] 公共 API 有文档注释
- [ ] 相关规格已更新

## 获取帮助

- 查看 `src/` 中的现有代码以了解模式
- 查看 `openspec/specs/` 了解设计意图
- 阅读 `CONTRIBUTING.md` 了解详细指南
- 查看 `docs/zh/` 获取用户文档

---

**记住**: 规格是契约，代码是实现。始终先规格，后实现。
