# AGENTS.md - AI 助手配置

> 本文档作为 AI 编程助手（Claude Code、Cursor 等）的系统提示词。它定义了所有代码变更必须遵循的规范驱动开发工作流。
>
> **[English](AGENTS.md) | [中文](AGENTS.zh.md)**

## 项目理念：规范驱动开发（SDD）

本项目严格遵循**规范驱动开发（Spec-Driven Development）**范式。所有的代码实现必须以 `/specs` 目录下的规范文档为**唯一事实来源（Single Source of Truth）**。

## 目录说明（Directory Context）

| 目录 | 用途 |
|------|------|
| `/specs/product/` | 产品需求文档（PRDs）- 用户故事、验收标准、功能定义 |
| `/specs/rfc/` | 征求意见稿（RFCs）- 技术设计文档、架构决策 |
| `/specs/api/` | API 接口规范 - OpenAPI.yaml、GraphQL schemas、接口契约 |
| `/specs/db/` | 数据库规范 - Schema 定义、数据模型 |
| `/specs/testing/` | 测试规范 - BDD Gherkin 特性文件、属性测试定义 |
| `/docs/` | 用户文档 - 安装指南、教程、部署文档 |
| `/changelog/` | 详细更新日志 - 每个版本的变更记录 |

## AI 助手工作流指令

当你（AI）被要求开发一个新功能、修改现有功能或修复 Bug 时，**必须严格按照以下工作流执行，不可跳过任何步骤**：

### Step 1: 审查与分析（Review & Analyze）

在编写任何代码之前：

1. **必须**阅读 `/specs/` 目录下的相关文档：
   - 产品需求：`/specs/product/`
   - 技术 RFC：`/specs/rfc/`
   - API 定义：`/specs/api/`
   - 测试规范：`/specs/testing/`

2. **如果用户指令与现有规范冲突**：
   - **立即停止编码**
   - 指出冲突点
   - 询问用户是否需要先更新规范

### Step 2: 规范优先（Spec-First Update）

对于新功能或需要修改接口/数据库结构的情况：

1. **必须首先提议修改或创建相应的 Spec 文档**：
   - 新功能 → 创建/更新 `/specs/product/*.md`
   - 架构变更 → 创建新的 RFC，命名格式为 `NNNN-short-description.md`
   - API 变更 → 更新 `/specs/api/openapi.yaml`
   - 数据库变更 → 更新 `/specs/db/` schemas

2. **等待用户确认**规范修改后，才能进入代码编写阶段

3. **创建新 RFC 时**，使用序号命名：
   - `0001-core-architecture.md`
   - `0002-feature-name.md`
   - 以此类推

### Step 3: 代码实现（Code Implementation）

编写代码时：

1. **100% 遵守规范中的定义**：
   - 遵循规范中的变量命名
   - 使用定义的 API 路径和数据类型
   - 实现定义的状态码和错误处理
   - 精确匹配接口契约

2. **不要在代码中擅自添加规范中未定义的功能**（No Gold-Plating）

3. **遵循语言特定的代码规范**：
   - Rust：`cargo fmt`、`cargo clippy`、完整的文档注释
   - 函数、变量、模块使用 `snake_case`
   - 结构体、枚举、trait 使用 `PascalCase`
   - 常量使用 `SCREAMING_SNAKE_CASE`

### Step 4: 测试验证（Test Against Spec）

测试要求：

1. **根据 `/specs/` 中的验收标准编写测试**：
   - 单元测试覆盖单个组件
   - 集成测试覆盖端到端流程
   - 属性测试验证不变量

2. **确保测试覆盖规范中描述的所有边界情况**

3. **属性测试必须验证 RFC 中定义的正确性属性**

4. **使用正确的测试标签格式**：
   ```
   Feature: [feature-name], Property N: [property description]
   ```

## 代码生成规则

### 规则 1: API 变更
任何对外部暴露的 API 变更**必须**同步修改 `/specs/api/` 文档。

### 规则 2: 架构决策
对于不确定的技术细节，请查阅 `/specs/rfc/` 下的架构约定。**不要自行捏造设计模式**。

### 规则 3: 无规范，不代码
如果没有对应请求变更的规范，**先创建规范**。永远不要在没有相应规范文档的情况下编写实现代码。

### 规则 4: 测试覆盖
所有新代码必须有对应的测试，如 `/specs/testing/` 中所规定。

### 规则 5: 文档同步
当代码变更时，更新相关文档：
- API 变更 → 更新 `/docs/en/API.md` 和 `/docs/zh/API.md`
- 配置变更 → 更新 `/docs/en/CONFIGURATION.md` 和 `/docs/zh/CONFIGURATION.md`
- 架构变更 → 更新 `/docs/en/ARCHITECTURE.md` 和 `/docs/zh/ARCHITECTURE.md`

## 项目特定约定

### 文件组织

```
src/
├── lib.rs           # 库入口点
├── main.rs          # CLI 入口点
├── config.rs        # 配置
├── engine.rs        # 推理引擎
├── error.rs         # 错误类型
├── types.rs         # 核心类型
├── kv_cache.rs      # KV Cache 管理器
├── scheduler.rs     # 调度器
├── tokenizer.rs     # 分词器
└── gpu_executor.rs  # GPU 执行器

tests/
└── integration_tests.rs  # 集成测试

specs/
├── README.md        # 规范概览
├── product/         # 产品需求
├── rfc/             # 技术设计文档
├── api/             # API 规范
├── db/              # 数据库 schemas
└── testing/         # BDD 测试规范
```

### Commit 消息格式

使用约定式提交格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**类型**：
- `feat` - 新功能
- `fix` - Bug 修复
- `docs` - 文档更新
- `style` - 代码格式化（无功能变更）
- `refactor` - 重构
- `test` - 测试相关
- `chore` - 构建/工具相关

**示例**：
```
feat(scheduler): add decode priority scheduling

Implement decode request priority over prefill requests in scheduling
to reduce latency of in-progress requests.

Refs: REQ-3.7
Closes #123
```

### PR 检查清单

在提交 PR 之前，确保：

- [ ] 代码通过 `cargo fmt --check`
- [ ] 代码通过 `cargo clippy --all-targets -- -D warnings`
- [ ] 所有测试通过 `cargo test`
- [ ] 新功能有对应的测试
- [ ] 公共 API 有文档注释
- [ ] `/specs/` 中的相关规范已更新
- [ ] `/docs/` 中的文档已更新

## 快速参考

### 常用命令

```bash
# 构建
cargo build --release

# 运行所有测试
cargo test

# 运行属性测试
cargo test -- --test-threads=1

# 运行覆盖率测试
cargo tarpaulin --out Html

# 格式检查
cargo fmt --check

# Lint 检查
cargo clippy --all-targets -- -D warnings

# 运行基准测试
cargo bench

# 生成文档
cargo doc --no-deps --open
```

### 测试覆盖目标

| 类型 | 数量 | 覆盖范围 |
|------|------|----------|
| 单元测试 | 78 | 核心模块 |
| 属性测试 | 15 | 不变量验证 |
| 集成测试 | 13 | 端到端流程 |
| 文档测试 | 29 | API 示例 |

### 需求引用格式

在实现或测试时，通过 ID 引用需求：
- 功能需求：`REQ-1`、`REQ-2.3`
- 正确性属性：`PROP-5`、`PROP-12`
- RFC 章节：`RFC-0001§2.3`

## 为什么这很重要

### 防范 AI 幻觉
AI 很容易在没有上下文的情况下"自由发挥"。强制它第一步读取 `/specs` 可以锚定其思考范围，防止虚构实现。

### 约束修改路径
通过声明"修改代码前先改规范"，保证了文档与代码永远同步（Document-Code Synchronization）。这防止了文档与实现之间的偏差。

### 提高 PR 质量
当 AI 帮你生成 Pull Request 时，实现会与业务逻辑高度一致，因为它是根据规范中定义的验收标准进行开发的。

### 支持代码审查
规范为审查代码变更提供了客观标准。审查者可以验证实现是否匹配规范，而不是依赖主观意见。

---

**记住**：规范是契约。代码是实现。契约永远优先。
