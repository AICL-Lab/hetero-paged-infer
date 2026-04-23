# Claude Code 配置

## OpenSpec 集成

本项目使用 OpenSpec 进行规格驱动开发。

### 关键命令

```bash
# 查看规格
npx @fission-ai/openspec@latest list --specs

# 查看变更
npx @fission-ai/openspec@latest list --changes

# 创建变更
npx @fission-ai/openspec@latest new change <名称>

# 验证
npx @fission-ai/openspec@latest validate --all
```

### 工作流

1. 修改前，运行 `/opsx:propose "<想法>"`
2. 审核生成的提案和规格
3. 使用 `/opsx:apply` 实施
4. 使用 `/opsx:archive` 归档

### 规格文件位置

- 活跃规格: `openspec/specs/`
- 活跃变更: `openspec/changes/`
- 已归档变更: `openspec/archive/`
- 项目上下文: `openspec/project.md`
- AI 助手指令: `openspec/AGENTS.md`

## 项目信息

**项目**: Hetero-Paged-Infer - Rust 异构推理引擎

**主要语言**: Rust (2021 Edition)

**注释语言**: 中文

### 构建命令

```bash
cargo build --release
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt
```

### 关键文件

| 文件 | 用途 |
|------|------|
| `src/engine.rs` | 核心推理引擎 |
| `src/scheduler.rs` | Continuous Batching 调度器 |
| `src/kv_cache.rs` | PagedAttention 内存管理 |
| `openspec/specs/heterogeneous-inference/spec.md` | 行为规格 |
| `openspec/specs/heterogeneous-inference/design.md` | 设计文档 |

## OpenSpec 格式说明

OpenSpec 使用英文结构关键字：

- `## ADDED Requirements` - 新增需求
- `## MODIFIED Requirements` - 修改需求
- `## REMOVED Requirements` - 删除需求
- `### Requirement: [名称]` - 需求标题
- `#### Scenario: [描述]` - 场景标题
- `**GIVEN**` / `**WHEN**` / `**THEN**` / `**AND**` - 场景步骤

需求和场景内容使用中文编写。
