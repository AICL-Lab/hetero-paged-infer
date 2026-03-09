# Hetero-Paged-Infer

[![CI](https://github.com/user/hetero-paged-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/user/hetero-paged-infer/actions/workflows/ci.yml)

异构推理系统 — 基于 PagedAttention 和 Continuous Batching 的 CPU-GPU 协同推理引擎。

## 特性

- **PagedAttention KV Cache** — 分页式显存管理，按需分配/释放物理块，支持 copy-on-write
- **Continuous Batching** — 连续批处理调度器，prefill/decode 分阶段管理，decode 优先调度
- **内存压力感知** — 可配置的内存阈值，自动拒绝新请求防止 OOM
- **CUDA Graph 支持** — decode 阶段可捕获 CUDA Graph 加速重复执行
- **模块化架构** — Tokenizer / Scheduler / GPU Executor / KV Cache Manager 均通过 trait 抽象，便于替换

## 架构

```
┌─────────────────────────────────────────────┐
│              InferenceEngine                │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  │
│  │Tokenizer │  │ Scheduler │  │   GPU    │  │
│  │(encode/  │  │(prefill/  │  │ Executor │  │
│  │ decode)  │  │ decode/   │  │(execute/ │  │
│  │          │  │ complete) │  │ graph)   │  │
│  └──────────┘  └─────┬─────┘  └──────────┘  │
│                      │                       │
│               ┌──────┴──────┐                │
│               │ KV Cache    │                │
│               │ Manager     │                │
│               │ (BlockPool  │                │
│               │  PageTable) │                │
│               └─────────────┘                │
└─────────────────────────────────────────────┘
```

## 项目结构

```
src/
├── lib.rs           # 库入口，模块声明与选择性导出
├── main.rs          # CLI 入口 (clap)
├── config.rs        # EngineConfig 配置、验证、JSON 序列化
├── error.rs         # 错误类型体系 (thiserror)
├── types.rs         # 核心数据结构 (Request, Sequence, ExecutionBatch, ...)
├── kv_cache.rs      # PagedAttention KV Cache 管理器
├── scheduler.rs     # Continuous Batching 调度器
├── tokenizer.rs     # 字符级 Tokenizer (测试用)
├── gpu_executor.rs  # GPU 执行器抽象 + Mock 实现
└── engine.rs        # 推理引擎编排器
tests/
└── integration_tests.rs  # 端到端集成测试
```

## 构建 & 运行

```bash
# 构建
cargo build

# 运行测试
cargo test

# 运行 (带输入)
cargo run -- --input "Hello, world!" --max-tokens 50

# 使用配置文件
cargo run -- --config config.json --input "Hello"
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `block_size` | 16 | 每个物理块容纳的 token 数 |
| `max_num_blocks` | 1024 | 最大物理块数量 |
| `max_batch_size` | 32 | 单次调度最大序列数 |
| `max_num_seqs` | 256 | 系统最大并发序列数 |
| `max_model_len` | 2048 | 模型最大上下文长度 |
| `max_total_tokens` | 4096 | 单批次最大 token 总数 |
| `memory_threshold` | 0.9 | 内存压力阈值 (0.0-1.0) |

## 工程质量

- **选择性导出** — `lib.rs` 使用精确的 `pub use` 避免命名空间污染
- **实例级请求 ID** — 避免全局静态计数器在测试间泄漏
- **完整指标追踪** — `EngineMetrics` 实时记录请求数、完成数、失败数、生成 token 数
- **Property-based Testing** — 使用 proptest 验证不变量 (block count, queue consistency, batch constraints)
- **分层错误体系** — `MemoryError` / `ConfigError` / `ValidationError` / `ExecutionError` / `SchedulerError` → `EngineError`
- **CI** — GitHub Actions: `cargo fmt` + `cargo clippy` + `cargo test`
- **代码风格** — `rustfmt.toml` + `.editorconfig`

## 依赖

| 依赖 | 用途 |
|------|------|
| `thiserror` | 派生错误类型 |
| `serde` / `serde_json` | 配置文件序列化 |
| `clap` | 命令行参数解析 |
| `log` / `env_logger` | 日志 |
| `proptest` (dev) | Property-based 测试 |

## License

MIT
