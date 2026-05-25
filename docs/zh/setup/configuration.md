# 配置指南

本页只记录仓库当前真实存在的配置面，不再描述未实现或已删除的能力。

## 当前支持的配置入口

1. 本地运行和启动服务时使用的 CLI 参数
2. 通过 `--config` 加载的 `config.json`
3. 代码中的 `EngineConfig`

当前 HTTP 服务只使用本地 Rust 引擎，不再支持外部命令桥接后端。

## CLI 参数

```bash
cargo run -- \
  --block-size 16 \
  --max-num-blocks 1024 \
  --max-batch-size 32 \
  --max-num-seqs 256 \
  --max-model-len 2048 \
  --max-total-tokens 4096 \
  --memory-threshold 0.9 \
  --input "你好" \
  --max-tokens 32 \
  --temperature 1.0 \
  --top-p 0.9
```

服务模式：

```bash
cargo run -- --serve
```

## JSON 配置

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "max_total_tokens": 4096,
  "memory_threshold": 0.9,
  "max_retry_attempts": 2,
  "tokenizer": {
    "kind": "simple",
    "path": null
  },
  "serving": {
    "host": "127.0.0.1",
    "port": 3000,
    "model_name": "hetero-infer"
  }
}
```

## 当前真实字段

| 字段 | 含义 |
| --- | --- |
| `block_size` | 每个 KV 块容纳的 token 数 |
| `max_num_blocks` | 可用 KV 块总数 |
| `max_batch_size` | 单次调度允许的最大序列数 |
| `max_num_seqs` | 系统允许的最大并发序列数 |
| `max_model_len` | 单请求最大总长度（输入 + 输出） |
| `max_total_tokens` | 单批次允许的最大 token 总数 |
| `memory_threshold` | 新 prefill 请求的准入阈值 |
| `max_retry_attempts` | GPU 超时后的最大重试次数 |
| `tokenizer.kind` | `simple` 或 `huggingface` |
| `tokenizer.path` | 当 `kind = "huggingface"` 时必须提供 |
| `serving.host` | HTTP 服务监听地址 |
| `serving.port` | HTTP 服务监听端口 |
| `serving.model_name` | HTTP API 返回的模型名 |

## 校验规则

- 所有容量相关数值都必须大于 0。
- `memory_threshold` 必须在 `(0.0, 1.0]` 范围内。
- `huggingface` tokenizer 必须提供 `tokenizer.path`。
- `serving.port` 必须大于 0。
- `serving.model_name` 不能为空。

## 代码中使用

```rust
use hetero_infer::EngineConfig;

let config = EngineConfig {
    block_size: 16,
    max_num_blocks: 1024,
    max_batch_size: 32,
    max_num_seqs: 256,
    max_model_len: 2048,
    max_total_tokens: 4096,
    memory_threshold: 0.9,
    ..Default::default()
};

config.validate()?;
# Ok::<(), hetero_infer::ConfigError>(())
```
