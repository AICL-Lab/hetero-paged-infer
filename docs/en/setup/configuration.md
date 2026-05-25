# Configuration Guide

This page documents only the configuration surface that exists in the repository today.

## Supported configuration paths

1. CLI flags for local runs and server startup
2. `config.json` loaded through `--config`
3. Programmatic `EngineConfig`

There is no external bridge backend anymore. The HTTP server always uses the local Rust engine.

## CLI flags

```bash
cargo run -- \
  --block-size 16 \
  --max-num-blocks 1024 \
  --max-batch-size 32 \
  --max-num-seqs 256 \
  --max-model-len 2048 \
  --max-total-tokens 4096 \
  --memory-threshold 0.9 \
  --input "hello" \
  --max-tokens 32 \
  --temperature 1.0 \
  --top-p 0.9
```

Server mode:

```bash
cargo run -- --serve
```

## JSON configuration

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

## Fields that actually matter

| Field | Meaning |
| --- | --- |
| `block_size` | Tokens per KV block |
| `max_num_blocks` | Total KV blocks available |
| `max_batch_size` | Max sequences per scheduled batch |
| `max_num_seqs` | Max concurrent sequences in the system |
| `max_model_len` | Max request length including prompt and generation |
| `max_total_tokens` | Max tokens processed in a single batch |
| `memory_threshold` | Admission threshold for new prefills |
| `max_retry_attempts` | Retry count for GPU timeout errors |
| `tokenizer.kind` | `simple` or `huggingface` |
| `tokenizer.path` | Required when `kind = "huggingface"` |
| `serving.host` | Bind host for HTTP server |
| `serving.port` | Bind port for HTTP server |
| `serving.model_name` | Model name returned by HTTP APIs |

## Validation rules

- All numeric capacity limits must be greater than zero.
- `memory_threshold` must be in `(0.0, 1.0]`.
- `tokenizer.path` is required for `huggingface`.
- `serving.port` must be greater than zero.
- `serving.model_name` must not be empty.

## Programmatic usage

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
