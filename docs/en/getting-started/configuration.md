# Configuration Guide

Complete configuration reference for Hetero-Paged-Infer.

## Configuration Methods

### 1. Command-Line Arguments

```bash
./hetero-infer \
  --block-size 16 \
  --max-num-blocks 1024 \
  --max-batch-size 32 \
  --memory-threshold 0.9 \
  --input "Hello" \
  --max-tokens 100
```

### 2. Configuration File

Create `config.json`:

```json
{
  "block_size": 16,
  "max_num_blocks": 1024,
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_model_len": 2048,
  "max_total_tokens": 4096,
  "memory_threshold": 0.9
}
```

Use with:

```bash
./hetero-infer --config config.json --input "Hello"
```

### 3. Environment Variables

```bash
export HETERO_BLOCK_SIZE=16
export HETERO_MAX_BLOCKS=1024
export HETERO_MEMORY_THRESHOLD=0.9
```

### 4. Programmatic

```rust
use hetero_infer::EngineConfig;

let config = EngineConfig {
    block_size: 16,
    max_num_blocks: 2048,
    max_batch_size: 64,
    ..Default::default()
};
```

## Configuration Reference

### Block Size

```json
{
  "block_size": 16
}
```

- **Default**: 16
- **Range**: 1-128
- **Impact**: Number of tokens per physical block

| Size | Fragmentation | Metadata | Best For |
|------|--------------|----------|----------|
| 8 | Low | High | Short sequences |
| 16 | Medium | Medium | **General use** |
| 32 | Higher | Low | Long sequences |

### Maximum Blocks

```json
{
  "max_num_blocks": 1024
}
```

Memory calculation:

```
Total Memory = max_num_blocks × block_size × layers × heads × head_dim × 2 × bytes
```

Example (FP16):
```
1024 × 16 × 32 × 32 × 128 × 2 × 2 = ~8.6 GB
```

### Batch Configuration

```json
{
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_total_tokens": 4096
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_batch_size | 32 | Sequences per batch |
| max_num_seqs | 256 | Concurrent sequences |
| max_total_tokens | 4096 | Tokens per batch |

### Memory Settings

```json
{
  "max_model_len": 2048,
  "memory_threshold": 0.9
}
```

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| max_model_len | 2048 | - | Max sequence length |
| memory_threshold | 0.9 | 0.0-1.0 | Admission control threshold |

## Generation Parameters

### Temperature

```rust
GenerationParams {
    temperature: 1.0,  // 0.0 = greedy, >1.0 = random
}
```

| Value | Behavior |
|-------|----------|
| 0.0 | Greedy decoding |
| 0.7 | Focused |
| 1.0 | Balanced |
| 1.5 | Creative |

### Top-p (Nucleus Sampling)

```rust
GenerationParams {
    top_p: 0.9,  // Sample from top 90% probability mass
}
```

### Max Tokens

```rust
GenerationParams {
    max_tokens: 100,
}
```

## Configuration Presets

### Low Latency (Interactive)

```json
{
  "block_size": 16,
  "max_num_blocks": 512,
  "max_batch_size": 8,
  "max_num_seqs": 64,
  "max_model_len": 1024,
  "max_total_tokens": 1024,
  "memory_threshold": 0.8
}
```

### High Throughput (Batch Processing)

```json
{
  "block_size": 32,
  "max_num_blocks": 4096,
  "max_batch_size": 128,
  "max_num_seqs": 1024,
  "max_model_len": 4096,
  "max_total_tokens": 16384,
  "memory_threshold": 0.95
}
```

### Memory Constrained

```json
{
  "block_size": 16,
  "max_num_blocks": 256,
  "max_batch_size": 16,
  "max_num_seqs": 128,
  "max_model_len": 1024,
  "max_total_tokens": 2048,
  "memory_threshold": 0.85
}
```

### Long Context

```json
{
  "block_size": 32,
  "max_num_blocks": 2048,
  "max_batch_size": 16,
  "max_num_seqs": 64,
  "max_model_len": 8192,
  "max_total_tokens": 8192,
  "memory_threshold": 0.9
}
```

## Validation Rules

| Parameter | Rule | Error |
|-----------|------|-------|
| block_size | > 0 | Must be positive |
| max_num_blocks | > 0 | Must be positive |
| max_batch_size | > 0 | Must be positive |
| max_model_len | ≥ block_size | Context must fit blocks |
| memory_threshold | 0.0-1.0 | Must be fraction |

## Monitoring Configuration

```rust
// Enable metrics
let config = EngineConfig {
    enable_metrics: true,
    metrics_port: 9090,
    ..Default::default()
};
```

## Performance Tuning

### GPU Optimization

```json
{
  "cuda_graph": true,
  "fp16": true,
  "kv_cache_dtype": "fp16"
}
```

### CPU Optimization

```json
{
  "num_threads": 8,
  "pin_memory": true
}
```

---

Next: [Architecture Overview](../architecture/overview.md)
