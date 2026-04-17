# Configuration Guide

## Overview

Hetero-Paged-Infer provides flexible configuration options to tune the inference engine for different hardware and workload requirements.

## Configuration Methods

### 1. Command-Line Arguments

When using the CLI:

```bash
cargo run --release -- \
  --block-size 16 \
  --max-num-blocks 1024 \
  --max-batch-size 32 \
  --input "Hello, world!" \
  --max-tokens 100
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | - | Path to configuration file |
| `--block-size` | 16 | Tokens per physical block |
| `--max-num-blocks` | 1024 | Maximum physical blocks |
| `--max-batch-size` | 32 | Maximum sequences per batch |
| `--max-num-seqs` | 256 | Maximum concurrent sequences |
| `--max-model-len` | 2048 | Maximum context length |
| `--max-total-tokens` | 4096 | Maximum tokens per batch |
| `--memory-threshold` | 0.9 | Memory pressure threshold (0.0-1.0) |
| `--input` | - | Input text for inference |
| `--max-tokens` | 100 | Maximum tokens to generate |
| `--temperature` | 1.0 | Sampling temperature |
| `--top-p` | 0.9 | Top-p sampling parameter |

### 2. Configuration File

Create a `config.json` file:

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

Load configuration:

```bash
cargo run --release -- --config config.json --input "Hello"
```

### 3. Programmatic Configuration

```rust
use hetero_infer::EngineConfig;

// Default configuration
let config = EngineConfig::default();

// Custom configuration
let config = EngineConfig {
    block_size: 16,
    max_num_blocks: 2048,
    max_batch_size: 64,
    max_num_seqs: 512,
    max_model_len": 4096,
    max_total_tokens": 8192,
    memory_threshold: 0.85,
};

// Validate before use
config.validate()?;
```

## Configuration Parameters

### Block Size (`block_size`)

Number of tokens stored in each physical KV Cache block.

- **Default**: 16
- **Range**: 1 to 128 (powers of 2 recommended)
- **Impact**: 
  - Smaller values: Less internal fragmentation, higher metadata overhead
  - Larger values: More internal fragmentation, lower metadata overhead

**Recommendation**: 16 for most use cases. Use 32 or 64 for very long sequences.

### Maximum Blocks (`max_num_blocks`)

Total number of physical blocks in the KV Cache pool.

- **Default**: 1024
- **Calculation**: `max_num_blocks × block_size` = total token capacity

**Memory Calculation**:
```
KV Cache Memory = max_num_blocks × block_size × num_layers × num_heads × head_dim × 2 × sizeof(dtype)
```

**Example** (FP16, 32 layers, 32 heads, 128 head_dim, block_size=16):
```
1024 blocks × 16 tokens × 32 layers × 32 heads × 128 dims × 2 (K+V) × 2 bytes = 8,589,934,592 bytes ≈ 8 GB
```

### Maximum Batch Size (`max_batch_size`)

Maximum number of sequences processed in a single batch.

- **Default**: 32
- **Trade-offs**:
  - Higher values: Better GPU utilization, higher latency
  - Lower values: Lower latency, potentially lower throughput

**Recommendation**: 32-64 for throughput optimization, 8-16 for latency-sensitive applications.

### Maximum Sequences (`max_num_seqs`)

Maximum number of concurrent sequences in the system.

- **Default**: 256
- **Purpose**: Limits memory used by request metadata and scheduler state

### Maximum Model Length (`max_model_len`)

Maximum sequence length (input + output) supported.

- **Default**: 2048
- **Note**: Requests exceeding this limit will be rejected or truncated

### Maximum Total Tokens (`max_total_tokens`)

Maximum total tokens in a single batch.

- **Default**: 4096
- **Purpose**: Prevents OOM from large batches with long sequences

**Calculation**:
```
sum(len(sequence) for sequence in batch) <= max_total_tokens
```

### Memory Threshold (`memory_threshold`)

Fraction of KV Cache blocks that triggers memory pressure.

- **Default**: 0.9 (90%)
- **Range**: 0.0 to 1.0
- **Behavior**:
  - Below threshold: Accept new prefill requests
  - Above threshold: Reject new prefill, continue decode

**Recommendation**: 
- 0.85-0.90: Balanced
- 0.95: Aggressive memory usage
- 0.70: Conservative, for bursty workloads

## Generation Parameters

### Maximum Tokens (`max_tokens`)

Maximum number of tokens to generate.

- **Default**: 100
- **Range**: 1 to (max_model_len - input_length)

### Temperature (`temperature`)

Controls randomness in sampling.

- **Default**: 1.0
- **Range**: 0.0 to 2.0
- **Behavior**:
  - 0.0: Greedy decoding (deterministic)
  - 0.7: Focused, coherent
  - 1.0: Balanced
  - >1.0: More random, creative

### Top-p (`top_p`)

Nucleus sampling threshold.

- **Default**: 0.9
- **Range**: 0.0 to 1.0
- **Behavior**: Sample from smallest set of tokens whose cumulative probability ≥ top_p

## Configuration Examples

### Low-Latency Configuration

For applications requiring fast response times:

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

### High-Throughput Configuration

For maximum request processing rate:

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

### Memory-Constrained Configuration

For limited GPU memory (e.g., 4GB):

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

### Long-Context Configuration

For processing long documents:

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

Configuration parameters are validated on engine creation:

| Parameter | Validation Rule | Error Message |
|-----------|----------------|---------------|
| `block_size` | > 0 | "block_size must be greater than 0" |
| `max_num_blocks` | > 0 | "max_num_blocks must be greater than 0" |
| `max_batch_size` | > 0 | "max_batch_size must be greater than 0" |
| `max_num_seqs` | > 0 | "max_num_seqs must be greater than 0" |
| `max_model_len` | ≥ block_size | "max_model_len must be at least block_size" |
| `max_total_tokens` | ≥ max_batch_size | "max_total_tokens must be at least max_batch_size" |
| `memory_threshold` | 0.0 - 1.0 | "memory_threshold must be between 0.0 and 1.0" |

## Performance Tuning Tips

### Memory Optimization

1. **Match block_size to sequence patterns**
   - If most sequences are ~100 tokens, use block_size=16 (less fragmentation)
   - If sequences are very long, use block_size=32 or 64

2. **Calculate max_num_blocks from GPU memory**
   ```
   max_num_blocks = available_gpu_memory / (block_size × token_size)
   ```

3. **Use memory_threshold for admission control**
   - Lower threshold: Better QoS, requests fail fast when busy
   - Higher threshold: Better utilization, potential queuing delays

### Throughput Optimization

1. **Increase max_batch_size**
   - Larger batches = better GPU utilization
   - Diminishing returns beyond 64-128

2. **Tune max_total_tokens**
   - Should accommodate your typical batch composition
   - Consider: avg_sequence_length × max_batch_size × 1.5

3. **Balance decode vs prefill batches**
   - The scheduler automatically prioritizes decode
   - Ensure enough capacity for mixed batches

### Latency Optimization

1. **Reduce max_batch_size**
   - Smaller batches = lower queuing delay
   - Trade-off: lower throughput

2. **Set appropriate memory_threshold**
   - Reject new requests early to focus on active ones
   - Prevents thrashing under heavy load

3. **Limit max_model_len**
   - Shorter sequences = faster processing
   - Match to your actual use case requirements

---

*For deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md).*
