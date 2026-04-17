# 配置指南

## 概述

Hetero-Paged-Infer 提供灵活的配置选项，可根据不同硬件和工作负载需求调整推理引擎。

## 配置方法

### 1. 命令行参数

使用 CLI 时：

```bash
cargo run --release -- \
  --block-size 16 \
  --max-num-blocks 1024 \
  --max-batch-size 32 \
  --input "你好，世界！" \
  --max-tokens 100
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | - | 配置文件路径 |
| `--block-size` | 16 | 每物理块 token 数 |
| `--max-num-blocks` | 1024 | 最大物理块数 |
| `--max-batch-size` | 32 | 每批次最大序列数 |
| `--max-num-seqs` | 256 | 最大并发序列数 |
| `--max-model-len` | 2048 | 最大上下文长度 |
| `--max-total-tokens` | 4096 | 每批次最大 token 数 |
| `--memory-threshold` | 0.9 | 内存压力阈值（0.0-1.0） |
| `--input` | - | 推理输入文本 |
| `--max-tokens` | 100 | 最大生成 token 数 |
| `--temperature` | 1.0 | 采样温度 |
| `--top-p` | 0.9 | Top-p 采样参数 |

### 2. 配置文件

创建 `config.json` 文件：

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

加载配置：

```bash
cargo run --release -- --config config.json --input "你好"
```

### 3. 编程方式配置

```rust
use hetero_infer::EngineConfig;

// 默认配置
let config = EngineConfig::default();

// 自定义配置
let config = EngineConfig {
    block_size: 16,
    max_num_blocks: 2048,
    max_batch_size: 64,
    max_num_seqs: 512,
    max_model_len: 4096,
    max_total_tokens: 8192,
    memory_threshold: 0.85,
};

// 使用前验证
config.validate()?;
```

## 配置参数详解

### 块大小（block_size）

每个物理 KV Cache 块存储的 token 数。

- **默认值**：16
- **范围**：1 到 128（建议 2 的幂）
- **影响**：
  - 较小值：内部碎片少，元数据开销高
  - 较大值：内部碎片多，元数据开销低

**建议**：大多数场景使用 16。超长序列可使用 32 或 64。

### 最大块数（max_num_blocks）

KV Cache 池中的物理块总数。

- **默认值**：1024
- **计算**：`max_num_blocks × block_size` = 总 token 容量

**内存计算**：
```
KV Cache 内存 = max_num_blocks × block_size × num_layers × num_heads × head_dim × 2 × sizeof(dtype)
```

**示例**（FP16，32 层，32 头，128 维，block_size=16）：
```
1024 blocks × 16 tokens × 32 layers × 32 heads × 128 dims × 2 (K+V) × 2 bytes = 8,589,934,592 bytes ≈ 8 GB
```

### 最大批次大小（max_batch_size）

单批次处理的最大序列数。

- **默认值**：32
- **权衡**：
  - 较高值：更好的 GPU 利用率，更高延迟
  - 较低值：更低延迟，可能降低吞吐率

**建议**：吞吐优化用 32-64，延迟敏感用 8-16。

### 最大序列数（max_num_seqs）

系统最大并发序列数。

- **默认值**：256
- **目的**：限制请求元数据和调度器状态的内存使用

### 最大模型长度（max_model_len）

支持的最大序列长度（输入 + 输出）。

- **默认值**：2048
- **注意**：超过此限制的请求将被拒绝或截断

### 最大总 Token 数（max_total_tokens）

单批次最大总 token 数。

- **默认值**：4096
- **目的**：防止长序列大批量导致的 OOM

**计算**：
```
sum(len(sequence) for sequence in batch) <= max_total_tokens
```

### 内存阈值（memory_threshold）

触发内存压力的 KV Cache 块比例。

- **默认值**：0.9（90%）
- **范围**：0.0 到 1.0
- **行为**：
  - 低于阈值：接受新 Prefill 请求
  - 高于阈值：拒绝新 Prefill，继续 Decode

**建议**：
- 0.85-0.90：平衡
- 0.95：激进内存使用
- 0.70：保守，适合突发负载

## 生成参数

### 最大 Token 数（max_tokens）

最大生成 token 数。

- **默认值**：100
- **范围**：1 到 (max_model_len - input_length)

### 温度（temperature）

控制采样随机性。

- **默认值**：1.0
- **范围**：0.0 到 2.0
- **行为**：
  - 0.0：贪婪解码（确定性）
  - 0.7：集中、连贯
  - 1.0：平衡
  - >1.0：更随机、更创意

### Top-p（top_p）

核采样阈值。

- **默认值**：0.9
- **范围**：0.0 到 1.0
- **行为**：从累积概率 ≥ top_p 的最小 token 集合中采样

## 配置示例

### 低延迟配置

需要快速响应的应用：

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

### 高吞吐配置

最大请求处理率：

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

### 内存受限配置

有限 GPU 内存（如 4GB）：

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

### 长上下文配置

处理长文档：

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

## 验证规则

引擎创建时验证配置参数：

| 参数 | 验证规则 | 错误信息 |
|------|----------|----------|
| `block_size` | > 0 | "block_size must be greater than 0" |
| `max_num_blocks` | > 0 | "max_num_blocks must be greater than 0" |
| `max_batch_size` | > 0 | "max_batch_size must be greater than 0" |
| `max_num_seqs` | > 0 | "max_num_seqs must be greater than 0" |
| `max_model_len` | ≥ block_size | "max_model_len must be at least block_size" |
| `max_total_tokens` | ≥ max_batch_size | "max_total_tokens must be at least max_batch_size" |
| `memory_threshold` | 0.0 - 1.0 | "memory_threshold must be between 0.0 and 1.0" |

## 性能调优建议

### 内存优化

1. **根据序列模式选择 block_size**
   - 大多数序列约 100 token：用 block_size=16（碎片少）
   - 序列很长：用 block_size=32 或 64

2. **从 GPU 内存计算 max_num_blocks**
   ```
   max_num_blocks = available_gpu_memory / (block_size × token_size)
   ```

3. **使用 memory_threshold 进行准入控制**
   - 较低阈值：更好的 QoS，繁忙时快速失败
   - 较高阈值：更好的利用率，可能排队延迟

### 吞吐优化

1. **增大 max_batch_size**
   - 更大批次 = 更好的 GPU 利用率
   - 超过 64-128 收益递减

2. **调整 max_total_tokens**
   - 应适应典型的批次组成
   - 计算：平均序列长度 × max_batch_size × 1.5

3. **平衡 Decode 和 Prefill 批次**
   - 调度器自动优先 Decode
   - 确保有足够容量用于混合批次

### 延迟优化

1. **减小 max_batch_size**
   - 较小批次 = 更低排队延迟
   - 权衡：吞吐降低

2. **设置合适的 memory_threshold**
   - 尽早拒绝新请求，专注活跃请求
   - 防止重负载下抖动

3. **限制 max_model_len**
   - 较短序列 = 更快处理
   - 匹配实际用例需求

---

*部署说明见 [DEPLOYMENT.md](./DEPLOYMENT.md)。*
