# 配置指南

Hetero-Paged-Infer 完整配置参考。

## 配置方式

### 1. 命令行参数

```bash
./hetero-infer \
  --block-size 16 \
  --max-num-blocks 1024 \
  --max-batch-size 32 \
  --memory-threshold 0.9 \
  --input "Hello" \
  --max-tokens 100
```

### 2. 配置文件

创建 `config.json`：

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

使用方式：

```bash
./hetero-infer --config config.json --input "Hello"
```

### 3. 环境变量

```bash
export HETERO_BLOCK_SIZE=16
export HETERO_MAX_BLOCKS=1024
export HETERO_MEMORY_THRESHOLD=0.9
```

### 4. 编程方式

```rust
use hetero_infer::EngineConfig;

let config = EngineConfig {
    block_size: 16,
    max_num_blocks: 2048,
    max_batch_size: 64,
    ..Default::default()
};
```

## 配置参考

### 块大小 (Block Size)

```json
{
  "block_size": 16
}
```

- **默认值**：16
- **范围**：1-128
- **影响**：每个物理块包含的 Token 数量

| 大小 | 内存碎片 | 元数据开销 | 适用场景 |
|------|---------|-----------|---------|
| 8 | 低 | 高 | 短序列 |
| 16 | 中等 | 中等 | **通用场景** |
| 32 | 较高 | 低 | 长序列 |

### 最大块数 (Maximum Blocks)

```json
{
  "max_num_blocks": 1024
}
```

内存计算：

```
总内存 = max_num_blocks × block_size × layers × heads × head_dim × 2 × bytes
```

示例 (FP16)：
```
1024 × 16 × 32 × 32 × 128 × 2 × 2 = ~8.6 GB
```

### 批处理配置

```json
{
  "max_batch_size": 32,
  "max_num_seqs": 256,
  "max_total_tokens": 4096
}
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_batch_size | 32 | 每批处理的序列数 |
| max_num_seqs | 256 | 并发序列数上限 |
| max_total_tokens | 4096 | 每批 Token 总数上限 |

### 内存设置

```json
{
  "max_model_len": 2048,
  "memory_threshold": 0.9
}
```

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| max_model_len | 2048 | - | 最大序列长度 |
| memory_threshold | 0.9 | 0.0-1.0 | 准入控制阈值 |

## 生成参数

### Temperature（温度）

```rust
GenerationParams {
    temperature: 1.0,  // 0.0 = 贪婪解码, >1.0 = 更随机
}
```

| 值 | 行为 |
|----|------|
| 0.0 | 贪婪解码（确定性） |
| 0.7 | 聚焦输出 |
| 1.0 | 平衡 |
| 1.5 | 创造性 |

### Top-p（核采样）

```rust
GenerationParams {
    top_p: 0.9,  // 从概率质量最高的 90% 中采样
}
```

### Max Tokens（最大 Token 数）

```rust
GenerationParams {
    max_tokens: 100,
}
```

## 配置预设

### 低延迟（交互式）

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

### 高吞吐（批处理）

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

### 内存受限

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

### 长上下文

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

| 参数 | 规则 | 错误信息 |
|------|------|---------|
| block_size | > 0 | 必须为正数 |
| max_num_blocks | > 0 | 必须为正数 |
| max_batch_size | > 0 | 必须为正数 |
| max_model_len | ≥ block_size | 上下文必须能装入块 |
| memory_threshold | 0.0-1.0 | 必须为小数 |

## 监控配置

```rust
// 启用指标监控
let config = EngineConfig {
    enable_metrics: true,
    metrics_port: 9090,
    ..Default::default()
};
```

## 性能调优

### GPU 优化

```json
{
  "cuda_graph": true,
  "fp16": true,
  "kv_cache_dtype": "fp16"
}
```

### CPU 优化

```json
{
  "num_threads": 8,
  "pin_memory": true
}
```

---

下一步：[架构概览](../architecture/overview.md)
