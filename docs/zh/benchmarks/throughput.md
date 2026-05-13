# 吞吐量

Token 生成吞吐量是 LLM 推理系统的关键指标。

## 吞吐量因素

1. **批次大小**：更大批次分摊内核启动开销
2. **内存带宽**：KV Cache 访问模式影响吞吐量
3. **GPU 利用率**：连续批处理最大化硬件使用

## 连续批处理影响

| 配置 | Tokens/sec | 提升 |
|------|:----------:|:----:|
| 静态批处理 (batch=32) | 1,000 | 基准 |
| 动态批处理 | 1,200 | +20% |
| **连续批处理** | **1,500** | **+50%** |

## 优化技术

### 1. 分块 Prefill

大提示词被分成块，避免阻塞 decode：

```
提示词: 1000 tokens
块大小: 256 tokens

迭代 1: [Decode: 8 seq] + [Prefill 块 1: 256 tokens]
迭代 2: [Decode: 9 seq] + [Prefill 块 2: 256 tokens]
```

### 2. 前缀缓存

公共前缀被缓存和共享。

### 3. CUDA Graphs

对于固定批大小的 decode 阶段，CUDA graphs 消除内核启动开销。

## 相关

- [连续批处理架构](/zh/architecture/continuous-batching)
- [延迟指标](/zh/benchmarks/latency)