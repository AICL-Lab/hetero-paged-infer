# 延迟

响应延迟对交互式应用至关重要。

## 延迟组成

- **TTFT (Time to First Token)**：从请求提交到第一个 token 生成的时间
- **ITL (Inter-Token Latency)**：连续 token 之间的时间间隔

## 调度影响

### Decode 优先策略

调度器优先处理 decode 序列以最小化 ITL：

```rust
fn schedule(&mut self) -> SchedulerOutput {
    // Decode 优先：低延迟
    for seq in self.decode_queue.iter() {
        batch.add_decode(seq);
    }
    
    // Prefill 次之：新请求等待
    for seq in self.prefill_queue.iter() {
        if !batch.is_full() {
            batch.add_prefill(seq);
        }
    }
}
```

## 延迟 vs 吞吐量权衡

- **交互应用**：batch ≤ 16 (ITL < 20ms)
- **批处理**：batch = 64 (最大吞吐量)

## 相关

- [吞吐量指标](/zh/benchmarks/throughput)
- [内存管理](/zh/architecture/memory-management)