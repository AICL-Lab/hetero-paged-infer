# 基准方法学

本页定义文档中性能结论的“证据规则”。

::: warning 口径说明
本页将结论分成三类：
1. 本仓库已测得
2. 基于 mock executor 的模拟结果
3. 来自外部论文或参考项目的结论
:::

## 事实来源

1. 来自 `benches/engine_benchmark.rs` 的 `cargo bench` 输出
2. `tests/integration_tests.rs` 中的引擎/运行时测试
3. 明确标注为 estimate 的架构推导结果

## 如何给结论分类

### 1. 本仓库已测得

只有当读者可以重新运行仓库中的 benchmark 或测试，并观察到同类结果时，才能使用这个标签。

### 2. 基于 mock executor 的模拟结果

当某个说法依赖 `MockGPUExecutor`、调度模拟或架构推演，而不是真实 CUDA 执行时，应使用这个标签。

### 3. 来自外部论文或参考项目的结论

当某个说法来自 PagedAttention、Orca、FlashAttention 等论文，或来自 vLLM、TensorRT-LLM 等成熟系统时，应使用这个标签。

## 复现路径

```bash
cargo bench
cargo test --test integration_tests
```

请在仓库根目录运行这些命令。如果某个 benchmark 页面无法回溯到这些命令，或无法回溯到明确标注的 estimate，那么它就不应该被写成“证明”。

## 当前刻意不宣称的内容

- 真实 CUDA 内核吞吐表
- 生产级 TTFT/ITL 报告
- 在本仓库中执行完成的跨引擎 apples-to-apples 对比

## 建议阅读顺序

- [基准总览](/zh/benchmarks/)
- [内存效率](/zh/benchmarks/memory-efficiency)
- [吞吐量](/zh/benchmarks/throughput)
- [延迟](/zh/benchmarks/latency)
