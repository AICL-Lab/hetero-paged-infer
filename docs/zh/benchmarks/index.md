# 性能基准

本节把性能说明写成一个“证据系统”：告诉读者哪些结论已经在仓库中测得，哪些只是基于 mock executor 的模拟，哪些来自外部论文或参考项目。

::: warning 口径说明
本页将结论分成三类：
1. 本仓库已测得
2. 基于 mock executor 的模拟结果
3. 来自外部论文或参考项目的结论
:::

## 证据阶梯

| 类别 | 在本项目中的含义 | 当前例子 |
|------|------------------|----------|
| 本仓库已测得 | 可以通过 `cargo bench` 或运行时/集成测试复现 | 引擎创建、调度 step、批处理循环、KV Cache 分配与释放 |
| 基于 mock executor 的模拟结果 | 依赖 mock executor 代替真实 GPU 内核时观察到的行为 | decode 优先调度、prefill/decode 混合批次、graph capture API 形状 |
| 来自外部论文或参考项目的结论 | 明确借用论文或成熟系统的结论 | PagedAttention 的内存碎片改善、真实 CUDA 内核的吞吐预期 |

## 当前已经证明的内容

- 仓库在 `benches/engine_benchmark.rs` 中提供了 Criterion 基准。
- `tests/integration_tests.rs` 覆盖了请求完成、内存跟踪与连续批处理。
- 默认引擎仍然使用 `MockGPUExecutor`，因此这里**不会**宣称已经得到生产级 GPU 吞吐或延迟数字。

## 如何阅读这些基准页面

1. 先看 [方法学](/zh/benchmarks/methodology)，理解哪些证据可以成立。
2. 阅读每个页面时按“证据类别”理解，而不是把所有数字都当成同等强度的结论。
3. 如果某条说法依赖先验工作，请继续查看 [对比](/zh/comparison/) 与 [参考](/zh/references/)。

## 基准页面导航

- [方法学](/zh/benchmarks/methodology) — 证据来源与复现实践
- [内存效率](/zh/benchmarks/memory-efficiency) — 哪些内存结论已测得，哪些只是推导
- [吞吐量](/zh/benchmarks/throughput) — 当前吞吐证据能证明什么
- [延迟](/zh/benchmarks/latency) — 当前延迟说法的边界
