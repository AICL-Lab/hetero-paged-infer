# 延迟

延迟页面最需要严守口径：仓库已经有调度策略和测试，但还没有生产级的 wall-clock 延迟报告。

::: warning 口径说明
本页将结论分成三类：
1. 本仓库已测得
2. 基于 mock executor 的模拟结果
3. 来自外部论文或参考项目的结论
:::

## 本仓库已测得

- 集成测试验证了请求完成、prefill/decode 混合执行，以及与内存压力相关的控制流。
- 仓库目前**没有**发布基于真实 GPU 执行得到的 TTFT 或 ITL 实测表。
- `cargo bench` 覆盖了编排开销，但它不等于用户可见的服务延迟。

## 基于 mock executor 的模拟结果

- 调度器采用 decode 优先，这是仓库当前用于保护在途请求延迟的核心策略。
- 内存阈值处理逻辑的目标是在接近 OOM 之前暂停新 prefill。
- GPU executor trait 暴露了 CUDA graph capture 接口，但当前 mock executor 会复用普通执行路径，因此本地尚未证明 graph 带来的延迟收益。

## 来自外部论文或参考项目的结论

- 分块 prefill、CUDA graphs 等低延迟技巧有外部系统文献支持。
- 它们解释了架构为何这样设计，但不能被直接读成本仓库已经测得的 TTFT/ITL 数字。

## 当前证明状态

| 问题 | 当前答案 |
|------|----------|
| 仓库是否具备面向低延迟的调度策略？ | 是，代码与集成测试中都有体现 |
| 仓库是否发布了生产级 TTFT/ITL 实测数据？ | 否 |
| 外部低延迟数字能否直接当成本仓库 benchmark 结果？ | 否 |

## 本页值得评估什么

- 延迟控制机制是否已经存在并被文档化。
- 文档是否明确把未测得的 GPU 结论标成未来工作。
- 当真实内核接入后，是否已经为新增测量结果预留了清晰位置。

## 相关链接

- [基准方法学](/zh/benchmarks/methodology)
- [吞吐量](/zh/benchmarks/throughput)
- [内存管理](/zh/architecture/memory-management)
- [论文阅读指南](/zh/references/papers)
