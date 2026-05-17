# 内存效率

本页聚焦于仓库目前究竟能证明哪些 KV Cache 行为，而不是直接复述外部论文中的漂亮数字。

::: warning 口径说明
本页将结论分成三类：
1. 本仓库已测得
2. 基于 mock executor 的模拟结果
3. 来自外部论文或参考项目的结论
:::

## 本仓库已测得

### 已被基准和测试覆盖的内容

- `cargo bench` 会在 `benches/engine_benchmark.rs` 中测量 `allocate_sequence` 与 `allocate_block` 路径。
- `tests/integration_tests.rs` 会检查内存利用率在分配后上升、请求完成后下降。
- 端到端请求流测试要求工作完成后释放 KV Cache 状态。

### 这能证明什么

| 结论 | 证据类别 | 当前证明方式 |
|------|----------|--------------|
| KV Cache 的分配与扩容路径存在且可被基准覆盖 | 本仓库已测得 | Criterion 覆盖了高分配频率路径 |
| 引擎可以观测内存利用率 | 本仓库已测得 | 集成测试断言负载变化会改变利用率 |
| 请求完成后会释放工作状态 | 本仓库已测得 | 端到端测试要求运行结束后利用率较低 |

## 基于 mock executor 的模拟结果

- 仓库中的分块分配器与页表模型意味着：每个活跃序列的浪费上界主要来自最后一个未填满的块。
- Copy-on-write 共享与引用计数支持“分支工作负载更省内存”的设计论证。
- 这些都是有架构依据的推导，但在收集真实 GPU 内存轨迹之前，它们仍然属于**推导**而不是实测。

## 来自外部论文或参考项目的结论

- 常见的“朴素分配浪费 40-60%，PagedAttention 可降到 <5%”这一表述，来源于 vLLM/PagedAttention 论文及其相关文献，而不是本仓库本地实测。
- 这些数字仍然适合帮助读者建立直觉，但应当视为外部继承结论，而不是本项目已经完成的 benchmark 结果。

## 本页今天真正证明的内容

- 内存管理设计已经落地并被执行。
- 仓库已有分配行为与清理行为的证据。
- 目前**还没有**真实 VRAM 采样或硬件级碎片分析报告。

## 相关链接

- [基准方法学](/zh/benchmarks/methodology)
- [PagedAttention 架构](/zh/architecture/paged-attention)
- [内存管理](/zh/architecture/memory-management)
- [论文阅读指南](/zh/references/papers)
