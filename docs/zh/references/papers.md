# 论文引用

本页不是原始文献列表，而是一份阅读指南：每篇论文都说明它与本仓库的具体关系。

## Efficient Memory Management for Large Language Model Serving with PagedAttention

**作者：** Woosuk Kwon 等
**会议：** SOSP 2023
**链接：** [论文](https://arxiv.org/abs/2309.06180) · [参考实现](https://github.com/vllm-project/vllm)

- **它为何与本仓库相关：** 它为分块式 KV Cache 分配提供了最核心的术语和系统论证。
- **它影响了哪个子系统：** `KVCacheManager`、内存管理文档，以及内存效率基准的叙述方式。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了页/块模型、按需增长和面向共享的思路；刻意没有把论文中的生产性能数字直接当成本仓库 benchmark 结果。

## Orca: A Distributed Serving System for Transformer-Based Generative Language Models

**作者：** Gyeongmin Yu 等
**会议：** OSDI 2022
**链接：** [论文](https://arxiv.org/abs/2205.11000)

- **它为何与本仓库相关：** Orca 是理解迭代级调度与连续批处理最直接的背景文献。
- **它影响了哪个子系统：** `Scheduler`、prefill/decode 混合流，以及延迟/吞吐方法学中的措辞。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了调度直觉与请求混合模型；刻意没有把 Orca 的分布式 serving 范围描述成当前仓库已具备的能力。

## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**作者：** Tri Dao 等
**会议：** NeurIPS 2022
**链接：** [论文](https://arxiv.org/abs/2205.14135) · [代码](https://github.com/Dao-AILab/flash-attention)

- **它为何与本仓库相关：** 它解释了为什么讨论注意力内核时应该关注内存流量，而不只是 FLOPs。
- **它影响了哪个子系统：** 未来 GPU executor 的方向，以及文档中对注意力内核瓶颈的讨论方式。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了 IO-aware 优化目标作为设计参考；刻意没有把论文中的加速数字写成本仓库已经复现的结果。

## FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**作者：** Tri Dao
**会议：** ICLR 2024
**链接：** [论文](https://arxiv.org/abs/2307.08691) · [代码](https://github.com/Dao-AILab/flash-attention)

- **它为何与本仓库相关：** 它为未来真实 GPU 后端设定了更明确的性能目标。
- **它影响了哪个子系统：** executor 接口规划，以及对生产差距的比较语言。
- **采纳了什么，以及刻意没有采纳什么：** 采纳了“kernel 层工作划分很重要”的观点；刻意没有宣称本仓库已经实现了 FlashAttention 级别的内核。

## 如何使用本页

请把这些论文与[基准方法学](/zh/benchmarks/methodology)一起阅读。凡是依赖这些论文的说法，都应标成“继承结论”，而不是“本地实测”。
