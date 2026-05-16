---
title: Hetero-Paged-Infer
---

<FlagshipHero
  eyebrow="Rust LLM Serving Engine"
  title="把分页注意力与连续批处理做成可讲清楚、可验证、可展示的项目"
  summary="Hetero-Paged-Infer 不是只做一个功能列表式 Demo，而是把现代推理引擎的关键思想——分页式 KV Cache、Decode 优先调度、OpenAI 兼容服务、测试与工程边界——整理成一个可以面向面试官与社区读者展开说明的技术白皮书站。"
  primary-text="阅读白皮书"
  primary-href="/zh/whitepaper/"
  secondary-text="查看架构"
  secondary-href="/zh/architecture/overview"
/>

<ProofStrip :items="[
  { label: '核心语言', value: 'Rust', detail: '以 Trait 抽象、内存安全与系统工程可读性为基础。' },
  { label: '内存模型', value: '分页式', detail: '用固定块管理 KV Cache，减少浪费并保留明确的内存账本。' },
  { label: '服务模型', value: '连续批处理', detail: '以 Decode 优先策略组织批次，并提供 OpenAI 兼容接口。' },
  { label: '质量门槛', value: 'CI + Tests', detail: '公开叙述需要同时能在代码、文档与基准中对上。' },
]" />

## Why this project matters｜为什么这个项目值得关注

很多开源推理引擎要么非常强悍但难以学习，要么容易阅读却缺少系统设计的力度。Hetero-Paged-Infer 想占据中间位置：既足够清楚，适合学习；又足够严谨，能建立技术信任。

## What makes it distinctive｜它的独特之处

1. 以 Rust 为第一叙事语言的系统工程视角
2. 明确区分控制面与计算面
3. 比典型兴趣型推理仓库更强调测试与类型安全
4. 用白皮书式说明替代功能清单式营销

## Why now｜为什么是现在

今天讨论 LLM Serving，已经不能只看吞吐数字。读者、评审者和未来贡献者同样关心：内存到底怎么管理，为什么 Decode 请求优先，接口设计是否真的靠近部署现实。

Hetero-Paged-Infer 的价值在于，它把这些问题直接纳入产品表达本身。首页不是一个简单的跳转目录，而是在说明：架构、证据与文档应该彼此支撑，才能形成可信的技术叙事。

- 通过[白皮书](/zh/whitepaper/)理解问题定义与技术立场。
- 通过[架构章节](/zh/architecture/overview)查看控制面与计算面的组织方式。
- 通过[性能基准](/zh/benchmarks/)区分当前证据与未来性能目标。

## Architecture at a glance｜架构速览

这个引擎被组织成一组可以单独理解的子系统，而不是一个难以解释的大文件。你可以顺着请求进入编排器，再看到调度器如何混排 Prefill 与 Decode，最后理解 KV Cache 管理器怎样以固定块追踪内存。

<SectionGrid :cards="[
  { title: '系统概览', summary: '先看控制面 / 计算面的分工，以及整体执行循环如何组织。', href: '/zh/architecture/overview' },
  { title: 'PagedAttention', summary: '理解基于块的 KV 分配如何减少浪费，并让内存账本更清楚。', href: '/zh/architecture/paged-attention' },
  { title: '连续批处理', summary: '查看 Decode 优先的调度模型，以及批处理槽位如何动态填充。', href: '/zh/architecture/continuous-batching' },
  { title: '设计说明', summary: '阅读更完整的架构取舍、约束与后续技术方向。', href: '/zh/architecture/design' },
]" />

## Proof, not slogans｜先看证据，不看口号

这个项目最有说服力的时候，不是喊出更多口号，而是让一个判断可以从多个方向被核对：实现结构、基准备注、API 文档，以及与前人工作的关联。首页因此更强调可追溯性，而不是宣传语气。

<SectionGrid :cards="[
  { title: '性能基准', summary: '查看内存、吞吐与延迟页面，并注意当前 Mock GPU 证据的边界。', href: '/zh/benchmarks/' },
  { title: 'API 接口', summary: '检查 OpenAI 兼容接口，以及核心请求与响应类型定义。', href: '/zh/api/' },
  { title: '项目对比', summary: '把它放回推理引擎语境中理解，而不是假装所有能力都已完成。', href: '/zh/comparison/' },
  { title: '参考资料', summary: '沿着论文与相关项目，追溯设计依据与技术来源。', href: '/zh/references/' },
]" />

## Where this project is already strong｜这个项目已经很强的地方

- **系统拆解清楚** —— 引擎、调度器、KV Cache 管理器、GPU 执行器都被当成有明确职责的架构单元。
- **文档强调证据链** —— 白皮书、架构说明与基准备注更像技术审阅材料，而不只是功能浏览页。
- **Rust-first 的设计纪律** —— Trait 与显式接口让控制流程更容易测试，也更容易推理。
- **贴近真实 Serving 问题** —— OpenAI 兼容服务、批处理行为、内存压力这些主题都被当作工程问题认真处理。

## Where it is still intentionally incomplete｜它仍然有意保持未完成的部分

这里并没有把自己包装成一个已经收尾的生产级 Serving 平台。它的一部分价值，恰恰来自对未完成部分的坦诚。

- 真实硬件上的 GPU 性能工作，与当前基于 Mock Executor 的基准叙事仍然是分开的。
- 部署与生产强化更多是清晰列出的方向，而不是已经完全兑现的保证。
- 文档站正在继续扩展成更完整的白皮书与参考系统，因此不同章节目前成熟度并不完全一致。

这种坦诚本身就是旗舰叙事的一部分：帮助读者分清哪些已经被证明，哪些仍然是有意识保留的下一步。

## Suggested reading path｜建议阅读路径

如果你第一次进入这个项目，建议把它当成一个技术案例来阅读，而不是一个产品宣传册。

<SectionGrid :cards="[
  { title: '1. 白皮书', summary: '先理解这个引擎为什么存在、如何定位、边界在哪里。', href: '/zh/whitepaper/' },
  { title: '2. 架构', summary: '继续进入子系统布局，理解调度器、缓存与执行器之间的控制流。', href: '/zh/architecture/overview' },
  { title: '3. 性能基准', summary: '确认当前已经给出的证据，以及明确推迟的部分。', href: '/zh/benchmarks/' },
  { title: '4. 项目对比', summary: '查看它如何在相邻推理系统中为自己定位。', href: '/zh/comparison/' },
  { title: '5. 参考资料', summary: '最后回到论文与项目来源，理解设计依据。', href: '/zh/references/' },
]" />
