# Paged-Infer 路线图

> **当前定位**：Serving 控制面的**架构练习作品**（v0.1.0 脚手架）。
> 计算后端为 CPU 参考执行器 + Mock GPU 执行器，这是有意为之的边界：
> 本仓库的价值在调度器、分页 KV 内存管理与资源不变量，不在计算 kernel。
>
> 本仓库目前处于**低优先级维护状态**：旗舰投入在
> [tiny-llm](https://github.com/AICL-Lab/tiny-llm)，kernel 深度投入在
> [cuflash-attn](https://github.com/AICL-Lab/cuflash-attn)。

## 已完成（v0.1.0）

- [x] PagedAttention 风格 BlockPool + PageTable
- [x] Continuous batching 调度（prefill/decode 动态混合）
- [x] 内存压力感知与 OOM 防护
- [x] OpenAI 兼容服务（/v1/completions、/v1/chat/completions、SSE）
- [x] 资源不变量属性测试（used + free == total 等）

## 阶段 1：巩固（低成本，面试前做一次）

- [ ] README 补一节「调度器设计讲解」：状态机、准入控制、抢占策略（面试讲述用）
- [ ] 属性测试补充：请求取消/失败后的资源归还穷举场景

## 阶段 2：方向选择（二选一，勿两个都做）

**选项 A：与 tiny-llm 对接（推荐，形成 mini-vLLM 故事）**
- [ ] 定义 EngineBackend trait，把 tiny-llm 作为真实执行后端接入
- [ ] 真实模型的端到端 serving：分页 KV + 连续批处理 + 真实 token 生成
- [ ] 并发压测：资源守恒、尾延迟、失败传播

**选项 B：转向主流框架贡献（对求职更高效）**
- [ ] 把本仓库的调度练习转化为对 vLLM / SGLang 的理解与 PR
- [ ] 从调度器/内存管理相关的 good-first-issue 入手
- [ ] 本仓库保持冻结，README 注明"练习作品，真实生产经验见上游贡献"

## 明确不做

- 不在没有真实执行后端前声称任何 serving 性能数字
- 不重复实现 vLLM 已有的完整功能栈
