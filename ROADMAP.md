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

- [x] README 补一节「调度器设计讲解」：状态机、准入控制、抢占策略（面试讲述用）
- [x] 属性测试补充：请求取消/失败后的资源归还穷举场景

## 阶段 2：方向选择（二选一，勿两个都做）

**选项 A：与 tiny-llm 对接（推荐，形成 mini-vLLM 故事）**
- [x] 定义 EngineBackend trait，把 tiny-llm 作为真实执行后端接入
      （`GPUExecutorTrait` + `TinyLlmExecutor` 适配器，C ABI 经 `tiny_llm_ffi`）
- [~] 真实模型的端到端 serving：分页 KV + 连续批处理 + 真实 token 生成
      - 引擎驱动 + 真实后端的接入流程已验证（`cargo test --features tiny-llm`，
        3 并发请求、KV 生命周期、资源守恒、能力声明）
      - tokenizer 词表对齐已完成：`--tokenizer <tokenizer.json>` 启用 HF tokenizer
        （真实 BOS/EOS/PAD 探测），差分测试与 tiny-llm 权威 fixture 逐 id 对齐，
        真实后端端到端与 llama.cpp 同 prompt greedy 输出逐 token 一致
        （`tests/tokenizer_real_diff.rs`、`tests/tiny_llm_text_e2e.rs`）
      - 待完善：分页 KV（策略 1）暂未启用（当前策略 2 连续 KV）
- [x] 并发压测：资源守恒、尾延迟、失败传播
      （Mock/CPU 后端先行，真实 tiny-llm 后端接入后直接切换 executor 复用场景：
      `tests/concurrency_stress.rs` 断言 + `benches/concurrency_benchmark.rs` 性能基线）

**选项 B：转向主流框架贡献（对求职更高效）**
- [ ] 把本仓库的调度练习转化为对 vLLM / SGLang 的理解与 PR
- [ ] 从调度器/内存管理相关的 good-first-issue 入手
- [ ] 本仓库保持冻结，README 注明"练习作品，真实生产经验见上游贡献"

## 明确不做

- 不在没有真实执行后端前声称任何 serving 性能数字
- 不重复实现 vLLM 已有的完整功能栈
