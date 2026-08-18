# Paged-Infer 路线图

> **当前定位**：Serving 控制面的**架构练习作品**（v0.2.0，`phase-2-e` 冻结）。
> 计算后端双路径：默认 CPU 参考执行器（CI/确定性）；`tiny-llm` feature 下接入
> 真实 CUDA 后端，**分页 KV 策略 1（block_tables）默认启用**。本仓库的价值在
> 调度器、分页 KV 内存管理与资源不变量，不在计算 kernel。

## 已完成（v0.1.0）

- [x] PagedAttention 风格 BlockPool + PageTable
- [x] Continuous batching 调度（prefill/decode 动态混合）
- [x] 内存压力感知与 OOM 防护
- [x] OpenAI 兼容服务（/v1/completions、/v1/chat/completions、SSE）
- [x] 资源不变量属性测试（used + free == total 等）

## P0 正确性修复（已完成）

- [x] T0：fmt / clippy 恢复 CI 绿色
- [x] T1：执行输出契约校验（坏后端不得卡死请求）
- [x] T2：调度器内存水位线 + decode 增长预留
- [x] T3：修复 pending 队头阻塞（HOL）
- [x] T4：后端序列生命周期钩子 + tiny-llm KV 释放
- [x] T5：超时重试声明幂等性
- [x] T6：浮点参数 NaN 校验
- [x] T7：Unicode stop 序列字节偏移修复
- [x] T8：文档与代码事实对齐

## P1（部分完成）

- [x] T9：Chat Completions 应用真实 chat template（Qwen2 `<|im_start|>`，HF tokenizer）
- [x] T10：引擎指标 /metrics + CB on/off benchmark
- [x] T11：tiny-llm 策略 1（分页 KV C ABI，跨仓库）——ABI v2 + 真实块表，默认策略 1；llama.cpp 逐 token 对齐 + 3 并发 e2e 通过（2026-08-18）
- [x] T12：明确降级（选项 A）——保留 `BufferedDecoder`，文档明确 HF tokenizer
  流式为"请求结束时的一个完整文本 chunk"，"token-level streaming" 表述限定
  `SimpleTokenizer`（T8 已完成）

## 阶段 1：巩固（低成本，面试前做一次）

- [x] README 补一节「调度器设计讲解」：状态机、准入控制、抢占策略（面试讲述用）
- [x] 属性测试补充：请求取消/失败后的资源归还穷举场景

## 阶段 2：方向选择（二选一，勿两个都做）

**选项 A：与 tiny-llm 对接（推荐，形成 mini-vLLM 故事）**
- [x] 定义 EngineBackend trait，把 tiny-llm 作为真实执行后端接入
      （`GPUExecutorTrait` + `TinyLlmExecutor` 适配器，C ABI 经 `tiny_llm_ffi`）
- [x] 真实模型的端到端 serving：分页 KV + 连续批处理 + 真实 token 生成
      - 引擎驱动 + 真实后端已验证（`cargo test --features tiny-llm`，
        3 并发请求、KV 生命周期、资源守恒、能力声明）
      - tokenizer 词表对齐：`--tokenizer <tokenizer.json>` 启用 HF tokenizer
        （真实 BOS/EOS/PAD 探测），差分测试与 tiny-llm 权威 fixture 逐 id 对齐
        （`tests/tokenizer_real_diff.rs`、`tests/tiny_llm_text_e2e.rs`）
      - **分页 KV 策略 1 默认启用**（真实 `block_tables`）；
        `PAGED_INFER_TINY_LLM_STRATEGY=2` 可回退连续 KV。请求 1 与 llama.cpp
        greedy 全序列对齐；请求 2 因 W8A16 vs Q4_K_M 量化分歧，断言为
        前缀一致 + EOS（见 `tiny_llm_text_e2e.rs`）
- [x] 并发压测：资源守恒、尾延迟、失败传播
      （Mock/CPU 后端先行，真实 tiny-llm 后端接入后直接切换 executor 复用场景：
      `tests/concurrency_stress.rs` 断言 + `benches/concurrency_benchmark.rs` 性能基线）

**选项 B：转向主流框架贡献（对求职更高效）**
- [ ] 把本仓库的调度练习转化为对 vLLM / SGLang 的理解与 PR
- [ ] 从调度器/内存管理相关的 good-first-issue 入手
- [ ] 本仓库保持冻结，README 注明"练习作品，真实生产经验见上游贡献"

## 明确不做（冻结边界）

- 不实现抢占（vLLM 式 swap / preempt-resume）
- 不实现 chunked prefill
- 不实现 prefix caching
- 不实现生产级 CUDA kernel（真实 kernel 在 tiny-llm 仓库）
- 不声称生产级 serving 吞吐数字（tiny-llm 后端用于正确性与资源守恒，不是压测旗舰）
- 不重复实现 vLLM 已有的完整功能栈

后续工作指向（冻结期内只记不做）：

- tiny-llm：paged decode 直接读 pool、分页路径接 CUDA Graphs（Phase 4 候选）
- 上游：把调度练习转化为 vLLM / SGLang 小 PR（选项 B）
