# Serving 评测方法论（指标口径唯一权威）

本文件定义 `benchmarks/serving/` 全部实验的指标口径与实验协议。
任何进入 README / 结果页 / 面试材料的数字，口径必须与此一致；
不一致的数字禁止引用。

通用测量纪律（环境模板 / 复现要求 / 无实测不写数字）沿用
[`tiny-llm/docs/performance/benchmark-methodology.md`](https://github.com/open-infra-ai/tiny-llm/blob/master/docs/performance/benchmark-methodology.md)
§1-§3，本文不复制，只定义 serving 层增量。

## 1. 指标定义（客户端侧，loadgen 产出）

| 指标 | 口径 | 分位 |
|------|------|------|
| **TTFT** | 响应头就绪（请求体发送完成）→ 首个**非空文本** SSE chunk 的墙钟。空文本前导帧不计，避免扭曲首 token 语义 | p50/p95/p99 |
| **ITL** | 同一 SSE 流相邻非空文本 chunk 的到达间隔（首 chunk 除外） | p50/p95/p99 |
| **TPOT** | 逐请求 `(总生成耗时 − TTFT) / (输出tokens − 1)`，取分布。与 ITL mean 双口径并列（对齐 vLLM 惯例） | p50/p95/p99 |
| **生成吞吐** | 稳态窗口内 Σ输出tokens / Δt（客户端侧）；与引擎侧 `paged_engine_tokens_generated_total` 两次采样差并列报告，差额即客户/网络开销，如实列出 | 均值 |
| **请求吞吐** | 稳态窗口内 completed req/s | 均值 |
| **成功率** | 完成（200 + `[DONE]`）/ 提交。失败归类：`timeout` / `http_429`（内存压力或并发上限的配置性拒绝，单独计数）/ `http_4xx` / `http_5xx` / `connection` / `stream_error`（SSE 内 error 载荷）/ `no_done`（流提前结束，即使已收到部分 chunk 也判失败——成功率不掺水） | 计数 |
| **KV 利用率** | 稳态窗口 1s 采样 `/metrics` 的 `paged_engine_kv_utilization` | mean/max/p95 |
| **调度延迟** | 双口径：(a) Criterion Mock 后端纯调度路径（`benches/`，隔离调度器开销与计算时间）；(b) `paged_engine_step_duration_us`（W5 引入） | 分布 |

**completion_tokens 来源**：优先最终帧 `usage.completion_tokens`
（`tokens_source="usage"`）；缺失时回退非空 chunk 计数
（`tokens_source="chunks"`，报表中必须标注）。

## 2. 负载模型（两种都必须在档）

1. **闭环饱和**（`--mode closed`）：固定并发槽 1/2/4/8，一个请求完成立即补发。
   测最大吞吐与尾延迟。
2. **开环泊松**（`--mode poisson`）：到达率 λ 按指数间隔，请求独立并发。
   λ 取 0.5×/1.0×/2.0× 饱和容量，画 **TTFT p95 vs λ 的 SLO 曲线**——
   这是 serving 岗位最常问的负载形态。

闭环与开环回答不同问题：闭环给"上限"，开环给"给定到达率下的延迟代价"。
只报闭环是 serving 评测的常见缺陷，本体系两者强制并列。

## 3. 数据集

| 数据集 | 用途 |
|--------|------|
| `datasets/synth/{short,work,long}.jsonl` | 受控实验：三档输入长度分布（32-64 / 128-256 / 512-1024 token 级），固定种子可复现；`prompt_tokens` 为 1.35 token/word 估算值，仅用于分布描述 |
| `datasets/synth/smoke.jsonl` | 冒烟：3 条手工 prompt，验证管线连通 |
| ShareGPT-1000 子集（W2 引入） | 真实性交叉验证：长尾输入分布最能暴露调度问题（大 prefill 挤兑小请求）；若获取受限，以合成分布为准并声明 |
| 重复 prefix 集（W5 引入） | prefix caching 专用：同一 system prompt × 200 条不同问句 |

统一参数：`max_tokens=128`、greedy（`temperature=0`，全链路一致）、
条目上限 prompt tokens ≤ `max_model_len − max_tokens`。

## 4. 实验协议

1. **三件套绑定**：每次 run 的 `metadata.json` 必须记录
   双仓 `git rev-parse HEAD`（paged-infer + tiny-llm）、`nvidia-smi` 快照、
   驱动/CUDA 版本、编译选项（`CMAKE_BUILD_TYPE`/CUDA arch）、模型文件路径、
   完整负载参数。`run_sweep.sh` 默认拒绝 dirty worktree（`--allow-dirty`
   显式放行并在 metadata 记录 `dirty: true`）。
2. **预热**：`--warmup-secs ≥ 30`（warmup 流量与测量窗口同负载形态，结果丢弃）。
3. **重复与收敛**：每个 (并发, 分布) 组合跑 3 次，报告均值与 min/max 波动；
   run-to-run 偏差 >10% 视为未收敛，须排查（笔记本卡散热降频是常见原因，
   写入结果说明而非隐藏）。
4. **横向可比**：三个后端（paged-infer / llama-server / vLLM）使用
   同一 `loadgen` 二进制、同一数据集、同一参数矩阵；量化格式差异
   （W8A16 vs Q4_K_M vs FP16）必须在结果表头声明——比值是完整路径差，
   不是同量化对比。
5. **硬件口径**：所有数字标注硬件（如 RTX 3060 Laptop 6GB / 驱动 / CUDA）。
   笔记本卡的功耗墙与散热限制写进口径声明，不外推到桌面/数据中心卡。
6. **负结果归档**：vLLM 启动失败、某并发档 429 风暴、偏差未收敛——
   全部作为结果归档（命令 + 输出 + 原因），不改写不隐藏。

## 5. 结果归档结构

```
results/<date>-<gpu-slug>/
├── metadata.json        # 三件套 + 参数矩阵（schema 见下）
├── <engine>_<mode>_c<N|rate>_<dataset>/
│   ├── per_request.jsonl   # loadgen 逐请求记录
│   └── summary.json        # 分位汇总（loadgen stdout 同构）
├── *.csv                # 跨组合汇总表（plots.py 生成）
└── *.png                # 图表（每张带 commit+日期+硬件 caption）
```

`metadata.json` schema：

```json
{
  "date": "2026-08-30",
  "hardware": {"gpu": "RTX 3060 Laptop", "vram_mib": 6144, "driver": "…", "cuda_toolkit": "12.0"},
  "commits": {"paged_infer": "sha", "tiny_llm": "sha", "dirty": false},
  "build": {"profile": "Release", "cuda_archs": "…"},
  "model": {"path": "…", "backend_quant": "W8A16|Q4_K_M|FP16"},
  "matrix": [{"engine": "…", "mode": "closed|poisson", "concurrency": 4,
              "rate": null, "dataset": "work", "max_tokens": 128,
              "warmup_s": 30, "repeats": 3}]
}
```

## 6. 图表规范（plots.py）

- TTFT/ITL p50/p95/p99 按并发分组条形图（对数轴）
- 吞吐 vs 并发折线（三引擎叠图）
- KV 利用率箱线
- SLO 曲线：TTFT p95 vs λ（泊松档）
- prefix 命中前后 TTFT 散点（W5）
- 每张图 caption：`<engine> @ <commit>, <date>, <gpu>`
