# Serving 评测体系

对 paged-serving（及对照基线 llama-server / vLLM）做可复现的 serving 级
负载实验。**指标口径与方法论的唯一权威是
[`methodology.md`](methodology.md)**——任何数字引用必须与其口径一致。

## 目录结构

```
benchmarks/serving/
├── README.md              # 本文件：入口与运行方法
├── methodology.md         # 指标口径 + 实验协议（唯一权威）
├── datasets/
│   └── synth/
│       ├── gen_synth.py   # 三档合成分布生成器（固定种子）
│       ├── smoke.jsonl    # 3 条冒烟 prompt
│       └── {short,work,long}.jsonl   # gen_synth.py 产出
├── run_sweep.sh           # 矩阵编排：dirty 检查 + 双层 metadata + loadgen
├── plots.py               # 只读取权威 summary.json 生成图表
├── validate_results.py    # 检查正式结果的必需产物与 JSON schema
├── RESULT_REPORT_TEMPLATE.md # 人工结论、限制与复现命令模板
└── results/<date>-<gpu>/  # 原始请求 + run summary + 环境/模型 metadata + 图表
```

压测客户端是 [`src/bin/loadgen.rs`](../../src/bin/loadgen.rs)（闭环饱和 /
开环泊松双模式，同一二进制零改动覆盖三个后端）。

## 快速开始

```bash
# 0. 构建（仓库根目录）
cargo build --release --bin loadgen
# 真实后端服务（可选；CPU 参考后端可直接 --serve）
cargo build --release --features tiny-llm   # 需 TINY_LLM_DIR

# 1. 生成数据集
python3 benchmarks/serving/datasets/synth/gen_synth.py \
    --outdir benchmarks/serving/datasets/synth

# 2. 启动被测服务（三选一）
# paged-serving（tiny-llm 真实后端）：
#   PAGED_SERVING_TINY_LLM_MAX_SEQS=8 ./target/release/paged-serving --serve \
#       --port 3000 --tokenizer <tokenizer.json> ...（模型经配置加载）
# llama-server（基线）：
#   llama-server -m <model.gguf> -c 2048 --parallel 8 --cont-batching --port 8080
# vLLM（若显存允许）：
#   vllm serve Qwen/Qwen2.5-0.5B-Instruct --enforce-eager \
#       --gpu-memory-utilization 0.75 --max-model-len 2048 --port 8000

# 3. 冒烟（3 条 prompt，验证管线连通）
./target/release/loadgen --base-url http://127.0.0.1:3000 \
    --engine paged-serving --model paged-serving \
    --mode closed --concurrency 2 \
    --dataset benchmarks/serving/datasets/synth/smoke.jsonl \
    --requests 4 --warmup-secs 0 --max-tokens 16 --out /tmp/smoke.jsonl

# 4. 完整矩阵（默认闭环并发 1/2/4/8 × work 分布 × 3 次重复）
cd benchmarks/serving
./run_sweep.sh --base-url http://127.0.0.1:3000 --engine paged-serving \
    --model paged-serving --model-path ../../../models/<model>.gguf \
    --backend-quant W8A16 --tokenizer <tokenizer.json> --cuda-archs 86

# 5. 检查产物并生成图表
python3 validate_results.py results/<date>-<gpu>/
python3 plots.py results/<date>-<gpu>/

# 6. 准备发布正式结果前（要求填写 report.md、模型 SHA-256 和汇总图表）
cp RESULT_REPORT_TEMPLATE.md results/<date>-<gpu>/report.md
python3 validate_results.py --formal results/<date>-<gpu>/
```

## 结果索引

| 日期 | 硬件 | 内容 | 目录 |
|------|------|------|------|
| （尚无正式结果；空表是刻意状态，不以脚手架代替数据） | | | |

首个正式报告必须同时覆盖：

- 正确性 canary 后的真实 CUDA 后端；
- closed-loop 并发 1/2/4/8 与 Poisson 到达率矩阵；
- `paged-serving`、`llama-server`、可运行时的 vLLM 使用同一 loadgen、数据集与请求参数；
- 每个数字绑定硬件、驱动、双仓 commit、模型 SHA-256、量化格式和原始请求数据；
- 失败、OOM、429、无法启动的对照与 token coverage 不足均写进 `report.md`。

`validate_results.py` 只检查产物完整性，不会把通过检查误写成性能结论；只有人工填写
`report.md` 的结论和限制后，结果才可被 README、简历或面试材料引用。

## 纪律提醒（摘要，全文见 methodology.md）

- 无实测不写数字；每个数字绑定双仓 commit + 硬件 + 复现命令
- 失败归类不掺水：无 `[DONE]` 判失败；429 单独计数
- TTFT 从请求发送前开始计时；SSE chunk 间隔不冒充 token 级 ITL
- usage 缺失时必须提供 tokenizer；token coverage 不足则 tok/s 留空
- 量化格式差异（W8A16 vs Q4_K_M vs FP16）必须在表头声明
- 负结果归档（vLLM 跑不起来也是结果）
- 笔记本卡散热/功耗墙写进口径，不外推
