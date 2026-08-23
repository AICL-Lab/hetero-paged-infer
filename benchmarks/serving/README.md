# Serving 评测体系

对 paged-infer（及对照基线 llama-server / vLLM）做可复现的 serving 级
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
# paged-infer（tiny-llm 真实后端）：
#   PAGED_INFER_TINY_LLM_MAX_SEQS=8 ./target/release/paged-infer --serve \
#       --port 3000 --tokenizer <tokenizer.json> ...（模型经配置加载）
# llama-server（基线）：
#   llama-server -m <model.gguf> -c 2048 --parallel 8 --cont-batching --port 8080
# vLLM（若显存允许）：
#   vllm serve Qwen/Qwen2.5-0.5B-Instruct --enforce-eager \
#       --gpu-memory-utilization 0.75 --max-model-len 2048 --port 8000

# 3. 冒烟（3 条 prompt，验证管线连通）
./target/release/loadgen --base-url http://127.0.0.1:3000 \
    --engine paged-infer --model paged-infer \
    --mode closed --concurrency 2 \
    --dataset benchmarks/serving/datasets/synth/smoke.jsonl \
    --requests 4 --warmup-secs 0 --max-tokens 16 --out /tmp/smoke.jsonl

# 4. 完整矩阵（默认闭环并发 1/2/4/8 × work 分布 × 3 次重复）
cd benchmarks/serving
./run_sweep.sh --base-url http://127.0.0.1:3000 --engine paged-infer \
    --model paged-infer --model-path ../../../models/<model>.gguf \
    --backend-quant W8A16 --tokenizer <tokenizer.json> --cuda-archs 86

# 5. 图表
python3 plots.py results/<date>-<gpu>/
```

## 结果索引

| 日期 | 硬件 | 内容 | 目录 |
|------|------|------|------|
| （W6 全功能复测后回填，先留空位不写假数字） | | | |

## 纪律提醒（摘要，全文见 methodology.md）

- 无实测不写数字；每个数字绑定双仓 commit + 硬件 + 复现命令
- 失败归类不掺水：无 `[DONE]` 判失败；429 单独计数
- TTFT 从请求发送前开始计时；SSE chunk 间隔不冒充 token 级 ITL
- usage 缺失时必须提供 tokenizer；token coverage 不足则 tok/s 留空
- 量化格式差异（W8A16 vs Q4_K_M vs FP16）必须在表头声明
- 负结果归档（vLLM 跑不起来也是结果）
- 笔记本卡散热/功耗墙写进口径，不外推
