#!/usr/bin/env bash
# run_sweep.sh —— serving 评测矩阵编排（方法论见 methodology.md）
#
# 职责：dirty 检查 → metadata 采集（双仓 commit + 硬件）→ 按矩阵跑 loadgen
#       → 汇总 summary。服务需预先启动（见 README 的启动命令）。
#
# 用法：
#   ./run_sweep.sh --base-url http://127.0.0.1:3000 --engine paged-infer
#   ./run_sweep.sh --base-url http://127.0.0.1:8080 --engine llama-server \
#       --modes closed --concurrencies "1 2 4" --allow-dirty
#
# 选项：
#   --base-url <url>        目标服务（必填）
#   --engine <name>         引擎标签，用于结果目录与报表（必填）
#   --modes <list>          "closed poisson"（默认 closed）
#   --concurrencies <list>  闭环并发档（默认 "1 2 4 8"）
#   --rates <list>          泊松到达率档 req/s（默认 "0.5 1.0 2.0"）
#   --datasets <list>       数据集名（默认 "work"；可选 short/work/long/smoke）
#   --requests <n>          每组合测量请求数（默认 64）
#   --warmup-secs <n>       预热秒数（默认 30）
#   --max-tokens <n>        每请求生成上限（默认 128）
#   --repeats <n>           每组合重复次数（默认 3，收敛判定用）
#   --allow-dirty           放行 dirty worktree（metadata 记录 dirty:true）
#   --tiny-llm-dir <dir>    tiny-llm 构建目录（记录其 commit 用；默认 ../..//tiny-llm）

set -euo pipefail
cd "$(dirname "$0")"

# ---------- 参数 ----------
BASE_URL="" ENGINE="" MODES="closed" CONCS="1 2 4 8" RATES="0.5 1.0 2.0"
DATASETS="work" REQUESTS=64 WARMUP=30 MAX_TOKENS=128 REPEATS=3
ALLOW_DIRTY=0 TINY_LLM_DIR="../../../tiny-llm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2;;
    --engine) ENGINE="$2"; shift 2;;
    --modes) MODES="$2"; shift 2;;
    --concurrencies) CONCS="$2"; shift 2;;
    --rates) RATES="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --requests) REQUESTS="$2"; shift 2;;
    --warmup-secs) WARMUP="$2"; shift 2;;
    --max-tokens) MAX_TOKENS="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --allow-dirty) ALLOW_DIRTY=1; shift;;
    --tiny-llm-dir) TINY_LLM_DIR="$2"; shift 2;;
    *) echo "未知参数: $1"; exit 2;;
  esac
done
[[ -n "$BASE_URL" && -n "$ENGINE" ]] || { echo "需要 --base-url 与 --engine"; exit 2; }

# ---------- dirty 检查（方法论 §4.1）----------
DIRTY=false
if ! git diff --quiet HEAD -- 2>/dev/null || [[ -n "$(git status --porcelain)" ]]; then
  if [[ "$ALLOW_DIRTY" -eq 1 ]]; then
    DIRTY=true
    echo "⚠ worktree 为 dirty（--allow-dirty 放行，metadata 将记录）"
  else
    echo "✗ worktree 为 dirty：先提交或 --allow-dirty（方法论 §4.1）"
    exit 1
  fi
fi

# ---------- 结果目录与 metadata ----------
DATE=$(date +%F)
GPU_SLUG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '-' | tr -cd '[:alnum:]-_' || echo "cpu")
OUTDIR="results/${DATE}-${GPU_SLUG}"
mkdir -p "$OUTDIR"

PINFER_COMMIT=$(git rev-parse HEAD)
TINYLLM_COMMIT=$(git -C "$TINY_LLM_DIR" rev-parse HEAD 2>/dev/null || echo "unavailable")
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")

cat > "$OUTDIR/metadata.json" <<EOF
{
  "date": "$DATE",
  "hardware": {"gpu": "$GPU_NAME", "vram": "$VRAM", "driver": "$DRIVER"},
  "commits": {"paged_infer": "$PINFER_COMMIT", "tiny_llm": "$TINYLLM_COMMIT", "dirty": $DIRTY},
  "engine_under_test": "$ENGINE",
  "base_url": "$BASE_URL",
  "protocol": {"requests": $REQUESTS, "warmup_s": $WARMUP, "max_tokens": $MAX_TOKENS, "repeats": $REPEATS}
}
EOF
echo "metadata -> $OUTDIR/metadata.json"

# ---------- loadgen 二进制 ----------
LOADGEN="../../target/release/loadgen"
if [[ ! -x "$LOADGEN" ]]; then
  echo "loadgen release 二进制不存在，构建中…"
  (cd ../.. && cargo build --release --bin loadgen)
fi

# ---------- 矩阵执行 ----------
run_one() { # $1=mode $2=param(并发或速率) $3=dataset $4=repeat
  local mode="$1" param="$2" ds="$3" rep="$4"
  local dsfile="datasets/synth/${ds}.jsonl"
  [[ -f "$dsfile" ]] || { echo "✗ 数据集不存在: $dsfile（先跑 gen_synth.py）"; return 1; }
  local tag
  if [[ "$mode" == "closed" ]]; then tag="c${param}"; else tag="rate${param}"; fi
  local dir="$OUTDIR/${ENGINE}_${mode}_${tag}_${ds}_r${rep}"
  mkdir -p "$dir"
  local mode_args
  if [[ "$mode" == "closed" ]]; then
    mode_args="--mode closed --concurrency $param"
  else
    mode_args="--mode poisson --rate $param"
  fi
  echo "▶ $ENGINE $mode $tag $ds (repeat $rep)"
  # shellcheck disable=SC2086
  "$LOADGEN" --base-url "$BASE_URL" $mode_args \
    --dataset "$dsfile" --requests "$REQUESTS" --warmup-secs "$WARMUP" \
    --max-tokens "$MAX_TOKENS" --out "$dir/per_request.jsonl" \
    | tee "$dir/summary.txt"
}

for ds in $DATASETS; do
  for mode in $MODES; do
    if [[ "$mode" == "closed" ]]; then
      for c in $CONCS; do
        for r in $(seq 1 "$REPEATS"); do run_one closed "$c" "$ds" "$r"; done
      done
    elif [[ "$mode" == "poisson" ]]; then
      for rate in $RATES; do
        for r in $(seq 1 "$REPEATS"); do run_one poisson "$rate" "$ds" "$r"; done
      done
    fi
  done
done

echo
echo "=== sweep 完成：$OUTDIR ==="
echo "下一步：python3 plots.py $OUTDIR 生成图表"
