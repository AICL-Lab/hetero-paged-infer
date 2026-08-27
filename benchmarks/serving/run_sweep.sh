#!/usr/bin/env bash
# run_sweep.sh —— serving 评测矩阵编排（方法论见 methodology.md）
#
# 职责：dirty 检查 → metadata 采集（双仓 commit + 硬件）→ 按矩阵跑 loadgen
#       → 汇总 summary。服务需预先启动（见 README 的启动命令）。
#
# 用法：
#   ./run_sweep.sh --base-url http://127.0.0.1:3000 --engine paged-serving
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
#   --model <name>          OpenAI API model 字段（默认 paged-serving）
#   --tokenizer <path>      usage 缺失时用于统计完整输出文本 token 数
#   --model-path <path>     被测模型路径（仅写入 run_metadata.json）
#   --model-sha256 <hex>    模型文件 SHA-256；省略时对可访问的 --model-path 自动计算
#   --backend-quant <name>  后端量化口径，如 W8A16/Q4_K_M/FP16
#   --engine-dir <dir>      被测引擎 git 目录（记录 commit；默认仓库根目录）
#   --cuda-archs <list>     构建 CUDA arch 口径（仅写 metadata）
#   --results-dir <dir>     指定实验根目录；默认 results/<date>-<gpu>
#   --allow-dirty           放行 dirty worktree（metadata 记录 dirty:true）
#   --tiny-llm-dir <dir>    tiny-llm 仓库目录（默认 ../../../tiny-llm）

set -euo pipefail
cd "$(dirname "$0")"

# ---------- 参数 ----------
BASE_URL="" ENGINE="" MODES="closed" CONCS="1 2 4 8" RATES="0.5 1.0 2.0"
DATASETS="work" REQUESTS=64 WARMUP=30 MAX_TOKENS=128 REPEATS=3
API_MODEL="paged-serving" TOKENIZER="" MODEL_PATH="" MODEL_SHA256="" BACKEND_QUANT="unspecified"
ENGINE_DIR="../.." CUDA_ARCHS="unspecified" RESULTS_DIR=""
ENGINE_DIR_SET=0 ALLOW_DIRTY=0 TINY_LLM_DIR="../../../tiny-llm"

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
    --model) API_MODEL="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;
    --model-path) MODEL_PATH="$2"; shift 2;;
    --model-sha256) MODEL_SHA256="$2"; shift 2;;
    --backend-quant) BACKEND_QUANT="$2"; shift 2;;
    --engine-dir) ENGINE_DIR="$2"; ENGINE_DIR_SET=1; shift 2;;
    --cuda-archs) CUDA_ARCHS="$2"; shift 2;;
    --results-dir) RESULTS_DIR="$2"; shift 2;;
    --allow-dirty) ALLOW_DIRTY=1; shift;;
    --tiny-llm-dir) TINY_LLM_DIR="$2"; shift 2;;
    *) echo "未知参数: $1"; exit 2;;
  esac
done
[[ -n "$BASE_URL" && -n "$ENGINE" ]] || { echo "需要 --base-url 与 --engine"; exit 2; }
[[ "$REQUESTS" =~ ^[1-9][0-9]*$ ]] || { echo "--requests 必须是正整数"; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats 必须是正整数"; exit 2; }
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "--max-tokens 必须是正整数"; exit 2; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "--warmup-secs 必须是非负整数"; exit 2; }
for mode in $MODES; do
  [[ "$mode" == "closed" || "$mode" == "poisson" ]] || {
    echo "--modes 只支持 closed / poisson"; exit 2;
  }
done
if [[ -n "$TOKENIZER" && ! -f "$TOKENIZER" ]]; then
  echo "tokenizer 不存在: $TOKENIZER"; exit 2
fi
if [[ -n "$MODEL_SHA256" && ! "$MODEL_SHA256" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo "--model-sha256 必须是 64 位十六进制 SHA-256"; exit 2
fi
if [[ -z "$MODEL_SHA256" && -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]; then
  MODEL_SHA256=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
fi
if [[ "$ENGINE" != "paged-serving" && "$ENGINE_DIR_SET" -eq 0 ]]; then
  echo "非 paged-serving 后端必须用 --engine-dir 指向其 git checkout，以记录真实 commit"
  exit 2
fi
git -C "$ENGINE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "--engine-dir 不是 git worktree: $ENGINE_DIR"; exit 2;
}

# ---------- dirty 检查（方法论 §4.1）----------
PAGED_SERVING_DIRTY=false TINYLLM_DIRTY=false ENGINE_DIRTY=false
if [[ -n "$(git -C ../.. status --porcelain)" ]]; then PAGED_SERVING_DIRTY=true; fi
if git -C "$TINY_LLM_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
   [[ -n "$(git -C "$TINY_LLM_DIR" status --porcelain)" ]]; then
  TINYLLM_DIRTY=true
fi
if [[ -n "$(git -C "$ENGINE_DIR" status --porcelain)" ]]; then ENGINE_DIRTY=true; fi
DIRTY=false
if [[ "$PAGED_SERVING_DIRTY" == true || "$TINYLLM_DIRTY" == true || "$ENGINE_DIRTY" == true ]]; then
  DIRTY=true
  if [[ "$ALLOW_DIRTY" -eq 1 ]]; then
    echo "⚠ 相关 worktree 为 dirty（--allow-dirty 放行，metadata 将逐仓记录）"
  else
    echo "✗ 相关 worktree 为 dirty：先提交或 --allow-dirty（方法论 §4.1）"
    exit 1
  fi
fi

# ---------- 结果目录与 metadata ----------
DATE=$(date +%F)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
if [[ -z "$GPU_NAME" ]]; then GPU_NAME="n/a"; GPU_SLUG="cpu";
else GPU_SLUG=$(printf '%s' "$GPU_NAME" | tr ' ' '-' | tr -cd '[:alnum:]-_'); fi
OUTDIR=${RESULTS_DIR:-"results/${DATE}-${GPU_SLUG}"}
mkdir -p "$OUTDIR"

PAGED_SERVING_COMMIT=$(git rev-parse HEAD)
TINYLLM_COMMIT=$(git -C "$TINY_LLM_DIR" rev-parse HEAD 2>/dev/null || echo "unavailable")
ENGINE_COMMIT=$(git -C "$ENGINE_DIR" rev-parse HEAD 2>/dev/null || echo "unavailable")
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "n/a")
CUDA_TOOLKIT=$(nvcc --version 2>/dev/null | tail -1 || echo "n/a")

python3 - "$OUTDIR/metadata.json" "$DATE" "$GPU_NAME" "$VRAM" "$DRIVER" \
  "$CUDA_TOOLKIT" "$PAGED_SERVING_COMMIT" "$TINYLLM_COMMIT" "$DIRTY" "$PAGED_SERVING_DIRTY" \
  "$TINYLLM_DIRTY" "$CUDA_ARCHS" <<'PY'
import json
import sys
from pathlib import Path

(
    path, date, gpu, vram, driver, cuda_toolkit, paged_serving_commit,
    tiny_commit, dirty, paged_serving_dirty, tiny_dirty, cuda_archs,
) = sys.argv[1:]
metadata = {
    "schema_version": 1,
    "date": date,
    "hardware": {"gpu": gpu, "vram": vram, "driver": driver},
    "software": {"cuda_toolkit": cuda_toolkit},
    "commits": {
        "paged_serving": paged_serving_commit,
        "tiny_llm": tiny_commit,
        "dirty": dirty.lower() == "true",
        "paged_serving_dirty": paged_serving_dirty.lower() == "true",
        "tiny_llm_dirty": tiny_dirty.lower() == "true",
    },
    "build": {"profile": "release", "cuda_archs": cuda_archs},
}
Path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
PY
echo "metadata -> $OUTDIR/metadata.json"

# ---------- loadgen 二进制 ----------
LOADGEN="../../target/release/loadgen"
echo "构建当前 HEAD 的 loadgen release 二进制…"
(cd ../.. && cargo build --locked --release --bin loadgen)

# ---------- 矩阵执行 ----------
run_one() { # $1=mode $2=param(并发或速率) $3=dataset $4=repeat
  local mode="$1" param="$2" ds="$3" rep="$4"
  local dsfile="datasets/synth/${ds}.jsonl"
  [[ -f "$dsfile" ]] || { echo "✗ 数据集不存在: $dsfile（先跑 gen_synth.py）"; return 1; }
  local tag
  if [[ "$mode" == "closed" ]]; then tag="c${param}"; else tag="rate${param}"; fi
  local dir="$OUTDIR/${ENGINE}_${mode}_${tag}_${ds}_r${rep}"
  mkdir -p "$dir"
  local -a mode_args
  if [[ "$mode" == "closed" ]]; then
    mode_args=(--mode closed --concurrency "$param")
  else
    mode_args=(--mode poisson --rate "$param")
  fi
  local -a tokenizer_args=()
  if [[ -n "$TOKENIZER" ]]; then tokenizer_args=(--tokenizer "$TOKENIZER"); fi

  python3 - "$dir/run_metadata.json" "$ENGINE" "$ENGINE_COMMIT" "$ENGINE_DIRTY" "$API_MODEL" \
    "$MODEL_PATH" "$MODEL_SHA256" "$BACKEND_QUANT" "$TOKENIZER" "$mode" "$param" "$ds" \
    "$rep" "$REQUESTS" "$WARMUP" "$MAX_TOKENS" <<'PY'
import json
import sys
from pathlib import Path

(
    path, engine, commit, engine_dirty, api_model, model_path, model_sha256, quant, tokenizer, mode,
    parameter, dataset, repeat, requests, warmup, max_tokens,
) = sys.argv[1:]
metadata = {
    "schema_version": 1,
    "engine": {
        "name": engine,
        "commit": commit,
        "dirty": engine_dirty.lower() == "true",
    },
    "model": {
        "api_name": api_model,
        "path": model_path or None,
        "sha256": model_sha256 or None,
        "backend_quant": quant,
        "tokenizer": tokenizer or None,
    },
    "load": {
        "mode": mode,
        "concurrency": int(float(parameter)) if mode == "closed" else None,
        "rate": float(parameter) if mode == "poisson" else None,
        "dataset": dataset,
        "repeat": int(repeat),
        "requests": int(requests),
        "warmup_secs": int(warmup),
        "max_tokens": int(max_tokens),
    },
}
Path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
PY
  echo "▶ $ENGINE $mode $tag $ds (repeat $rep)"
  "$LOADGEN" --base-url "$BASE_URL" "${mode_args[@]}" \
    --dataset "$dsfile" --requests "$REQUESTS" --warmup-secs "$WARMUP" \
    --max-tokens "$MAX_TOKENS" --engine "$ENGINE" --model "$API_MODEL" \
    --out "$dir/per_request.jsonl" --summary-out "$dir/summary.json" \
    "${tokenizer_args[@]}" | tee "$dir/stdout.log"
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
python3 validate_results.py "$OUTDIR"
echo "下一步：python3 plots.py $OUTDIR 生成图表"
