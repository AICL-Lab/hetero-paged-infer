#!/usr/bin/env python3
"""gen_synth.py —— 三档合成负载数据集生成器（可复现，固定种子）

生成三个 jsonl 数据集，供 serving 评测的受控实验使用：
  short.jsonl  —— 32-64 token 级 prompt（轻量请求）
  work.jsonl   —— 128-256 token 级 prompt（典型工作负载）
  long.jsonl   —— 512-1024 token 级 prompt（长 prefill 压力）

口径说明（写入每条的 prompt_tokens 为**近似值**）：
  英文自然文本在 Qwen2.5 BPE 词表下约 1.2-1.5 token/word，
  本生成器按 1.35 token/word 估算目标词数。精确 token 数以服务端
  tokenizer 为准；prompt_tokens 字段仅用于报表核对与分布描述。

用法：
    python3 gen_synth.py [--n 200] [--seed 42] [--outdir .]
"""
import argparse
import json
import random
from pathlib import Path

# 常用词表（避免生僻词触发 byte-fallback，保持 token 估算稳定）
VOCAB = (
    "the of and to in a is that for it as was with be by on not he i this are or "
    "his from at which but have an had they you were their one its can will would "
    "there been when who if more no out so up said what about into than them than "
    "system serving scheduler request batch token cache memory block sequence model "
    "kernel latency throughput prefill decode attention quantization inference "
    "engine queue priority pressure threshold allocation fragment eviction stream "
    "concurrent parallel pipeline buffer tensor weight gradient optimization "
    "profiling benchmark experiment measurement percentile distribution arrival "
    "interval timeout retry error success rate utilization saturation bottleneck"
).split()

TIERS = {
    "short": (32, 64),
    "work": (128, 256),
    "long": (512, 1024),
}
TOKENS_PER_WORD = 1.35  # Qwen2.5 BPE 对英文自然文本的经验比率


def gen_prompt(rng: random.Random, target_tokens: int) -> tuple[str, int]:
    n_words = max(4, round(target_tokens / TOKENS_PER_WORD))
    words = [rng.choice(VOCAB) for _ in range(n_words)]
    # 每 12 词一句，模拟自然文本的边界结构
    sentences = []
    for i in range(0, n_words, 12):
        chunk = " ".join(words[i : i + 12])
        sentences.append(chunk.capitalize() + ".")
    return " ".join(sentences), target_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="每档条目数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（可复现）")
    ap.add_argument("--outdir", default=".", help="输出目录")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for tier, (lo, hi) in TIERS.items():
        path = outdir / f"{tier}.jsonl"
        with open(path, "w") as f:
            for _ in range(args.n):
                target = rng.randint(lo, hi)
                prompt, approx = gen_prompt(rng, target)
                f.write(
                    json.dumps({"prompt": prompt, "prompt_tokens": approx}) + "\n"
                )
        print(f"written {path} ({args.n} entries, {lo}-{hi} token tier)")


if __name__ == "__main__":
    main()
