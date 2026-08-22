#!/usr/bin/env python3
"""plots.py —— 从 results/<date>-<gpu>/ 生成评测图表（规范见 methodology.md §6）

用法：
    python3 plots.py results/2026-08-30-RTX-3060-Laptop

输入：结果目录（含 metadata.json 与 <engine>_<mode>_<tag>_<ds>_r<N>/ 子目录）
输出：同目录下
    ttft_by_concurrency.png    闭环 TTFT p50/p95/p99 × 并发（分组条形，对数轴）
    throughput_by_concurrency.png  闭环吞吐 × 并发折线
    slo_curve.png              泊松 TTFT p95 vs λ（SLO 曲线）
    summary_table.csv          跨组合汇总表
每张图 caption：<engine(s)> @ <commit>, <date>, <gpu>（方法论 §6）
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = round((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def load_run(run_dir: Path):
    """解析单次 run：返回指标字典（跨请求聚合）。"""
    records = []
    with open(run_dir / "per_request.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    ok = [r for r in records if r.get("ok")]
    ttfts = sorted(r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None)
    itls = sorted(v for r in ok for v in r.get("itl_ms", []))
    tpots = sorted(
        (r["duration_ms"] - r["ttft_ms"]) / (r["completion_tokens"] - 1)
        for r in ok
        if r.get("ttft_ms") is not None and (r.get("completion_tokens") or 0) > 1
    )
    total_tokens = sum(r.get("completion_tokens") or 0 for r in ok)
    wall = max((r["duration_ms"] for r in records), default=0.0) / 1000.0
    errors = {}
    for r in records:
        if not r.get("ok"):
            cls = r.get("error_class") or "unknown"
            errors[cls] = errors.get(cls, 0) + 1
    return {
        "n": len(records),
        "ok": len(ok),
        "ttft": {q: percentile(ttfts, q) for q in (50, 95, 99)},
        "itl": {q: percentile(itls, q) for q in (50, 95, 99)},
        "tpot": {q: percentile(tpots, q) for q in (50, 95, 99)},
        "throughput_tok_s": total_tokens / wall if wall > 0 else 0.0,
        "errors": errors,
    }


DIR_RE = re.compile(
    r"^(?P<engine>.+)_(?P<mode>closed|poisson)_(?P<tag>c[\d.]+|rate[\d.]+)_(?P<ds>\w+)_r(?P<rep>\d+)$"
)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    root = Path(sys.argv[1])
    meta = json.loads((root / "metadata.json").read_text())
    commit = meta.get("commits", {}).get("paged_infer", "?")[:10]
    date = meta.get("date", "?")
    gpu = meta.get("hardware", {}).get("gpu", "?")
    caption = f"@ {commit}, {date}, {gpu}"

    # 聚合：(engine, mode, tag, ds) -> [run 指标]
    groups = {}
    for d in sorted(root.iterdir()):
        m = DIR_RE.match(d.name)
        if not m or not (d / "per_request.jsonl").exists():
            continue
        key = (m["engine"], m["mode"], m["tag"], m["ds"])
        groups.setdefault(key, []).append(load_run(d))

    if not groups:
        print(f"✗ {root} 下无结果子目录")
        sys.exit(1)

    # 跨重复取均值（方法论 §4.3：报告均值与波动）
    def agg(runs, path):
        vals = []
        for r in runs:
            cur = r
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if cur is not None:
                vals.append(cur)
        return (sum(vals) / len(vals)) if vals else None

    rows = []
    for (engine, mode, tag, ds), runs in sorted(groups.items()):
        rows.append(
            {
                "engine": engine,
                "mode": mode,
                "tag": tag,
                "dataset": ds,
                "repeats": len(runs),
                "success": f"{agg(runs, ['ok']):.0f}/{agg(runs, ['n']):.0f}",
                "ttft_p50": agg(runs, ["ttft", "p50"]),
                "ttft_p95": agg(runs, ["ttft", "p95"]),
                "ttft_p99": agg(runs, ["ttft", "p99"]),
                "itl_p50": agg(runs, ["itl", "p50"]),
                "tpot_p50": agg(runs, ["tpot", "p50"]),
                "throughput_tok_s": agg(runs, ["throughput_tok_s"]),
            }
        )

    # CSV 汇总
    import csv

    csv_path = root / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (f"{v:.2f}" if isinstance(v, float) else v)
                    for k, v in r.items()
                }
            )
    print(f"written {csv_path}")

    def num(tag):
        return float(tag.lstrip("crate"))

    # 图 1：闭环 TTFT 分位 × 并发
    closed = [r for r in rows if r["mode"] == "closed"]
    if closed:
        fig, ax = plt.subplots(figsize=(9, 5))
        concs = sorted({num(r["tag"]) for r in closed})
        width = 0.25
        for i, q in enumerate(("ttft_p50", "ttft_p95", "ttft_p99")):
            vals = []
            for c in concs:
                match = [r[q] for r in closed if num(r["tag"]) == c and r[q] is not None]
                vals.append(sum(match) / len(match) if match else 0)
            ax.bar([x + i * width for x in range(len(concs))], vals, width, label=q.replace("ttft_", "TTFT "))
        ax.set_yscale("log")
        ax.set_xticks([x + width for x in range(len(concs))])
        ax.set_xticklabels([str(c) for c in concs])
        ax.set_xlabel("concurrency")
        ax.set_ylabel("TTFT (ms, log)")
        ax.set_title(f"TTFT percentiles by concurrency — {caption}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "ttft_by_concurrency.png", dpi=120)
        print(f"written {root / 'ttft_by_concurrency.png'}")

        # 图 2：吞吐 × 并发
        fig, ax = plt.subplots(figsize=(8, 5))
        for engine in {r["engine"] for r in closed}:
            sub = sorted((r for r in closed if r["engine"] == engine), key=lambda r: num(r["tag"]))
            ax.plot([num(r["tag"]) for r in sub], [r["throughput_tok_s"] for r in sub], marker="o", label=engine)
        ax.set_xlabel("concurrency")
        ax.set_ylabel("throughput (tok/s)")
        ax.set_title(f"Throughput by concurrency — {caption}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "throughput_by_concurrency.png", dpi=120)
        print(f"written {root / 'throughput_by_concurrency.png'}")

    # 图 3：泊松 SLO 曲线
    poisson = [r for r in rows if r["mode"] == "poisson"]
    if poisson:
        fig, ax = plt.subplots(figsize=(8, 5))
        for engine in {r["engine"] for r in poisson}:
            sub = sorted((r for r in poisson if r["engine"] == engine), key=lambda r: num(r["tag"]))
            ax.plot(
                [num(r["tag"]) for r in sub],
                [r["ttft_p95"] for r in sub],
                marker="s",
                label=engine,
            )
        ax.set_xlabel("arrival rate λ (req/s)")
        ax.set_ylabel("TTFT p95 (ms)")
        ax.set_title(f"SLO curve: TTFT p95 vs arrival rate — {caption}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(root / "slo_curve.png", dpi=120)
        print(f"written {root / 'slo_curve.png'}")


if __name__ == "__main__":
    main()
