#!/usr/bin/env python3
"""从 serving sweep 结果生成跨引擎汇总与图表。

单次 run 的权威聚合来自 loadgen 生成的 ``summary.json``。本脚本不再从
逐请求 duration 反推测量墙钟，避免并发多波次时高估吞吐。

用法：
    python3 plots.py results/2026-08-30-RTX-3060-Laptop

输出：
    ttft_by_concurrency.png       闭环 TTFT p95 × 并发
    throughput_by_concurrency.png 闭环输出 token 吞吐 × 并发
    slo_curve.png                 泊松 TTFT p95 × 到达率
    summary_table.csv             全指标与重复波动
"""

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DIR_RE = re.compile(
    r"^(?P<engine>.+)_(?P<mode>closed|poisson)_"
    r"(?P<tag>c[\d.]+|rate[\d.]+)_(?P<ds>\w+)_r(?P<rep>\d+)$"
)


def nested(data, *path):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def load_run(run_dir: Path):
    """读取 loadgen 权威汇总，并验证逐请求原始数据仍然存在。"""
    summary_path = run_dir / "summary.json"
    requests_path = run_dir / "per_request.jsonl"
    if not summary_path.is_file() or not requests_path.is_file():
        raise ValueError(f"不完整 run：{run_dir} 必须同时含 summary.json 与 per_request.jsonl")
    summary = json.loads(summary_path.read_text())
    if summary.get("schema_version") != 1:
        raise ValueError(f"不支持的 summary schema：{summary_path}")
    return summary


def values_at(runs, *path):
    return [value for run in runs if (value := nested(run, *path)) is not None]


def aggregate(runs, *path):
    values = values_at(runs, *path)
    if not values:
        return None
    return sum(values) / len(values)


def spread(runs, *path):
    values = values_at(runs, *path)
    if not values:
        return None, None, None
    return sum(values) / len(values), min(values), max(values)


def tag_value(tag: str) -> float:
    if tag.startswith("rate"):
        return float(tag[4:])
    if tag.startswith("c"):
        return float(tag[1:])
    raise ValueError(f"未知负载标签：{tag}")


def series_label(engine: str, dataset: str) -> str:
    return f"{engine}/{dataset}"


def write_csv(root: Path, rows):
    csv_path = root / "summary_table.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.3f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    print(f"written {csv_path}")


def plot_closed_ttft(root: Path, rows, caption: str):
    closed = [row for row in rows if row["mode"] == "closed"]
    if not closed:
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    series = sorted({(row["engine"], row["dataset"]) for row in closed})
    plotted = False
    for engine, dataset in series:
        subset = sorted(
            (
                row
                for row in closed
                if row["engine"] == engine
                and row["dataset"] == dataset
                and row["ttft_p95"] is not None
            ),
            key=lambda row: tag_value(row["tag"]),
        )
        if not subset:
            continue
        x_values = [tag_value(row["tag"]) for row in subset]
        means = [row["ttft_p95"] for row in subset]
        lows = [row["ttft_p95_min"] for row in subset]
        highs = [row["ttft_p95_max"] for row in subset]
        axis.plot(x_values, means, marker="o", label=series_label(engine, dataset))
        axis.fill_between(x_values, lows, highs, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_yscale("log")
    axis.set_xlabel("concurrency")
    axis.set_ylabel("TTFT p95 (ms, log)")
    axis.set_title(f"TTFT p95 by concurrency — {caption}")
    axis.legend()
    fig.tight_layout()
    path = root / "ttft_by_concurrency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"written {path}")


def plot_closed_throughput(root: Path, rows, caption: str):
    closed = [row for row in rows if row["mode"] == "closed"]
    if not closed:
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    series = sorted({(row["engine"], row["dataset"]) for row in closed})
    plotted = False
    for engine, dataset in series:
        subset = sorted(
            (
                row
                for row in closed
                if row["engine"] == engine
                and row["dataset"] == dataset
                and row["throughput_tok_s"] is not None
            ),
            key=lambda row: tag_value(row["tag"]),
        )
        if not subset:
            continue
        x_values = [tag_value(row["tag"]) for row in subset]
        means = [row["throughput_tok_s"] for row in subset]
        lows = [row["throughput_tok_s_min"] for row in subset]
        highs = [row["throughput_tok_s_max"] for row in subset]
        axis.plot(
            x_values,
            means,
            marker="o",
            label=series_label(engine, dataset),
        )
        axis.fill_between(x_values, lows, highs, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        print("skip throughput_by_concurrency.png: token 计数覆盖不足")
        return
    axis.set_xlabel("concurrency")
    axis.set_ylabel("output throughput (tok/s)")
    axis.set_title(f"Output throughput by concurrency — {caption}")
    axis.legend()
    fig.tight_layout()
    path = root / "throughput_by_concurrency.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"written {path}")


def plot_poisson_slo(root: Path, rows, caption: str):
    poisson = [row for row in rows if row["mode"] == "poisson"]
    if not poisson:
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    series = sorted({(row["engine"], row["dataset"]) for row in poisson})
    plotted = False
    for engine, dataset in series:
        subset = sorted(
            (
                row
                for row in poisson
                if row["engine"] == engine
                and row["dataset"] == dataset
                and row["ttft_p95"] is not None
            ),
            key=lambda row: tag_value(row["tag"]),
        )
        if not subset:
            continue
        x_values = [tag_value(row["tag"]) for row in subset]
        means = [row["ttft_p95"] for row in subset]
        lows = [row["ttft_p95_min"] for row in subset]
        highs = [row["ttft_p95_max"] for row in subset]
        axis.plot(
            x_values,
            means,
            marker="s",
            label=series_label(engine, dataset),
        )
        axis.fill_between(x_values, lows, highs, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xlabel("arrival rate λ (req/s)")
    axis.set_ylabel("TTFT p95 (ms)")
    axis.set_title(f"SLO curve — {caption}")
    axis.legend()
    fig.tight_layout()
    path = root / "slo_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"written {path}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    root = Path(sys.argv[1])
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(f"缺少 metadata.json：{metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    commit = metadata.get("commits", {}).get("paged_serving", "?")[:10]
    date = metadata.get("date", "?")
    gpu = metadata.get("hardware", {}).get("gpu", "?")
    caption = f"@ {commit}, {date}, {gpu}"

    groups = {}
    for directory in sorted(root.iterdir()):
        match = DIR_RE.match(directory.name)
        if not match:
            continue
        key = (match["engine"], match["mode"], match["tag"], match["ds"])
        groups.setdefault(key, []).append(load_run(directory))

    if not groups:
        raise SystemExit(f"{root} 下没有完整的结果子目录")

    rows = []
    for (engine, mode, tag, dataset), runs in sorted(groups.items()):
        ttft_p95, ttft_p95_min, ttft_p95_max = spread(runs, "ttft_ms", "p95")
        throughput, throughput_min, throughput_max = spread(
            runs, "throughput", "output_tokens_per_second"
        )
        rows.append(
            {
                "engine": engine,
                "mode": mode,
                "tag": tag,
                "dataset": dataset,
                "repeats": len(runs),
                "success_rate_pct": aggregate(runs, "requests", "success_rate_pct"),
                "ttft_p50": aggregate(runs, "ttft_ms", "p50"),
                "ttft_p95": ttft_p95,
                "ttft_p95_min": ttft_p95_min,
                "ttft_p95_max": ttft_p95_max,
                "ttft_p99": aggregate(runs, "ttft_ms", "p99"),
                "inter_chunk_p50": aggregate(
                    runs, "inter_chunk_latency_ms", "p50"
                ),
                "tpot_p50": aggregate(runs, "tpot_ms", "p50"),
                "tpot_p95": aggregate(runs, "tpot_ms", "p95"),
                "token_coverage_pct": aggregate(
                    runs, "completion_tokens", "coverage_pct"
                ),
                "throughput_tok_s": throughput,
                "throughput_tok_s_min": throughput_min,
                "throughput_tok_s_max": throughput_max,
                "throughput_req_s": aggregate(
                    runs, "throughput", "successful_requests_per_second"
                ),
            }
        )

    write_csv(root, rows)
    plot_closed_ttft(root, rows, caption)
    plot_closed_throughput(root, rows, caption)
    plot_poisson_slo(root, rows, caption)


if __name__ == "__main__":
    main()
