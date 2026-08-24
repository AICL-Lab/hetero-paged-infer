#!/usr/bin/env python3
"""检查 serving 结果包的结构与可追溯产物。

用法：
    python3 validate_results.py results/2026-08-30-L40S
    python3 validate_results.py --formal results/2026-08-30-L40S

默认检查一次 sweep 所需的 metadata、每个 run 的原始请求、汇总、运行日志和 JSON schema。
--formal 额外要求人工报告、汇总 CSV、至少两张图以及每个 run 的模型 SHA-256。
它不评价性能数值，也不隐藏失败请求。
"""

import json
import re
import sys
from pathlib import Path


RUN_DIR_RE = re.compile(
    r"^.+_(?:closed|poisson)_(?:c[\d.]+|rate[\d.]+)_\w+_r\d+$"
)
RUN_FILES = ("run_metadata.json", "per_request.jsonl", "summary.json", "stdout.log")


def load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"无效 JSON：{path}: {error}")
        return None


def require_mapping(value, path: Path, errors: list[str]):
    if not isinstance(value, dict):
        errors.append(f"JSON 根节点必须是对象：{path}")
        return False
    return True


def main() -> int:
    args = sys.argv[1:]
    formal = False
    if args[:1] == ["--formal"]:
        formal = True
        args = args[1:]
    if len(args) != 1:
        print(__doc__)
        return 2

    root = Path(args[0])
    errors: list[str] = []
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        errors.append(f"缺少实验根 metadata.json：{metadata_path}")
    else:
        metadata = load_json(metadata_path, errors)
        if metadata is not None and require_mapping(metadata, metadata_path, errors):
            for key in ("schema_version", "date", "hardware", "software", "commits", "build"):
                if key not in metadata:
                    errors.append(f"metadata.json 缺少字段 {key}")

    runs = sorted(path for path in root.iterdir() if path.is_dir() and RUN_DIR_RE.match(path.name)) if root.is_dir() else []
    if not runs:
        errors.append(f"没有匹配的 run 目录：{root}")

    for run in runs:
        for name in RUN_FILES:
            path = run / name
            if not path.is_file():
                errors.append(f"缺少 {path}")
        request_path = run / "per_request.jsonl"
        if request_path.is_file() and request_path.stat().st_size == 0:
            errors.append(f"原始请求文件为空：{request_path}")

        run_metadata_path = run / "run_metadata.json"
        if run_metadata_path.is_file():
            run_metadata = load_json(run_metadata_path, errors)
            if run_metadata is not None and require_mapping(run_metadata, run_metadata_path, errors):
                for key in ("schema_version", "engine", "model", "load"):
                    if key not in run_metadata:
                        errors.append(f"{run_metadata_path} 缺少字段 {key}")
                if formal and not run_metadata.get("model", {}).get("sha256"):
                    errors.append(f"正式结果缺少模型 SHA-256：{run_metadata_path}")

        summary_path = run / "summary.json"
        if summary_path.is_file():
            summary = load_json(summary_path, errors)
            if summary is not None and require_mapping(summary, summary_path, errors):
                if summary.get("schema_version") != 1:
                    errors.append(f"不支持的 summary schema：{summary_path}")

    if formal:
        report = root / "report.md"
        if not report.is_file() or not report.read_text().strip():
            errors.append(f"正式结果缺少人工报告：{report}")
        if not (root / "summary_table.csv").is_file():
            errors.append(f"正式结果缺少汇总 CSV：{root / 'summary_table.csv'}")
        if len(list(root.glob("*.png"))) < 2:
            errors.append(f"正式结果至少需要两张图：{root}")

    if errors:
        print("结果包校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1

    level = "正式" if formal else "基础"
    print(f"{level}结果包结构校验通过：{root}（{len(runs)} 个 run）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
