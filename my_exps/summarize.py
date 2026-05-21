# SPDX-License-Identifier: Apache-2.0
"""
汇总 my_exps/ 下 3 套 sweep 的结果,打印同点对比表。

用法:
    .venv/bin/python my_exps/summarize.py
    .venv/bin/python my_exps/summarize.py --results-root results
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd


def _safe(value, fmt: str = "?") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "?"
    try:
        return format(value, fmt) if fmt != "?" else str(value)
    except (TypeError, ValueError):
        return str(value)


def _find_top1_from_files(files: list[str]) -> dict | None:
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) == 0:
                continue
            top = df.iloc[0].to_dict()
            top["__file__"] = f
            rows.append(top)
        except Exception as e:
            print(f"  [WARN] failed to read {f}: {e}", file=sys.stderr)
    if not rows:
        return None
    # 若同一 pattern 命中多个文件(多次 run),选 tokens/s 最高的(最新一次大概率)
    return max(rows, key=lambda r: r.get("tokens/s", 0) or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="目录名,通常为 results/")
    args = parser.parse_args()

    workloads = ["short", "mid", "long"]
    configs = [
        ("agg_baseline",   "01 agg 4-GPU"),
        ("disagg_pd44",    "02 disagg locked"),
        ("disagg_open",    "03 disagg open"),
        ("agg_8gpu_2x",    "04 agg 2x replicas (8-GPU)"),
    ]

    table = []
    for wl in workloads:
        row = {"workload": wl}
        for cfg_dir, cfg_label in configs:
            # aic 实际落盘:results/<cfg>/<model_org>/<run_signature>/exp_*_<wl>/best_config_topn.csv
            # 用 recursive ** 兼容不同嵌套深度
            pattern = os.path.join(
                args.results_root, cfg_dir, "**", f"exp_*_{wl}", "best_config_topn.csv"
            )
            files = sorted(glob.glob(pattern, recursive=True))
            if not files:
                row[cfg_label] = None
                continue
            top = _find_top1_from_files(files)
            if top is None:
                row[cfg_label] = None
            else:
                row[cfg_label] = top
        table.append(row)

    # 打印宽表
    print("=" * 100)
    print(f"  {'workload':<8} | {'config':<22} | {'cluster tps':>11} | {'tps/gpu':>9} | {'gpus':>5} | {'TTFT':>7} | {'TPOT':>7}")
    print("-" * 100)
    for row in table:
        wl = row["workload"]
        for cfg_dir, cfg_label in configs:
            data = row.get(cfg_label)
            if data is None:
                print(f"  {wl:<8} | {cfg_label:<22} |  (no data)")
                continue
            cluster_tps = data.get("tokens/s", 0) or 0
            per_gpu = data.get("tokens/s/gpu", 0) or 0
            total_gpus = data.get("num_total_gpus") or data.get("total_gpus") or "?"
            ttft = data.get("ttft", data.get("TTFT", "?"))
            tpot = data.get("tpot", data.get("TPOT", "?"))
            print(
                f"  {wl:<8} | {cfg_label:<22} | "
                f"{_safe(cluster_tps, ',.1f'):>11} | "
                f"{_safe(per_gpu, ',.1f'):>9} | "
                f"{str(total_gpus):>5} | "
                f"{_safe(ttft, '.1f'):>7} | "
                f"{_safe(tpot, '.2f'):>7}"
            )
        print("-" * 100)

    # 关键比值
    print("\n  关键比值:")
    print("  ─ 用户原始 target: 8-GPU disagg ≥ 2.0× of 4-GPU agg(super-linear,极难)")
    print("  ─ 真正公平 target: 8-GPU disagg  >  8-GPU agg-2x(才说明 disagg 有价值)")
    for row in table:
        wl = row["workload"]
        agg4 = row.get("01 agg 4-GPU")
        locked = row.get("02 disagg locked")
        opened = row.get("03 disagg open")
        agg8 = row.get("04 agg 2x replicas (8-GPU)")
        if not agg4:
            continue
        agg4_tps = agg4.get("tokens/s", 0) or 0
        if agg4_tps == 0:
            continue
        line = f"    {wl:<8}: "
        if locked:
            r = (locked.get("tokens/s") or 0) / agg4_tps
            line += f"locked/agg4={r:.2f}x  "
        if opened:
            r = (opened.get("tokens/s") or 0) / agg4_tps
            line += f"open/agg4={r:.2f}x  "
        if agg8:
            agg8_tps = agg8.get("tokens/s", 0) or 0
            r_8x = agg8_tps / agg4_tps
            line += f"agg8/agg4={r_8x:.2f}x  "
            # 更重要的: disagg vs agg8 (公平对照)
            if opened:
                r_vs_8 = (opened.get("tokens/s") or 0) / agg8_tps if agg8_tps else 0
                line += f"|  open/agg8={r_vs_8:.2f}x"
        print(line)

    # 打印 mid 点的 top-1 详情(agg + disagg locked + disagg open 各打一段)
    mid_row = next((r for r in table if r["workload"] == "mid"), None)
    if mid_row:
        for cfg_dir, cfg_label in configs:
            data = mid_row.get(cfg_label)
            if not data:
                continue
            print(f"\n  {cfg_label} / mid 点 top-1 关键参数:")
            keys_of_interest = [
                "num_total_gpus",
                "concurrency", "request_rate",
                "(p)parallel", "(p)bs", "(p)workers", "(p)seq/s/worker", "(p)memory",
                "(d)parallel", "(d)bs", "(d)workers", "(d)seq/s/worker", "(d)memory",
                "tokens/s", "tokens/s/gpu", "tokens/s/user", "ttft", "tpot",
            ]
            for k in keys_of_interest:
                v = data.get(k)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    print(f"    {k:<22} = {v}")
            print(f"    [source] {data['__file__']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
