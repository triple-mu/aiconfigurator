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


def _discover_workloads(results_root: str, configs: list) -> list[str]:
    """从 results/<config>/**/exp_<config_tag>_<workload>/ 目录名里抽出 workload 后缀。"""
    seen = set()
    for cfg_dir, _label, _budget in configs:
        pat = os.path.join(results_root, cfg_dir, "**", "exp_*")
        for d in glob.glob(pat, recursive=True):
            if not os.path.isdir(d):
                continue
            name = os.path.basename(d)
            # exp 名形如 "exp_<tag>_<workload>"; workload 是最后一段
            parts = name.split("_")
            if len(parts) >= 3:
                seen.add(parts[-1])
    # 排序:已知顺序优先,其他按字母
    known_order = ["short", "mid", "long", "prefillH", "decodeH", "bothH",
                   "earlyTurn", "typicalTurn", "tailTurn"]
    return sorted(seen, key=lambda w: (known_order.index(w) if w in known_order else 999, w))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="目录名,通常为 results/")
    parser.add_argument("--workloads", default=None,
                        help="逗号分隔的 workload 列表;不给时自动发现")
    args = parser.parse_args()

    # (results 子目录, 显示名, GPU 预算)—— 预算用于 agg 模式把 per-worker tps 还原成 cluster tps
    configs = [
        ("agg_baseline",        "01 agg 4-GPU",                  4),
        ("disagg_pd44",         "02 disagg locked",              8),
        ("disagg_open",         "03 disagg open",                8),
        ("agg_8gpu_2x",         "04 agg 2x replicas (8-GPU)",    8),
        ("disagg_open_extra",   "05 disagg open (extra wl)",     8),
        ("agg2x_extra",         "06 agg 2x (extra wl)",          8),
        ("disagg_swebench",     "07 disagg open (SWEBench)",     8),
        ("agg2x_swebench",      "08 agg 2x (SWEBench)",          8),
    ]

    if args.workloads:
        workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    else:
        workloads = _discover_workloads(args.results_root, configs)
    if not workloads:
        workloads = ["short", "mid", "long"]   # 兜底

    table = []
    for wl in workloads:
        row = {"workload": wl}
        for cfg_dir, cfg_label, budget in configs:
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
            if top is not None:
                # 关键:aic 的 tokens/s 是 per-worker;
                # agg 模式 cluster = per-worker × (budget / num_gpu_per_worker)
                # disagg 模式 1 replica 已经 = budget,所以 ratio 应该=1
                used_gpus = top.get("num_total_gpus") or budget
                replicas = max(1, int(budget) // max(1, int(used_gpus)))
                top["__cluster_tokens_per_s__"] = (top.get("tokens/s") or 0) * replicas
                top["__cluster_gpus__"] = int(used_gpus) * replicas
                top["__replicas__"] = replicas
                top["__budget__"] = budget
            if top is None:
                row[cfg_label] = None
            else:
                row[cfg_label] = top
        table.append(row)

    # 打印宽表
    print("=" * 115)
    print(f"  {'workload':<8} | {'config':<28} | {'cluster tps':>11} | {'tps/gpu':>9} | {'gpus':>6} | {'TTFT':>7} | {'TPOT':>7}")
    print("  " + "-" * 113)
    for row in table:
        wl = row["workload"]
        for cfg_dir, cfg_label, _budget in configs:
            data = row.get(cfg_label)
            if data is None:
                print(f"  {wl:<8} | {cfg_label:<28} |  (no data)")
                continue
            cluster_tps = data.get("__cluster_tokens_per_s__", 0) or 0
            per_gpu = data.get("tokens/s/gpu", 0) or 0
            gpus_str = f"{data.get('__cluster_gpus__', '?')}"
            if data.get("__replicas__", 1) > 1:
                gpus_str = f"{data['__cluster_gpus__']} (×{data['__replicas__']})"
            ttft = data.get("ttft", "?")
            tpot = data.get("tpot", "?")
            print(
                f"  {wl:<8} | {cfg_label:<28} | "
                f"{_safe(cluster_tps, ',.1f'):>11} | "
                f"{_safe(per_gpu, ',.1f'):>9} | "
                f"{gpus_str:>6} | "
                f"{_safe(ttft, '.1f'):>7} | "
                f"{_safe(tpot, '.2f'):>7}"
            )
        print("  " + "-" * 113)

    # 关键比值(用 cluster tps,不用 per-worker tps)
    print("\n  关键比值(均使用 cluster 总吞吐):")
    print("  ─ 用户原始 target:  8-GPU disagg ≥ 2.0× of 4-GPU agg(super-linear,极难)")
    print("  ─ 公平 baseline:    8-GPU disagg > 8-GPU agg-2x replicas(才说明 disagg 有真实价值)")
    for row in table:
        wl = row["workload"]
        def _c(label: str) -> float:
            d = row.get(label)
            return float(d.get("__cluster_tokens_per_s__", 0) or 0) if d else 0.0
        agg4 = _c("01 agg 4-GPU")
        locked = _c("02 disagg locked")
        opened = _c("03 disagg open")
        agg8 = _c("04 agg 2x replicas (8-GPU)")
        if agg4 == 0:
            continue
        line = f"    {wl:<8}: "
        line += f"locked/agg4={locked/agg4:.2f}x  "
        line += f"open/agg4={opened/agg4:.2f}x  "
        if agg8:
            line += f"agg8/agg4={agg8/agg4:.2f}x  |  "
            line += f"open/agg8={opened/agg8:.2f}x  "
            line += f"locked/agg8={locked/agg8:.2f}x"
        print(line)

    # 打印 mid 点的 top-1 详情(agg + disagg locked + disagg open 各打一段)
    mid_row = next((r for r in table if r["workload"] == "mid"), None)
    if mid_row:
        for cfg_dir, cfg_label, _budget in configs:
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
