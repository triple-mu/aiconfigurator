# SPDX-License-Identifier: Apache-2.0
"""
分析 SWEBench multi-turn agent trace 抽出 aic 用的 (ISL, OSL, prefix) 分布。

输入:
    JSONL,每行一个 session,字段 {"filename": ..., "messages": [...], "tools": [...]}
    messages 是 OpenAI-style chat:role ∈ {system, user, assistant, tool}

输出:
    - 每 turn 的 (ISL, OSL, prefix) 表(CSV)
    - 总体 P50/P95/mean/max 统计
    - 推荐 aic workload 点(基于 P50 / P95)

用法:
    python my_exps/analyze_swebench_workload.py path/to/data.json [--tokenizer cl100k_base] [--out csv]

注意:
    - SWEBench trace 没标 tool schema 的 prompt token,我们近似为:
      ISL_k = tokens(system + user + asst_1 + tool_1 + ... + asst_{k-1} + tool_{k-1})
      用 tool/asst content 的字符 + JSON 编码大致估算
    - prefix_k = ISL_{k-1} + OSL_{k-1} (假设 sglang 的 prefix-aware router 命中前一轮全部 KV)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def _load_jsonl(path: str) -> list[dict]:
    sessions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    return sessions


def _content_to_str(content) -> str:
    """OpenAI-style content can be str or list of parts; flatten to str for token counting."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                # tool_calls, image, text, etc. — keep JSON to preserve token weight
                out.append(json.dumps(part, ensure_ascii=False))
            else:
                out.append(str(part))
        return "\n".join(out)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content) if content is not None else ""


def _msg_to_str(msg: dict) -> str:
    """把单条 message(含 tool_calls / tool_call_id 等)拼成"模型实际看到的字符串"近似。"""
    role = msg.get("role", "")
    parts = [f"<|{role}|>"]
    content = _content_to_str(msg.get("content"))
    if content:
        parts.append(content)
    # assistant 有时把 tool 调用放在 tool_calls 里(非 content)
    if msg.get("tool_calls"):
        parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
    if msg.get("tool_call_id"):
        parts.append(f"tool_call_id={msg['tool_call_id']}")
    if msg.get("name"):
        parts.append(f"name={msg['name']}")
    return "\n".join(parts)


def _get_encoder(name: str):
    import tiktoken
    return tiktoken.get_encoding(name)


def analyze_session(session: dict, encoder) -> list[dict]:
    """
    Return list of per-turn dict: {turn_index, isl, osl, prefix, role_at_turn}
    A "turn" = one assistant call (model.generate event).
    """
    messages = session["messages"]
    tools = session.get("tools") or []
    # tool definitions 也算 system-level prompt 的一部分(每 turn 都重复)
    tool_prompt = json.dumps(tools, ensure_ascii=False) if tools else ""
    tool_prompt_tokens = len(encoder.encode(tool_prompt)) if tool_prompt else 0

    # 预计算每条 message 的 token 长度
    token_lens = [len(encoder.encode(_msg_to_str(m))) for m in messages]

    turns = []
    cumulative_input = tool_prompt_tokens  # 工具描述每次都在 prompt 里
    prev_isl = 0
    prev_osl = 0

    for i, msg in enumerate(messages):
        role = msg["role"]
        tlen = token_lens[i]
        if role == "assistant":
            # 这条是模型生成的 → turn k
            isl_k = cumulative_input  # 前面累积的所有 token
            osl_k = tlen              # 当前 assistant message 长度
            prefix_k = prev_isl + prev_osl  # 上轮已生成的 KV 全部命中
            # 第 1 轮 prefix = 0(没有上轮);后续 prefix 不超过 ISL_k
            prefix_k = min(prefix_k, isl_k)
            turns.append({
                "turn_index": len(turns) + 1,
                "isl": isl_k,
                "osl": osl_k,
                "prefix": prefix_k,
                "new_input_tokens": isl_k - prefix_k,  # 本轮真正要新跑的 prefill
            })
            cumulative_input += tlen
            prev_isl = isl_k
            prev_osl = osl_k
        else:
            cumulative_input += tlen
    return turns


def percentile_summary(values: list[float], name: str) -> dict:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(arr),
        "mean": float(arr.mean()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def _round_to_aic(v: float) -> int:
    """把 P50/P95 数字 round 到 aic workload 表里好看的整数。"""
    # round 到 nearest 100 或 500 看大小
    if v < 1000:
        return int(round(v / 100) * 100)
    if v < 5000:
        return int(round(v / 500) * 500)
    return int(round(v / 1000) * 1000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to JSONL session file (one session per line)")
    parser.add_argument("--tokenizer", default="cl100k_base",
                        help="tiktoken encoder name (default: cl100k_base, 接近 GPT-4/GLM)")
    parser.add_argument("--out", default=None, help="optional CSV output of per-turn stats")
    args = parser.parse_args()

    sessions = _load_jsonl(args.input)
    print(f"Loaded {len(sessions)} sessions from {args.input}")

    encoder = _get_encoder(args.tokenizer)
    print(f"Using tokenizer: {args.tokenizer}")
    print()

    all_turns = []
    turns_per_session = []
    for s_idx, session in enumerate(sessions):
        turns = analyze_session(session, encoder)
        for t in turns:
            t["session_idx"] = s_idx
            t["filename"] = session.get("filename", "?")
        all_turns.extend(turns)
        turns_per_session.append(len(turns))

    # 总体统计
    isl_all  = [t["isl"]    for t in all_turns]
    osl_all  = [t["osl"]    for t in all_turns]
    pre_all  = [t["prefix"] for t in all_turns]
    new_all  = [t["new_input_tokens"] for t in all_turns]

    print("=" * 88)
    print(f"  Sessions: {len(sessions)},  Total turns: {len(all_turns)},  "
          f"Turns/session avg={np.mean(turns_per_session):.1f}, max={max(turns_per_session)}")
    print("=" * 88)
    print(f"  {'metric':<16} | {'n':>5} | {'mean':>7} | {'p25':>7} | {'p50':>7} | "
          f"{'p75':>7} | {'p90':>7} | {'p95':>7} | {'p99':>7} | {'max':>7}")
    print("  " + "-" * 86)
    for name, vals in [
        ("ISL (input)",     isl_all),
        ("OSL (output)",    osl_all),
        ("prefix",          pre_all),
        ("new_prefill",     new_all),
    ]:
        s = percentile_summary(vals, name)
        print(f"  {s['name']:<16} | {s['n']:>5} | "
              f"{s['mean']:>7.0f} | {s['p25']:>7.0f} | {s['p50']:>7.0f} | "
              f"{s['p75']:>7.0f} | {s['p90']:>7.0f} | {s['p95']:>7.0f} | "
              f"{s['p99']:>7.0f} | {s['max']:>7.0f}")

    # 按 turn 位置 (前期/中期/后期) 分桶
    print()
    print("=" * 88)
    print("  按 turn 位置分桶(看多轮累积效应)")
    print("=" * 88)
    print(f"  {'bucket':<16} | {'n':>5} | {'mean isl':>9} | {'mean osl':>9} | "
          f"{'mean prefix':>12} | {'mean new_pf':>12}")
    print("  " + "-" * 86)
    buckets = {
        "1-5 (开局)":  [t for t in all_turns if 1 <= t["turn_index"] <= 5],
        "6-20 (中段)": [t for t in all_turns if 6 <= t["turn_index"] <= 20],
        "21-50":       [t for t in all_turns if 21 <= t["turn_index"] <= 50],
        "51-100":      [t for t in all_turns if 51 <= t["turn_index"] <= 100],
        "101+":        [t for t in all_turns if t["turn_index"] >= 101],
    }
    for name, group in buckets.items():
        if not group:
            continue
        isl_m = np.mean([t["isl"] for t in group])
        osl_m = np.mean([t["osl"] for t in group])
        pre_m = np.mean([t["prefix"] for t in group])
        new_m = np.mean([t["new_input_tokens"] for t in group])
        print(f"  {name:<16} | {len(group):>5} | {isl_m:>9.0f} | {osl_m:>9.0f} | "
              f"{pre_m:>12.0f} | {new_m:>12.0f}")

    # aic workload 推荐
    print()
    print("=" * 88)
    print("  推荐的 aic workload 点(基于实际分布)")
    print("=" * 88)
    isl_p50 = _round_to_aic(np.percentile(isl_all, 50))
    isl_p95 = _round_to_aic(np.percentile(isl_all, 95))
    osl_p50 = _round_to_aic(np.percentile(osl_all, 50))
    osl_p95 = _round_to_aic(np.percentile(osl_all, 95))
    pre_p50 = _round_to_aic(np.percentile(pre_all, 50))
    pre_p95 = _round_to_aic(np.percentile(pre_all, 95))

    print(f"  P50 点 (代表 median turn,aic YAML 用):")
    print(f"      isl: {isl_p50},  osl: {osl_p50},  prefix: {pre_p50}")
    print(f"  P95 点 (代表 tail turn,aic YAML 用):")
    print(f"      isl: {isl_p95},  osl: {osl_p95},  prefix: {pre_p95}")

    # 输出 CSV 可选
    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["session_idx", "filename", "turn_index",
                                              "isl", "osl", "prefix", "new_input_tokens"])
            w.writeheader()
            w.writerows(all_turns)
        print(f"\n  per-turn CSV written to: {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
