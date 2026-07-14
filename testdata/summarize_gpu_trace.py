#!/usr/bin/env python3
"""Summarize GPU kernel events from a Torch Profiler Chrome trace."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


CATEGORY_PATTERNS = (
    ("ua2d", re.compile(r"unified_attention_2d", re.I)),
    ("ua3d_main", re.compile(r"unified_attention_3d", re.I)),
    ("ua3d_merge", re.compile(r"reduce_segments", re.I)),
    ("gemm", re.compile(r"LLMM1|qwen35_gemv", re.I)),
    (
        "gdn",
        re.compile(
            r"gated_delta|chunk_delta|chunk_fwd_kernel_o|recompute_w_u|"
            r"causal_conv|fused_recurrent",
            re.I,
        ),
    ),
    ("norm_rope", re.compile(r"rms_norm|rotary|rope", re.I)),
    ("kv_cache", re.compile(r"reshape_and_cache|cache", re.I)),
    ("gemm", re.compile(r"gemm|tensile|hipblas|rocblas|^Cijk_", re.I)),
    ("memcpy", re.compile(r"memcpy|copy_kernel", re.I)),
)


def load_events(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["traceEvents"] if isinstance(payload, dict) else payload


def category_for(name: str) -> str:
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return category
    return "other"


def summarize(path: Path, pattern: re.Pattern[str] | None) -> dict:
    grouped: dict[str, list[float]] = defaultdict(list)
    categories: dict[str, list[float]] = defaultdict(list)
    intervals: list[tuple[float, float]] = []
    total_us = 0.0
    event_count = 0
    for event in load_events(path):
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        name = event.get("name", "<unnamed>")
        if pattern is not None and pattern.search(name) is None:
            continue
        duration_us = float(event.get("dur", 0.0))
        start_us = float(event.get("ts", 0.0))
        grouped[name].append(duration_us)
        categories[category_for(name)].append(duration_us)
        intervals.append((start_us, start_us + duration_us))
        total_us += duration_us
        event_count += 1

    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    busy_us = sum(end - start for start, end in merged)
    span_us = merged[-1][1] - merged[0][0] if merged else 0.0
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]

    rows = []
    for name, durations in grouped.items():
        duration_us = sum(durations)
        rows.append(
            {
                "name": name,
                "category": category_for(name),
                "calls": len(durations),
                "total_ms": duration_us / 1000.0,
                "mean_us": duration_us / len(durations),
                "share": duration_us / total_us if total_us else 0.0,
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    category_rows = []
    for category, durations in categories.items():
        duration_us = sum(durations)
        category_rows.append(
            {
                "category": category,
                "calls": len(durations),
                "total_ms": duration_us / 1000.0,
                "share": duration_us / total_us if total_us else 0.0,
            }
        )
    category_rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return {
        "trace": str(path),
        "kernel_events": event_count,
        "total_kernel_ms": total_us / 1000.0,
        "gpu_span_ms": span_us / 1000.0,
        "gpu_busy_union_ms": busy_us / 1000.0,
        "gpu_idle_gap_ms": max(span_us - busy_us, 0.0) / 1000.0,
        "gaps_over_5us": sum(gap > 5.0 for gap in gaps),
        "gap_over_5us_ms": sum(gap for gap in gaps if gap > 5.0) / 1000.0,
        "categories": category_rows,
        "rows": rows,
    }


def subtract_summaries(full: dict, prefill: dict, decode_tokens: int) -> dict:
    if decode_tokens <= 0:
        raise ValueError("--decode-tokens must be positive")
    prefill_categories = {row["category"]: row for row in prefill["categories"]}
    categories = []
    names = {row["category"] for row in full["categories"]} | set(prefill_categories)
    full_categories = {row["category"]: row for row in full["categories"]}
    for name in names:
        full_row = full_categories.get(name, {})
        prefill_row = prefill_categories.get(name, {})
        categories.append(
            {
                "category": name,
                "calls_per_token": (
                    full_row.get("calls", 0) - prefill_row.get("calls", 0)
                )
                / decode_tokens,
                "ms_per_token": max(
                    full_row.get("total_ms", 0.0) - prefill_row.get("total_ms", 0.0),
                    0.0,
                )
                / decode_tokens,
            }
        )
    categories.sort(key=lambda row: row["ms_per_token"], reverse=True)
    full_rows = {row["name"]: row for row in full["rows"]}
    prefill_rows = {row["name"]: row for row in prefill["rows"]}
    rows = []
    for name in set(full_rows) | set(prefill_rows):
        full_row = full_rows.get(name, {})
        prefill_row = prefill_rows.get(name, {})
        delta_ms = full_row.get("total_ms", 0.0) - prefill_row.get("total_ms", 0.0)
        delta_calls = full_row.get("calls", 0) - prefill_row.get("calls", 0)
        if delta_ms <= 0.0 and delta_calls <= 0:
            continue
        rows.append(
            {
                "name": name,
                "category": category_for(name),
                "calls_per_token": max(delta_calls, 0) / decode_tokens,
                "ms_per_token": max(delta_ms, 0.0) / decode_tokens,
            }
        )
    rows.sort(key=lambda row: row["ms_per_token"], reverse=True)
    return {
        "full_trace": full["trace"],
        "prefill_trace": prefill["trace"],
        "decode_tokens": decode_tokens,
        "kernel_ms_per_token": max(
            full["total_kernel_ms"] - prefill["total_kernel_ms"], 0.0
        )
        / decode_tokens,
        "gpu_span_ms_per_token": max(
            full["gpu_span_ms"] - prefill["gpu_span_ms"], 0.0
        )
        / decode_tokens,
        "gpu_idle_gap_ms_per_token": max(
            full["gpu_idle_gap_ms"] - prefill["gpu_idle_gap_ms"], 0.0
        )
        / decode_tokens,
        "categories": categories,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--include", help="regular expression matched against kernel name")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--subtract", type=Path, help="prefill trace to subtract")
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.top <= 0:
        raise ValueError("--top must be positive")
    pattern = re.compile(args.include) if args.include else None

    summary = summarize(args.trace, pattern)
    if args.subtract is not None:
        result = subtract_summaries(
            summary, summarize(args.subtract, pattern), args.decode_tokens
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    summary["rows"] = summary["rows"][: args.top]

    if args.as_json:
        print(
            json.dumps(
                {
                    **summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(
        f"trace={args.trace} kernel_events={summary['kernel_events']} "
        f"total_kernel_ms={summary['total_kernel_ms']:.6f} "
        f"gpu_span_ms={summary['gpu_span_ms']:.6f} "
        f"gpu_idle_gap_ms={summary['gpu_idle_gap_ms']:.6f}"
    )
    print("categories:")
    for row in summary["categories"]:
        print(
            f"  {row['total_ms']:12.6f} {row['share'] * 100:8.2f}% "
            f"{row['calls']:8d}  {row['category']}"
        )
    print(f"{'total_ms':>12} {'share':>9} {'calls':>8} {'mean_us':>12}  name")
    for row in summary["rows"]:
        print(
            f"{row['total_ms']:12.6f} {row['share'] * 100:8.2f}% "
            f"{row['calls']:8d} {row['mean_us']:12.3f}  {row['name']}"
        )


if __name__ == "__main__":
    main()
