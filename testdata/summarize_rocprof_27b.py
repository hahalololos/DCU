#!/usr/bin/env python3
"""汇总 Qwen3.5-27B 热点 rocprof CSV。

Linear CSV 来自 ``profile_hotspots_4b.py linear_shapes``。该命令按固定形状顺序
执行，每个被测 kernel 在 rocprof 合并 CSV 中保留四次采样。本脚本用 Index 区间和
kernel 名双重校验形状映射，避免把不同 Linear 形状的同名 kernel 混在一起。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


COUNTERS = (
    "SQ_WAVES",
    "SQ_INSTS_VALU",
    "SQ_INSTS_SALU",
    "SQ_LDS_BANK_CONFLICT",
    "FETCH_SIZE",
    "WRITE_SIZE",
    "TCC_HIT_sum",
    "TCC_MISS_sum",
)
RESOURCE_FIELDS = (
    "grd",
    "wgr",
    "lds",
    "scr",
    "arch_vgpr",
    "accum_vgpr",
    "sgpr",
    "wave_size",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def numeric_median(rows: list[dict[str, str]], field: str) -> float | int:
    values = [float(row[field]) for row in rows if row.get(field, "") != ""]
    if not values:
        raise ValueError(f"字段 {field} 没有数值")
    value = statistics.median(values)
    return int(value) if value.is_integer() else value


def summarize(rows: list[dict[str, str]], label: str) -> dict:
    if not rows:
        raise ValueError(f"{label} 未匹配到 kernel")
    kernel_names = sorted({row["KernelName"] for row in rows})
    if len(kernel_names) != 1:
        raise ValueError(f"{label} 匹配到多个 kernel: {kernel_names}")
    result = {
        "label": label,
        "kernel_name": kernel_names[0],
        "sample_count": len(rows),
        "index": [int(row["Index"]) for row in rows],
        "resources": {
            field: numeric_median(rows, field) for field in RESOURCE_FIELDS
        },
        "counters": {field: numeric_median(rows, field) for field in COUNTERS},
    }
    hit = float(result["counters"]["TCC_HIT_sum"])
    miss = float(result["counters"]["TCC_MISS_sum"])
    result["l2_hit_ratio"] = hit / (hit + miss) if hit + miss else None
    return result


def select_name(rows: list[dict[str, str]], needle: str) -> list[dict[str, str]]:
    return [row for row in rows if needle in row["KernelName"]]


def select_indexed(
    rows: list[dict[str, str]],
    start: int,
    end: int,
    needle: str,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if start <= int(row["Index"]) <= end and needle in row["KernelName"]
    ]
    if len(selected) != 4:
        raise ValueError(
            f"Index {start}..{end} / {needle} 预期 4 行，实际 {len(selected)} 行"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ua2d_rows = read_csv(args.input_dir / "ua2d.csv")
    ua3d_rows = read_csv(args.input_dir / "ua3d.csv")
    linear_rows = read_csv(args.input_dir / "linear.csv")

    # Index 区间对应 linear_shapes 的固定执行顺序。F.linear 的 PostGSU 辅助核不计入
    # 主核资源统计；需要时可从原始 CSV 单独查看。
    shapes = {
        "gdn_qkvz": {
            "shape": [16384, 1, 5120],
            "f_linear": (2, 5, "Cijk_"),
            "llmm1": (6, 9, "LLGemm1_kernel"),
        },
        "gdn_ba": {
            "shape": [96, 1, 5120],
            "f_linear": (12, 19, "Cijk_Alik"),
            "llmm1": (20, 23, "LLGemm1_kernel"),
        },
        "attn_qkv_gate": {
            "shape": [14336, 1, 5120],
            "f_linear": (26, 33, "Cijk_Alik"),
            "llmm1": (34, 37, "LLGemm1_kernel"),
        },
        "attn_out": {
            "shape": [5120, 1, 6144],
            "f_linear": (40, 43, "Cijk_Alik"),
            "llmm1": (44, 47, "LLGemm1_kernel"),
        },
        "mlp_gate_up": {
            "shape": [34816, 1, 5120],
            "f_linear": (50, 53, "Cijk_Alik"),
            "llmm1": (54, 57, "LLGemm1_kernel"),
        },
        "mlp_down": {
            "shape": [5120, 1, 17408],
            "f_linear": (60, 63, "Cijk_Alik"),
        },
        "lm_head": {
            "shape": [248320, 1, 5120],
            "f_linear": (67, 70, "Cijk_Alik"),
            "llmm1": (71, 74, "LLGemm1_kernel"),
        },
    }

    linear = {}
    for shape_name, spec in shapes.items():
        entry = {"shape_m_n_k": spec["shape"]}
        for implementation in ("f_linear", "llmm1"):
            if implementation not in spec:
                continue
            start, end, needle = spec[implementation]
            entry[implementation] = summarize(
                select_indexed(linear_rows, start, end, needle),
                f"{shape_name}/{implementation}",
            )
        linear[shape_name] = entry

    output = {
        "source": {
            "model": "Qwen3.5-27B",
            "context_len": 22294,
            "rocprof_note": "计数器采集会重放应用，仅用于瓶颈分类，不用于耗时对比",
        },
        "ua2d": summarize(
            select_name(ua2d_rows, "kernel_qwen35_unified_attention_2d_v2b.kd"),
            "ua2d/main",
        ),
        "ua3d": {
            "main": summarize(
                select_name(ua3d_rows, "kernel_unified_attention_3d.kd"),
                "ua3d/main",
            ),
            "merge": summarize(
                select_name(ua3d_rows, "reduce_segments.kd"),
                "ua3d/merge",
            ),
        },
        "linear": linear,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
