#!/usr/bin/env python3
"""Aggregate paired Qwen3.5-27B service traces into optimization priorities."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


BANDS = ("4-8K", "8-16K", "16-32K")
QUANTILES = ("p50", "p90")
ROUNDS = (1, 2)
WEIGHTS = {"4-8K": 0.20, "8-16K": 0.50, "16-32K": 0.30}


def median(values: list[float]) -> float:
    return statistics.median(values)


def by_key(rows: list[dict], key: str) -> dict[str, dict]:
    return {row[key]: row for row in rows}


def baseline_paths(root: Path, band: str) -> list[Path]:
    if band == "8-16K":
        return [
            root / "baseline-hot2/8-16K_throughput/result.json",
            root / "baseline-hot3/8-16K_throughput/result.json",
        ]
    return [root / f"baseline/{band}_throughput/result.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/profile27b"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root

    baselines = {}
    for band in BANDS:
        rows = [json.loads(path.read_text()) for path in baseline_paths(root, band)]
        baselines[band] = {
            "completed": min(row["completed"] for row in rows),
            "failed": max(row["failed"] for row in rows),
            "output_throughput": median([row["output_throughput"] for row in rows]),
            "request_throughput": median(
                [row["request_throughput"] for row in rows]
            ),
            "p99_ttft_ms": median([row["p99_ttft_ms"] for row in rows]),
            "p99_tpot_ms": median([row["p99_tpot_ms"] for row in rows]),
            "total_input_tokens": rows[0]["total_input_tokens"],
            "total_output_tokens": rows[0]["total_output_tokens"],
            "prefill_wall_share": median(
                [sum(row["ttfts"]) / row["duration"] for row in rows]
            ),
        }
        baselines[band]["decode_wall_share"] = (
            1.0 - baselines[band]["prefill_wall_share"]
        )

    samples = {}
    all_valid = True
    for band in BANDS:
        for quantile in QUANTILES:
            sample = f"{band}-{quantile}"
            rounds = []
            for round_index in ROUNDS:
                prefill = json.loads(
                    (root / f"summaries/{sample}-r{round_index}-o1.json").read_text()
                )
                decode = json.loads(
                    (
                        root / f"summaries/{sample}-r{round_index}-decode.json"
                    ).read_text()
                )
                short_result = json.loads(
                    (
                        root / f"captures/{sample}-r{round_index}-o1/result.json"
                    ).read_text()
                )
                full_result = json.loads(
                    (
                        root / f"captures/{sample}-r{round_index}-o65/result.json"
                    ).read_text()
                )
                valid = (
                    short_result["completed"] == 1
                    and short_result["failed"] == 0
                    and full_result["completed"] == 1
                    and full_result["failed"] == 0
                    and short_result["output_lens"] == [1]
                    and full_result["output_lens"] == [65]
                    and short_result["input_lens"] == full_result["input_lens"]
                    and full_result["generated_texts"][0].startswith(
                        short_result["generated_texts"][0]
                    )
                )
                all_valid &= valid
                rounds.append(
                    {
                        "round": round_index,
                        "valid": valid,
                        "input_tokens": short_result["input_lens"][0],
                        "prefill_kernel_ms": prefill["total_kernel_ms"],
                        "prefill_gap_ms": prefill["gpu_idle_gap_ms"],
                        "prefill_categories": {
                            row["category"]: row["share"]
                            for row in prefill["categories"]
                        },
                        "decode_kernel_ms_per_token": decode["kernel_ms_per_token"],
                        "decode_gap_ms_per_token": decode["gpu_idle_gap_ms_per_token"],
                        "decode_categories": {
                            row["category"]: row["ms_per_token"]
                            for row in decode["categories"]
                        },
                        "decode_rows": decode["rows"],
                    }
                )

            categories_prefill = set().union(
                *(row["prefill_categories"] for row in rounds)
            )
            categories_decode = set().union(
                *(row["decode_categories"] for row in rounds)
            )
            samples[sample] = {
                "band": band,
                "quantile": quantile,
                "input_tokens": rounds[0]["input_tokens"],
                "rounds_valid": all(row["valid"] for row in rounds),
                "prefill_kernel_ms": median(
                    [row["prefill_kernel_ms"] for row in rounds]
                ),
                "prefill_gap_ms": median([row["prefill_gap_ms"] for row in rounds]),
                "prefill_categories": {
                    category: median(
                        [row["prefill_categories"].get(category, 0.0) for row in rounds]
                    )
                    for category in categories_prefill
                },
                "decode_kernel_ms_per_token": median(
                    [row["decode_kernel_ms_per_token"] for row in rounds]
                ),
                "decode_gap_ms_per_token": median(
                    [row["decode_gap_ms_per_token"] for row in rounds]
                ),
                "decode_categories": {
                    category: median(
                        [row["decode_categories"].get(category, 0.0) for row in rounds]
                    )
                    for category in categories_decode
                },
                "prefill_round_delta_pct": abs(
                    rounds[1]["prefill_kernel_ms"] / rounds[0]["prefill_kernel_ms"]
                    - 1.0
                )
                * 100.0,
                "decode_round_delta_pct": abs(
                    rounds[1]["decode_kernel_ms_per_token"]
                    / rounds[0]["decode_kernel_ms_per_token"]
                    - 1.0
                )
                * 100.0,
            }

    bands = {}
    for band in BANDS:
        band_samples = [samples[f"{band}-{quantile}"] for quantile in QUANTILES]
        prefill_categories = set().union(
            *(row["prefill_categories"] for row in band_samples)
        )
        decode_categories = set().union(
            *(row["decode_categories"] for row in band_samples)
        )
        bands[band] = {
            "prefill_kernel_ms": median(
                [row["prefill_kernel_ms"] for row in band_samples]
            ),
            "prefill_gap_ms": median([row["prefill_gap_ms"] for row in band_samples]),
            "prefill_categories": {
                category: median(
                    [row["prefill_categories"].get(category, 0.0) for row in band_samples]
                )
                for category in prefill_categories
            },
            "decode_kernel_ms_per_token": median(
                [row["decode_kernel_ms_per_token"] for row in band_samples]
            ),
            "decode_gap_ms_per_token": median(
                [row["decode_gap_ms_per_token"] for row in band_samples]
            ),
            "decode_categories": {
                category: median(
                    [row["decode_categories"].get(category, 0.0) for row in band_samples]
                )
                for category in decode_categories
            },
        }

    priority_shares = defaultdict(float)
    for band in BANDS:
        weight = WEIGHTS[band]
        prefill_wall = baselines[band]["prefill_wall_share"]
        decode_wall = baselines[band]["decode_wall_share"]
        prefill = bands[band]
        decode_total = prefill["decode_kernel_ms_per_token"]
        priority_shares["decode_gemm"] += (
            weight
            * decode_wall
            * prefill["decode_categories"].get("gemm", 0.0)
            / decode_total
        )
        priority_shares["ua2d"] += (
            weight * prefill_wall * prefill["prefill_categories"].get("ua2d", 0.0)
        )
        priority_shares["ua3d"] += (
            weight
            * decode_wall
            * (
                prefill["decode_categories"].get("ua3d_main", 0.0)
                + prefill["decode_categories"].get("ua3d_merge", 0.0)
            )
            / decode_total
        )
        priority_shares["gdn"] += weight * (
            prefill_wall * prefill["prefill_categories"].get("gdn", 0.0)
            + decode_wall
            * prefill["decode_categories"].get("gdn", 0.0)
            / decode_total
        )
        priority_shares["gpu_idle_gap"] += weight * (
            prefill_wall
            * prefill["prefill_gap_ms"]
            / (prefill["prefill_kernel_ms"] + prefill["prefill_gap_ms"])
            + decode_wall
            * prefill["decode_gap_ms_per_token"]
            / (decode_total + prefill["decode_gap_ms_per_token"])
        )

    priorities = []
    for name, share in priority_shares.items():
        priorities.append(
            {
                "name": name,
                "weighted_e2e_share": share,
                "gain_if_1_1x": share * (1.0 - 1.0 / 1.1),
                "gain_if_1_2x": share * (1.0 - 1.0 / 1.2),
            }
        )
    priorities.sort(key=lambda row: row["weighted_e2e_share"], reverse=True)

    decode_rows: dict[str, list[float]] = defaultdict(list)
    decode_calls: dict[str, list[float]] = defaultdict(list)
    for band in BANDS:
        for quantile in QUANTILES:
            sample = f"{band}-{quantile}"
            for round_index in ROUNDS:
                decode = json.loads(
                    (
                        root / f"summaries/{sample}-r{round_index}-decode.json"
                    ).read_text()
                )
                for row in decode["rows"]:
                    decode_rows[row["name"]].append(row["ms_per_token"])
                    decode_calls[row["name"]].append(row["calls_per_token"])
    top_decode_kernels = [
        {
            "name": name,
            "median_ms_per_token": median(values),
            "median_calls_per_token": median(decode_calls[name]),
        }
        for name, values in decode_rows.items()
    ]
    top_decode_kernels.sort(
        key=lambda row: row["median_ms_per_token"], reverse=True
    )

    result = {
        "all_pairs_valid": all_valid,
        "baselines": baselines,
        "samples": samples,
        "bands": bands,
        "priorities": priorities,
        "top_decode_kernels": top_decode_kernels[:20],
        "max_prefill_round_delta_pct": max(
            row["prefill_round_delta_pct"] for row in samples.values()
        ),
        "max_decode_round_delta_pct": max(
            row["decode_round_delta_pct"] for row in samples.values()
        ),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
