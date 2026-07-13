#!/usr/bin/env python3
"""Select deterministic P50/P90 requests for paired service profiling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[int], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--band", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-output-tokens", type=int, default=65)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    all_rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    input_lens = result["input_lens"]
    output_lens = result["output_lens"]
    num_prompts = result.get("num_prompts", len(input_lens))
    rows = all_rows[:num_prompts]
    if result.get("failed") != 0 or result.get("completed") != num_prompts:
        raise ValueError("baseline must complete every requested row without failures")
    if not (len(rows) == len(input_lens) == len(output_lens)):
        raise ValueError("dataset and detailed result lengths do not match")

    eligible = [
        index
        for index, output_len in enumerate(output_lens)
        if output_len >= args.min_output_tokens
    ]
    if len(eligible) < 2:
        raise ValueError("fewer than two requests naturally generated enough tokens")

    targets = {"p50": percentile(input_lens, 0.50), "p90": percentile(input_lens, 0.90)}
    selected: dict[str, int] = {}
    used: set[int] = set()
    for label, target in targets.items():
        choices = [index for index in eligible if index not in used]
        index = min(
            choices,
            key=lambda item: (abs(input_lens[item] - target), -output_lens[item], item),
        )
        selected[label] = index
        used.add(index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "band": args.band,
        "baseline_result": str(args.result),
        "dataset": str(args.dataset),
        "min_output_tokens": args.min_output_tokens,
        "targets": targets,
        "selected": {},
    }
    for label, index in selected.items():
        subset = args.output_dir / f"{args.band}-{label}.jsonl"
        subset.write_text(json.dumps(rows[index], ensure_ascii=False) + "\n")
        manifest["selected"][label] = {
            "zero_based_index": index,
            "one_based_line": index + 1,
            "input_tokens": input_lens[index],
            "baseline_output_tokens": output_lens[index],
            "subset": str(subset),
        }
    manifest_path = args.output_dir / f"{args.band}-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
