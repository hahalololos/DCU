#!/usr/bin/env python3
"""重放 OpenCompass 保存的 prompt，用于 4B 数值路径精度筛选。"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DATASET_MAX_TOKENS = {
    "hotpotqa": 1024,
    "gov_report": 1024,
    "retrieval_multi_point": 96,
    "aggregation_keyword_aggregation": 128,
}

ROLE_MAP = {
    "HUMAN": "user",
    "BOT": "assistant",
    "SYSTEM": "system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8002/v1/chat/completions",
    )
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def request_completion(
    endpoint: str,
    model: str,
    origin_prompt: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
) -> str:
    messages = [
        {
            "role": ROLE_MAP.get(str(item["role"]).upper(),
                                 str(item["role"]).lower()),
            "content": item["prompt"],
        }
        for item in origin_prompt
    ]
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": "Bearer EMPTY",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    key = item[0]
    try:
        return int(key), key
    except ValueError:
        return 2**31 - 1, key


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for dataset, max_tokens in DATASET_MAX_TOKENS.items():
        source = args.baseline_dir / f"{dataset}.json"
        rows = json.loads(source.read_text(encoding="utf-8"))
        replayed: dict[str, Any] = {}
        exact = 0

        for index, (key, row) in enumerate(sorted(rows.items(), key=sort_key), 1):
            candidate = request_completion(
                args.endpoint,
                args.model,
                row["origin_prompt"],
                max_tokens,
                args.timeout,
            )
            baseline = row.get("prediction", "")
            exact += int(candidate == baseline)
            replayed[key] = {
                **row,
                "baseline_prediction": baseline,
                "prediction": candidate,
            }
            if index % 5 == 0 or index == len(rows):
                print(
                    f"{dataset}: {index}/{len(rows)}, exact={exact}",
                    flush=True,
                )

        output = args.output_dir / f"{dataset}.json"
        output.write_text(
            json.dumps(replayed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary[dataset] = {
            "total": len(rows),
            "exact": exact,
            "exact_rate": exact / len(rows) if rows else 0.0,
        }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
