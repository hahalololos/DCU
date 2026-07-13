#!/usr/bin/env python3
"""Summarize GPU kernel events from a Torch Profiler Chrome trace."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--include", help="regular expression matched against kernel name")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    if args.top <= 0:
        raise ValueError("--top must be positive")
    pattern = re.compile(args.include) if args.include else None

    payload = json.loads(args.trace.read_text())
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    grouped: dict[str, list[float]] = defaultdict(list)
    total_us = 0.0
    event_count = 0
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        name = event.get("name", "<unnamed>")
        if pattern is not None and pattern.search(name) is None:
            continue
        duration_us = float(event.get("dur", 0.0))
        grouped[name].append(duration_us)
        total_us += duration_us
        event_count += 1

    rows = []
    for name, durations in grouped.items():
        duration_us = sum(durations)
        rows.append(
            {
                "name": name,
                "calls": len(durations),
                "total_ms": duration_us / 1000.0,
                "mean_us": duration_us / len(durations),
                "share": duration_us / total_us if total_us else 0.0,
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    rows = rows[: args.top]

    if args.as_json:
        print(
            json.dumps(
                {
                    "trace": str(args.trace),
                    "kernel_events": event_count,
                    "total_kernel_ms": total_us / 1000.0,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(
        f"trace={args.trace} kernel_events={event_count} "
        f"total_kernel_ms={total_us / 1000.0:.6f}"
    )
    print(f"{'total_ms':>12} {'share':>9} {'calls':>8} {'mean_us':>12}  name")
    for row in rows:
        print(
            f"{row['total_ms']:12.6f} {row['share'] * 100:8.2f}% "
            f"{row['calls']:8d} {row['mean_us']:12.3f}  {row['name']}"
        )


if __name__ == "__main__":
    main()
