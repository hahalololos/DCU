#!/usr/bin/env python3
"""Run isolated Qwen3.5-4B/27B hotspot shapes for ROCm profiling."""

import argparse
import math
import os
import statistics
import time

import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.ops import triton_unified_attention as ua


def run_attention(
    mode: str,
    context_len: int,
    repeats: int,
    num_segments: int,
    model_size: str,
    verify: bool,
    ua2d_experiment: int,
    compare_ua2d_experiment: int | None,
) -> None:
    torch.manual_seed(0)
    ua._is_qwen35_ua3d_scalar_block_candidate = lambda **_: mode == "ua3d"
    device = torch.device("cuda")
    query_len = 4096 if mode == "ua2d" else 1
    num_query_heads = 16 if model_size == "4b" else 24
    num_kv_heads = 4
    head_size = 256
    block_size = 528 if model_size == "4b" else 784
    num_blocks = math.ceil(context_len / block_size)
    if mode == "ua2d":
        os.environ["VLLM_ROCM_QWEN_UA2D_EXPERIMENT"] = str(ua2d_experiment)

    query = torch.randn(
        query_len,
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([context_len], dtype=torch.int32, device=device)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        1, -1
    )
    output = torch.empty_like(query)
    seq_threshold_3d = 32
    if mode in ("ua3d", "ua3d_generic"):
        segment_output = torch.empty(
            seq_threshold_3d,
            num_query_heads,
            num_segments,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        segment_max = torch.empty(
            seq_threshold_3d,
            num_query_heads,
            num_segments,
            dtype=torch.float32,
            device=device,
        )
        segment_expsum = torch.empty_like(segment_max)
    else:
        segment_output = None
        segment_max = None
        segment_expsum = None

    def invoke() -> None:
        ua.unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens,
            max_seqlen_q=query_len,
            max_seqlen_k=context_len,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=(
                seq_threshold_3d if mode in ("ua3d", "ua3d_generic") else None
            ),
            num_par_softmax_segments=(
                num_segments if mode in ("ua3d", "ua3d_generic") else None
            ),
            softmax_segm_output=segment_output,
            softmax_segm_max=segment_max,
            softmax_segm_expsum=segment_expsum,
        )

    if mode == "ua2d" and compare_ua2d_experiment is not None:
        experiments = (compare_ua2d_experiment, ua2d_experiment)

        def invoke_experiment(experiment: int) -> None:
            os.environ["VLLM_ROCM_QWEN_UA2D_EXPERIMENT"] = str(experiment)
            invoke()

        for experiment in experiments:
            for _ in range(3):
                invoke_experiment(experiment)
        torch.cuda.synchronize()

        outputs = {}
        for experiment in experiments:
            invoke_experiment(experiment)
            torch.cuda.synchronize()
            outputs[experiment] = output.clone()

        measurements = {experiment: [] for experiment in experiments}
        for round_idx in range(7):
            order = experiments if round_idx % 2 == 0 else experiments[::-1]
            for experiment in order:
                measurements[experiment].append(
                    _time_cuda(
                        lambda experiment=experiment: invoke_experiment(
                            experiment
                        ),
                        repeats,
                    )
                )

        baseline_ms = statistics.median(
            measurements[compare_ua2d_experiment]
        )
        candidate_ms = statistics.median(measurements[ua2d_experiment])
        diff = (
            outputs[ua2d_experiment].float()
            - outputs[compare_ua2d_experiment].float()
        ).abs()
        print(
            f"mode=ua2d_ab model={model_size} context={context_len} "
            f"baseline_experiment={compare_ua2d_experiment} "
            f"candidate_experiment={ua2d_experiment} "
            f"baseline_median_ms={baseline_ms:.6f} "
            f"candidate_median_ms={candidate_ms:.6f} "
            f"speedup={baseline_ms / candidate_ms:.6f} "
            f"baseline_p99_ms={_p99(measurements[compare_ua2d_experiment]):.6f} "
            f"candidate_p99_ms={_p99(measurements[ua2d_experiment]):.6f} "
            f"bitwise={torch.equal(outputs[ua2d_experiment], outputs[compare_ua2d_experiment])} "
            f"max_abs_diff={diff.max().item():.8f}"
        )
        os.environ["VLLM_ROCM_QWEN_UA2D_EXPERIMENT"] = str(
            ua2d_experiment
        )
        return

    for _ in range(3):
        invoke()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        invoke()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"mode={mode} model={model_size} context={context_len} "
        f"segments={num_segments} repeats={repeats} "
        f"average_ms={elapsed * 1000 / repeats:.6f}"
    )
    if verify and mode == "ua3d":
        candidate = output.clone()
        ua._is_qwen35_ua3d_scalar_block_candidate = lambda **_: False
        invoke()
        torch.cuda.synchronize()
        diff = (candidate.float() - output.float()).abs()
        print(
            f"verify_equal={torch.equal(candidate, output)} "
            f"max_abs_diff={diff.max().item():.8f}"
        )
    elif verify and mode == "ua2d":
        candidate = output.clone()
        previous = os.environ.get("VLLM_ROCM_QWEN_UA2D_EXPERIMENT")
        os.environ["VLLM_ROCM_QWEN_UA2D_EXPERIMENT"] = "5"
        try:
            invoke()
            torch.cuda.synchronize()
        finally:
            if previous is None:
                os.environ.pop("VLLM_ROCM_QWEN_UA2D_EXPERIMENT", None)
            else:
                os.environ["VLLM_ROCM_QWEN_UA2D_EXPERIMENT"] = previous
        diff = (candidate.float() - output.float()).abs()
        print(
            f"verify_equal={torch.equal(candidate, output)} "
            f"max_abs_diff={diff.max().item():.8f}"
        )


def run_llmm1_lm_head(repeats: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    output_features = 248_320
    input_features = 2_560
    x = torch.rand(1, input_features, dtype=torch.bfloat16, device=device)
    weight = torch.rand(
        output_features,
        input_features,
        dtype=torch.bfloat16,
        device=device,
    )

    for _ in range(3):
        ops.LLMM1(weight, x, 4)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        ops.LLMM1(weight, x, 4)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        f"mode=llmm1_lm_head repeats={repeats} "
        f"average_ms={elapsed * 1000 / repeats:.6f}"
    )


LINEAR_SHAPES = {
    "4b": (
        ("gdn_qkvz", 12_288, 2_560),
        ("gdn_ba", 64, 2_560),
        ("mlp_gate_up", 18_432, 2_560),
        ("mlp_down", 2_560, 9_216),
        ("lm_head", 248_320, 2_560),
    ),
    "27b": (
        ("gdn_qkvz", 16_384, 5_120),
        ("gdn_ba", 96, 5_120),
        ("attn_qkv_gate", 14_336, 5_120),
        ("attn_out", 5_120, 6_144),
        ("mlp_gate_up", 34_816, 5_120),
        ("mlp_down", 5_120, 17_408),
        ("lm_head", 248_320, 5_120),
    ),
}

LLMM1_ROWS_SHAPES_27B = (
    ("gdn_qkvz", 16_384, 5_120, 48),
    ("gdn_ba", 96, 5_120, 48),
    ("attn_qkv_gate", 14_336, 5_120, 16),
    ("mlp_gate_up", 34_816, 5_120, 64),
)
LLMM1_ROWS_PER_BLOCK = (2, 4, 8, 16)


def _benchmark_cuda(fn, repeats: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def _time_cuda(fn, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def _p99(values: list[float]) -> float:
    return sorted(values)[math.ceil(len(values) * 0.99) - 1]


def run_llmm1_rows(repeats: int, rounds: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    for name, output_features, input_features, calls_per_token in (
        LLMM1_ROWS_SHAPES_27B
    ):
        x = torch.rand(1, input_features, dtype=torch.bfloat16, device=device)
        weight = torch.rand(
            output_features,
            input_features,
            dtype=torch.bfloat16,
            device=device,
        )
        runners = {
            "linear": lambda: torch.nn.functional.linear(x, weight),
            **{
                f"row{rows}": lambda rows=rows: ops.LLMM1(weight, x, rows)
                for rows in LLMM1_ROWS_PER_BLOCK
            },
        }

        for runner in runners.values():
            for _ in range(5):
                runner()
        torch.cuda.synchronize()

        outputs = {backend: runner() for backend, runner in runners.items()}
        torch.cuda.synchronize()
        row4_output = outputs["row4"]
        linear_output = outputs["linear"]

        measurements = {backend: [] for backend in runners}
        backend_order = list(runners)
        for round_idx in range(rounds):
            order = backend_order if round_idx % 2 == 0 else backend_order[::-1]
            for backend in order:
                measurements[backend].append(_time_cuda(runners[backend], repeats))

        medians = {
            backend: statistics.median(values)
            for backend, values in measurements.items()
        }
        row4_median = medians["row4"]
        best_rows = min(LLMM1_ROWS_PER_BLOCK, key=lambda rows: medians[f"row{rows}"])
        best_median = medians[f"row{best_rows}"]
        saved_ms_per_call = row4_median - best_median
        print(
            f"mode=llmm1_rows model=27b name={name} m={output_features} n=1 "
            f"k={input_features} calls_per_token={calls_per_token} "
            f"best_row={best_rows} saved_ms_per_call={saved_ms_per_call:.6f} "
            f"saved_ms_per_token={saved_ms_per_call * calls_per_token:.6f}"
        )
        for backend in runners:
            output = outputs[backend]
            diff = (output.float() - linear_output.float()).abs()
            bitwise_row4 = torch.equal(output, row4_output)
            speedup_vs_row4 = row4_median / medians[backend]
            print(
                f"backend={backend} median_ms={medians[backend]:.6f} "
                f"p99_ms={_p99(measurements[backend]):.6f} "
                f"speedup_vs_row4={speedup_vs_row4:.6f} "
                f"bitwise_row4={bitwise_row4} "
                f"max_abs_diff_vs_linear={diff.max().item():.8f} "
                f"finite={torch.isfinite(output).all().item()}"
            )

        del x, weight, outputs
        torch.cuda.empty_cache()


def run_llmm1_gfx936_shapes(repeats: int, rounds: int) -> None:
    device = torch.device("cuda")
    input_features = 5_120
    shape_calls = (
        ("gdn_qkvz", 16_384, 48),
        ("attn_qkv_gate", 14_336, 16),
        ("mlp_gate_up", 34_816, 64),
    )
    for seed in (0, 1, 17):
        torch.manual_seed(seed)
        for name, output_features, calls_per_token in shape_calls:
            x = (
                torch.rand(1, input_features, dtype=torch.bfloat16, device=device)
                * 2
                - 1
            )
            weight = (
                torch.rand(
                    output_features,
                    input_features,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 2
                - 1
            )
            runners = {
                "linear": lambda: torch.nn.functional.linear(x, weight),
                "llmm1": lambda: ops.LLMM1(weight, x, 4),
                **{
                    f"v{variant}": lambda variant=variant: ops.LLMM1Gfx936(
                        weight, x, variant
                    )
                    for variant in range(1, 5)
                },
            }
            for runner in runners.values():
                for _ in range(5):
                    runner()
            torch.cuda.synchronize()

            outputs = {backend: runner() for backend, runner in runners.items()}
            torch.cuda.synchronize()
            measurements = {backend: [] for backend in runners}
            order = list(runners)
            for round_idx in range(rounds):
                round_order = order if round_idx % 2 == 0 else order[::-1]
                for backend in round_order:
                    measurements[backend].append(
                        _time_cuda(runners[backend], repeats)
                    )

            medians = {
                backend: statistics.median(values)
                for backend, values in measurements.items()
            }
            llmm1_output = outputs["llmm1"]
            linear_output = outputs["linear"]
            print(
                f"mode=llmm1_gfx936_shapes seed={seed} name={name} "
                f"m={output_features} n=1 k={input_features} "
                f"calls_per_token={calls_per_token}"
            )
            for backend, output in outputs.items():
                diff = (output.float() - linear_output.float()).abs()
                saved_per_call = medians["llmm1"] - medians[backend]
                print(
                    f"backend={backend} median_ms={medians[backend]:.6f} "
                    f"p99_ms={_p99(measurements[backend]):.6f} "
                    f"speedup_vs_llmm1={medians['llmm1'] / medians[backend]:.6f} "
                    f"saved_ms_per_token={saved_per_call * calls_per_token:.6f} "
                    f"bitwise_llmm1={torch.equal(output, llmm1_output)} "
                    f"max_abs_diff_vs_linear={diff.max().item():.8f} "
                    f"mean_abs_diff_vs_linear={diff.mean().item():.8f} "
                    f"finite={torch.isfinite(output).all().item()}"
                )
            del x, weight, outputs
            torch.cuda.empty_cache()


def run_linear_shapes(model_size: str, repeats: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    for name, output_features, input_features in LINEAR_SHAPES[model_size]:
        x = torch.rand(1, input_features, dtype=torch.bfloat16, device=device)
        weight = torch.rand(
            output_features,
            input_features,
            dtype=torch.bfloat16,
            device=device,
        )
        linear_ms = _benchmark_cuda(
            lambda: torch.nn.functional.linear(x, weight), repeats
        )
        llmm1_ms = None
        # The current LLMM1 launch derives its thread count from K and cannot
        # launch safely once the resulting block exceeds the HIP thread limit.
        if input_features <= 8_192 and output_features % 4 == 0:
            llmm1_ms = _benchmark_cuda(lambda: ops.LLMM1(weight, x, 4), repeats)
        result = (
            f"mode=linear model={model_size} name={name} "
            f"m={output_features} n=1 k={input_features} "
            f"linear_ms={linear_ms:.6f}"
        )
        if llmm1_ms is not None:
            result += (
                f" llmm1_ms={llmm1_ms:.6f} "
                f"speedup={linear_ms / llmm1_ms:.4f}x"
            )
        print(result)
        del x, weight
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "ua2d",
            "ua3d",
            "ua3d_generic",
            "llmm1_lm_head",
            "llmm1_rows",
            "llmm1_gate_up_gfx936",
            "llmm1_gfx936_shapes",
            "linear_shapes",
        ),
    )
    parser.add_argument("--context-len", type=int, default=22_258)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--model-size", choices=("4b", "27b"), default="4b")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--ua2d-experiment", type=int, default=5)
    parser.add_argument("--compare-ua2d-experiment", type=int)
    args = parser.parse_args()

    if args.mode == "linear_shapes":
        run_linear_shapes(args.model_size, args.repeats)
    elif args.mode == "llmm1_gfx936_shapes":
        run_llmm1_gfx936_shapes(args.repeats, args.rounds)
    elif args.mode == "llmm1_gate_up_gfx936":
        run_llmm1_gfx936_shapes(args.repeats, args.rounds)
    elif args.mode == "llmm1_rows":
        run_llmm1_rows(args.repeats, args.rounds)
    elif args.mode == "llmm1_lm_head":
        run_llmm1_lm_head(args.repeats)
    else:
        run_attention(
            args.mode,
            args.context_len,
            args.repeats,
            args.segments,
            args.model_size,
            args.verify,
            args.ua2d_experiment,
            args.compare_ua2d_experiment,
        )


if __name__ == "__main__":
    main()
