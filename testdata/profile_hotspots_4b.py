#!/usr/bin/env python3
"""Run isolated Qwen3.5-4B hotspot shapes for ROCm profiling."""

import argparse
import math
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
) -> None:
    torch.manual_seed(0)
    ua._is_qwen35_ua3d_scalar_block_candidate = lambda **_: mode == "ua3d"
    device = torch.device("cuda")
    query_len = 4096 if mode == "ua2d" else 1
    num_query_heads = 16 if model_size == "4b" else 24
    num_kv_heads = 4
    head_size = 256
    block_size = 528
    num_blocks = math.ceil(context_len / block_size)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("ua2d", "ua3d", "ua3d_generic", "llmm1_lm_head")
    )
    parser.add_argument("--context-len", type=int, default=22_258)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--model-size", choices=("4b", "27b"), default="4b")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.mode == "llmm1_lm_head":
        run_llmm1_lm_head(args.repeats)
    else:
        run_attention(
            args.mode,
            args.context_len,
            args.repeats,
            args.segments,
            args.model_size,
            args.verify,
        )


if __name__ == "__main__":
    main()
