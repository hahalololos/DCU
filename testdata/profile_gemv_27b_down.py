#!/usr/bin/env python3
"""Prototype gfx936 BF16 GEMV kernels for Qwen3.5-27B MLP down projection."""

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def gemv_dot_kernel(
    weight_ptr,
    x_ptr,
    out_ptr,
    m: tl.constexpr,
    k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_idx = k_start + offs_k
        weight = tl.load(
            weight_ptr + offs_m[:, None] * k + k_idx[None, :],
            mask=(offs_m[:, None] < m) & (k_idx[None, :] < k),
            other=0.0,
        )
        x = tl.load(x_ptr + k_idx, mask=k_idx < k, other=0.0)
        acc += tl.dot(weight, x[:, None])
    acc_vector = tl.reshape(acc, (BLOCK_M,))
    tl.store(out_ptr + offs_m, acc_vector, mask=offs_m < m)


def invoke(weight: torch.Tensor, x: torch.Tensor, block_m: int, block_k: int,
           warps: int):
    out = torch.empty(weight.shape[0], dtype=weight.dtype, device=weight.device)
    gemv_dot_kernel[(triton.cdiv(weight.shape[0], block_m), )](
        weight,
        x,
        out,
        m=weight.shape[0],
        k=weight.shape[1],
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=1,
    )
    return out


def bench(fn, repeats: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(0)
    m, k = 5_120, 17_408
    weight = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(k, dtype=torch.bfloat16, device="cuda")
    reference = torch.nn.functional.linear(x, weight)
    baseline_ms = bench(lambda: torch.nn.functional.linear(x, weight), args.repeats)
    print(f"baseline_ms={baseline_ms:.6f}")
    for block_m in (16, 32):
        for block_k in (32, 64, 128):
            for warps in (1, 2, 4):
                try:
                    output = invoke(weight, x, block_m, block_k, warps)
                    torch.cuda.synchronize()
                    diff = (output.float() - reference.float()).abs()
                    elapsed_ms = bench(
                        lambda: invoke(weight, x, block_m, block_k, warps),
                        args.repeats,
                    )
                    print(
                        f"block_m={block_m} block_k={block_k} warps={warps} "
                        f"elapsed_ms={elapsed_ms:.6f} "
                        f"speedup={baseline_ms / elapsed_ms:.4f}x "
                        f"max_abs_diff={diff.max().item():.6f} "
                        f"equal={torch.equal(output, reference)}"
                    )
                except Exception as exc:
                    print(
                        f"block_m={block_m} block_k={block_k} warps={warps} "
                        f"error={type(exc).__name__}:{exc}"
                    )


if __name__ == "__main__":
    main()
