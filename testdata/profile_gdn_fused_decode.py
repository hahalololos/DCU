#!/usr/bin/env python3
"""Prototype Qwen3.5 decode conv + recurrent fusion on ROCm/DCU.

The candidate keeps a tiny second kernel only for sliding the convolution
state.  The main kernel consumes the raw projection and old convolution state
directly, avoiding the full mixed_qkv write/read between causal-conv and the
packed recurrent kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict

import torch
import triton

from profile_gdn_4b_27b import GDNShape, SHAPES, benchmark_cuda, percentile
from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.triton_utils import tl


@triton.jit
def _fused_conv_recurrent_decode_kernel(
    mixed_qkv,
    conv_state,
    conv_weight,
    conv_bias,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    state_indices,
    scale,
    stride_mixed_tok: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_h_tok: tl.constexpr,
    stride_indices: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(state_indices + i_n * stride_indices).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx < 0:
        tl.store(p_o, tl.zeros([BV], tl.float32), mask=mask_v)
        return

    q_off = i_h * K + o_k
    k_off = H * K + i_h * K + o_k
    v_off = 2 * H * K + i_hv * V + o_v
    p_x = mixed_qkv + i_n * stride_mixed_tok

    q0 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + q_off * stride_conv_state_dim,
        mask=mask_k,
        other=0.0,
    )
    q1 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + q_off * stride_conv_state_dim
        + stride_conv_state_tok,
        mask=mask_k,
        other=0.0,
    )
    q2 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + q_off * stride_conv_state_dim
        + 2 * stride_conv_state_tok,
        mask=mask_k,
        other=0.0,
    )
    qx = tl.load(p_x + q_off, mask=mask_k, other=0.0)
    qw = conv_weight + q_off * stride_weight_dim
    q_acc = tl.zeros([BK], tl.float32)
    if HAS_BIAS:
        q_acc += tl.load(conv_bias + q_off, mask=mask_k, other=0.0)
    q_acc += q0 * tl.load(qw, mask=mask_k, other=0.0)
    q_acc += q1 * tl.load(
        qw + stride_weight_width, mask=mask_k, other=0.0
    )
    q_acc += q2 * tl.load(
        qw + 2 * stride_weight_width, mask=mask_k, other=0.0
    )
    q_acc += qx * tl.load(
        qw + 3 * stride_weight_width, mask=mask_k, other=0.0
    )
    q_acc = q_acc / (1.0 + tl.exp(-q_acc))
    b_q = q_acc.to(mixed_qkv.dtype.element_ty).to(tl.float32)

    k0 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + k_off * stride_conv_state_dim,
        mask=mask_k,
        other=0.0,
    )
    k1 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + k_off * stride_conv_state_dim
        + stride_conv_state_tok,
        mask=mask_k,
        other=0.0,
    )
    k2 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + k_off * stride_conv_state_dim
        + 2 * stride_conv_state_tok,
        mask=mask_k,
        other=0.0,
    )
    kx = tl.load(p_x + k_off, mask=mask_k, other=0.0)
    kw = conv_weight + k_off * stride_weight_dim
    k_acc = tl.zeros([BK], tl.float32)
    if HAS_BIAS:
        k_acc += tl.load(conv_bias + k_off, mask=mask_k, other=0.0)
    k_acc += k0 * tl.load(kw, mask=mask_k, other=0.0)
    k_acc += k1 * tl.load(
        kw + stride_weight_width, mask=mask_k, other=0.0
    )
    k_acc += k2 * tl.load(
        kw + 2 * stride_weight_width, mask=mask_k, other=0.0
    )
    k_acc += kx * tl.load(
        kw + 3 * stride_weight_width, mask=mask_k, other=0.0
    )
    k_acc = k_acc / (1.0 + tl.exp(-k_acc))
    b_k = k_acc.to(mixed_qkv.dtype.element_ty).to(tl.float32)

    v0 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + v_off * stride_conv_state_dim,
        mask=mask_v,
        other=0.0,
    )
    v1 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + v_off * stride_conv_state_dim
        + stride_conv_state_tok,
        mask=mask_v,
        other=0.0,
    )
    v2 = tl.load(
        conv_state
        + state_idx * stride_conv_state_seq
        + v_off * stride_conv_state_dim
        + 2 * stride_conv_state_tok,
        mask=mask_v,
        other=0.0,
    )
    vx = tl.load(p_x + v_off, mask=mask_v, other=0.0)
    vw = conv_weight + v_off * stride_weight_dim
    v_acc = tl.zeros([BV], tl.float32)
    if HAS_BIAS:
        v_acc += tl.load(conv_bias + v_off, mask=mask_v, other=0.0)
    v_acc += v0 * tl.load(vw, mask=mask_v, other=0.0)
    v_acc += v1 * tl.load(
        vw + stride_weight_width, mask=mask_v, other=0.0
    )
    v_acc += v2 * tl.load(
        vw + 2 * stride_weight_width, mask=mask_v, other=0.0
    )
    v_acc += vx * tl.load(
        vw + 3 * stride_weight_width, mask=mask_v, other=0.0
    )
    v_acc = v_acc / (1.0 + tl.exp(-v_acc))
    b_v = v_acc.to(mixed_qkv.dtype.element_ty).to(tl.float32)

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q *= scale

    p_h = (
        h0
        + state_idx * stride_h_tok
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(
        x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x
    )
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)

    p_ht = (
        ht
        + state_idx * stride_h_tok
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    tl.store(p_ht, b_h.to(ht.dtype.element_ty), mask=mask_h)


@triton.jit
def _slide_conv_state_kernel(
    mixed_qkv,
    conv_state,
    state_indices,
    stride_mixed_tok: tl.constexpr,
    stride_state_seq: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_tok: tl.constexpr,
    stride_indices: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    i_n, i_block = tl.program_id(0), tl.program_id(1)
    state_idx = tl.load(state_indices + i_n * stride_indices).to(tl.int64)
    if state_idx < 0:
        return
    offs = i_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < DIM
    base = (
        conv_state
        + state_idx * stride_state_seq
        + offs * stride_state_dim
    )
    col1 = tl.load(base + stride_state_tok, mask=mask, other=0.0)
    col2 = tl.load(base + 2 * stride_state_tok, mask=mask, other=0.0)
    x = tl.load(
        mixed_qkv + i_n * stride_mixed_tok + offs, mask=mask, other=0.0
    )
    tl.store(base, col1, mask=mask)
    tl.store(base + stride_state_tok, col2, mask=mask)
    tl.store(base + 2 * stride_state_tok, x, mask=mask)


def make_inputs(
    shape: GDNShape,
    batch_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
):
    torch.manual_seed(0)
    mixed = torch.randn(batch_size, shape.qkv_dim, device=device, dtype=dtype)
    weight = torch.randn(
        shape.qkv_dim, shape.conv_width, device=device, dtype=dtype
    ) * 0.02
    bias = torch.randn(shape.qkv_dim, device=device, dtype=dtype) * 0.02
    conv_storage = torch.randn(
        batch_size,
        shape.conv_width - 1,
        shape.qkv_dim,
        device=device,
        dtype=dtype,
    ) * 0.02
    conv_state = conv_storage.transpose(-1, -2)
    a = torch.randn(batch_size, shape.num_v_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.zeros(shape.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)
    recurrent = torch.randn(
        batch_size,
        shape.num_v_heads,
        shape.head_v_dim,
        shape.head_k_dim,
        device=device,
        dtype=state_dtype,
    ) * 0.01
    out = torch.empty(
        batch_size,
        1,
        shape.num_v_heads,
        shape.head_v_dim,
        device=device,
        dtype=dtype,
    )
    indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    return {
        "mixed": mixed,
        "weight": weight,
        "bias": bias,
        "conv_state": conv_state,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "recurrent": recurrent,
        "out": out,
        "indices": indices,
    }


def run_reference(shape: GDNShape, tensors: dict[str, torch.Tensor]):
    mixed = causal_conv1d_update(
        tensors["mixed"],
        tensors["conv_state"],
        tensors["weight"],
        bias=tensors["bias"],
        activation="silu",
        conv_state_indices=tensors["indices"],
        validate_data=False,
    )
    return fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=tensors["a"],
        b=tensors["b"],
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        scale=shape.head_k_dim**-0.5,
        initial_state=tensors["recurrent"],
        out=tensors["out"],
        ssm_state_indices=tensors["indices"],
        use_qk_l2norm_in_kernel=True,
    )


def run_candidate(
    shape: GDNShape,
    tensors: dict[str, torch.Tensor],
    bv: int,
    num_warps: int,
    num_stages: int,
):
    mixed = tensors["mixed"]
    conv_state = tensors["conv_state"]
    recurrent = tensors["recurrent"]
    indices = tensors["indices"]
    grid = (triton.cdiv(shape.head_v_dim, bv), mixed.shape[0] * shape.num_v_heads)
    _fused_conv_recurrent_decode_kernel[grid](
        mixed_qkv=mixed,
        conv_state=conv_state,
        conv_weight=tensors["weight"],
        conv_bias=tensors["bias"],
        a=tensors["a"],
        b=tensors["b"],
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        o=tensors["out"],
        h0=recurrent,
        ht=recurrent,
        state_indices=indices,
        scale=shape.head_k_dim**-0.5,
        stride_mixed_tok=mixed.stride(0),
        stride_conv_state_seq=conv_state.stride(0),
        stride_conv_state_dim=conv_state.stride(1),
        stride_conv_state_tok=conv_state.stride(2),
        stride_weight_dim=tensors["weight"].stride(0),
        stride_weight_width=tensors["weight"].stride(1),
        stride_a_tok=tensors["a"].stride(0),
        stride_b_tok=tensors["b"].stride(0),
        stride_h_tok=recurrent.stride(0),
        stride_indices=indices.stride(0),
        H=shape.num_k_heads,
        HV=shape.num_v_heads,
        K=shape.head_k_dim,
        V=shape.head_v_dim,
        BK=triton.next_power_of_2(shape.head_k_dim),
        BV=bv,
        HAS_BIAS=True,
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _slide_conv_state_kernel[(mixed.shape[0], triton.cdiv(shape.qkv_dim, 256))](
        mixed_qkv=mixed,
        conv_state=conv_state,
        state_indices=indices,
        stride_mixed_tok=mixed.stride(0),
        stride_state_seq=conv_state.stride(0),
        stride_state_dim=conv_state.stride(1),
        stride_state_tok=conv_state.stride(2),
        stride_indices=indices.stride(0),
        DIM=shape.qkv_dim,
        BLOCK_N=256,
        num_warps=4,
        num_stages=1,
    )
    return tensors["out"], recurrent


def clone_inputs(tensors: dict[str, torch.Tensor]):
    cloned = {name: tensor.clone() for name, tensor in tensors.items()}
    # Preserve the runtime [N, DIM, width-1] non-contiguous layout.
    storage = tensors["conv_state"].transpose(-1, -2).clone()
    cloned["conv_state"] = storage.transpose(-1, -2)
    return cloned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(SHAPES), default="4b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bv", type=int, choices=(16, 32, 64, 128), default=32)
    parser.add_argument("--num-warps", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-stages", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    shape = SHAPES[args.model]
    device = torch.device("cuda")
    base = make_inputs(shape, args.batch_size, torch.bfloat16, torch.float32, device)

    ref_check = clone_inputs(base)
    cand_check = clone_inputs(base)
    run_reference(shape, ref_check)
    run_candidate(shape, cand_check, args.bv, args.num_warps, args.num_stages)
    torch.cuda.synchronize()

    correctness = {}
    for name in ("out", "recurrent", "conv_state"):
        ref = ref_check[name]
        cand = cand_check[name]
        diff = (ref.float() - cand.float()).abs()
        correctness[name] = {
            "equal": bool(torch.equal(ref, cand)),
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        }

    ref_perf = clone_inputs(base)
    cand_perf = clone_inputs(base)
    ref_times, _ = benchmark_cuda(
        lambda: run_reference(shape, ref_perf), args.warmup, args.repeats
    )
    cand_times, _ = benchmark_cuda(
        lambda: run_candidate(
            shape, cand_perf, args.bv, args.num_warps, args.num_stages
        ),
        args.warmup,
        args.repeats,
    )
    ref_median = statistics.median(ref_times)
    cand_median = statistics.median(cand_times)
    print(
        json.dumps(
            {
                "model": args.model,
                "shape": asdict(shape),
                "batch_size": args.batch_size,
                "config": {
                    "BV": args.bv,
                    "num_warps": args.num_warps,
                    "num_stages": args.num_stages,
                },
                "correctness": correctness,
                "reference": {
                    "median_ms": ref_median,
                    "p90_ms": percentile(ref_times, 0.9),
                    "min_ms": min(ref_times),
                },
                "candidate": {
                    "median_ms": cand_median,
                    "p90_ms": percentile(cand_times, 0.9),
                    "min_ms": min(cand_times),
                },
                "speedup": ref_median / cand_median,
                "improvement_pct": (ref_median / cand_median - 1.0) * 100.0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
