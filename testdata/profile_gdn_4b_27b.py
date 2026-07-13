#!/usr/bin/env python3
"""Profile isolated Qwen3.5 Gated DeltaNet paths on ROCm/DCU.

The script intentionally uses the exact 4B/27B GDN tensor shapes without
loading model weights.  It is suitable for quick kernel timing, Torch Profiler
traces, and rocprof counter collection before an end-to-end model run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton

from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode_kernel,
)
from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_offsets
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)


@dataclass(frozen=True)
class GDNShape:
    name: str
    num_layers: int
    hidden_size: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_width: int = 4

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def qkv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def qkvz_dim(self) -> int:
        return self.qkv_dim + self.value_dim

    @property
    def ba_dim(self) -> int:
        return 2 * self.num_v_heads


SHAPES = {
    "4b": GDNShape(
        name="4b",
        num_layers=24,
        hidden_size=2560,
        num_k_heads=16,
        num_v_heads=32,
    ),
    "27b": GDNShape(
        name="27b",
        num_layers=48,
        hidden_size=5120,
        num_k_heads=16,
        num_v_heads=48,
    ),
}


def parse_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtypes[name]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def benchmark_cuda(
    invoke: Callable[[], object], warmup: int, repeats: int
) -> tuple[list[float], object]:
    result: object = None
    for _ in range(warmup):
        result = invoke()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends):
        start.record()
        result = invoke()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)], result


def export_trace(
    invoke: Callable[[], object], trace_path: Path, profile_repeats: int
) -> None:
    from torch.profiler import ProfilerActivity, profile

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(profile_repeats):
            invoke()
    torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))


def unique_tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    """Count unique storages so transposed/views are not double-counted."""
    storages: dict[tuple[str, int | None, int], int] = {}
    for tensor in tensors.values():
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def make_prefill_core(
    shape: GDNShape,
    seq_len: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    q = torch.randn(
        1,
        seq_len,
        shape.num_k_heads,
        shape.head_k_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1,
        seq_len,
        shape.num_v_heads,
        shape.head_v_dim,
        device=device,
        dtype=dtype,
    )
    # G is the log decay. Keep it negative and small to avoid unstable states.
    g = -torch.rand(
        1, seq_len, shape.num_v_heads, device=device, dtype=dtype
    ) * 0.1
    beta = torch.sigmoid(
        torch.randn(1, seq_len, shape.num_v_heads, device=device, dtype=dtype)
    )
    state = torch.zeros(
        1,
        shape.num_v_heads,
        shape.head_v_dim,
        shape.head_k_dim,
        device=device,
        dtype=state_dtype,
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.long)

    def invoke() -> object:
        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=shape.head_k_dim**-0.5,
            initial_state=state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    return invoke, {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "state": state,
        "cu_seqlens": cu_seqlens,
    }


def make_decode_core(
    shape: GDNShape,
    batch_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
    decode_bv: int | None = None,
    decode_num_warps: int = 1,
    decode_num_stages: int = 3,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    mixed_qkv = torch.randn(
        batch_size, shape.qkv_dim, device=device, dtype=dtype
    )
    a = torch.randn(batch_size, shape.num_v_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.zeros(shape.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)
    state = torch.zeros(
        batch_size,
        shape.num_v_heads,
        shape.head_v_dim,
        shape.head_k_dim,
        device=device,
        dtype=state_dtype,
    )
    out = torch.empty(
        batch_size,
        1,
        shape.num_v_heads,
        shape.head_v_dim,
        device=device,
        dtype=dtype,
    )
    state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)

    if decode_bv is None:

        def invoke() -> object:
            return fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=shape.head_k_dim**-0.5,
                initial_state=state,
                out=out,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )

    else:
        if decode_bv > shape.head_v_dim:
            raise ValueError("decode-bv cannot exceed the value head dimension")
        grid = (
            triton.cdiv(shape.head_v_dim, decode_bv),
            batch_size * shape.num_v_heads,
        )

        def invoke() -> object:
            fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                o=out,
                h0=state,
                ht=state,
                ssm_state_indices=state_indices,
                scale=shape.head_k_dim**-0.5,
                stride_mixed_qkv_tok=mixed_qkv.stride(0),
                stride_a_tok=a.stride(0),
                stride_b_tok=b.stride(0),
                stride_init_state_token=state.stride(0),
                stride_final_state_token=state.stride(0),
                stride_indices_seq=state_indices.stride(0),
                H=shape.num_k_heads,
                HV=shape.num_v_heads,
                K=shape.head_k_dim,
                V=shape.head_v_dim,
                BK=triton.next_power_of_2(shape.head_k_dim),
                BV=decode_bv,
                SOFTPLUS_THRESHOLD=20.0,
                USE_QK_L2NORM_IN_KERNEL=True,
                num_warps=decode_num_warps,
                num_stages=decode_num_stages,
            )
            return out, state

    return invoke, {
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state": state,
        "out": out,
        "state_indices": state_indices,
    }


def make_decode_conv(
    shape: GDNShape,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    mixed_qkv = torch.randn(
        batch_size, shape.qkv_dim, device=device, dtype=dtype
    )
    weight = torch.randn(
        shape.qkv_dim, shape.conv_width, device=device, dtype=dtype
    ) * 0.02
    # The runtime cache is stored [N, width-1, dim] and transposed before use.
    state_storage = torch.zeros(
        batch_size,
        shape.conv_width - 1,
        shape.qkv_dim,
        device=device,
        dtype=dtype,
    )
    conv_state = state_storage.transpose(-1, -2)
    state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)

    def invoke() -> object:
        return causal_conv1d_update(
            mixed_qkv,
            conv_state,
            weight,
            bias=None,
            activation="silu",
            conv_state_indices=state_indices,
            validate_data=False,
        )

    return invoke, {
        "mixed_qkv": mixed_qkv,
        "weight": weight,
        "state_storage": state_storage,
        "conv_state": conv_state,
        "state_indices": state_indices,
    }


def make_decode_pipeline(
    shape: GDNShape,
    batch_size: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    conv_invoke, tensors = make_decode_conv(shape, batch_size, dtype, device)
    a = torch.randn(batch_size, shape.num_v_heads, device=device, dtype=dtype)
    b = torch.randn_like(a)
    A_log = torch.zeros(shape.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)
    recurrent_state = torch.zeros(
        batch_size,
        shape.num_v_heads,
        shape.head_v_dim,
        shape.head_k_dim,
        device=device,
        dtype=state_dtype,
    )
    out = torch.empty(
        batch_size,
        1,
        shape.num_v_heads,
        shape.head_v_dim,
        device=device,
        dtype=dtype,
    )
    state_indices = tensors["state_indices"]

    def invoke() -> object:
        mixed_qkv = conv_invoke()
        return fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=shape.head_k_dim**-0.5,
            initial_state=recurrent_state,
            out=out,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

    tensors.update(
        {
            "a": a,
            "b": b,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "recurrent_state": recurrent_state,
            "out": out,
        }
    )
    return invoke, tensors


def make_prefill_conv(
    shape: GDNShape,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    mixed_qkv = torch.randn(seq_len, shape.qkv_dim, device=device, dtype=dtype)
    x = mixed_qkv.transpose(0, 1)
    weight = torch.randn(
        shape.qkv_dim, shape.conv_width, device=device, dtype=dtype
    ) * 0.02
    state_storage = torch.zeros(
        1,
        shape.conv_width - 1,
        shape.qkv_dim,
        device=device,
        dtype=dtype,
    )
    conv_state = state_storage.transpose(-1, -2)
    query_start_loc = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    cache_indices = torch.tensor([0], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor([False], device=device, dtype=torch.bool)

    def invoke() -> object:
        return causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=None,
            conv_states=conv_state,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation="silu",
            validate_data=False,
        )

    return invoke, {
        "mixed_qkv": mixed_qkv,
        "x": x,
        "weight": weight,
        "state_storage": state_storage,
        "conv_state": conv_state,
        "query_start_loc": query_start_loc,
        "cache_indices": cache_indices,
        "has_initial_state": has_initial_state,
    }


def make_layout(
    shape: GDNShape,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    mixed_qkvz = torch.randn(
        num_tokens, shape.qkvz_dim, device=device, dtype=dtype
    )
    ba = torch.randn(num_tokens, shape.ba_dim, device=device, dtype=dtype)

    def invoke() -> object:
        mixed_qkv, z = mixed_qkvz.split([shape.qkv_dim, shape.value_dim], dim=-1)
        z = z.reshape(num_tokens, shape.num_v_heads, shape.head_v_dim)
        b, a = ba.chunk(2, dim=-1)
        # Match Qwen3_5GatedDeltaNet.forward exactly.
        return mixed_qkv, z, b.contiguous(), a.contiguous()

    return invoke, {"mixed_qkvz": mixed_qkvz, "ba": ba}


def make_projection(
    shape: GDNShape,
    mode: str,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    if mode == "qkvz-proj":
        in_features, out_features = shape.hidden_size, shape.qkvz_dim
    elif mode == "ba-proj":
        in_features, out_features = shape.hidden_size, shape.ba_dim
    elif mode == "out-proj":
        in_features, out_features = shape.value_dim, shape.hidden_size
    else:
        raise ValueError(f"unsupported projection mode: {mode}")
    x = torch.randn(num_tokens, in_features, device=device, dtype=dtype)
    weight = torch.randn(out_features, in_features, device=device, dtype=dtype) * 0.02

    def invoke() -> object:
        return F.linear(x, weight)

    return invoke, {"x": x, "weight": weight}


def build_benchmark(
    args: argparse.Namespace,
    shape: GDNShape,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
) -> tuple[Callable[[], object], dict[str, torch.Tensor]]:
    if args.mode == "prefill-core":
        return make_prefill_core(shape, args.seq_len, dtype, state_dtype, device)
    if args.mode == "decode-core":
        return make_decode_core(
            shape,
            args.batch_size,
            dtype,
            state_dtype,
            device,
            args.decode_bv,
            args.decode_num_warps,
            args.decode_num_stages,
        )
    if args.mode == "prefill-conv":
        return make_prefill_conv(shape, args.seq_len, dtype, device)
    if args.mode == "decode-conv":
        return make_decode_conv(shape, args.batch_size, dtype, device)
    if args.mode == "decode-pipeline":
        return make_decode_pipeline(
            shape, args.batch_size, dtype, state_dtype, device
        )
    if args.mode == "layout":
        return make_layout(shape, args.seq_len, dtype, device)
    if args.mode in ("qkvz-proj", "ba-proj", "out-proj"):
        return make_projection(shape, args.mode, args.seq_len, dtype, device)
    raise ValueError(f"unsupported mode: {args.mode}")


def run_decode_sweep(
    args: argparse.Namespace,
    shape: GDNShape,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
) -> None:
    rows = []
    for bv in (16, 32, 64, 128):
        for num_warps in (1, 2, 4):
            for num_stages in (1, 2, 3):
                invoke, tensors = make_decode_core(
                    shape,
                    args.batch_size,
                    dtype,
                    state_dtype,
                    device,
                    bv,
                    num_warps,
                    num_stages,
                )
                times_ms, result = benchmark_cuda(
                    invoke, args.warmup, args.repeats
                )
                rows.append(
                    {
                        "BV": bv,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                        "median_ms": statistics.median(times_ms),
                        "p90_ms": percentile(times_ms, 0.90),
                        "min_ms": min(times_ms),
                    }
                )
                del result, invoke, tensors
                torch.cuda.empty_cache()
    rows.sort(key=lambda row: row["median_ms"])
    print(
        json.dumps(
            {
                "mode": "decode-sweep",
                "model": args.model,
                "shape": asdict(shape),
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "state_dtype": args.state_dtype,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_prefill_h_sweep(
    args: argparse.Namespace,
    shape: GDNShape,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
) -> None:
    bt = 64
    num_chunks = triton.cdiv(args.seq_len, bt)
    k = torch.randn(
        1,
        args.seq_len,
        shape.num_k_heads,
        shape.head_k_dim,
        device=device,
        dtype=dtype,
    ) * 0.02
    v = torch.randn(
        1,
        args.seq_len,
        shape.num_v_heads,
        shape.head_v_dim,
        device=device,
        dtype=dtype,
    ) * 0.02
    w = torch.randn(
        1,
        args.seq_len,
        shape.num_v_heads,
        shape.head_k_dim,
        device=device,
        dtype=dtype,
    ) * 0.02
    g = -torch.rand(
        1,
        args.seq_len,
        shape.num_v_heads,
        device=device,
        dtype=torch.float32,
    ).cumsum(dim=1) * 1e-5
    h = torch.empty(
        1,
        num_chunks,
        shape.num_v_heads,
        shape.head_v_dim,
        shape.head_k_dim,
        device=device,
        dtype=dtype,
    )
    v_new = torch.empty_like(v)
    initial_state = torch.zeros(
        1,
        shape.num_v_heads,
        shape.head_v_dim,
        shape.head_k_dim,
        device=device,
        dtype=state_dtype,
    )
    cu_seqlens = torch.tensor([0, args.seq_len], device=device, dtype=torch.long)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, bt)
    kernel = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn.fn
    rows = []
    for bv in (16, 32, 64):
        grid = (triton.cdiv(shape.head_v_dim, bv), shape.num_v_heads)
        for num_warps in (1, 2, 4):
            for num_stages in (1, 2, 3, 4):

                def invoke() -> object:
                    kernel[grid](
                        k=k,
                        v=v,
                        w=w,
                        v_new=v_new,
                        g=g,
                        gk=None,
                        h=h,
                        h0=initial_state,
                        ht=None,
                        cu_seqlens=cu_seqlens,
                        chunk_offsets=chunk_offsets,
                        T=args.seq_len,
                        H=shape.num_v_heads,
                        Hg=shape.num_k_heads,
                        K=shape.head_k_dim,
                        V=shape.head_v_dim,
                        BT=bt,
                        BV=bv,
                        USE_G=True,
                        USE_GK=False,
                        USE_INITIAL_STATE=True,
                        STORE_FINAL_STATE=False,
                        SAVE_NEW_VALUE=True,
                        IS_VARLEN=True,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    return h, v_new

                try:
                    times_ms, result = benchmark_cuda(
                        invoke, args.warmup, args.repeats
                    )
                except Exception as error:
                    rows.append(
                        {
                            "BV": bv,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "error": str(error),
                        }
                    )
                    continue
                rows.append(
                    {
                        "BV": bv,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                        "median_ms": statistics.median(times_ms),
                        "p90_ms": percentile(times_ms, 0.90),
                        "min_ms": min(times_ms),
                    }
                )
                del result
    rows.sort(key=lambda row: row.get("median_ms", float("inf")))

    def run_config(bv: int, num_warps: int, num_stages: int) -> None:
        grid = (triton.cdiv(shape.head_v_dim, bv), shape.num_v_heads)
        kernel[grid](
            k=k,
            v=v,
            w=w,
            v_new=v_new,
            g=g,
            gk=None,
            h=h,
            h0=initial_state,
            ht=None,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=args.seq_len,
            H=shape.num_v_heads,
            Hg=shape.num_k_heads,
            K=shape.head_k_dim,
            V=shape.head_v_dim,
            BT=bt,
            BV=bv,
            USE_G=True,
            USE_GK=False,
            USE_INITIAL_STATE=True,
            STORE_FINAL_STATE=False,
            SAVE_NEW_VALUE=True,
            IS_VARLEN=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    run_config(32, 2, 2)
    torch.cuda.synchronize()
    reference_h = h.clone()
    reference_v_new = v_new.clone()
    candidate = (16, 1, 1) if args.model == "4b" else (32, 2, 1)
    run_config(*candidate)
    torch.cuda.synchronize()
    correctness = {
        "reference": {"BV": 32, "num_warps": 2, "num_stages": 2},
        "candidate": {
            "BV": candidate[0],
            "num_warps": candidate[1],
            "num_stages": candidate[2],
        },
        "h_equal": torch.equal(reference_h, h),
        "h_max_abs_diff": (reference_h.float() - h.float()).abs().max().item(),
        "v_new_equal": torch.equal(reference_v_new, v_new),
        "v_new_max_abs_diff": (
            (reference_v_new.float() - v_new.float()).abs().max().item()
        ),
    }
    print(
        json.dumps(
            {
                "mode": "prefill-h-sweep",
                "model": args.model,
                "shape": asdict(shape),
                "seq_len": args.seq_len,
                "dtype": args.dtype,
                "state_dtype": args.state_dtype,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "correctness": correctness,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "layout",
            "qkvz-proj",
            "ba-proj",
            "out-proj",
            "prefill-conv",
            "prefill-core",
            "decode-conv",
            "decode-core",
            "decode-pipeline",
            "decode-sweep",
            "prefill-h-sweep",
        ),
    )
    parser.add_argument("--model", choices=tuple(SHAPES), default="4b")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--state-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float32",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument("--decode-bv", type=int, choices=(16, 32, 64, 128))
    parser.add_argument("--decode-num-warps", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--decode-num-stages", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA/ROCm device is required")
    if args.seq_len <= 0 or args.batch_size <= 0:
        raise ValueError("seq-len and batch-size must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.profile_repeats <= 0:
        raise ValueError("invalid warmup/repeat count")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    shape = SHAPES[args.model]
    dtype = parse_dtype(args.dtype)
    state_dtype = parse_dtype(args.state_dtype)

    if args.mode == "decode-sweep":
        run_decode_sweep(args, shape, dtype, state_dtype, device)
        return
    if args.mode == "prefill-h-sweep":
        run_prefill_h_sweep(args, shape, dtype, state_dtype, device)
        return

    invoke, tensors = build_benchmark(args, shape, dtype, state_dtype, device)
    tensor_bytes = unique_tensor_bytes(tensors)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times_ms, result = benchmark_cuda(invoke, args.warmup, args.repeats)
    del result

    if args.trace_path is not None:
        export_trace(invoke, args.trace_path, args.profile_repeats)

    summary = {
        "mode": args.mode,
        "model": args.model,
        "shape": asdict(shape),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "state_dtype": args.state_dtype,
        "decode_config": (
            {
                "BV": args.decode_bv,
                "num_warps": args.decode_num_warps,
                "num_stages": args.decode_num_stages,
            }
            if args.mode == "decode-core" and args.decode_bv is not None
            else None
        ),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p90_ms": percentile(times_ms, 0.90),
        "p99_ms": percentile(times_ms, 0.99),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "all_ms": times_ms,
        "input_tensor_mib": tensor_bytes / (1024**2),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "trace_path": str(args.trace_path) if args.trace_path else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
