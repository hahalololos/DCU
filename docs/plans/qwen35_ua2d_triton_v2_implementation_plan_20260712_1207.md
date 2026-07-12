# Qwen3.5 UA2D Triton V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用合成 27B 证据驱动实现和筛选 Qwen3.5 UA2D Triton V2，在每轮最多一次完整 27B 启动的约束下，争取长档 kernel 至少提升 8% 且保持平台精度零扣分目标。

**Architecture:** 保留 `c274c6b` 专用 UA2D 为稳定基线，先建立无需模型权重的 27B 合成 A/B 与 rocprof 基线，再实现边界感知 V2-A；仅当 V2-A 未达门槛时实现 block-aligned V2-B 和一轮 ROCm 资源布局实验。通过合成 27B 和 4B 门禁的唯一候选才获得一次完整 27B 服务启动；若 Triton 无法达到门槛，清理实验代码并另写 HIP/MFMA 实施计划。

**Tech Stack:** Python 3.10、PyTorch 2.10、vLLM 0.18.1、Triton/ROCm、gfx936、rocprof、pytest、OpenCompass、Bash、Git。

## Global Constraints

- 始终使用简体中文记录设计、计划和实验结论。
- 本地只阅读、编辑和 Git；构建、运行、kernel 测试、profile 和服务测试只在远端执行。
- 每次远端操作先执行：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
```

- 远端源码目录固定为 `/public/home/xdzs2026_c203/haha/vllm_cscc`，`PYTHONPATH` 必须优先指向该源码树。
- 远端是共享 root 容器；启动服务前检查 GPU、显存和端口，只停止本工作区进程。
- 不创建、重启、停止或删除 PRA26 容器；容器不可用时按项目规则停止远端工作并请用户处理。
- 不修改比赛脚本、模型权重、tokenizer、chat template、模型结构、scheduler、锁定参数或 bench 统计口径。
- 不截断输入/输出，不跳样本/层/token，不使用投机解码、draft model、外挂预测器或测试集缓存。
- BF16 输入和输出；online softmax 与 accumulator 保持 FP32；允许改变归约顺序，但必须确定且通过精度门禁。
- 当前稳定基线为 `vllm_cscc` `haha` 分支提交 `c274c6b`；所有实验必须可回退到该实现。
- 纯 Python/Triton 改动无需 CMake；已有 editable 安装时重启 Python 或 vLLM 进程即可。
- 若后续证据触发 HIP/C++，停止本计划并另写计划；构建必须遵循 `竞赛环境vllm增量编译guide.md`。
- 每轮结构优化最多启动一次完整 27B 服务；所有候选先通过合成 27B 和 4B 门禁。
- 每次 vLLM 源码改动和测试结果按时间戳简记至 `docs/progress.md`。
- `DCU/` 与 `vllm_cscc/` 是独立 Git 仓库：测试脚本/文档提交到 DCU，源码/测试提交到 vllm_cscc，不混合提交。

---

## File Map

### DCU 仓库

- Create: `testdata/benchmark_qwen35_ua2d_27b.py` — 构造 27B 合成 UA2D 形状、运行变体 A/B、输出 JSON 统计。
- Reuse: `testdata/rocprof_hotspots_metrics.txt` — rocprof 计数器配置，不修改。
- Reuse: `testdata/start_vllm_4b.sh`、`testdata/run_throughput_4b.sh` — 4B 服务门禁，不修改。
- Reuse: `testdata/start_vllm.sh`、`testdata/run_throughput.sh`、`testdata/run_accuracy.sh` — 单次 27B 验证，不修改。
- Modify: `docs/progress.md` — 记录每个候选的源码哈希、micro、profile、端到端和取舍。

### vllm_cscc 仓库

- Modify: `vllm/envs.py` — 增加临时实验变体选择，默认保持稳定基线。
- Modify: `vllm/v1/attention/ops/triton_unified_attention.py` — V2-A、条件式 V2-B、严格 dispatch 和最终清理。
- Modify: `tests/kernels/attention/test_triton_unified_attention.py` — 变体选择、block 边界、GQA=6、长 query、确定性和回退测试。

## Interfaces

实施期间使用以下稳定接口：

```python
# vllm/envs.py
VLLM_ROCM_QWEN_UA2D_EXPERIMENT: int = 0

# 0: c274c6b baseline
# 1: V2-A boundary-aware second-block loading
# 2: V2-B block-aligned traversal
# 3+: 仅用于有 profile 依据的一轮 ROCm launch/resource 实验
```

```python
# testdata/benchmark_qwen35_ua2d_27b.py
@dataclass(frozen=True)
class Shape:
    query_len: int
    kv_len: int
    num_query_heads: int = 24
    num_kv_heads: int = 4
    head_size: int = 256
    block_size: int = 784

```

`benchmark_variant(shape: Shape, variant: int, warmups: int, repeats: int)` 返回
`dict[str, object]`，字段固定为 `shape`、`variant`、`median_ms`、`min_ms`、
`max_ms` 和 `samples_ms`。

所有候选仍通过现有 `unified_attention(...)` 入口调用，未命中 Qwen3.5/gfx936/BF16/causal prefill guard 时必须使用现有通用路径。

---

### Task 1: 建立可重复的合成 27B UA2D 基线工具

**Files:**
- Create: `testdata/benchmark_qwen35_ua2d_27b.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: 当前 `unified_attention(...)` 和 `VLLM_ROCM_QWEN_UA2D_FASTPATH=1`。
- Produces: `Shape`、`benchmark_variant()` 和机器可读 JSON，供 V2-A/V2-B 使用同一形状与统计口径。

- [ ] **Step 1: 写入合成 benchmark 脚本**

创建 `testdata/benchmark_qwen35_ua2d_27b.py`，核心代码必须包含以下完整结构：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


@dataclass(frozen=True)
class Shape:
    query_len: int
    kv_len: int
    num_query_heads: int = 24
    num_kv_heads: int = 4
    head_size: int = 256
    block_size: int = 784


DEFAULT_SHAPES = (
    Shape(query_len=4096, kv_len=8192),
    Shape(query_len=4096, kv_len=16384),
    Shape(query_len=4096, kv_len=24576),
    Shape(query_len=4096, kv_len=32768),
)


def make_tensors(shape: Shape) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_blocks = math.ceil(shape.kv_len / shape.block_size)
    query = torch.randn(
        shape.query_len,
        shape.num_query_heads,
        shape.head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    key_cache = torch.randn(
        num_blocks,
        shape.block_size,
        shape.num_kv_heads,
        shape.head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "output": torch.empty_like(query),
        "cu_query_lens": torch.tensor(
            [0, shape.query_len], dtype=torch.int32, device=device
        ),
        "kv_lens": torch.tensor(
            [shape.kv_len], dtype=torch.int32, device=device
        ),
        "block_tables": torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).view(1, -1),
    }


def invoke(shape: Shape, tensors: dict[str, torch.Tensor]) -> None:
    unified_attention(
        q=tensors["query"],
        k=tensors["key_cache"],
        v=tensors["value_cache"],
        out=tensors["output"],
        cu_seqlens_q=tensors["cu_query_lens"],
        seqused_k=tensors["kv_lens"],
        max_seqlen_q=shape.query_len,
        max_seqlen_k=shape.kv_len,
        softmax_scale=shape.head_size**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=tensors["block_tables"],
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=None,
        num_par_softmax_segments=None,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
    )


def benchmark_variant(
    shape: Shape,
    variant: int,
    warmups: int,
    repeats: int,
) -> dict[str, object]:
    os.environ["VLLM_ROCM_QWEN_UA2D_FASTPATH"] = "1"
    if variant == 0:
        os.environ.pop("VLLM_ROCM_QWEN_UA2D_EXPERIMENT", None)
    else:
        os.environ["VLLM_ROCM_QWEN_UA2D_EXPERIMENT"] = str(variant)
    tensors = make_tensors(shape)
    for _ in range(warmups):
        invoke(shape, tensors)
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke(shape, tensors)
        end.record()
        end.synchronize()
        samples_ms.append(float(start.elapsed_time(end)))

    return {
        "shape": asdict(shape),
        "variant": variant,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="0")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--query-len", type=int)
    parser.add_argument("--kv-len", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [int(value) for value in args.variants.split(",")]
    shapes = DEFAULT_SHAPES
    if args.query_len is not None or args.kv_len is not None:
        if args.query_len is None or args.kv_len is None:
            raise SystemExit("--query-len and --kv-len must be provided together")
        shapes = (Shape(query_len=args.query_len, kv_len=args.kv_len),)

    results = [
        benchmark_variant(shape, variant, args.warmups, args.repeats)
        for shape in shapes
        for variant in variants
    ]
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行静态门禁**

Run:

```bash
python -m py_compile testdata/benchmark_qwen35_ua2d_27b.py
git diff --check -- testdata/benchmark_qwen35_ua2d_27b.py
```

Expected: 两条命令均退出码 `0`，无输出。

- [ ] **Step 3: 同步到远端并确认导入环境**

Run remotely:

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
which python
which vllm
python -m pip show vllm | rg 'Editable project location|Location'
python -c 'import vllm; print(vllm.__file__)'
```

Expected: `python`、`vllm` 位于 `.venv`；`vllm.__file__` 和 editable location 指向远端 `vllm_cscc`。

- [ ] **Step 4: 采集 `c274c6b` 合成 27B baseline**

Run remotely:

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
mkdir -p testdata/experiments/UA2D27-SYNTH-BASE-20260712
python testdata/benchmark_qwen35_ua2d_27b.py \
  --variants 0 --warmups 5 --repeats 20 \
  --output testdata/experiments/UA2D27-SYNTH-BASE-20260712/results.json
```

Expected: 四个形状均输出有限正数 `median_ms`；无 VM fault、NaN、导入错误或回退警告。

- [ ] **Step 5: 记录 baseline 并提交 DCU 工具**

在 `docs/progress.md` 顶部追加时间戳、远端源码哈希、四个形状的 median/min/max 和实验目录。

Run:

```bash
git add testdata/benchmark_qwen35_ua2d_27b.py docs/progress.md
git commit -m "test: add synthetic Qwen3.5 27B UA2D benchmark"
```

Expected: DCU 仓库生成一个只包含 benchmark 和进展记录的提交。

---

### Task 2: 以测试驱动增加 V2-A 变体选择与正确性覆盖

**Files:**
- Modify: `vllm_cscc/vllm/envs.py`
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`

**Interfaces:**
- Consumes: Task 1 的 `variant=0/1` 约定。
- Produces: `VLLM_ROCM_QWEN_UA2D_EXPERIMENT=1` 启动 V2-A；默认 `0` 保持 `c274c6b`。

- [ ] **Step 1: 写入失败的环境变量和选择测试**

在 `tests/kernels/attention/test_triton_unified_attention.py` 增加：

```python
@pytest.mark.parametrize("value", [0, 1, 2, 3])
def test_qwen35_ua2d_experiment_env_accepts_nonnegative_values(
    monkeypatch: pytest.MonkeyPatch,
    value: int,
) -> None:
    monkeypatch.setenv("VLLM_ROCM_QWEN_UA2D_EXPERIMENT", str(value))
    assert envs.VLLM_ROCM_QWEN_UA2D_EXPERIMENT == value


def test_qwen35_ua2d_experiment_env_rejects_negative_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ROCM_QWEN_UA2D_EXPERIMENT", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _ = envs.VLLM_ROCM_QWEN_UA2D_EXPERIMENT


@pytest.mark.parametrize(
    ("experiment", "expected"),
    [(0, "baseline"), (1, "v2a"), (2, "v2b"), (3, "resource")],
)
def test_qwen35_ua2d_variant_name(experiment: int, expected: str) -> None:
    assert ua._qwen35_ua2d_variant_name(experiment) == expected
```

- [ ] **Step 2: 运行测试并确认失败**

Run remotely:

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
cd vllm_cscc
pytest -q tests/kernels/attention/test_triton_unified_attention.py \
  -k 'experiment_env or variant_name'
```

Expected: FAIL，原因是 `VLLM_ROCM_QWEN_UA2D_EXPERIMENT` 或 `_qwen35_ua2d_variant_name` 尚不存在。

- [ ] **Step 3: 实现最小环境变量和选择函数**

在 `vllm/envs.py` 类型声明区增加：

```python
VLLM_ROCM_QWEN_UA2D_EXPERIMENT: int = 0
```

在 `environment_variables` 映射中增加：

```python
"VLLM_ROCM_QWEN_UA2D_EXPERIMENT": lambda: env_non_negative_int(
    "VLLM_ROCM_QWEN_UA2D_EXPERIMENT", 0
),
```

如果当前文件没有 `env_non_negative_int`，在其它环境解析 helper 附近增加：

```python
def env_non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value
```

在 `triton_unified_attention.py` 的 Qwen3.5 guard 附近增加：

```python
def _qwen35_ua2d_variant_name(experiment: int) -> str:
    return {
        0: "baseline",
        1: "v2a",
        2: "v2b",
    }.get(experiment, "resource")
```

- [ ] **Step 4: 运行选择测试并确认通过**

Run remotely:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py \
  -k 'experiment_env or variant_name'
```

Expected: 新增测试全部 PASS。

- [ ] **Step 5: 写入失败的 V2-A GPU 正确性测试**

将现有 `test_qwen35_ua2d_configurations_and_boundaries` 增加 `experiment` 参数：

```python
@pytest.mark.parametrize("experiment", [0, 1])
```

并在测试内增加：

```python
monkeypatch.setenv("VLLM_ROCM_QWEN_UA2D_EXPERIMENT", str(experiment))
```

另增加精确覆盖“tile 不跨 block 时不得依赖第二 block-table 项”的测试：

```python
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Qwen3.5 UA2D fastpath is ROCm-specific",
)
@torch.inference_mode()
def test_qwen35_ua2d_v2a_non_crossing_tile_matches_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_default_device("cuda")
    set_random_seed(0)
    query_len = 32
    kv_len = 783
    block_size = 784
    num_query_heads = 24
    num_kv_heads = 4
    head_size = 256
    query = torch.randn(
        query_len, num_query_heads, head_size, dtype=torch.bfloat16
    )
    key_cache = torch.randn(
        1, block_size, num_kv_heads, head_size, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0, query_len], dtype=torch.int32)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    block_tables = torch.tensor([[0]], dtype=torch.int32)

    def run(experiment: int) -> torch.Tensor:
        monkeypatch.setenv("VLLM_ROCM_QWEN_UA2D_FASTPATH", "1")
        monkeypatch.setenv(
            "VLLM_ROCM_QWEN_UA2D_EXPERIMENT", str(experiment)
        )
        output = torch.full_like(query, torch.nan)
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens,
            max_seqlen_q=query_len,
            max_seqlen_k=kv_len,
            softmax_scale=head_size**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_tables,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=None,
            num_par_softmax_segments=None,
            softmax_segm_output=None,
            softmax_segm_max=None,
            softmax_segm_expsum=None,
        )
        return output

    torch.testing.assert_close(run(1), run(0), atol=1.5e-2, rtol=1e-2)
```

- [ ] **Step 6: 运行 GPU 测试并确认 V2-A 尚未实现**

Run remotely:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py \
  -k 'qwen35_ua2d_v2a or qwen35_ua2d_configurations_and_boundaries'
```

Expected: variant `1` 尚未有独立实现或 dispatch，测试失败或无法证明路径命中。

- [ ] **Step 7: 实现 V2-A kernel 和严格 dispatch**

在当前 `kernel_qwen35_unified_attention_2d` 后新增同签名的
`kernel_qwen35_unified_attention_2d_v2a`。从当前稳定 kernel 复制实现，只将 full-prefix
和 diagonal/partial 两处第二 block 加载条件替换为：

```python
tile_crosses_block = tile_start_offset + TILE_SIZE > next_block_start
second_physical_block_idx = tl.load(
    block_tables_ptr + block_table_offset + logical_block_idx + 1,
    mask=tile_crosses_block,
    other=first_physical_block_idx,
).to(tl.int64)
```

其它 Q/K/V load、mask、online softmax、FP32 accumulator、epilogue 和 store 逐行保持与
`kernel_qwen35_unified_attention_2d` 一致。

在两个 kernel 定义之后增加唯一 selector：

```python
def _select_qwen35_specialized_ua2d_kernel(experiment: int):
    if experiment == 1:
        return kernel_qwen35_unified_attention_2d_v2a
    return kernel_qwen35_unified_attention_2d
```

并补充身份测试：

```python
def test_qwen35_ua2d_v2a_kernel_selection() -> None:
    assert (
        ua._select_qwen35_specialized_ua2d_kernel(1)
        is ua.kernel_qwen35_unified_attention_2d_v2a
    )
    assert (
        ua._select_qwen35_specialized_ua2d_kernel(0)
        is ua.kernel_qwen35_unified_attention_2d
    )
```

在 dispatch 中读取：

```python
qwen_ua2d_experiment = envs.VLLM_ROCM_QWEN_UA2D_EXPERIMENT
```

并通过 selector 启动：

```python
qwen_kernel = _select_qwen35_specialized_ua2d_kernel(
    qwen_ua2d_experiment
)

qwen_kernel[(total_num_q_blocks, num_kv_heads)](
    output_ptr=out,
    query_ptr=q,
    key_cache_ptr=k,
    value_cache_ptr=v,
    block_tables_ptr=block_table,
    seq_lens_ptr=seqused_k,
    scale=softmax_scale,
    num_query_heads=num_query_heads,
    num_queries_per_kv=num_queries_per_kv,
    block_table_stride=block_table.stride(0),
    query_stride_0=q.stride(0),
    query_stride_1=q.stride(1),
    output_stride_0=out.stride(0),
    output_stride_1=out.stride(1),
    BLOCK_SIZE=block_size,
    TILE_SIZE=TILE_SIZE_PREFILL,
    HEAD_SIZE=head_size,
    stride_k_cache_0=k.stride(0),
    stride_k_cache_1=k.stride(1),
    stride_k_cache_2=k.stride(2),
    stride_k_cache_3=k.stride(3),
    stride_v_cache_0=v.stride(0),
    stride_v_cache_1=v.stride(1),
    stride_v_cache_2=v.stride(2),
    stride_v_cache_3=v.stride(3),
    query_start_len_ptr=cu_seqlens_q,
    BLOCK_Q=BLOCK_Q,
    num_seqs=num_seqs,
    BLOCK_M=BLOCK_M,
    num_warps=num_warps_2d,
    num_stages=num_stages_2d,
)
```

不得为 V2-A 扩大现有 Qwen3.5 guard。

- [ ] **Step 8: 运行完整 Qwen3.5 UA2D 定向测试**

Run remotely:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py -k qwen35
```

Expected: 所有 Qwen3.5 guard、边界、长 query、确定性和 V2-A 测试 PASS；无 VM fault。

- [ ] **Step 9: 静态验证并提交 vllm_cscc**

Run locally:

```bash
python -m py_compile \
  vllm_cscc/vllm/envs.py \
  vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py \
  vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py
git -C vllm_cscc diff --check
git -C vllm_cscc add \
  vllm/envs.py \
  vllm/v1/attention/ops/triton_unified_attention.py \
  tests/kernels/attention/test_triton_unified_attention.py
git -C vllm_cscc commit -m "perf: add boundary-aware Qwen3.5 UA2D variant"
```

Expected: 生成独立 vllm_cscc 提交，默认实验值 `0` 不改变稳定路径。

---

### Task 3: 对 V2-A 做合成 27B A/B 与 rocprof 决策

**Files:**
- Modify: `testdata/benchmark_qwen35_ua2d_27b.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: Task 2 的 variant `0/1`。
- Produces: V2-A 的晋级、组合优化或淘汰结论。

- [ ] **Step 1: 验证 benchmark 能在同进程交替运行 0/1**

Run remotely:

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
python testdata/benchmark_qwen35_ua2d_27b.py \
  --variants 0,1 --warmups 5 --repeats 20 \
  --output testdata/experiments/UA2D27-V2A-20260712/results.json
```

Expected: 每个 shape 有 variant `0` 和 `1` 两条记录；没有因 Triton 编译混用导致错误。

- [ ] **Step 2: 计算逐形状提升率**

Run remotely:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("testdata/experiments/UA2D27-V2A-20260712/results.json")
rows = json.loads(path.read_text())
grouped = {}
for row in rows:
    key = (row["shape"]["query_len"], row["shape"]["kv_len"])
    grouped.setdefault(key, {})[row["variant"]] = row["median_ms"]
for key, values in sorted(grouped.items()):
    gain = (values[0] - values[1]) / values[0] * 100
    print(key, f"baseline={values[0]:.4f}ms", f"v2a={values[1]:.4f}ms", f"gain={gain:.2f}%")
PY
```

Expected: 四个 shape 均输出 baseline、V2-A 和 gain；缺失任一 variant 时脚本失败而不是静默跳过。

- [ ] **Step 3: 定位 rocprof 可执行文件并采集两种变体**

Run remotely:

```bash
ROCPROF="$(command -v rocprof || true)"
if [ -z "$ROCPROF" ]; then
  ROCPROF="$(rg --files /opt/dtk | rg '/rocprof$' | head -n1)"
fi
test -x "$ROCPROF"
mkdir -p testdata/experiments/UA2D27-V2A-20260712/rocprof

for variant in 0 1; do
  VLLM_ROCM_QWEN_UA2D_EXPERIMENT="$variant" \
  "$ROCPROF" \
    -i testdata/rocprof_hotspots_metrics.txt \
    -o "testdata/experiments/UA2D27-V2A-20260712/rocprof/v${variant}.csv" \
    python testdata/benchmark_qwen35_ua2d_27b.py \
      --variants "$variant" --query-len 4096 --kv-len 24576 \
      --warmups 3 --repeats 3
done
```

Expected: 两个变体均生成 rocprof 输出；能识别 UA2D kernel 名称和计数器记录。若平台实际只提供 `rocprofv2`，先保存 `--help` 输出并按其 input/output 语法等价执行，不修改指标集合。

- [ ] **Step 4: 应用 V2-A 决策门槛**

按以下规则只选择一个结论：

- `16K/24K/32K` 代表形状均值或中位收益 `>=8%`，且 `8K` 回退不超过 `2%`：跳过 Task 5，进入 Task 7。
- 收益 `5%-8%`，且 block-table load、SALU、bank conflict 或 occupancy 至少一项明确改善：允许执行 Task 6 的一轮资源组合。
- 收益 `<5%`：保留实验提交但默认关闭，执行 Task 5。
- 任一关键形状回退 `>2%` 或出现不确定/错误：修复一次；仍失败则回滚 V2-A 并执行 Task 5。

- [ ] **Step 5: 记录并提交实验结论**

在 `docs/progress.md` 写明 vllm_cscc 哈希、四个 shape、rocprof 对比、决策分支和实验目录。

Run:

```bash
git add docs/progress.md
git commit -m "docs: record Qwen3.5 UA2D V2-A results"
```

---

### Task 4: 为 V2-B 先补齐 block tail 和跨 block 测试

> 仅在 Task 3 判定 V2-A 未达到 `8%` 时执行。

**Files:**
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`

**Interfaces:**
- Consumes: variant `2` 约定。
- Produces: 覆盖 `528/544/784` 的 block-aligned tail、diagonal 和 GQA=6 失败测试。

- [ ] **Step 1: 扩展现有 reference 参数矩阵并加入 kernel 选择测试**

将现有 `test_qwen35_ua2d_configurations_and_boundaries` 增加：

```python
@pytest.mark.parametrize("experiment", [0, 1, 2])
```

在函数参数中加入 `experiment: int`，并在原有 FASTPATH 设置后加入：

```python
monkeypatch.setenv("VLLM_ROCM_QWEN_UA2D_EXPERIMENT", str(experiment))
```

在原有 shape 参数列表中追加两组 544 block case：

```python
pytest.param(
    16,
    544,
    [1, 2, 3, 5, 6],
    [543, 544, 545, 1088, 1089],
    id="4b-544-small-query-kv-block-boundaries",
),
pytest.param(
    16,
    544,
    [15, 16, 17, 31, 32],
    [1087, 1088, 1089, 545, 544],
    id="4b-544-partial-q-block-boundaries",
),
```

增加 selector 身份测试，确保 variant `2` 不是静默落入 baseline：

```python
def test_qwen35_ua2d_v2b_kernel_selection() -> None:
    assert (
        ua._select_qwen35_specialized_ua2d_kernel(2)
        is ua.kernel_qwen35_unified_attention_2d_v2b
    )
```

- [ ] **Step 2: 运行测试并确认 variant 2 失败**

Run remotely:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py \
  -k 'v2b_block_tails'
```

Expected: FAIL，因为 `_select_qwen35_specialized_ua2d_kernel` 或 V2-B kernel 尚未实现。

- [ ] **Step 3: 提交纯测试重构**

Run:

```bash
git -C vllm_cscc add tests/kernels/attention/test_triton_unified_attention.py
git -C vllm_cscc commit -m "test: cover Qwen3.5 UA2D block-aligned tails"
```

Expected: 此提交只包含测试参数矩阵和 selector 失败测试，不包含 V2-B kernel。

---

### Task 5: 实现并筛选 V2-B block-aligned 遍历

> 仅在 Task 4 已执行时继续。

**Files:**
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- Test: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: variant `2`、现有 dense `ref_paged_attn` 对照和扩展后的边界参数矩阵。
- Produces: `kernel_qwen35_unified_attention_2d_v2b`，或明确淘汰结论。

- [ ] **Step 1: 新增 V2-B 独立 kernel**

新增 `kernel_qwen35_unified_attention_2d_v2b`，入口参数与稳定 kernel 一致。初始化 Q、
`M/L/acc`、query/KV metadata 后，使用以下编译期 block 分解：

```python
FULL_TILES_PER_BLOCK: tl.constexpr = BLOCK_SIZE // TILE_SIZE
BLOCK_TAIL: tl.constexpr = BLOCK_SIZE % TILE_SIZE
num_visible_blocks = cdiv_fn(max_visible_key + 1, BLOCK_SIZE)

for logical_block_idx in range(0, num_visible_blocks):
    physical_block_idx = tl.load(
        block_tables_ptr + block_table_offset + logical_block_idx
    ).to(tl.int64)
    block_abs_start = logical_block_idx * BLOCK_SIZE

    for tile_in_block in range(0, FULL_TILES_PER_BLOCK):
        seq_offset = block_abs_start + tile_in_block * TILE_SIZE + offs_t
        # K/V 只使用当前 physical_block_idx；full-prefix 不构造 causal mask，
        # diagonal/最后序列块应用 seq_offset <= query_abs 和 <= max_visible_key。
        # 使用与稳定 kernel 完全相同的 FP32 M/L/acc 更新顺序。

    if BLOCK_TAIL > 0:
        tail_offset = tl.arange(0, TILE_SIZE)
        seq_offset = block_abs_start + FULL_TILES_PER_BLOCK * TILE_SIZE + tail_offset
        tail_mask = tail_offset < BLOCK_TAIL
        # masked load K/V；同时应用 max_visible_key 与 causal mask；
        # 继续更新同一组 M/L/acc，不写中间状态。
```

实现要求：

- `BLOCK_SIZE`、`TILE_SIZE`、`FULL_TILES_PER_BLOCK`、`BLOCK_TAIL` 都是 constexpr；
- 每个 logical block 只加载一次 physical block id；
- 不跨 block 拼接 tile；
- 528/544/784 均编译并通过测试；
- 不改变 output layout 和现有 guard。

dispatch 中仅当 `VLLM_ROCM_QWEN_UA2D_EXPERIMENT == 2` 时启动 V2-B。
将 selector 扩展为：

```python
def _select_qwen35_specialized_ua2d_kernel(experiment: int):
    if experiment == 1:
        return kernel_qwen35_unified_attention_2d_v2a
    if experiment == 2:
        return kernel_qwen35_unified_attention_2d_v2b
    return kernel_qwen35_unified_attention_2d
```

- [ ] **Step 2: 运行 V2-B 和全部 Qwen3.5 测试**

Run remotely:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py -k qwen35
```

Expected: block tail、长 query、GQA=6、误差、确定性和 fallback 测试全部 PASS。

- [ ] **Step 3: 提交 V2-B 源码**

Run:

```bash
git -C vllm_cscc add \
  vllm/v1/attention/ops/triton_unified_attention.py
git -C vllm_cscc commit -m "perf: add block-aligned Qwen3.5 UA2D variant"
```

- [ ] **Step 4: 运行 0/1/2 合成 27B A/B**

Run remotely:

```bash
python testdata/benchmark_qwen35_ua2d_27b.py \
  --variants 0,1,2 --warmups 5 --repeats 20 \
  --output testdata/experiments/UA2D27-V2B-20260712/results.json
```

Expected: 四个 shape 三种 variant 均有结果；V2-B 无编译异常和 VM fault。

- [ ] **Step 5: 对最优 V2-B 代表形状采集 rocprof**

使用 Task 3 的 rocprof 命令，将 variant 改为 `2`，输出到
`testdata/experiments/UA2D27-V2B-20260712/rocprof/v2.csv`。

Expected: 获得与 baseline 同指标集合的 kernel 记录。

- [ ] **Step 6: 应用 V2-B 门槛并记录**

- `>=8%` 且中档无回退：选择 V2-B，进入 Task 7。
- `5%-8%` 且资源指标明显改善：允许 Task 6 一轮组合。
- `<5%`、端到端上限不足或 tail 成本抵消：保留历史提交但不晋级。

在 `docs/progress.md` 记录结果并提交：

```bash
git add docs/progress.md
git commit -m "docs: record Qwen3.5 UA2D V2-B results"
```

---

### Task 6: 只执行一轮有 profile 依据的 Triton 资源布局实验

> 仅在 V2-A 或 V2-B 收益为 `5%-8%` 且 rocprof 明确显示可改善资源指标时执行。

**Files:**
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: 当前最优 variant 和对应 rocprof 瓶颈。
- Produces: 唯一一个 resource variant，编号从 `3` 开始；不得并行保留多个无证据配置。

- [ ] **Step 1: 将 profile 结论写成单一假设**

只允许选择以下一种：

- VGPR/occupancy 受限：缩短 `S/P/alpha` live range；
- wave 数不足：只扫描 `waves_per_eu=1/2/4`；
- MFMA 形状不佳：只扫描平台确认支持的 `matrix_instr_nonkdim`/k-pack；
- LDS bank conflict：只调整会改变 LDS 映射的 launch/compiler 配置。

先在 `docs/progress.md` 写明基线计数器、假设和唯一变量，禁止同时修改算法与多个 launch 参数。

- [ ] **Step 2: 写失败的 variant 选择测试**

例如选择 variant `3` 时增加：

```python
def test_qwen35_ua2d_resource_variant_name() -> None:
    assert ua._qwen35_ua2d_variant_name(3) == "resource"
```

若需要新 launch helper，先测试精确返回值：

```python
def test_qwen35_ua2d_resource_launch_config() -> None:
    assert ua._qwen35_ua2d_launch_config(3) == {
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 2,
    }
```

实际值必须来自 Step 1 的单一假设，不得无依据改成其它组合。

- [ ] **Step 3: 实现最小 resource variant**

使用独立 constexpr 或 launch kwargs，只改变 Step 1 选择的一个因素；默认 variant `0` 和已实现的 V2-A/V2-B 不变。

- [ ] **Step 4: 运行 Qwen3.5 测试、合成 A/B 和 rocprof**

Run remotely:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py -k qwen35
python testdata/benchmark_qwen35_ua2d_27b.py \
  --variants 0,3 --warmups 5 --repeats 20 \
  --output testdata/experiments/UA2D27-RESOURCE-20260712/results.json
```

随后按 Task 3 采集 variant `3` rocprof。

Expected: 正确性通过；只有达到 `>=8%` 且中档无超过 `2%` 回退才晋级。

- [ ] **Step 5: 提交或清理 resource variant**

若晋级：

```bash
git -C vllm_cscc add vllm/v1/attention/ops/triton_unified_attention.py \
  tests/kernels/attention/test_triton_unified_attention.py
git -C vllm_cscc commit -m "perf: tune Qwen3.5 UA2D ROCm resource layout"
```

若不晋级：用 `apply_patch` 删除 variant `3` 源码和测试，只在 `docs/progress.md` 保留结果；不得使用破坏性 Git 回退命令。

---

### Task 7: 对唯一晋级候选执行 4B 服务与完整精度门禁

**Files:**
- Reuse: `testdata/start_vllm_4b.sh`
- Reuse: `testdata/run_throughput_4b.sh`
- Reuse: `testdata/run_accuracy.sh` 或 `testdata/replay_accuracy_4b.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: 唯一晋级 variant `WINNER`，值为 `1`、`2` 或有依据的 resource variant。
- Produces: 是否允许支付一次 27B 启动成本的结论。

- [ ] **Step 1: 同步源码并检查共享 GPU/端口**

Run remotely:

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
rocm-smi --showmemuse --showuse
ss -ltnp | rg ':8001|:8002' || true
pgrep -af 'vllm|EngineCore|run_throughput' || true
```

Expected: 确认不会干扰队友；如资源被占用则等待，不停止队友进程。

- [ ] **Step 2: 运行 4B baseline 两轮热态测试**

启动时设置：

```bash
export VLLM_ROCM_QWEN_UA2D_EXPERIMENT=0
cd /public/home/xdzs2026_c203/haha/testdata
./start_vllm_4b.sh
```

服务就绪后使用 `curl --noproxy '*'` 验证，再运行：

```bash
for round in 1 2; do
  for bucket in 4-8K 8-16K 16-32K; do
    ./run_throughput_4b.sh "$bucket" 10
    cp "test_4b/${bucket}_throughput/result.json" \
      "experiments/UA2D-V2-4B-BASE-${round}-${bucket}.json"
  done
done
```

Expected: 六轮均 `10/10`；保存 request/output throughput、TTFT P99、TPOT P99、输入/输出 token。

- [ ] **Step 3: 只停止本工作区 baseline 服务并启动候选**

记录 baseline PID，只终止该 PID；确认显存释放后：

```bash
export VLLM_ROCM_QWEN_UA2D_EXPERIMENT="$WINNER"
./start_vllm_4b.sh
```

Expected: 服务使用同一源码、模型和参数，唯一差异为 UA2D variant。

- [ ] **Step 4: 运行候选两轮三档测试**

使用 Step 2 相同循环，将结果保存为 `UA2D-V2-4B-CAND-*`。

Expected: 完成率 `100%`；中长档 request throughput 不出现可重复回退；至少一档有明确提升；TPOT 无异常恶化。

- [ ] **Step 5: 检查逐样本输出与 token 数**

对 baseline/candidate 的 detailed result 比较每条请求输入长度、输出 token 数和文本。若 output throughput 上升但 request throughput、TTFT 和 kernel micro 不支持，判为输出变化而非有效收益。

- [ ] **Step 6: 运行四类完整精度门禁**

优先使用现有同一批 baseline prompt 的重放口径。先自动定位同时包含四类 JSON 的历史
baseline 目录；项目进展已确认该 109 条归档存在。设置大小写代理绕过：

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
BASELINE_DIR="$({
  rg -l 'origin_prompt' experiments -g 'hotpotqa.json' 2>/dev/null || true
} | while read -r path; do
  dir="$(dirname "$path")"
  test -f "$dir/gov_report.json" || continue
  test -f "$dir/retrieval_multi_point.json" || continue
  test -f "$dir/aggregation_keyword_aggregation.json" || continue
  echo "$dir"
done | head -n1)"
test -n "$BASELINE_DIR"
python replay_accuracy_4b.py \
  --baseline-dir "$BASELINE_DIR" \
  --output-dir experiments/UA2D-V2-4B-ACCURACY \
  --endpoint http://127.0.0.1:8001/v1/chat/completions \
  --model Qwen3.5-4B
```

如果 `test -n "$BASELINE_DIR"` 失败，将“缺少已记录的 109 条 4B baseline prompt/prediction
归档”记为硬阻塞并停止 Task 7，不停止当前 baseline 服务，不用 27B 专用的
`run_accuracy.sh` 代替，也不用少量 smoke 代替最终精度门禁。

Expected: hotpotqa、gov_report、retrieval、aggregation 均满足平台零扣分预测；重复输出确定。

- [ ] **Step 7: 应用 4B 晋级门槛并记录**

只有全部满足才进入 Task 8：

- 三档完成率 `100%`；
- 中长档 request throughput 无稳定回退；
- 至少一档有明确正收益；
- TPOT/TTFT 无异常；
- 四类精度预测平台扣分 `0`。

在 `docs/progress.md` 记录完整对照并提交：

```bash
git add docs/progress.md
git commit -m "docs: record Qwen3.5 UA2D V2 4B gate"
```

---

### Task 8: 只启动一次 27B 并完成全部候选验证

**Files:**
- Reuse: `testdata/start_vllm.sh`
- Reuse: `testdata/run_throughput.sh`
- Reuse: `testdata/run_accuracy.sh`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: Task 7 通过的唯一 `WINNER`。
- Produces: 保留、提交榜单或回滚的最终 27B 证据。

- [ ] **Step 1: 启动前确认环境、GPU和端口**

Run remotely:

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
which python; which vllm; python -m pip show vllm
rocm-smi --showmemuse --showuse
ss -ltnp | rg ':8001' || true
pgrep -af 'vllm|EngineCore' || true
```

Expected: editable location 为远端 `vllm_cscc`；GPU和端口可独占使用。资源不满足则不启动。

- [ ] **Step 2: 设置唯一候选并启动 27B 一次**

Run remotely:

```bash
export VLLM_ROCM_QWEN_UA2D_EXPERIMENT="$WINNER"
cd /public/home/xdzs2026_c203/haha/testdata
mkdir -p experiments
./start_vllm.sh > experiments/UA2D-V2-27B-service.log 2>&1
```

在独立终端监控日志，不重复启动。服务就绪后：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8001/v1/models
```

Expected: 返回 vLLM 模型信息；若响应来自 Squid/代理，修正 NO_PROXY 后重试，不误判服务故障。

- [ ] **Step 3: 预热并连续运行中长短三档**

Run remotely while the same service remains alive:

```bash
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

for round in 1 2; do
  ./run_throughput.sh 8-16K 10
  cp test/8-16K_throughput/result.json \
    "experiments/UA2D-V2-27B-mid-${round}.json"

  ./run_throughput.sh 16-32K 10
  cp test/16-32K_throughput/result.json \
    "experiments/UA2D-V2-27B-long-${round}.json"
done

./run_throughput.sh 4-8K 10
cp test/4-8K_throughput/result.json \
  experiments/UA2D-V2-27B-short-1.json
```

Expected: 五轮均完成 `10/10`；无 SLA 趋势、OOM、重试或服务崩溃。

- [ ] **Step 4: 在同一服务上完成必要精度 smoke 或完整门禁**

先四类各 5 条：

```bash
for dataset in hotpotqa gov_report retrieval_multi_point aggregation_keyword_aggregation; do
  ./run_accuracy.sh "$dataset" 5
done
```

若任一 smoke 与预期不符，候选不提交；若资源窗口允许且计划提交榜单，在同一服务上运行：

```bash
./run_accuracy.sh all
```

Expected: 不扩大误差门限，不因服务启动成本跳过已有异常。

- [ ] **Step 5: 与证据链对齐并给出最终结论**

同时比较：

- 合成 27B kernel A/B；
- 4B 严格端到端 A/B；
- 当前 `c274c6b` 历史 27B 和榜单 `11.03/16.72/14.75`；
- 本轮同一服务两轮中长档的方差。

若数据受外部并发、代理、冷编译或显存残留污染，标记无效，不据此回滚，也不立即进行第二次 27B 启动。

- [ ] **Step 6: 更新 `docs/progress.md` 并提交结论**

记录启动耗时、源码哈希、所有结果目录、完成率、三档指标、精度和最终决定。

```bash
git add docs/progress.md
git commit -m "docs: record Qwen3.5 UA2D V2 27B validation"
```

---

### Task 9: 固化赢家或清理 Triton 实验并决定 HIP 入口

**Files:**
- Modify: `vllm_cscc/vllm/envs.py`
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: Task 8 最终结论。
- Produces: 干净的提交候选，或干净回到稳定基线并触发独立 HIP/MFMA 计划。

- [ ] **Step 1A: 若候选通过，写失败的默认选择测试**

删除“实验值决定正式默认”的长期依赖。先将测试改为要求默认 dispatch 命中赢家 kernel，显式回滚开关仍能命中 `c274c6b` baseline。

例如赢家为 V2-A：

```python
def test_qwen35_ua2d_default_variant_is_v2a() -> None:
    assert ua._qwen35_ua2d_default_variant_name() == "v2a"
```

Run remotely and expect FAIL before cleanup implementation。

- [ ] **Step 2A: 固化赢家并删除其它实验代码**

- 将赢家设为 Qwen3.5 专用默认 kernel；
- 保留现有 `VLLM_ROCM_QWEN_UA2D_FASTPATH=0` 作为整体回退；
- 删除 `VLLM_ROCM_QWEN_UA2D_EXPERIMENT` 类型、环境映射和所有未晋级 kernel；
- 删除只服务于实验编号的测试；
- 保留赢家边界、长 query、精度容限和确定性覆盖。

- [ ] **Step 3A: 运行最终验证并提交 vllm_cscc**

Run locally and remotely:

```bash
python -m py_compile \
  vllm_cscc/vllm/envs.py \
  vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py \
  vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py
git -C vllm_cscc diff --check
```

Remote:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py -k qwen35
```

Expected: 全部 PASS；环境注册表不再包含临时实验变量。

Commit:

```bash
git -C vllm_cscc add \
  vllm/envs.py \
  vllm/v1/attention/ops/triton_unified_attention.py \
  tests/kernels/attention/test_triton_unified_attention.py
git -C vllm_cscc commit -m "perf: enable validated Qwen3.5 UA2D V2"
```

- [ ] **Step 1B: 若所有 Triton 候选未达门槛，清理实验路径**

使用 `apply_patch` 删除 V2-A/V2-B/resource kernel、临时环境变量和实验 dispatch；保留
`c274c6b` 稳定 kernel 及其原有测试。不得使用 `git reset --hard` 或 `git checkout --`。

- [ ] **Step 2B: 验证稳定基线恢复并提交清理**

Run:

```bash
pytest -q tests/kernels/attention/test_triton_unified_attention.py -k qwen35
git -C vllm_cscc diff --check
git -C vllm_cscc add \
  vllm/envs.py \
  vllm/v1/attention/ops/triton_unified_attention.py \
  tests/kernels/attention/test_triton_unified_attention.py
git -C vllm_cscc commit -m "chore: remove unprofitable UA2D V2 experiments"
```

Expected: Qwen3.5 测试 PASS，默认行为等价于 `c274c6b`。

- [ ] **Step 3B: 记录 HIP/MFMA 触发证据**

在 `docs/progress.md` 写明：

- Triton 最优收益及未达门槛原因；
- VGPR/LDS/bank conflict/occupancy 证据；
- 为什么继续扫 Triton 参数不再合理；
- HIP 原型需要解决的单一首要瓶颈。

Commit:

```bash
git add docs/progress.md
git commit -m "docs: record UA2D HIP escalation evidence"
```

完成后停止实施，不直接编写 HIP/C++；先基于实测证据创建独立 HIP/MFMA 设计与实施计划。

---

## Final Verification Checklist

- [ ] `vllm_cscc` 工作树仅包含本计划范围内的已提交改动。
- [ ] DCU 工作树未意外提交 `docs/rankings/` 或 `mcp-ssh` 的用户改动。
- [ ] `python -m py_compile` 和 `git diff --check` 通过。
- [ ] 远端 `which python`、`which vllm`、`pip show vllm` 指向 `.venv` 与源码树。
- [ ] Qwen3.5 UA2D guard、4B/27B、block boundary、GQA=6、long query、确定性测试通过。
- [ ] 合成 27B 记录包含 8K/16K/24K/32K 和 baseline/candidate 交替结果。
- [ ] rocprof 记录包含延迟、VGPR、LDS、bank conflict、L2和 wave 指标。
- [ ] 4B 三档完成率 100%，中长档 request throughput 无回退，四类精度预测零扣分。
- [ ] 每轮只为最终候选启动一次完整 27B 服务。
- [ ] 27B 结果与合成 kernel 和 4B A/B 方向一致，或明确标记污染/无效。
- [ ] `docs/progress.md` 记录所有候选、哈希、实验目录、保留/淘汰原因。
- [ ] 最终源码不保留无效实验环境变量和未使用 kernel。
