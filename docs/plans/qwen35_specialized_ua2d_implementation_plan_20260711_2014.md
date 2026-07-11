# Qwen3.5 Specialized UA2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 gfx936 上的 Qwen3.5-4B/27B BF16 文本 prefill 新增独立 UA2D Triton kernel，并在严格正确性门禁下提高长上下文吞吐率。

**Architecture:** 在 `unified_attention()` 保留现有通用 2D/3D 路径，只对已验证的 Qwen3.5 形态分流到独立 `kernel_qwen35_unified_attention_2d`。专用 kernel 保持 paged KV、多序列映射和单 kernel 在线 softmax，但把 KV 循环拆成无需逐元素 causal mask 的完全可见 prefix 与保留 mask 的 diagonal/尾块两段。

**Tech Stack:** Python 3.10、PyTorch BF16、Triton、ROCm/gfx936、pytest、Torch profiler、rocprof、vLLM benchmark。

## Global Constraints

- 只修改 `vllm_cscc` 源码；不修改模型、tokenizer、chat template、比赛脚本、scheduler 或锁定参数。
- 本地只编辑和 Git；远端 `/public/home/xdzs2026_c203/haha` 只构建、运行和测试。
- 每次远端命令先执行 `cd /public/home/xdzs2026_c203/haha && source .venv/bin/activate && source env.sh`。
- 本轮是纯 Python/Triton 改动，不运行 CMake；已有 editable 安装下同步源码并重启进程即可。
- 专用路径仅支持 gfx936、BF16、head size 256、4 KV heads、4B `(16,4,4)` 或 27B `(24,4,6)`、causal prefill，且无 alibi/sinks/softcap/qq-bias/mm-prefix/sliding-window/FP8。
- 任何 guard 不满足时必须使用现有通用路径。
- 4B `16–32K` output throughput 提升至少 `5%`，短/中档回退不超过 `3%`，才进入 27B 验证。

---

## File Map

- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py` — 专用 kernel、严格 guard 与 launch 分流。
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py` — 4B/27B 数值、边界、专用 dispatch 和回退测试。
- Read/execute: `testdata/profile_hotspots_4b.py` — 使用最新 profile 的真实 4B UA2D 形态做 micro/rocprof，不修改该文件。
- Modify: `docs/progress.md` — 按时间戳记录每轮源码实验、测试和结论。

### Task 1: 先建立专用 dispatch 的失败测试

**Files:**
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Test: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`

**Interfaces:**
- Consumes: 现有 `_is_qwen35_ua2d_fastpath_candidate`。
- Produces: `_select_ua2d_kernel(qwen_fastpath_enabled: bool)`，返回专用或通用 Triton kernel 对象。

- [ ] **Step 1: 写纯 Python dispatch 失败测试**

在测试文件 import 区加入：

```python
from vllm.v1.attention.ops import triton_unified_attention as ua
```

新增：

```python
def test_qwen35_ua2d_kernel_selector() -> None:
    assert ua._select_ua2d_kernel(False) is ua.kernel_unified_attention_2d
    assert (
        ua._select_ua2d_kernel(True)
        is ua.kernel_qwen35_unified_attention_2d
    )
```

- [ ] **Step 2: 本地运行并确认因接口不存在而失败**

Run:

```bash
cd /home/haha/DCU/vllm_cscc
python -m pytest tests/kernels/attention/test_triton_unified_attention.py::test_qwen35_ua2d_kernel_selector -q
```

Expected: FAIL，错误包含 `has no attribute '_select_ua2d_kernel'` 或
`kernel_qwen35_unified_attention_2d` 未定义。若本地环境因无 ROCm 无法导入，则只执行
`python -m py_compile tests/kernels/attention/test_triton_unified_attention.py`，并把 pytest
失败原因记录为环境限制，真正的红灯在远端确认。

- [ ] **Step 3: 添加最小 selector 接口，保持专用分支仍未实现**

在 `kernel_unified_attention_2d` 后、`kernel_unified_attention_3d` 前临时声明专用名称，
并在 Python helper 区加入：

```python
kernel_qwen35_unified_attention_2d = None


def _select_ua2d_kernel(qwen_fastpath_enabled: bool):
    if qwen_fastpath_enabled:
        return kernel_qwen35_unified_attention_2d
    return kernel_unified_attention_2d
```

此时测试仍应失败，因为专用返回值为 `None`；不要让 selector 测试提前变绿。

- [ ] **Step 4: 静态检查并提交红灯测试**

Run:

```bash
cd /home/haha/DCU/vllm_cscc
python -m py_compile tests/kernels/attention/test_triton_unified_attention.py vllm/v1/attention/ops/triton_unified_attention.py
git diff --check
```

Expected: 两条命令均退出 `0`。

Commit:

```bash
git add tests/kernels/attention/test_triton_unified_attention.py vllm/v1/attention/ops/triton_unified_attention.py
git commit -m "test: define Qwen3.5 UA2D dispatch contract"
```

### Task 2: 实现独立 Qwen3.5 单 kernel 分段循环

**Files:**
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- Test: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`

**Interfaces:**
- Consumes: `find_seq_idx`、现有 Q/K/V/cache strides、`query_start_len_ptr`、scalar block table。
- Produces: `kernel_qwen35_unified_attention_2d`，参数只包含 Qwen3.5 BF16 文本路径实际需要的数据。

- [ ] **Step 1: 将专用 kernel 的固定接口写入源码**

专用 kernel 参数固定为：

```python
@triton.jit
def kernel_qwen35_unified_attention_2d(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
```

固定 `HEAD_SIZE=256` 时不再传 `HEAD_SIZE_PADDED`；`offs_d = tl.arange(0, HEAD_SIZE)`。

- [ ] **Step 2: 实现 query/GQA 有效行映射**

使用与已验证通用 kernel 相同的索引，保留 27B GQA=6 padding：

```python
q_block_global_idx = tl.program_id(0)
kv_head_idx = tl.program_id(1)
seq_idx = find_seq_idx(
    query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
)
q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
q_block_local_idx = q_block_global_idx - q_block_start_idx
query_start = tl.load(query_start_len_ptr + seq_idx)
query_stop = tl.load(query_start_len_ptr + seq_idx + 1)
query_len = query_stop - query_start
if q_block_local_idx * BLOCK_Q >= query_len:
    return

offs_m = tl.arange(0, BLOCK_M)
offs_d = tl.arange(0, HEAD_SIZE)
offs_t = tl.arange(0, TILE_SIZE)
query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
query_head = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
valid_row = (
    (offs_m < BLOCK_Q * num_queries_per_kv)
    & (query_pos < query_len)
    & (query_head < num_query_heads)
)
```

- [ ] **Step 3: 实现共享的 BF16 tile 读取与在线 softmax 更新代码块**

每个循环体保持下列运算顺序，prefix 与 diagonal 仅在 `score_mask` 上不同：

```python
K = tl.load(key_cache_ptr + k_offset, mask=tile_mask[None, :], other=0.0)
V = tl.load(value_cache_ptr + v_offset, mask=tile_mask[:, None], other=0.0)
S = scale * tl.dot(Q, K)
S = tl.where(valid_row[:, None] & score_mask, S, float("-inf"))
m_j = tl.maximum(M, tl.max(S, axis=1))
m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
P = tl.exp(S - m_j[:, None])
l_j = tl.sum(P, axis=1)
alpha = tl.exp(M - m_j)
acc = acc * alpha[:, None] + tl.dot(P.to(V.dtype), V)
L = L * alpha + l_j
M = m_j
```

K/V offset 必须复用现有 scalar block-table 公式；每个 tile 最多加载两个物理 block id，
不得恢复逐 lane 的非 power-of-two div/mod。

- [ ] **Step 4: 实现 full-prefix 与 diagonal 边界**

绝对 query 位置为 `context_len + query_pos`。完全可见 prefix 的 tile 数按最早有效 query
位置计算：

```python
seq_len = tl.load(seq_lens_ptr + seq_idx)
context_len = seq_len - query_len
first_query_abs = context_len + q_block_local_idx * BLOCK_Q
last_query_rel = tl.minimum(
    q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv,
    query_len - 1,
)
max_visible_key = context_len + last_query_rel
full_tile_end = tl.maximum(0, first_query_abs // TILE_SIZE)
tile_end = (max_visible_key + TILE_SIZE) // TILE_SIZE
```

第一段 `for j in range(0, full_tile_end)` 使用：

```python
score_mask = tl.full((BLOCK_M, TILE_SIZE), True, tl.int1)
```

第二段 `for j in range(full_tile_end, tile_end)` 使用：

```python
seq_offset = j * TILE_SIZE + offs_t
query_abs = context_len + query_pos[:, None]
score_mask = seq_offset[None, :] <= query_abs
```

两段都用 `tile_mask = seq_offset <= max_visible_key`，最后执行：

```python
acc = acc / L[:, None]
tl.store(output_ptr + output_offset, acc, mask=valid_row[:, None])
```

- [ ] **Step 5: 删除 Task 1 的临时 `None` 声明并让 selector 测试变绿**

保留真实 `@triton.jit` 定义，并使用：

```python
def _select_ua2d_kernel(qwen_fastpath_enabled: bool):
    return (
        kernel_qwen35_unified_attention_2d
        if qwen_fastpath_enabled
        else kernel_unified_attention_2d
    )
```

- [ ] **Step 6: 本地静态验证并提交 kernel 本体**

Run:

```bash
cd /home/haha/DCU/vllm_cscc
python -m py_compile vllm/v1/attention/ops/triton_unified_attention.py tests/kernels/attention/test_triton_unified_attention.py
git diff --check
```

Expected: 退出 `0`，无 whitespace error。

Commit:

```bash
git add vllm/v1/attention/ops/triton_unified_attention.py tests/kernels/attention/test_triton_unified_attention.py
git commit -m "feat: add specialized Qwen3.5 UA2D kernel"
```

### Task 3: 接入严格 guard，并补齐数值与回退测试

**Files:**
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`

**Interfaces:**
- Consumes: `_is_qwen35_ua2d_fastpath_candidate`、`_select_ua2d_kernel`、专用 kernel。
- Produces: `unified_attention()` 中专用 launch；所有非候选仍调用 generic launch。

- [ ] **Step 1: 写 4B/27B generic-vs-specialized 参数化失败测试**

抽取现有 Qwen35 测试的数据构造为 `_run_qwen35_ua2d_case`，然后新增：

```python
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Qwen3.5 UA2D specialized kernel is ROCm-specific",
)
@pytest.mark.parametrize(
    ("num_query_heads", "block_size"),
    [(16, 528), (24, 784)],
)
@pytest.mark.parametrize(
    ("query_lens", "kv_lens"),
    [
        ([3, 5, 17], [527, 1057, 1569]),
        ([31, 32, 33], [22258, 16385, 8193]),
    ],
)
def test_qwen35_specialized_ua2d_matches_generic(
    monkeypatch,
    num_query_heads,
    block_size,
    query_lens,
    kv_lens,
) -> None:
    generic = _run_qwen35_ua2d_case(
        monkeypatch, False, num_query_heads, block_size, query_lens, kv_lens
    )
    specialized = _run_qwen35_ua2d_case(
        monkeypatch, True, num_query_heads, block_size, query_lens, kv_lens
    )
    torch.testing.assert_close(specialized, generic, atol=1.5e-2, rtol=1e-2)
```

Expected initial result: FAIL，因为 `unified_attention()` 尚未 launch 专用 kernel。

- [ ] **Step 2: 将 2D launch 分成专用和通用两个显式调用**

不要为两个不同签名的 kernel 构造一个混合 kwargs 字典。使用：

```python
if uses_2d_kernel and qwen_ua2d_fastpath_enabled:
    kernel_qwen35_unified_attention_2d[(total_num_q_blocks, num_kv_heads)](
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
        TILE_SIZE=64,
        HEAD_SIZE=256,
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
        BLOCK_M=32,
        num_warps=4,
        num_stages=1,
    )
elif uses_2d_kernel:
    launch_generic_ua2d_with_existing_arguments()
else:
    launch_generic_ua3d_with_existing_arguments()
```

上面两个 `launch_generic_*` 名称仅表示移动现有源码块，不在源码中创建这两个函数：把当前
`kernel_unified_attention_2d` 从 `output_ptr=out` 到 `num_stages=num_stages_2d` 的完整调用
原样放入 `elif`，把当前 `kernel_unified_attention_3d` 及其后续 reduce 调用原样放入
`else`。移动前后用 `git diff --word-diff` 核对通用调用的参数和值没有变化。

- [ ] **Step 3: 补齐 guard 回退测试**

参数化以下任一变化，断言 `_is_qwen35_ua2d_fastpath_candidate` 返回 `False`：

```python
@pytest.mark.parametrize(
    "override",
    [
        {"max_seqlen_q": 1},
        {"num_query_heads": 8, "num_queries_per_kv": 2},
        {"use_alibi_slopes": True},
        {"use_qq_bias": True},
        {"use_mm_prefix": True},
        {"softcap": 50.0},
        {"sinks": object()},
        {"output_scale": object()},
        {"window_size": (1023, 0)},
    ],
)
def test_qwen35_specialized_guard_rejects_unsupported_features(
    monkeypatch, override
) -> None:
    monkeypatch.setattr(ua.current_platform, "is_rocm", lambda: True)
    args = _make_qwen35_guard_args()
    args.update(override)
    assert not ua._is_qwen35_ua2d_fastpath_candidate(**args)
```

- [ ] **Step 4: 远端同步并执行正确性矩阵**

先确认 GPU、端口和队友进程，不启动服务。同步本地 `vllm_cscc` 后：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
which python
which vllm
python -m pip show vllm
cd vllm_cscc
python -m pytest \
  tests/kernels/attention/test_triton_unified_attention.py::test_qwen35_ua2d_kernel_selector \
  tests/kernels/attention/test_triton_unified_attention.py::test_qwen35_specialized_ua2d_matches_generic \
  tests/kernels/attention/test_triton_unified_attention.py::test_qwen35_specialized_guard_rejects_unsupported_features \
  tests/kernels/attention/test_triton_unified_attention.py::test_qwen35_27b_ua2d_configurations_and_boundaries \
  tests/kernels/attention/test_triton_unified_attention.py::test_qwen35_27b_ua2d_default_long_query_matches_generic \
  -q
```

Expected: 全部通过，无 NaN/Inf、VM fault、超时或随机失败；`which python/vllm` 均位于
`.venv`，editable location 为远端 `vllm_cscc`。

- [ ] **Step 5: 运行现有通用 UA2D 回归**

```bash
python -m pytest tests/kernels/attention/test_triton_unified_attention.py -q
```

Expected: 现有测试全部通过；因显存限制跳过的参数必须记录，不能把 OOM 当作功能通过。

- [ ] **Step 6: 提交 dispatch 和测试**

```bash
cd /home/haha/DCU/vllm_cscc
git add vllm/v1/attention/ops/triton_unified_attention.py tests/kernels/attention/test_triton_unified_attention.py
git commit -m "test: validate specialized Qwen3.5 UA2D path"
```

### Task 4: 用真实热点形态调优资源和单 kernel 延迟

**Files:**
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- Read/execute: `testdata/profile_hotspots_4b.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: 已通过正确性矩阵的专用 kernel。
- Produces: 固化的 `TILE_SIZE/BLOCK_M/num_warps/num_stages` 与 micro/rocprof 证据。

- [ ] **Step 1: 建立 generic 与 specialized 交替 micro 基线**

在远端分别设置：

```bash
export VLLM_ROCM_QWEN_UA2D_FASTPATH=0
python testdata/profile_hotspots_4b.py ua2d --context-len 22258 --repeats 20
export VLLM_ROCM_QWEN_UA2D_FASTPATH=1
python testdata/profile_hotspots_4b.py ua2d --context-len 22258 --repeats 20
```

按 `off/on/on/off` 至少两轮记录平均值；首次 Triton 编译不计入统计。Expected: specialized
输出与 Task 3 数值测试一致，且平均延迟低于 generic；若不低于 generic，先定位 kernel
内部原因，不进入端到端测试。

- [ ] **Step 2: 依次扫描有限参数矩阵**

只允许以下四组，每改一组均重新执行 Task 3 专用正确性测试和 Step 1 micro：

```text
A: TILE=64, BLOCK_M=32, WARPS=4, STAGES=1
B: TILE=32, BLOCK_M=32, WARPS=4, STAGES=1
C: TILE=64, BLOCK_M=16, WARPS=4, STAGES=1
D: TILE=64, BLOCK_M=32, WARPS=8, STAGES=1
```

27B GQA=6 下 `BLOCK_M=16` 的 `BLOCK_Q=2`，必须重新验证 padded row 和相邻 q-block。

- [ ] **Step 3: 对最优候选采集 rocprof 计数器**

使用与最新报告一致的 rocprof 方法，至少记录：

```text
elapsed time, VGPR, LDS, L2 hit, SQ_WAVES, SQ_LDS_BANK_CONFLICT
```

Expected: kernel 延迟下降；VGPR、LDS 或 bank conflict 至少一项有合理改善。若计数器变好但
延迟变差，以延迟为准淘汰候选。

- [ ] **Step 4: 固化单一配置并删除实验参数分支**

源码中只保留最优常量，例如：

```python
QWEN35_UA2D_TILE_SIZE = 64
QWEN35_UA2D_BLOCK_M = 32
QWEN35_UA2D_NUM_WARPS = 4
QWEN35_UA2D_NUM_STAGES = 1
```

不要新增运行时环境变量扫描矩阵。

- [ ] **Step 5: 记录并提交 micro 结论**

向 `docs/progress.md` 追加时间、提交、形态、generic/specialized 延迟、计数器和结论，只提交
本轮新增段落，不覆盖用户已有未提交内容。vLLM 常量改动在 `vllm_cscc` 独立提交：

```bash
cd /home/haha/DCU/vllm_cscc
git add vllm/v1/attention/ops/triton_unified_attention.py
git commit -m "perf: tune Qwen3.5 UA2D for gfx936"
```

### Task 5: 完成 4B 端到端门禁与必要的 27B 验证

**Files:**
- Modify: `docs/progress.md`
- Verify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`

**Interfaces:**
- Consumes: Task 4 固化的专用 kernel。
- Produces: 是否保留实现的最终性能决定和可复现实验记录。

- [ ] **Step 1: 检查远端共享资源并建立 4B baseline**

检查 GPU、8001/8002 等端口和队友进程；不停止他人任务。若 GPU 可用，加载环境并以当前
提交、`VLLM_ROCM_QWEN_UA2D_FASTPATH=0` 启动 4B，使用：

```bash
cd /public/home/xdzs2026_c203/haha/testdata
./run_throughput_4b.sh 4-8K 10
./run_throughput_4b.sh 8-16K 10
./run_throughput_4b.sh 16-32K 10
```

若实际脚本档位只接受 `16-32K`，按仓库当前脚本帮助输出使用完全相同的 baseline/candidate
参数；不得修改脚本解析或锁定参数。

- [ ] **Step 2: 相同环境运行 specialized candidate**

只切换 `VLLM_ROCM_QWEN_UA2D_FASTPATH=1`，重启本工作区服务并按相同顺序运行三档各 10 条。
使用 `curl --noproxy '*'` 或设置大小写 `NO_PROXY`；502/503 先确认是否来自 Squid。

- [ ] **Step 3: 审核 4B 门禁**

同时核对完成率、输入长度、生成 token、request/output throughput、TTFT、TPOT 和日志：

```text
16–32K output throughput >= baseline * 1.05
4–8K output throughput >= baseline * 0.97
8–16K output throughput >= baseline * 0.97
completion = 10/10 for every band
```

若不满足，回滚专用 launch 和 kernel 提交，保留测试中对现有通用路径有价值的独立修复；
不要以单次噪声或输出变短宣称成功。

- [ ] **Step 4: 4B 通过后按需运行 27B**

只有共享 GPU 空闲且 4B 通过时启动 27B。先确认实际 guard 命中，再运行三档短测；共享存储
加载约半小时属于正常现象。Expected: 完成率 `100%`、无数值或时延异常、吞吐不回退。

- [ ] **Step 5: 最终静态和定向回归**

本地：

```bash
cd /home/haha/DCU/vllm_cscc
python -m py_compile vllm/v1/attention/ops/triton_unified_attention.py tests/kernels/attention/test_triton_unified_attention.py
git diff --check
git status --short --branch
```

远端：重跑 Task 3 的专用测试集合。Expected: 全部通过，vLLM 工作树仅包含预期提交。

- [ ] **Step 6: 更新进度并做完成审计**

在 `docs/progress.md` 记录最终保留或回滚结论、4B/27B 数据、测试数量、实验目录和 vLLM
提交哈希。逐项对照设计文档的 guard、语义、测试和性能门禁；只有全部证据成立才将目标标记
完成。
