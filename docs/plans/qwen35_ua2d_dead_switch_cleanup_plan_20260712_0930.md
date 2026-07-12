# Qwen3.5 UA2D 失效开关清理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除不再影响 Qwen3.5 专用 UA2D 执行路径的 `VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE` 环境变量及其遗留状态。

**Architecture:** 专用 UA2D kernel 继续固定使用已验证的标量 block-table 地址计算。环境变量注册、dispatch 临时变量和测试真假参数化全部删除；通用 UA2D 显式保持 `SCALAR_BLOCK_TABLE=False`。

**Tech Stack:** Python 3.10、pytest、Triton、ROCm/gfx936、Git。

## Global Constraints

- 只修改 `vllm_cscc` 源码与定向测试，不修改模型、比赛脚本、scheduler 或推理语义。
- 本地只编辑和静态检查；GPU 定向测试仅在远端竞赛环境执行。
- 保持 Qwen3.5 专用 UA2D 默认开启及 `TILE32/BLOCK_M32/warps4/stages1` 不变。
- 外部设置旧变量时通过独立的废弃变量集合静默忽略，不恢复运行时开关能力，也不产生告警。

---

### Task 1: 建立失效环境变量删除契约并清理实现

**Files:**
- Modify: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Modify: `vllm_cscc/vllm/envs.py`
- Modify: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: `vllm.envs.environment_variables` 环境变量注册表。
- Produces: 不再注册 `VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE`；专用 UA2D 固定使用自身标量地址计算。

- [ ] **Step 1: 写失败的删除契约测试**

在测试文件中导入 `vllm.envs`，新增：

```python
from vllm import envs


def test_qwen35_removed_scalar_block_table_env() -> None:
    assert "VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE" not in envs.environment_variables
```

- [ ] **Step 2: 运行测试并确认红灯**

Run:

```bash
cd /home/haha/DCU/vllm_cscc
python3 -m pytest tests/kernels/attention/test_triton_unified_attention.py::test_qwen35_removed_scalar_block_table_env -q
```

Expected: 断言失败，因为旧变量仍存在于 `environment_variables`。若本地 pytest 因缺少 `tblib` 无法收集，使用不加载项目 conftest 的 Python 导入检查证明变量当前存在，并记录环境限制。

- [ ] **Step 3: 删除环境变量和无效 dispatch 状态**

从 `vllm/envs.py` 删除类型声明及注册 lambda。从 `unified_attention()` 删除
`scalar_block_table_2d` 初始化和赋值，将通用 kernel launch 参数固定为：

```python
SCALAR_BLOCK_TABLE=False,
```

专用 kernel 源码及 launch 参数保持不变。

- [ ] **Step 4: 简化 GPU 测试参数化**

删除 `scalar_block_table: bool` 参数、对应的 `pytest.mark.parametrize` 以及测试中的
`monkeypatch.setenv("VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE", ...)`。保留所有模型形状、query/KV 边界及五次确定性检查。

- [ ] **Step 5: 验证绿灯和静态质量**

Run:

```bash
cd /home/haha/DCU/vllm_cscc
python3 -m py_compile vllm/envs.py vllm/v1/attention/ops/triton_unified_attention.py tests/kernels/attention/test_triton_unified_attention.py
git diff --check
rg "VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE|scalar_block_table_2d" vllm
```

Expected: 前两条退出 `0`；旧变量不再出现在类型声明、环境变量注册表或 attention dispatch，
只允许出现在 `envs.py` 的废弃变量集合及对应删除/兼容契约测试中；
`scalar_block_table_2d` 无匹配。

- [ ] **Step 6: 更新进度并提交**

在 `docs/progress.md` 的 `2026-07-12` 下记录删除原因、行为不变和验证结果。分别提交源码仓库与主文档仓库：

```bash
cd /home/haha/DCU/vllm_cscc
git add vllm/envs.py vllm/v1/attention/ops/triton_unified_attention.py tests/kernels/attention/test_triton_unified_attention.py
git commit -m "refactor: remove dead UA2D scalar block-table switch"

cd /home/haha/DCU
git add docs/progress.md docs/plans/qwen35_ua2d_dead_switch_cleanup_plan_20260712_0930.md vllm_cscc
git commit -m "docs: 记录 UA2D 失效开关清理"
```

- [ ] **Step 7: 推送源码与文档仓库**

```bash
git -C /home/haha/DCU/vllm_cscc push origin haha
git -C /home/haha/DCU push origin main
```

Expected: 两个远端分别更新到本地 HEAD。
