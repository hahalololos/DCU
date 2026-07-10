# Qwen3.5 `kernel_unified_attention_2d` 优化实施方案

生成日期：2026-07-09

## 1. 目标

在不修改模型权重、tokenizer、chat template、推理语义和原始比赛测试脚本的前提下，针对
Qwen3.5 长上下文 chunked prefill 中的 `kernel_unified_attention_2d` 热点做可回滚优化。

主目标：

| 指标 | 目标 |
| --- | --- |
| `16k-32k` | 明显降低 TTFT 和 `kernel_unified_attention_2d` 总时长 |
| `8k-16k` | 提升或至少不回退 |
| `4k-8k` | 不作为主收益来源，但不能明显回退 |
| 正确性 | attention 输出数值误差在现有 Triton 路径可接受范围内 |
| 稳定性 | 4B smoke、27B 三档短测、必要精度回归通过 |

## 2. 当前判断

已有 profile 表明：

| 档位 | 主要瓶颈 |
| --- | --- |
| `4k-8k` | GEMM / Linear |
| `8k-16k` | GEMM + prefill attention 混合 |
| `16k-32k` | `kernel_unified_attention_2d` prefill attention |

`P0A gfx936 skinny GEMM` 修复后没有吞吐提升，说明下一轮不应继续押注 decode skinny
GEMM。长档大突破应优先处理 `kernel_unified_attention_2d`。

当前实现位置：

| 文件 | 关注点 |
| --- | --- |
| `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py` | `kernel_unified_attention_2d()` 和 `unified_attention()` |
| `vllm_cscc/vllm/v1/attention/backends/triton_attn.py` | `TritonAttentionImpl.forward()` 调用入口 |
| `vllm_cscc/vllm/v1/attention/backends/rocm_aiter_unified_attn.py` | AITER unified attention 对照 |
| `vllm_cscc/vllm/v1/attention/backends/rocm_aiter_fa.py` | AITER FA 对照 |
| `vllm_cscc/vllm/v1/attention/ops/prefix_prefill.py` | 可复用的 prefill kernel 参考 |
| `vllm_cscc/vllm/v1/attention/ops/chunked_prefill_paged_decode.py` | prefix prefill + paged decode 组合参考 |

## 3. 优化假设

### H1：现成 ROCm/AITER 后端可能直接优于当前 Triton UA2D

先用后端 A/B 快速确认是否有可直接启用的收益。若 AITER 在 `gfx936` 上可运行且
`kernel_unified_attention_2d` 热点明显下降，优先走低代码改动方案。

### H2：Qwen3.5 固定形态可以绕开通用 kernel 开销

比赛模型与脚本使 prefill attention 形态较稳定：

| 参数 | 预期值 |
| --- | --- |
| dtype | bf16 |
| head size | 256 |
| query heads | 24 |
| kv heads | 4 |
| GQA ratio | 6 |
| block size | 16 |
| max batched tokens | 4096 |
| attention | causal decoder |
| 多模态 / alibi / sinks / softcap | 常规文本评测中应为关闭 |

在这些条件下可以新增严格 guard 的专用 fast path，减少通用逻辑、调优 tile 和 launch
形态。

### H3：结构性收益来自 prefill 拆分，而不是只调单个常量

当前 2D kernel 对每个 Q block 串行扫描 prefix tiles。对于 16k-32k，prefix 很长，
单次 kernel 时间巨大。较大突破需要考虑：

- full prefix 区间与 diagonal causal 区间分开处理；
- 当前 chunk 的 contiguous Q/K/V 部分复用 flash/context prefill kernel；
- paged prefix 与当前 chunk 的 softmax state 合并。

## 4. 分阶段实施

### 阶段 0：形态与命中率确认

目的：避免写错优化对象。

实施：

1. 增加临时 debug 统计，记录 `unified_attention()` 中以下字段：
   - `q.shape`
   - `k.shape`
   - `v.shape`
   - `num_seqs`
   - `max_seqlen_q`
   - `max_seqlen_k`
   - `num_query_heads`
   - `num_kv_heads`
   - `num_queries_per_kv`
   - `head_size`
   - `block_size`
   - `TILE_SIZE_PREFILL`
   - `window_size`
   - `softcap/alibi/sinks/qq_bias/mm_prefix` 是否启用
2. debug 必须由环境变量控制，默认关闭。
3. 远端跑 4B `16-32K 10` smoke，确认热点形态。

建议环境变量：

```bash
VLLM_ROCM_UA2D_DEBUG_SHAPES=1
```

验收：

| 结果 | 处理 |
| --- | --- |
| 形态符合 Qwen3.5 guard | 进入阶段 1/2 |
| 形态不符合 | 先修正方案，不写专用 kernel |

### 阶段 1：后端 A/B 快速实验

目的：先确认有没有现成可用的高性能后端。

实验矩阵：

| 实验 | 后端/环境 | 期望 |
| --- | --- | --- |
| B0 | 当前 `TRITON_ATTN` | 复现 baseline |
| B1 | `ROCM_AITER_UNIFIED_ATTN` | 替换 UA2D |
| B2 | `ROCM_AITER_FA` | 验证 prefill/decode 分流 |
| B3 | `ROCM_ATTN` | 验证 ROCm split 路径 |

优先不修改原始测试脚本。可通过环境变量或源码后端优先级做可回滚实验。

验收：

| 指标 | 成功标准 |
| --- | --- |
| 服务启动 | 无 import/backend/arch 错误 |
| `16k-32k` | throughput 上升，TTFT P99 不恶化 |
| profile | `kernel_unified_attention_2d` 总时长下降或消失 |
| 正确性 | smoke 输出无异常，后续精度回归不明显下降 |

若 B1/B2/B3 有显著收益，优先固化后端选择；否则进入阶段 2。

### 阶段 2：Qwen3.5 专用 UA2D fast path

目的：在当前 Triton UA2D 内新增严格条件保护的专用路径。

入口：

```text
vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py::unified_attention()
```

新增 guard：

| 条件 | 要求 |
| --- | --- |
| 平台 | ROCm，优先 `gfx936` |
| dtype | bf16 |
| attention | causal |
| `head_size` | 256 |
| `num_query_heads` | 24 |
| `num_kv_heads` | 4 |
| `num_queries_per_kv` | 6 |
| `block_size` | 16 |
| `max_seqlen_q` | `> 1` |
| 特性 | no alibi、no sinks、no softcap、no qq_bias、no mm_prefix |
| KV dtype | 先只支持 bf16 KV cache，不碰 fp8 |

建议新增环境变量：

```bash
VLLM_ROCM_QWEN_UA2D_FASTPATH=0
VLLM_ROCM_QWEN_UA2D_TILE=32
VLLM_ROCM_QWEN_UA2D_BLOCK_M=16
```

默认先关闭，验证稳定后再考虑对 `gfx936 + Qwen3.5` 默认开启。

第一版 kernel 目标：

1. 保持 paged KV cache 语义，避免改 KV 管理。
2. 保持多 seq 支持，仍使用 `query_start_len_ptr` 映射 seq。
3. 固化不需要的 feature 分支：
   - 删除 alibi 分支；
   - 删除 softcap 分支；
   - 删除 sinks 初始化；
   - 删除 qq_bias；
   - 删除 mm_prefix；
   - 删除 fp8 output。
4. 显式调参：
   - `TILE_SIZE=32/64`
   - `BLOCK_M=16/32`
   - `num_warps=4/8`
   - `num_stages=1/2`

注意：Triton 的 `tl.arange` 维度通常更适合 power-of-two。GQA ratio 6 与
`BLOCK_M=16` 不完全匹配，第一版不要冒险改成非 power-of-two block；先通过
`BLOCK_M/TILE_SIZE/warps` A/B 找收益。

验收：

| 指标 | 成功标准 |
| --- | --- |
| micro | 专用 kernel 单次耗时低于 generic UA2D |
| 4B smoke | 10/10 成功，无 VM fault |
| 16-32K | total/output throughput 明显上升 |
| 数值 | 与 generic UA2D 误差可接受 |

### 阶段 3：prefix full tiles 与 diagonal tiles 拆分

目的：减少 causal mask 与长 prefix 下的通用开销。

思路：

1. 对每个 q block，将 key 范围分为：
   - `context_len` 之前的 full prefix，所有 query token 都可见；
   - 当前 chunk 内的 diagonal causal 区间，需要 causal mask。
2. full prefix 路径去掉 causal `seq_mask` 计算。
3. diagonal 路径保留 causal mask。
4. 两段共享在线 softmax 状态 `M/L/acc`。

预期收益：

| 来源 | 说明 |
| --- | --- |
| full prefix | 16k-32k 中占比大，去 mask 后更利于编译器优化 |
| diagonal | 保留语义但只覆盖较短当前 chunk |

风险：

| 风险 | 应对 |
| --- | --- |
| softmax state 合并错误 | 对比 generic UA2D 输出 |
| 边界 token 处理错误 | 覆盖 q_len 非 4096、最后一个 chunk、短 prompt |

### 阶段 4：复用 contiguous current-chunk prefill

目的：把当前 chunk 的 Q/K/V contiguous attention 从 paged cache 路径中拆出来。

可复用参考：

| 文件 | 用途 |
| --- | --- |
| `prefix_prefill.py` | `context_attention_fwd()` 已处理 contiguous prefill |
| `chunked_prefill_paged_decode.py` | 已有 prefix/context + paged decode 组合思路 |
| `triton_merge_attn_states.py` | softmax state 合并参考 |

结构：

1. prefix 部分：paged KV prefix attention，输出 prefix softmax state。
2. current chunk 部分：使用 contiguous Q/K/V flash/context prefill，输出 suffix state。
3. merge：按 online softmax 规则合并 prefix/suffix 输出。

这是高收益但高风险阶段，只有阶段 2/3 收益不足时再推进。

### 阶段 5：固化与清理

完成条件：

1. 删除临时 debug 输出或默认关闭。
2. 保留一个明确的 feature flag 和回退路径。
3. 更新 `docs/progress.md`，记录每个实验的吞吐、P99、profile 变化。
4. 若默认启用 fast path，必须写清 guard 条件和回滚方式。

## 5. 测试与验证流程

本地只做阅读和源码修改；远端只做构建、运行和测试。

同步到远端后执行：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
```

Python-only 改动：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e ./vllm_cscc --no-build-isolation --no-deps
```

如涉及 HIP/C++/CMake：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
./build_vllm_wheel_install.sh
```

基础检查：

```bash
which python
which vllm
python -m pip show vllm
```

4B smoke：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd testdata
./start_vllm_4b.sh
./run_throughput_4b.sh 16-32K 10
```

27B 三档短测：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd testdata
./start_vllm.sh
./run_throughput.sh 4k-8k 10
./run_throughput.sh 8k-16k 10
./run_throughput.sh 16k-32k 10
```

## 6. 记录模板

每次实验记录到 `docs/progress.md`：

| 字段 | 内容 |
| --- | --- |
| 实验编号 | 如 `UA2D-B1`、`UA2D-FP-T64-B16-W4` |
| 改动 | 后端、tile、block、warps、guard |
| 数据 | throughput、output throughput、TTFT P99、TPOT P99 |
| profile | `kernel_unified_attention_2d` 调用次数、总时长、平均时长 |
| 结论 | 继续、回滚、扩大验证 |

## 7. 回滚策略

| 改动类型 | 回滚方式 |
| --- | --- |
| 后端 A/B | 恢复默认 backend priority / 关闭环境变量 |
| Qwen fast path | `VLLM_ROCM_QWEN_UA2D_FASTPATH=0` |
| 参数实验 | 恢复默认 `TILE_SIZE=32` 与 generic UA2D |
| 结构性拆分 | guard 不命中时回退 generic UA2D |

任何实验若出现 VM fault、输出异常、完成率下降或 P99 明显恶化，应立即回退到 generic
`kernel_unified_attention_2d`。

## 8. 推荐执行顺序

1. `UA2D-S0`：形态统计，确认 guard 命中面。
2. `UA2D-B1/B2/B3`：AITER/ROCm 后端 A/B。
3. `UA2D-FP1`：新增 Qwen fast path 空壳和 feature flag，默认关闭。
4. `UA2D-FP2`：实现专用 Triton kernel，先保持 `TILE_SIZE=32/BLOCK_M=16`。
5. `UA2D-FP3`：扫 `TILE_SIZE=64`、`BLOCK_M=32`、`num_warps=4/8`。
6. `UA2D-FP4`：若收益不足，拆 full prefix 与 diagonal。
7. `UA2D-FP5`：若仍不足，推进 contiguous current-chunk prefill + state merge。

优先级判断：如果阶段 1 有 10% 以上长档收益，先固化后端；如果没有，阶段 2/3 是主线。
