# Qwen3.5 UAST-2 Attention 结构重写实施计划（2026-07-10 19:13 CST）

## 1. 计划元信息

| 项目 | 内容 |
| --- | --- |
| 计划版本 | `20260710-1913` |
| 路线编号 | `UAST-2` |
| 计划性质 | 停止 Attention 参数扫描后的结构性重写 |
| 主要目标 | 拆分 paged prefix 与 contiguous current chunk，并合并 online-softmax state |
| 优先模型 | Qwen3.5-4B；通过门禁后再申请 Qwen3.5-27B 验证 |
| 主要后端 | `TRITON_ATTN` / ROCm gfx936 |
| 第一版实现语言 | Python + Triton；不修改 `csrc/` |
| 默认状态 | 实验开关默认关闭，未命中严格 guard 时回退现有 UA2D |

所有实验目录继续使用：

```text
<实验编号>_YYYYMMDD_HHMM
```

建议编号：

```text
UAST2-CORR1_20260711_0900
UAST2-MICRO1_20260711_1100
UAST2-E2E0_20260711_1400
UAST2-E2E1_20260711_1500
```

## 2. 决策背景

当前不再继续以下工作：

- 扫描 `TILE/BLOCK_M/WARPS/STAGES`。
- 扩大 LLMM1 形状 allowlist。
- 继续优化 block-table 标量化等低上限局部路径。
- 通过叠加多个环境变量寻找偶然最优组合。

旧 profile 中 Unified Attention 约占全局 GPU 时间 `42.59%`，仍是长上下文的主要高收益
方向。当前 `kernel_unified_attention_2d` 对每个 Q block 扫描可见 prefix，并在循环内统一执行
causal mask、paged block-table 地址计算和 KV Cache 读取。对于 chunked prefill：

1. `context_len` 之前的历史 prefix 对当前 chunk 的所有 query 均完全可见，不需要 causal
   compare。
2. 当前 chunk 内部才需要 causal diagonal。
3. 当前 chunk 的 K/V 已经以连续张量形式传入 Attention，但现有 `TRITON_ATTN` 最终只从
   paged KV Cache 读取，未利用这份连续 K/V。

UAST-2 将两部分拆开计算，再通过数值稳定的 online-softmax state 合并恢复完整 attention
结果。

## 3. 当前源码事实与实现约束

### 3.1 当前调用路径

```text
Qwen3NextAttention.forward
  -> Attention.forward
  -> unified_kv_cache_update(key, value)
  -> unified_attention_with_output(query, key, value, output)
  -> TritonAttentionImpl.forward
  -> triton_unified_attention.unified_attention
  -> kernel_unified_attention_2d / kernel_unified_attention_3d
```

关键事实：

- `TritonAttentionImpl.forward()` 同时持有当前 chunk 的连续 `query/key/value` 和已经完成写入的
  paged KV Cache。
- `TritonAttentionBackend.forward_includes_kv_cache_update=False`，现有
  `kv_cache_dummy_dep` 已保证 cache update 先于 attention，UAST-2 不得破坏该依赖。
- 当前 `triton_unified_attention.unified_attention()` 只接收 query 和 paged K/V cache；需要
  为 UAST-2 显式传入当前 chunk 的连续 K/V。
- `TRITON_ATTN` 的 KV Cache 为 NHD：
  `[num_blocks, block_size, num_kv_heads, head_size]`。

### 3.2 不能直接复用的旧实现

`vllm/v1/attention/ops/prefix_prefill.py` 已包含“paged prefix + contiguous current chunk”的
计算思路，但不能直接作为本路线最终实现：

- 它使用的 K/V cache 维度和地址布局与当前 `TRITON_ATTN` NHD 布局不同。
- 它只写最终归一化 output，不输出 prefix/suffix LSE。
- 它在单个 kernel 内连续处理两部分，不能直接接入现有
  `triton_merge_attn_states.merge_attn_states()`。

`triton_prefill_attention.context_attention_fwd()` 可以处理连续 Q/K/V causal attention，但当前
同样不输出 LSE。因此第一版需要扩展或新增 partial-state kernel，不能只做 Python 层拼接。

### 3.3 可复用组件

- `triton_merge_attn_states.merge_attn_states()`：已有数值稳定的 prefix/suffix output + LSE
  合并公式。
- `tests/kernels/attention/test_merge_attn_states.py`：已有 PyTorch reference。
- `tests/kernels/attention/test_triton_unified_attention.py`：已有 Qwen3.5 4B/27B、block 边界、
  长 query 和 generic UA2D reference 测试框架。
- `TritonAttentionMetadataBuilder`：可预分配并复用 UAST-2 workspace，避免每层、每次 forward
  动态分配。

## 4. 目标结构

### 4.1 数据区间

对每个 sequence：

```text
seq_len = context_len + query_len

[0, context_len)
    历史 prefix，来自 paged KV Cache

[context_len, seq_len)
    当前 chunk，来自连续 current_key/current_value
```

### 4.2 计算图

```text
                       query
                         |
             +-----------+-----------+
             |                       |
             v                       v
  paged prefix attention     contiguous causal attention
  K/V: KV Cache              K/V: current_key/current_value
  range: [0, context_len)    range: [0, query_len)
  no causal mask             causal diagonal
             |                       |
             v                       v
   prefix_output/prefix_lse  suffix_output/suffix_lse
             |                       |
             +-----------+-----------+
                         |
                         v
              merge_attn_states
                         |
                         v
                    final output
```

### 4.3 Partial-state 定义

两个 attention 分支都必须使用 FP32 `M/L/acc` 进行 online softmax，并输出：

```text
partial_output = acc / L
partial_lse = M + log(L)
```

若 suffix kernel 使用 `exp2` 域，则写回 merge kernel 前必须转换为自然对数域：

```text
partial_lse = (M_log2 + log2(L)) * ln(2)
```

禁止直接合并两个已经归一化 output 而不携带 LSE。

### 4.4 Workspace 设计

第一版使用 metadata builder 预分配固定地址 workspace，避免在 attention forward 内调用
`torch.empty`：

```text
suffix_output: [max_num_batched_tokens, num_query_heads, head_size], bf16
prefix_lse:    [num_query_heads, max_num_batched_tokens], fp32
suffix_lse:    [num_query_heads, max_num_batched_tokens], fp32
```

内存复用策略：

- prefix kernel 直接把 `prefix_output` 写入最终 `output` buffer。
- suffix kernel 写入 `suffix_output` workspace。
- merge kernel 读取 `output` 作为 prefix input，并原位写回最终 `output`。
- 不分配第二份 prefix output，也不保留最终 output LSE。
- workspace 在各层顺序执行时复用；必须验证 eager、piecewise graph 和 capture 使用固定地址时
  均无跨层覆盖问题。

## 5. 严格命中条件与回退

新增实验开关：

```text
VLLM_ROCM_QWEN_UAST2_ATTENTION
```

第一版默认 `False`。仅同时满足以下条件时命中：

- ROCm gfx936。
- dtype 为 BF16。
- causal decoder self-attention。
- Qwen3.5 文本 full-attention 实际形态：
  - 4B：heads `16`、KV heads `4`、GQA `4`、head size `256`、block size `528`。
  - 27B：heads `24`、KV heads `4`、GQA `6`、head size `256`、block size `784`。
- `max_seqlen_q > 1`，decode `q_len=1` 继续走现有路径。
- 无 alibi、sinks、softcap、QQ bias、MM PrefixLM、sliding window 和 FP8 output。
- 当前 key/value 非空，shape 与 query metadata 一致。
- KV cache layout 为当前 `TRITON_ATTN` NHD。

任一条件不满足时无条件回退当前 `unified_attention()`。禁止静默修改输入、scheduler metadata、
query length 或 block table 以强制命中。

## 6. 分阶段实施

### 阶段 S0：冻结基线与接口契约

工作内容：

1. 固定当前 UA2D 默认版本、commit、环境变量和三档 4B baseline。
2. 记录 `TritonAttentionImpl.forward()` 中 current K/V、KV cache、query metadata 的 shape、
   stride、dtype，仅去重打印。
3. 确认真实 4B chunked prefill 的 `query_len/context_len/seq_len`，不做参数扫描。
4. 为 UAST-2 guard、fallback 和 workspace shape 编写 Python 单元测试。

完成条件：接口和布局记录完整，实验关闭时行为与当前版本一致。

### 阶段 S1：数据通路与 workspace

建议修改文件：

```text
vllm/v1/attention/backends/triton_attn.py
vllm/v1/attention/ops/triton_unified_attention.py
vllm/envs.py
```

工作内容：

1. `TritonAttentionImpl.forward()` 将裁剪后的 current `key/value` 传给 UAST-2 dispatcher。
2. 保持现有 `unified_kv_cache_update -> attention` 数据依赖不变。
3. 在 `TritonAttentionMetadataBuilder` 中预分配 suffix output 与两份 LSE workspace。
4. graph capture 使用相同固定地址和固定最大 shape；实际 kernel 仅处理
   `num_actual_tokens`。
5. 开关关闭、guard 未命中、workspace 不可用时完全回退原路径。

本阶段不改变 attention 数学计算，只打通接口。

### 阶段 S2：Paged Prefix Partial-State Kernel

新增建议文件：

```text
vllm/v1/attention/ops/triton_qwen35_uast2_attention.py
```

实现内容：

- 输入 query、NHD paged K/V cache、block table、seq lens、query start loc。
- 每个 sequence 计算 `context_len = seq_len - query_len`。
- 只遍历 `[0, context_len)`，绝不读取当前 chunk 区间。
- 删除 causal compare；保留最后一个 prefix tile 的边界 mask。
- 复用当前 UA2D 已验证的非 2 次幂 block 地址计算与严格有效行 mask。
- FP32 `M/L/acc`，写 normalized prefix output 与 FP32 natural-log LSE。
- `context_len=0` 时写 `prefix_lse=-inf`，prefix output 写零，禁止产生 NaN。

第一版保持单一固定实现，不增加 tile/warp 参数矩阵。

### 阶段 S3：Contiguous Current-Chunk Partial-State Kernel

实现选择：

1. 优先从 `triton_prefill_attention.py` 的 contiguous causal kernel 派生；或
2. 在新 UAST-2 文件中维护严格 Qwen3.5 专用版本，避免影响通用 encoder/prefill 路径。

实现内容：

- query、current key、current value 均使用连续张量。
- 每个 sequence 只计算当前 chunk 内 causal attention。
- 不读取 block table，不读取 paged KV Cache。
- 输出 normalized suffix output 与 FP32 natural-log suffix LSE。
- 正确处理最后一个不完整 Q/K tile。
- 保持 head size `256`、GQA `4/6` 语义；K/V head 映射必须与现有 UA2D 一致。

优先选择“新专用 kernel”，因为现有通用 contiguous kernel 没有 LSE 输出，直接修改会扩大
回归面。

### 阶段 S4：State Merge 与 dispatcher

工作内容：

1. 使用 `triton_merge_attn_states.merge_attn_states()` 合并 prefix/suffix。
2. merge 目标与 prefix output 共用最终 output buffer，验证原位读写安全。
3. 增加 ROCm merge correctness 测试；现有测试依赖 CUDA custom kernel，不能作为 gfx936
   唯一证据。
4. dispatcher 顺序固定：

```text
prefix partial -> suffix partial -> merge
```

5. 第一版不融合 merge epilogue。只有三 kernel 路径正确但 launch 开销抵消明显收益时，才进入
   S7 可选融合阶段。

### 阶段 S5：正确性与确定性门禁

先完成纯 kernel 对照，再启动端到端服务。

#### 形态矩阵

| 维度 | 测试值 |
| --- | --- |
| 模型形态 | 4B GQA=4/block=528；27B GQA=6/block=784 |
| head size | `256` |
| dtype | BF16 |
| query length | `1/2/3/5/15/16/17/31/32/63/64/65/4095/4096` |
| context length | `0`、block 边界前后、约 `4K/8K/16K/32K` |
| block table | 连续、随机非连续物理 block |
| 重复次数 | 同一输入至少 5 次 |

#### 必测边界

- `context_len=0`：merge 后应完全等于 contiguous causal attention。
- `query_len=1`：UAST-2 guard 不命中，确认回退现有 decode。
- prefix 在 `527/528/529` 和 `783/784/785`。
- 最后一个不完整 4096-token chunk。
- query tile 小于、等于、大于 kernel block。
- GQA=6 的 padded row 不得与相邻 Q block 重叠写。
- 当前 chunk 已写入 KV cache，但 prefix kernel必须排除该区间，防止重复计数。

#### Reference 层次

1. 小尺寸：PyTorch dense causal attention。
2. 中尺寸：现有 generic/current UA2D。
3. 4095/4096 长 query：当前正式 UA2D 与 UAST-2 对照，不构造超大 dense reference。

验收：

- 无 NaN、Inf、OOM、VM fault、越界和随机漂移。
- 重复运行输出稳定。
- BF16 `torch.testing.assert_close` 初始门槛沿用当前 UA2D：
  `atol=1.5e-2, rtol=1e-2`；同时记录 max/P99 absolute、relative diff。
- 任何明显超过当前 UA2D 误差范围的结果必须停止性能测试并定位。

### 阶段 S6：4B 性能与精度门禁

#### Microbenchmark

只做 UAST-2 off/on，不扫 kernel 参数：

| query/context | 目的 |
| --- | --- |
| `4096/0` | 纯 current chunk 成本 |
| `4096/4K` | 中短 prefix |
| `4096/8K` | 中档 |
| `4096/16K` | 中长档 |
| `4096/32K` | 长档上界 |

记录：总 kernel latency、prefix/suffix/merge 分项、P50/P99、重复波动。

#### 4B 端到端

1. 使用 8002、`gpu-memory-utilization=0.45`。
2. 检查共享 GPU、端口和队友进程。
3. 同一源码版本先跑开关关闭 baseline，再跑开启候选。
4. 三档各 10 条，至少重复两轮。
5. 保存 output length、generated text、TTFT/TPOT、完成率和服务日志。

保留门槛：

| 指标 | 门槛 |
| --- | --- |
| `16K-32K` output throughput | `>=10%` |
| `8K-16K` output throughput | `>=5%`，不得因输出长度变化伪造收益 |
| `4K-8K` | 回退 `<1%` |
| 核心指标重复波动 | `<3%` |
| 完成率 | `100%` |
| SLA | TTFT/TPOT 无超限风险 |

由于 prefix/suffix 拆分会改变 reduction 顺序，不能要求生成文本逐 token 完全相同，但必须：

- 运行官方四类 4B 完整精度 baseline/candidate。
- 四类精度均不回退到扣分区间。
- 输出变化必须记录并可归因于 BF16 reduction 顺序，不能出现随机漂移。

### 阶段 S7：可选的第二轮结构融合

仅当 S6 显示：

- prefix/suffix kernel 本身有明显收益；但
- 独立 merge 或额外 workspace 写回使端到端收益低于门槛，

才考虑：

1. 将 merge 融入 suffix 或专用 epilogue。
2. prefix 输出改为 compact partial state，减少 BF16 workspace 往返。
3. 单 kernel 内共享 `M/L/acc`，但仍保持 paged prefix 与 contiguous suffix 两种加载路径。

该阶段仍禁止参数矩阵调优，只比较结构版本 off/on。

## 7. 27B 升级门禁

本计划不直接授权启动 27B。升级前必须：

1. 本地静态检查通过。
2. 远端 4B/27B 形态定向 kernel correctness 均通过；27B 形态测试不等于启动 27B 服务。
3. 4B 三档完成率 `100%`。
4. 4B 至少中档或长档达到 S6 收益门槛。
5. 两轮核心指标波动 `<3%`。
6. 4B 四类完整精度无回退。
7. 汇报结果并获得用户确认。

获得确认后，27B 第一轮只做少量样本和 `16K-32K` 区分性测试；只有结果明确时才运行完整
三档与四类 OpenCompass。

## 8. 构建、同步与验证

第一版为 Python/Triton 改动，可使用 editable install：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e ./vllm_cscc --no-build-isolation --no-deps
```

每次测试前：

1. 本地修改并做 git diff 检查。
2. 同步源码到远端。
3. 校验关键文件 SHA256。
4. 激活 `.venv` 并加载 `env.sh`。
5. 确认导入路径来自 `/public/home/xdzs2026_c203/haha/vllm_cscc`。
6. 检查 GPU 和 8001/8002 占用。

基础检查：

```bash
which python
which vllm
python -m pip show vllm
```

若后续 S7 引入 HIP/C++，必须改用：

```bash
./build_vllm_wheel_install.sh
```

## 9. 合规边界

本路线允许：

- Attention kernel、PagedAttention 路径、online-softmax 与内存访问优化。
- 不改变有效计算结构和输出语义的 Triton/HIP kernel。
- 基于模型通用 shape、dtype、平台和 backend capability 的 dispatch。

本路线禁止：

- 修改原始比赛测试脚本。
- 修改模型、权重、tokenizer、chat template 或采样语义。
- 修改 `max-model-len`、`max-num-seqs`、`max-num-batched-tokens` 或 scheduler 行为。
- 截断 prefix/current chunk、跳过 token、跳过层或减少真实 attention 范围。
- 根据 prompt、数据集类别或 token 内容命中 fastpath。
- 通过输出长度变化解释或伪造 throughput 收益。

## 10. 回滚设计

| 层级 | 回滚方式 |
| --- | --- |
| 运行时 | `VLLM_ROCM_QWEN_UAST2_ATTENTION=0` |
| Guard | 任一约束不满足时回退当前 UA2D |
| 源码 | UAST-2 独立文件与独立 commit，不与其它主优化叠加 |
| Workspace | metadata 字段仅在 UAST-2 支持的平台/形态初始化 |
| 数值 | 保留当前 UA2D 作为同进程 reference 与正式 fallback |

禁止在 UAST-2 尚未通过精度门禁前默认开启。

## 11. 决策表

| 结果 | 决策 |
| --- | --- |
| 长档 `>=10%`、中档 `>=5%`、精度无回退 | 保留，汇报并申请 27B |
| micro 明显提升但端到端被 merge/workspace 抵消 | 进入 S7 融合 |
| 长档收益 `3%~10%` | 只允许一次明确的结构归因优化，不做参数扫描 |
| 收益 `<3%` | 淘汰 UAST-2，不继续增加复杂度 |
| 数值超标、精度回退、随机漂移或 VM fault | 立即关闭并回退当前 UA2D |
| 仅靠输出 token 减少提高吞吐 | 结果无效 |

## 12. 最近一轮具体执行清单

下一轮只执行 S0 和 S1：

1. 固定当前 UA2D commit、工作区和 4B baseline 结果。
2. 增加默认关闭的 `VLLM_ROCM_QWEN_UAST2_ATTENTION`。
3. 实现严格 guard 与 fallback 测试。
4. 将 current K/V 安全传入 UAST-2 dispatcher，不改变现有默认计算。
5. 在 metadata builder 中加入固定 workspace，并验证 eager/capture shape。
6. 远端只做导入、静态和小 shape smoke，不启动 27B。
7. 更新 `docs/progress.md`，再进入 S2 prefix partial-state kernel。

