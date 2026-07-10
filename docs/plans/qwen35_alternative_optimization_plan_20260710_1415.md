# Qwen3.5 其他优化方向实施计划（2026-07-10 14:15 CST）

## 1. 计划元信息

| 项目 | 内容 |
| --- | --- |
| 计划版本 | `20260710-1415` |
| 生成时间 | `2026-07-10 14:15:00 CST (+0800)` |
| 计划性质 | 在 UA2D 参数调优、Skinny/LLMM1 全局开关和 HipBLASLt 初筛之后的替代优化路线 |
| 排行榜基线 | 第 `51` 名，最终得分 `76.6248` |
| 当前三档吞吐 | `4K-8K=15.84`、`8K-16K=14.22`、`16K-32K=10.05 tok/s` |
| 当前扣分 | SLA `0`，精度系数约 `0.985`，精度扣分 `1.1669` |
| 当前提交源码 | `d3dc2e69f8830788fe48d56840868579e450d287` |
| 已完成的新增测试 | `CORR-K0~K3` 小边界 `8 passed`；4095/4096 长 query `2 passed` |

为与此前计划和实验区分，所有新增实验目录、日志和记录必须使用：

```text
<实验编号>_YYYYMMDD_HHMM
```

例如：

```text
ALT-A1_20260710_1500
ALT-B1_20260710_1700
ALT-C-MICRO1_20260711_0930
```

## 2. 已知事实与路线调整原因

### 2.1 已完成的 4B Decode/GEMM 初筛

| 实验 | 配置 | 4K-8K output | 8K-16K output | 16K-32K output | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| `DEC4B-D0` | LLMM1 on、HipBLASLt off | 67.54 | 41.84 | 29.44 | 当前基线 |
| `DEC4B-D1` | LLMM1 off、HipBLASLt off | 68.49 | 40.83 | 29.71 | TPOT 约改善 2%，吞吐收益不稳定，输出变化 |
| `DEC4B-D2` | LLMM1 off、HipBLASLt on | 34.93 | 25.28 | 19.84 | 三档严重退化，淘汰 |

结论：

1. 不继续投入全局 HipBLASLt 切换。
2. 不采用 LLMM1 全局开/关作为最终方案；需转为按实际 GEMM 形状选择性 dispatch。
3. 当前 UA2D 参数路线已经显著改善长档，但不足以覆盖中短档和 decode。
4. 后续优先寻找不依赖低精度、不过度改变 reduction 顺序、且能同时改善中档和长档的路径。

### 2.2 旧 profile 提供的约束

旧 baseline profile 中：

| 类别 | 全局占比 | 判断 |
| --- | ---: | --- |
| GEMM/Linear | 约 `51.76%` | 中短档主热点 |
| Unified Attention | 约 `42.59%` | 长档主热点 |
| GDN/linear attention | 约 `2.14%` | 暂不作为首选 |
| Triton pointwise/reduction | 约 `2.75%` | 仅在新 profile 证明占比上升后投入 |
| KV cache 写入 | 约 `0.07%` | 暂不投入 |
| Sampling | 约 `0.01%` | 暂不投入 |

该 profile 早于最新 UA2D 优化，只能用于排除明显低价值方向，不能直接决定最终 kernel 投入。

## 3. 总体目标与执行原则

### 3.1 本轮目标

- 找到至少一个 4B 可重复候选，使 `8K-16K` 或 `16K-32K` output throughput 提升
  `>=3%`，且其它档位不明显回退。
- 优先保持输出逐元素或生成文本稳定；任何输出变化必须进入精度归因流程。
- 不修改原始比赛脚本、模型、权重、tokenizer、chat template、采样参数和 scheduler 语义。
- 4B 达到升级门槛后，先汇报并询问用户是否运行 27B。

### 3.2 路线优先级

```text
ALT-A：ROCM_ATTN 分离式后端 A/B
    ↓ 无稳定收益
ALT-B：UA3D decode block-table 标量化
    ↓ 收益不足
ALT-C：LLMM1 按 GEMM 形状选择性 dispatch
    ↓ 仍不足以覆盖长档
ALT-D：UA2D full-prefix/causal-diagonal 结构拆分
    ↓ 新 profile 确认后
ALT-E：小算子融合或 GDN buffer/launch 优化
```

每条路线必须独立实验、独立提交候选，避免多个主变量叠加后无法归因。

## 4. ALT-A：ROCM_ATTN 分离式后端 A/B

### 4.1 优化假设

当前日志确认 full attention 使用 `TRITON_ATTN`。ROCm 源码还提供 `ROCM_ATTN`，其 forward
路径使用 `chunked_prefill_paged_decode`，将 prefill 与 decode 分离，而不是使用 unified
attention kernel。

该后端：

- 支持 `head_size=256`。
- 支持 block size 为 16 的倍数，包括 Qwen3.5 4B 的 `528` 和 27B 的 `784`。
- 对非 16/32 block size 使用 Triton KV cache update，不依赖不兼容的原生 HIP cache kernel。
- 初筛可只改变启动配置，不先修改源码。

### 4.2 实验矩阵

| 实验 | Attention 配置 | 目的 |
| --- | --- | --- |
| `ALT-A0` | 当前 `TRITON_ATTN` | 使用最近 `DEC4B-D0` 作为基线，并在必要时复测 |
| `ALT-A1` | `use_prefill_decode_attention=true` | 触发 `ROCM_ATTN` 自动选择 |
| `ALT-A2` | 显式 backend=`ROCM_ATTN` | 仅在 A1 未按预期命中时用于诊断 |

启动测试副本或等价命令增加：

```bash
--attention-config '{"use_prefill_decode_attention":true}'
```

不得修改原始 `testdata/start_vllm.sh` 和 `testdata/run_throughput.sh`。

### 4.3 执行顺序

1. 检查 GPU、8001/8002 和队友进程。
2. 使用 8002、4B、`gpu-memory-utilization=0.45` 启动 A1。
3. 从日志确认：模型、block size、backend=`ROCM_ATTN`、源码导入路径。
4. 先跑 `16K-32K 10`，判断分离式 prefill 是否有价值。
5. 长档未明显回退时补跑 `8K-16K 10`、`4K-8K 10`。
6. 保存生成文本和 output length，与 A0 对比。

### 4.4 决策门槛

| 结果 | 决策 |
| --- | --- |
| `16K-32K >= +5%` 或 `8K-16K >= +3%`，其它档回退 `<2%` | 重复两轮，准备完整三档验证 |
| 收益 `<2%` | 不固化后端，进入 ALT-B |
| 启动失败、VM fault、完成率下降、输出异常 | 立即淘汰 |
| 日志未命中 `ROCM_ATTN` | 结果无效，只做 backend 选择诊断 |

若输出文本发生变化，即使吞吐达标，也只能作为 27B 精度候选，不能直接提交。

## 5. ALT-B：UA3D Decode Block-Table 标量化

### 5.1 优化假设

UA2D 已通过标量化 block-table lookup 减少每个 lane 重复执行的非 2 次幂除法、取模和
block-table load。高频 decode 使用的 `kernel_unified_attention_3d` 仍保留原始逐 lane
地址计算。

Qwen3.5 的 cache block `528/784` 均大于常用 decode tile，一个 tile 最多跨越两个物理
cache block，因此可复用已验证的 2D 地址计算方式。

### 5.2 实现范围

新增实验环境变量：

```text
VLLM_ROCM_QWEN_UA3D_SCALAR_BLOCK_TABLE
```

第一版要求：

- 默认关闭。
- 仅在 Qwen3.5 4B/27B 文本 full attention guard 命中时启用。
- 仅在 `block_size >= TILE_SIZE_DECODE` 时启用。
- 不修改 `NUM_PAR_SOFTMAX_SEGMENTS`、tile、softmax reduction 顺序和输出布局。
- 只替换物理 block index 与 cache offset 的计算方式。

### 5.3 测试矩阵

| 维度 | 测试值 |
| --- | --- |
| 模型形态 | 4B GQA=4、27B GQA=6 |
| block size | `528`、`784` |
| query length | `1` |
| sequence length | block 边界前后，以及约 `4K/8K/16K/32K` |
| scalar | off/on |
| 重复次数 | 同一输入至少 5 次 |

验收：

- scalar off/on 输出逐元素完全一致；若无法 bitwise 相同则暂停并定位。
- 无越界、NaN、Inf 和 VM fault。
- microbenchmark 记录 kernel latency、均值和标准差。

### 5.4 端到端门槛

| 指标 | 门槛 |
| --- | --- |
| P99 TPOT | 至少一个档位下降 `>=1.5%` |
| Output throughput | 加权收益可重复，任一档不回退超过 `1%` |
| 输出 | output length 和 generated text 与关闭组一致 |
| 完成率 | `100%` |

若收益小于 `1%`，保持默认关闭，不继续扩大实现复杂度。

## 6. ALT-C：LLMM1 按 GEMM 形状选择性 Dispatch

### 6.1 优化假设

全局关闭 LLMM1 后三档 TPOT 均改善约 `2%`，说明当前宽泛 gate 中可能同时包含：

- LLMM1 明显更快的形状。
- 与 F.linear 接近的形状。
- LLMM1 实际更慢或数值误差更敏感的形状。

需要从全局布尔开关改为 profile 驱动的形状级选择。

### 6.2 阶段 C0：形状采集

新增默认关闭的去重统计：

```text
VLLM_ROCM_GEMM_DEBUG_SHAPES=1
```

记录：

```text
(n, m, k, dtype, bias, selected_backend)
```

要求：

- 只记录去重形状及累计调用次数，不逐 token 输出日志。
- 不改变 dispatch 和模型输出。
- 分离 prefill 与 decode 的调用统计。

### 6.3 阶段 C1：算子级 microbenchmark

对累计调用时间最高的形状比较：

- LLMM1。
- `torch.nn.functional.linear`。
- 必要时当前 rocBLAS 默认路径。

每个形状记录：

- warmup 和正式迭代次数。
- 平均/P50/P99 latency。
- max absolute/relative error。
- 重复确定性。
- 是否改变 greedy decode 输出。

### 6.4 阶段 C2：选择性 allowlist

只允许同时满足以下条件的形状进入 LLMM1：

- micro latency 相对 F.linear 提升 `>=5%`。
- 数值误差满足现有 bf16 Linear 标准。
- 4B 端到端 TPOT/吞吐有稳定收益。
- 输出变化已经完成记录；涉及 27B 时必须通过精度评测。

不允许基于测试 prompt、token 内容或数据集类别 dispatch，只能基于通用张量形状、dtype 和
平台能力。

### 6.5 决策门槛

| 结果 | 决策 |
| --- | --- |
| 选择性 dispatch 使 TPOT 改善 `>=2%` 且吞吐不回退 | 保留候选 |
| 能恢复平台精度系数且吞吐损失 `<1%` | 即使纯性能收益不足，也作为精度候选升级 27B |
| 形状间收益不稳定或输出漂移无法解释 | 回退 F.linear/当前默认，不继续投入 |

## 7. ALT-D：UA2D Full-Prefix / Causal-Diagonal 拆分

### 7.1 进入条件

满足以下任一条件才进入 ALT-D：

- ALT-A 无收益，最新 profile 中 UA2D 在 `16K-32K` 仍占 GPU 时间 `40%+`。
- 现有 UA2D micro 显示 causal mask、地址计算或 tile 控制仍有明确优化空间。
- 预计能在当前排行榜版本上继续改善长档 `>=10%`。

### 7.2 结构

```text
context_len 之前的 full prefix
    - 所有 query 可见
    - 删除逐元素 causal compare/mask

当前 chunk 的 causal diagonal
    - 保留完整 causal 语义

共享 online-softmax M/L/acc 状态
```

### 7.3 实现约束

- 不改变 KV cache、block table 和输出布局。
- 不改变 chunked prefill scheduler 行为。
- 保留严格 Qwen3.5 文本 attention guard。
- 未命中时回退当前 generic/fast UA2D。
- 第一版不同时引入 current-chunk contiguous FlashAttention 和状态 merge，避免变量过多。

### 7.4 边界测试

- `context_len=0`。
- 中间完整 chunk。
- 最后一个非 4096 token chunk。
- prefix 落在 `528/784` block 边界前后。
- query length 小于、等于和大于一个 `BLOCK_Q`。
- 4B GQA=4、27B GQA=6。

### 7.5 决策门槛

| 指标 | 门槛 |
| --- | --- |
| `16K-32K` | 在当前版本上继续提升 `>=10%` |
| `8K-16K` | 不回退，争取 `>=5%` |
| `4K-8K` | 回退 `<1%` |
| 数值/精度 | 满足 bf16 attention 标准，无新增精度扣分 |

## 8. ALT-E：仅在新 Profile 命中后的次级方向

以下方向不立即实现，只在最新版本 profile 证明累计占比明显上升时进入：

| 候选 | 进入条件 | 说明 |
| --- | --- | --- |
| Q/K RMSNorm + RoPE 融合 | full-attention pointwise/launch 累计 `>=5%` | 16 层 full attention 可复用，但需严测数值 |
| RoPE + KV cache update 融合 | cache update/launch 累计 `>=5%` | 旧 profile KV 写入仅 0.07%，无新证据不投入 |
| GDN 临时 output/buffer 复用 | GDN/zero/allocation 累计 `>=5%` | 当前 GDN 总占比仅约 2.14% |
| RMSNorm/residual 小算子融合 | pointwise 与 launch gap 显著 | 现有 RMSNorm 已具备 residual 融合，先避免重复工作 |
| 单请求 Python metadata 快路径 | CPU trace 显示 GPU 等待 CPU | 不允许改变 scheduler 参数或语义 |

## 9. 明确暂缓或淘汰的方向

| 方向 | 状态 | 原因 |
| --- | --- | --- |
| 全局 HipBLASLt preference | 淘汰 | 4B 三档吞吐下降 `32%~48%`，TPOT 接近翻倍 |
| 普通 gfx936 `wvSplitK` | 淘汰 | 已出现数值问题和 VM fault 风险 |
| LLMM1 全局关闭 | 不固化 | TPOT 小幅改善，但吞吐不稳定且输出变化 |
| KV Cache/Activation 量化 | 暂缓 | 当前已有精度扣分，风险过高 |
| Sampling 优化 | 暂缓 | 旧 profile 占比约 0.01% |
| Memcpy 优化 | 暂缓 | 设备侧 copy 非主瓶颈 |
| GDN 大规模重写 | 暂缓 | 旧 profile 占比约 2.14% |
| max-model-len、scheduler 修改 | 禁止 | 评测参数和 scheduler 行为已锁定 |
| 测试集缓存、跳过请求或计算 | 禁止 | 违反比赛约束 |

## 10. 4B 到 27B 的门禁

任何路线升级 27B 前必须：

1. 本地静态检查通过。
2. 远端定向数值测试通过。
3. 4B 三档完成率 `100%`。
4. 至少一个高权重档位达到计划收益门槛。
5. 结果至少重复两轮，核心指标波动 `<3%`。
6. 无 VM fault、NaN、OOM、服务重启和输出异常。
7. 汇报 4B 结果并获得用户确认。

27B 第一轮优先运行少量样本，只有候选能够区分收益或精度差异时才运行完整三档与四类
OpenCompass。

## 11. 实验记录与提交规范

每次源码改动并完成测试后更新 `docs/progress.md`，记录：

- 实验编号和开始/结束时间。
- commit hash、工作区状态和环境变量。
- 远端同步文件及 SHA256。
- 模型、backend、block size、源码导入路径。
- 三档 throughput、TTFT/TPOT P99、完成率。
- output length/generated text 是否变化。
- GPU/端口和队友任务状态。
- 结论：保留、重复、升级 27B、回滚或淘汰。

排行榜提交门槛保持：

- 精度系数 `1.0`。
- SLA 扣分 `0`。
- 完成率 `100%`。
- 相对当前排行榜版本预计最终得分至少增加 `1.0` 分。
- 安全目标仍为 `4K-8K>=19`、`8K-16K>=18`、`16K-32K>=14 tok/s`，最终得分
  `>=86.5`。

## 12. 最近一轮具体执行清单

当前只执行 ALT-A，不同时修改源码：

1. 固定 `DEC4B-D0` 结果和当前测试文件状态。
2. 使用时间戳目录启动 4B `ROCM_ATTN` 服务到 8002。
3. 确认日志真正命中 `ROCM_ATTN`。
4. 先跑 `16K-32K 10`。
5. 长档通过门槛后补跑 `8K-16K 10`、`4K-8K 10`。
6. 对比吞吐、P99、output length 和 generated text。
7. 若 ALT-A 无收益，停止服务并进入 ALT-B 设计，不直接叠加其它变量。
8. 完成测试后更新 `docs/progress.md`。

本计划当前不授权直接启动 27B；任何 27B 测试仍需在 4B 结果汇报后单独确认。
