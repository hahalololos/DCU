# “添财程序员队”前 23 名优化冲刺计划

生成日期：2026-07-10

## 1. 背景与目标

最新排行榜中，“添财程序员队”排名第 54 名，当前结果为：

| 指标 | 当前结果 | 第 23 名结果 | 当前差距 |
| --- | ---: | ---: | ---: |
| 最终得分 | 72.5063 | 85.2009 | 12.6946 分 |
| 精度扣分 | 1.1042 | 0 | 需归零 |
| `4K-8K` 输出吞吐 | 15.50 | 18.76 | `+21.0%` |
| `8K-16K` 输出吞吐 | 13.40 | 16.82 | `+25.5%` |
| `16K-32K` 输出吞吐 | 8.01 | 13.69 | `+70.9%` |

按排行榜和官方计分公式反推，当前吞吐原始得分约为 `73.6105`，精度系数约为
`0.985`，最终被扣至 `72.5063`。即使吞吐追平当前第 23 名，若仍保留该精度系数，
最终得分也只有约 `83.92`，因此精度归零是进入决赛的必要条件。

考虑排行榜仍会变化，本计划不以刚好达到 `85.2009` 为终点，而以以下安全目标作为提交门槛：

| 指标 | 冲刺目标 |
| --- | ---: |
| 最终得分 | `>= 86.5`，争取 `87+` |
| 精度扣分 | `0` |
| SLA 扣分 | `0` |
| `4K-8K` 输出吞吐 | `>= 19.0 tok/s` |
| `8K-16K` 输出吞吐 | `>= 18.0 tok/s` |
| `16K-32K` 输出吞吐 | `>= 14.0 tok/s` |
| 完成率 | `100%` |

## 2. 总体判断

当前 fastpath 方向有效：相较早期排行榜结果，三个档位均有明显提升。但 4B 上观察到的
长档大幅收益尚未充分迁移到正式 27B，且最新提交出现精度扣分。因此下一阶段不能继续仅靠
`TILE/BLOCK_M/WARPS` 的盲目调参，应按以下顺序推进：

1. 修复并验证 27B fastpath 正确性，消除精度扣分。
2. 在正式 27B 形态上完成 fastpath 吞吐量、时延和命中率 A/B。
3. 对 UA2D prefill 做结构性优化，主攻 `16K-32K`。
4. 对 decode 阶段做 profile 驱动的 kernel/GEMM 优化，补齐 `4K-16K`。
5. 每个候选都经过 4B 筛选、27B 三档验证、精度和 SLA 门禁后才能提交。

## 3. 版本与实验管理

### 3.1 固定提交版本

当前工作区存在未提交改动，且 `HEAD` 与可能用于排行榜提交的源码不完全一致。后续每次实验必须：

1. 为候选版本建立独立 commit。
2. 在实验记录中写入完整 commit hash。
3. 记录全部相关环境变量。
4. 保存启动日志、bench 结果和精度结果目录。
5. 排行榜提交后记录提交时间、commit hash 和平台结果。

禁止使用无法还原具体源码和环境变量的“工作区临时版本”提交。

### 3.2 实验编号

| 前缀 | 含义 |
| --- | --- |
| `CORR-*` | 正确性和精度修复 |
| `UA27-*` | 27B UA2D fastpath 参数实验 |
| `UAST-*` | UA2D 结构性优化 |
| `DEC-*` | decode profile 与 kernel 优化 |
| `SUB-*` | 提交候选全量验证 |

## 4. 阶段 P0：27B 正确性与精度归零

### 4.1 风险分析

Qwen3.5-27B 的文本 full attention 形态为：

| 参数 | 值 |
| --- | ---: |
| query heads | 24 |
| KV heads | 4 |
| GQA ratio | 6 |
| head size | 256 |
| attention block size | 784 |

当 `BLOCK_M=16`、`GQA=6` 时，`BLOCK_Q=2`，每个 Q block 实际只有
`2 * 6 = 12` 个有效行。若未屏蔽 `offs_m=12..15`，这些填充行会映射到下一个
Q block 的 token/head，并可能与相邻 Triton program 对同一输出地址产生写竞争。

当前工作区中的 `valid_block_row` mask 是必须保留并验证的正确性修复。若最新榜单提交未包含
该修复，应优先完成修复验证，不再基于旧提交继续叠加性能改动。

### 4.2 正确性测试矩阵

新增或扩展 27B 形态的 attention 对照测试，比较 generic UA2D 与 fastpath 输出：

| 维度 | 测试值 |
| --- | --- |
| GQA | `6` |
| `BLOCK_M` | `16`、`32` |
| query length | `1`、`2`、`3`、`15`、`16`、`17`、`4095`、`4096` |
| context/block 边界 | `783`、`784`、`785`、`1567`、`1568`、`1569` |
| fastpath | off / on |
| 重复次数 | 同一输入至少 5 次 |

验收内容：

- 无越界、VM fault、NaN、Inf。
- fastpath 与 generic 输出误差满足现有 bf16 attention 路径标准。
- 同一输入重复执行结果稳定，无竞争导致的随机输出变化。
- 最后一个不完整 Q block 的所有有效 token/head 均正确写回。

### 4.3 端到端精度门禁

执行顺序：

1. 4B 三档 smoke，确认通用路径没有回退。
2. 27B fastpath-off 精度 baseline。
3. 27B fastpath-on 完整四类 OpenCompass。
4. 对比问答、摘要、检索、聚合四类任务的相对精度变化。

P0 完成条件：

| 指标 | 门槛 |
| --- | --- |
| OpenCompass 精度系数 | 四类均进入无扣分区间 |
| 平台预期精度扣分 | `0` |
| 完成率 | `100%` |
| 输出稳定性 | 重复运行无随机漂移 |

未达到以上条件时，不进入新的激进性能优化，不提交排行榜。

## 5. 阶段 P1：27B UA2D fastpath 参数筛选

### 5.1 前置条件

- P0 正确性测试通过。
- 远端模型存储读取恢复正常，27B 服务能够稳定启动。
- 启动日志确认模型、后端、block size 和源码导入路径正确。
- GPU 和 8001 端口空闲，不影响队友服务。

### 5.2 参数矩阵

第一轮在 `16K-32K 10` 上筛选：

| 实验 | Fastpath | TILE | BLOCK_M | WARPS | STAGES | Scalar block table |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `UA27-F0` | off | 默认 | 默认 | 默认 | 默认 | off |
| `UA27-F1` | on | 64 | 16 | 4 | 1 | off |
| `UA27-F2` | on | 64 | 16 | 8 | 1 | off |
| `UA27-F3` | on | 64 | 32 | 4 | 1 | off |
| `UA27-F4` | on | 64 | 32 | 8 | 1 | off |
| `UA27-F5` | on | 32 | 16 | 4 | 3 | off |

`BLOCK_M=32` 同样存在 GQA 不整除产生的填充行，必须在 P0 的 mask 和数值测试通过后才能启用。

### 5.3 Scalar block-table 独立实验

当前 27B attention block size 为 `784`，每个 UA2D tile 只有 `32/64` 个 token。
scalar block-table 路径试图减少每个 lane 中非 2 次幂整除、取模及重复 block-table load。

该优化必须作为独立变量测试：

| 实验 | 基础参数 | Scalar block table |
| --- | --- | --- |
| `UA27-S0` | P1 最优参数 | off |
| `UA27-S1` | P1 最优参数 | on |

只有在三次重复实验均有稳定收益、数值对照通过时才允许默认开启。若收益小于 `3%`，保持关闭，
避免增加边界风险和维护复杂度。

### 5.4 筛选规则

每组记录：

- `qwen_candidate`、`qwen_fastpath`、`block_size`。
- output throughput。
- TTFT P50/P95/P99。
- TPOT P50/P95/P99。
- 完成率、异常和显存峰值。
- 三次重复测试的均值与标准差。

决策门槛：

| 结果 | 决策 |
| --- | --- |
| `16K-32K` 相对 F0 提升 `>= 20%` | 最优两组补跑其他档位 |
| 最优两组差距 `< 3%` | 选择资源占用更小、正确性更保守的配置 |
| P99 超过 F0 的 `1.5x` | 淘汰 |
| 完成率下降或出现输出异常 | 立即淘汰 |
| 27B 日志未确认 fastpath 命中 | 结果无效，先修 guard/部署 |

最优两组随后补跑 `8K-16K 10`、`4K-8K 10`；最终候选跑三档完整测试。

## 6. 阶段 P2：UA2D prefill 结构性优化

若 P1 完成后 `16K-32K` 仍低于 `12 tok/s`，或 profile 中
`kernel_unified_attention_2d` 仍是长档主热点，应停止继续做零散常量调优，进入结构性优化。

### 6.1 UAST-1：full prefix 与 causal diagonal 拆分

当前 UA2D 对每个 Q block 扫描完整 prefix，并在所有 tile 上执行通用 causal mask。
对 chunked prefill 可将 key 范围拆分为：

1. `context_len` 之前的 full prefix：当前 Q block 的所有 query 均可见。
2. 当前 chunk 内的 diagonal 区间：需要保留 causal mask。

实施要点：

- full prefix 路径删除逐元素 causal 比较和 mask。
- diagonal 路径保留完整因果语义。
- 两段共享在线 softmax 状态 `M/L/acc`。
- 不改变 block table、KV cache 和输出布局。

边界测试：

- 首个 chunk：`context_len=0`。
- 中间完整 chunk。
- 最后一个非 4096 token chunk。
- prefix 长度落在 `784` block 边界前后。
- query length 小于一个 `BLOCK_Q`。

验收门槛：

| 指标 | 目标 |
| --- | --- |
| `16K-32K` | 在 P1 最优版本上继续提升 `>= 10%` |
| `8K-16K` | 不回退，争取提升 `>= 5%` |
| 数值 | 与 P1 版本一致或在 bf16 可接受误差内 |
| 精度/SLA | 无新增扣分 |

### 6.2 UAST-2：连续 current chunk prefill + 状态合并

若 UAST-1 后长档仍不能接近目标，进一步将当前 chunk 从 paged KV 路径拆出：

1. prefix：使用 paged KV attention，输出 prefix softmax state。
2. current chunk：使用 contiguous Q/K/V context/flash prefill kernel。
3. merge：按照 online softmax 规则合并 prefix 和 current chunk 的输出状态。

可参考：

- `vllm/v1/attention/ops/prefix_prefill.py`
- `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`
- `vllm/v1/attention/ops/triton_merge_attn_states.py`

这是高收益、高风险候选。必须保留严格 Qwen3.5 文本 attention guard 和 generic UA2D 回退路径。

## 7. 阶段 P3：decode 全档优化

### 7.1 目标

`4K-8K` 主要受 GEMM/Linear 和 decode 固定开销影响；每条请求固定生成 1024 token，
因此 decode 优化可以同时改善三个档位，并补足仅优化 prefill 无法解决的短档差距。

P3 目标：

| 档位 | 目标提升 |
| --- | ---: |
| `4K-8K` | `>= 15%` |
| `8K-16K` | `>= 10%` |
| `16K-32K` | 不回退 |

### 7.2 分离 profile

在 P1/P2 最优版本上分别采集：

- 纯 prefill/首 token 时间线。
- 稳态 decode 100 个以上 token 的时间线。
- kernel 调用次数、总时间、平均时间、P99。
- GPU 利用率、带宽、kernel launch 间隙和同步。

按累计时间列出 decode Top 10，重点分类：

1. full-attention paged decode。
2. GDN/linear-attention state update。
3. QKV、O projection、MLP 小 batch GEMM。
4. RMSNorm、residual、RoPE 等高频小 kernel。
5. sampling、同步和内存复制。

### 7.3 候选选择规则

- 不再继续投入已证明端到端无收益的普通 skinny GEMM 路径。
- 只优化 profile 中累计占比靠前、可在三个档位复用的 kernel。
- 优先选择不改变数值语义的融合、layout、launch 和访存优化。
- 每次只引入一个主变量，避免无法归因。

候选优先级：

| 优先级 | 候选 | 适用条件 |
| --- | --- | --- |
| D1 | full-attention paged decode kernel 调优 | 长上下文 KV 读取占比高 |
| D2 | GDN state update/fallback 消除 | 48 层 linear attention 累计耗时高 |
| D3 | QKV/O/MLP 小 batch GEMM 专用 dispatch | GEMM 累计占比高且存在低效形态 |
| D4 | RMSNorm/residual 等小算子融合 | launch 间隙和小 kernel 占比高 |
| D5 | memcpy/同步路径削减 | profile 显示设备空洞明显 |

## 8. 暂缓方向

当前阶段暂不优先：

| 方向 | 原因 |
| --- | --- |
| KV Cache 量化 | 当前已有精度扣分，风险过高；先完成无损 kernel 优化 |
| Activation 动态量化 | 需要较完整精度回归，开发周期与风险较高 |
| 继续扩大 skinny GEMM 覆盖 | 已有端到端测试未见收益，且 gfx936 曾有数值/VM fault 风险 |
| scheduler 行为修改 | 评测口径明确锁定，合规风险高 |
| 修改 max model len 等锁定参数 | 违反比赛约束 |
| tokenizer/chat template/采样修改 | 改变精度和推理语义，禁止 |

## 9. 提交候选门禁

任何版本提交排行榜前必须满足：

### 9.1 基础门禁

- 本地 `python -m py_compile` 或对应静态检查通过。
- `git diff --check` 通过。
- 远端 Python、vLLM、editable location 均来自项目 `.venv` 和源码树。
- 4B smoke 通过。
- 27B 三档请求完成率 `100%`。
- 无 VM fault、OOM、NaN、服务重启或请求跳过。

### 9.2 性能门禁

- 每档至少重复三次，核心吞吐波动建议 `< 3%`。
- TTFT P99 和全局 TPOT P99 均满足 SLA。
- 加权收益不是由单次异常快结果产生。
- 相对当前排行榜版本，预计最终得分至少提升 `1.0` 分才提交，减少无效提交次数。

### 9.3 精度门禁

- 四类 OpenCompass 均完成。
- 精度系数为 `1.0`，预计平台精度扣分为 `0`。
- fastpath off/on 的输出差异已完成定位和记录。

### 9.4 决赛线门禁

最终冲刺版本至少达到以下组合之一：

| 方案 | `4K-8K` | `8K-16K` | `16K-32K` | 预计结果 |
| --- | ---: | ---: | ---: | --- |
| 均衡目标 | 19.0 | 18.0 | 14.0 | 约 86.5 分，具备安全边际 |
| 中档强化 | 18.0 | 18.5 | 13.0 | 约 86.1 分，仍需关注排行榜变化 |
| 长档强化 | 17.5 | 18.0 | 14.0 | 约 85.9 分，仍需关注排行榜变化 |

优先追求均衡目标，避免只优化单一档位。根据评分函数，当前每增加 `1 tok/s`，
`8K-16K` 和 `16K-32K` 的边际分值明显高于 `4K-8K`，但单独提升任一档都不足以跨越
12.7 分的总差距。

## 10. 推荐执行顺序

| 顺序 | 工作 | 产出/决策 |
| ---: | --- | --- |
| 1 | 固定榜上提交版本和当前候选版本 | 可复现 commit、环境变量和结果目录 |
| 2 | `CORR-1` GQA 填充行单测 | 确认无覆盖写和随机输出 |
| 3 | `CORR-2` 4B smoke + 27B 精度 | 精度扣分归零，否则继续修复 |
| 4 | `UA27-F0~F5` 参数筛选 | 得到 27B 最优 fastpath 参数 |
| 5 | `UA27-S0/S1` scalar block-table A/B | 收益稳定才启用 |
| 6 | 三档完整复测 | 判断是否已达到提交增益门槛 |
| 7 | `UAST-1` prefix/diagonal 拆分 | 主攻 `16K-32K` |
| 8 | decode 分离 profile | 确定 Top 10 kernel 和下一优化点 |
| 9 | `DEC-D1~D5` 中选择一个主候选 | 补齐 `4K-16K` |
| 10 | `SUB-1` 全量吞吐、SLA、精度验证 | 达到 86.5+ 后提交 |

## 11. 实验记录模板

每次源码改动并完成测试后，将简要结果追加到 `docs/progress.md`：

```markdown
## YYYY-MM-DD <实验编号> <标题>

- commit：`<完整 hash>`
- 改动：<核心源码和 guard>
- 环境变量：<fastpath/tile/block/warps/stages 等>
- 安装方式：editable / full wheel build
- 4B smoke：<通过/失败，关键指标>
- 27B 4K-8K：output throughput / TTFT P99 / TPOT P99 / 完成率
- 27B 8K-16K：output throughput / TTFT P99 / TPOT P99 / 完成率
- 27B 16K-32K：output throughput / TTFT P99 / TPOT P99 / 完成率
- 精度：问答 / 摘要 / 检索 / 聚合
- profile：Top kernels、调用次数、总时间
- GPU/端口：是否影响队友服务
- 结论：保留 / 继续调参 / 回滚 / 提交
```

## 12. 回滚策略

| 改动 | 回滚方式 |
| --- | --- |
| Qwen UA2D fastpath | `VLLM_ROCM_QWEN_UA2D_FASTPATH=0` |
| scalar block-table | `VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE=0` |
| 参数实验 | 恢复 P0 已验证的保守参数 |
| 结构性 UA2D | guard 未命中时回退 generic UA2D |
| decode 专用 kernel | dispatch 回退原 vLLM/PyTorch 路径 |

任一版本出现精度扣分、完成率下降、VM fault、无法稳定复现或 P99 SLA 风险时，立即回退到
最近一个完整通过精度和三档测试的 commit，不在有疑问的版本上继续叠加改动。
