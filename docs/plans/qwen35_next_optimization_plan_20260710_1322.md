# Qwen3.5 下一步优化计划（2026-07-10 13:22 CST）

## 1. 计划元信息

| 项目 | 内容 |
| --- | --- |
| 计划版本 | `20260710-1322` |
| 生成时间 | `2026-07-10 13:22:16 CST (+0800)` |
| 排行榜快照 | `docs/rankings/ranking.html`，队伍最后提交时间 `2026-07-10 11:20:32` |
| 当前提交源码 | `vllm_cscc` commit `d3dc2e69f8830788fe48d56840868579e450d287` |
| 当前排名/得分 | 第 `51` 名，最终得分 `76.6248` |
| 当前三档吞吐 | `4K-8K=15.84`、`8K-16K=14.22`、`16K-32K=10.05 tok/s` |
| 当前扣分 | SLA `0`，精度扣分 `1.1669` |

为避免与此前计划混淆，后续新增计划文件、实验目录和结果记录必须包含时间戳。统一采用：

```text
YYYYMMDD_HHMM
```

实验编号建议采用：

```text
<阶段>-<编号>_YYYYMMDD_HHMM
```

例如：`CORR27-A1_20260710_1530`、`DEC4B-D1_20260710_1700`。

## 2. 最新结果判断

相对上一次排行榜提交，最新结果变化为：

| 指标 | 上次 | 最新 | 变化 |
| --- | ---: | ---: | ---: |
| 最终得分 | 72.5063 | 76.6248 | `+4.1185` |
| `4K-8K` | 15.50 | 15.84 | `+2.2%` |
| `8K-16K` | 13.40 | 14.22 | `+6.1%` |
| `16K-32K` | 8.01 | 10.05 | `+25.5%` |

结论：`BLOCK_M=32 + scalar block-table` 路线已经在正式排行榜上体现出明显收益，主要改善
中长上下文 prefill，但对短档和 decode 的帮助有限。

最新吞吐原始分约为：

```text
76.6248 + 1.1669 = 77.7917
```

精度系数仍约为 `0.985`，与上一次提交基本相同。精度扣分金额上升是因为吞吐原始分提高，
不能据此判断最新改动进一步恶化了精度。`0.985` 可能对应一个任务落入 `k=0.94`，或两个
任务落入 `k=0.97`，必须通过分变量对照实验定位。

当前第 23 名分数为 `85.5287`。即使立即恢复精度系数 `1.0`，当前预计得分也只有
`77.7917`，因此下一阶段必须同时解决精度和性能问题。

## 3. 总体优先级

下一步按以下顺序执行：

1. 补齐正式默认 UA2D 配置的数值与确定性测试。
2. 使用 4B 对 decode/GEMM 低风险候选进行筛选。
3. 4B 达到升级门槛后，先向用户汇报并询问是否启动 27B 验证。
4. 使用 27B 分离定位精度扣分来源。
5. 对最新 27B 版本重新进行 prefill/decode profile。
6. 根据 profile 在 decode/GEMM 与 UA2D 结构性优化之间选择主线。
7. 完成三档、SLA、完成率和四类精度门禁后再提交排行榜。

本轮不继续盲目扫描 `TILE/BLOCK_M/WARPS/STAGES`，也暂不引入 KV Cache 量化或激活动态
量化。

## 4. 阶段 P0：正式默认 UA2D 正确性补齐

### 4.1 目的

当前源码默认值是：

```text
FASTPATH=true
TILE=64
BLOCK_M=32
NUM_WARPS=4
NUM_STAGES=0（TILE=64 时实际选择 1）
SCALAR_BLOCK_TABLE=true
```

现有 27B GQA 回归测试主要覆盖 `BLOCK_M=16 + scalar=false`，与正式提交默认配置不完全
一致。本阶段只补测试和定位问题，不新增性能逻辑。

### 4.2 测试矩阵

统一使用 27B attention 实际形态：

```text
query_heads=24
kv_heads=4
GQA=6
head_size=256
block_size=784
dtype=bf16
```

| 实验 | `BLOCK_M` | Scalar block table | 目的 |
| --- | ---: | --- | --- |
| `CORR-K0` | 16 | off | 保守对照 |
| `CORR-K1` | 16 | on | 单独验证标量 block-table |
| `CORR-K2` | 32 | off | 单独验证 `BLOCK_M=32` |
| `CORR-K3` | 32 | on | 正式默认配置 |

边界至少覆盖：

- query length：`1、2、3、5、6、15、16、17、31、32、4095、4096`。
- KV/block 边界：`783、784、785、1567、1568、1569`。
- 最后一个不完整 Q block。
- tile 未跨 KV block、实际跨入第二个 KV block、第二个 block 仅有少量有效 token。
- 同一输入重复执行至少 5 次。

### 4.3 验收门槛

- 与 PyTorch/reference paged attention 在现有 bf16 容差内一致。
- 无 NaN、Inf、越界和 VM fault。
- 重复运行结果稳定，不出现写竞争导致的随机漂移。
- `CORR-K3` 必须通过，否则不得继续使用当前默认配置做新优化。

## 5. 阶段 P1：4B decode/GEMM 低风险筛选

### 5.1 实验矩阵

一次只改变一个主变量：

| 实验 | Skinny/LLMM1 | HipBLASLt preference | 目的 |
| --- | --- | --- | --- |
| `DEC4B-D0` | on | 默认/off | 当前提交配置基线 |
| `DEC4B-D1` | off | 默认/off | 判断 LLMM1 是否实际有收益 |
| `DEC4B-D2` | off | on | 相对 D1 单独判断 HipBLASLt |

环境变量：

```bash
VLLM_ROCM_USE_SKINNY_GEMM=0/1
TORCH_BLAS_PREFER_HIPBLASLT=0/1
```

### 5.2 执行顺序

1. 同步本地源码到远端。
2. 激活项目 `.venv` 并加载 `env.sh`。
3. 检查 GPU、8001/8002 端口和队友进程。
4. 先跑 `4K-8K 10`，观察 decode/GEMM 主导档位。
5. 再跑 `8K-16K 10`。
6. 候选未回退时补跑 `16K-32K 10`。
7. 最优候选至少重复两轮；准备升级 27B 前，关键结果重复三轮。

### 5.3 记录内容

- output throughput。
- TTFT P50/P95/P99。
- TPOT P50/P95/P99。
- 完成率和异常。
- GPU 是否空闲、是否与队友任务重叠。
- 若可采集，记录 GEMM kernel 名称、调用次数和累计时间。

### 5.4 4B 决策门槛

| 结果 | 决策 |
| --- | --- |
| `4K-8K` 提升 `>=3%` 且其他档不回退 | 保留并准备 27B 验证 |
| 收益 `<2%` 或波动无法复现 | 不固化默认值 |
| TPOT/TTFT 明显恶化、数值异常或 VM fault | 立即回滚 |
| 所有低风险候选无收益 | 直接进入最新版本 profile，不继续扩展 skinny GEMM |

4B 完成后必须先汇报结果，并询问用户是否进行 27B 验证；未经确认不直接占用 27B 测试资源。

## 6. 阶段 P2：27B 精度扣分归因

### 6.1 前置条件

- P0 正式默认数值测试通过。
- P1 4B smoke 和候选筛选完成。
- 用户确认进行 27B 验证。
- 27B 模型存储读取正常，GPU 和 8001 端口空闲。

### 6.2 第一轮四因素对照

| 实验 | UA2D fastpath | Skinny/LLMM1 | 目的 |
| --- | --- | --- | --- |
| `CORR27-A0` | off | off | 最保守精度基线 |
| `CORR27-A1` | on | off | 判断 UA2D 对精度的影响 |
| `CORR27-A2` | off | on | 判断 LLMM1 对精度的影响 |
| `CORR27-A3` | on | on | 当前排行榜提交配置 |

第一轮先对问答、摘要、检索、聚合各跑少量样本。只对能够区分差异的实验组运行完整四类
OpenCompass，避免无意义地重复加载 27B。

### 6.3 UA2D 子矩阵

若精度差异指向 UA2D，再比较：

| 实验 | `BLOCK_M` | Scalar | 目的 |
| --- | ---: | --- | --- |
| `CORR27-U0` | 16 | off | 保守 UA2D |
| `CORR27-U1` | 32 | off | 定位 reduction/launch 变化 |
| `CORR27-U2` | 32 | on | 定位 scalar block-table |

### 6.4 完成条件

- 明确精度扣分来自 UA2D、LLMM1、两者组合或其它路径。
- 四类任务均完成完整评测。
- 最终候选精度系数为 `1.0`。
- 若高性能配置无法达到 `1.0`，优先选择无扣分的保守配置，不以吞吐收益交换精度系数。

## 7. 阶段 P3：最新 27B prefill/decode 重新画像

旧 profile 采集于 UA2D 大幅优化之前，不能直接作为下一轮代码投入依据。本阶段使用 P2 选出的
正确候选重新采集：

- 纯 prefill/首 token 时间线。
- 稳态 decode 至少 100 个 token 的时间线。
- kernel 调用次数、累计时间、平均时间和尾部时间。
- GPU 利用率、显存带宽、kernel launch 间隙、同步与 memcpy。

重点输出：

1. `4K-8K` decode Top 10。
2. `8K-16K` prefill 与 decode 时间占比。
3. `16K-32K` UA2D、GEMM 和 decode attention 时间占比。
4. QKV、O projection、MLP 的主要 GEMM 形状及当前 dispatch。
5. full-attention paged decode、GDN state update、RMSNorm/residual 的累计占比。

## 8. 阶段 P4：依据 profile 选择性能主线

### 8.1 评分边际参考

按当前吞吐位置和官方评分函数估算，每增加 `1 tok/s` 的原始分收益约为：

| 档位 | 预计分值收益 |
| --- | ---: |
| `4K-8K` | `+0.57` |
| `8K-16K` | `+1.42` |
| `16K-32K` | `+0.92` |

因此 `8K-16K` 是分值杠杆最高的档位，不能只追求长档 microbenchmark 数字。

### 8.2 主线 A：decode/GEMM

满足以下任一条件时优先选择 decode/GEMM：

- `4K-8K` 中 GEMM/Linear 累计占比最高。
- `8K-16K` decode 占比超过 prefill。
- UA2D 优化后 GPU 时间已经转移到 QKV/O/MLP 或高频小 kernel。

候选顺序：

1. 固化 P1 中经过验证的 LLMM1/HipBLASLt 选择。
2. 针对 Top GEMM 形状做 microbenchmark 和专用 dispatch。
3. 调优 full-attention paged decode kernel。
4. profile 显示 launch 开销显著时，再做 RMSNorm/residual 等小算子融合。

禁止继续扩大数值不稳定的普通 `wvSplitK` 覆盖。

### 8.3 主线 B：UA2D 结构性优化

满足以下条件时实施 full-prefix/causal-diagonal 拆分：

- `16K-32K` 中 `kernel_unified_attention_2d` 仍占 GPU 时间 `40%+`；或
- 预计在当前版本上仍能取得 `>=10%` 的长档端到端收益。

结构：

```text
full prefix attention
        +
causal diagonal attention
        +
共享 online-softmax M/L/acc 状态
```

必须保留严格 Qwen3.5 guard 和 generic UA2D 回退路径，并覆盖 context/block/chunk 边界。

## 9. 阶段 P5：提交门禁

任何新排行榜提交必须同时满足：

### 9.1 正确性与稳定性

- 4B smoke 通过。
- 27B 三档完成率 `100%`。
- 无 VM fault、OOM、NaN、服务重启和请求跳过。
- 四类 OpenCompass 完整结束，精度系数 `1.0`。

### 9.2 性能与 SLA

- 每档至少重复三次，核心吞吐波动建议 `<3%`。
- 各档 TTFT P99 和全局 TPOT P99 满足 SLA。
- 相对当前排行榜版本预计最终得分至少提升 `1.0` 分。
- 收益不能依赖共享 GPU 干扰、热服务偶然状态或异常快的单轮结果。

### 9.3 决赛线目标

| 指标 | 安全目标 |
| --- | ---: |
| 最终得分 | `>=86.5`，争取 `87+` |
| 精度扣分 | `0` |
| SLA 扣分 | `0` |
| `4K-8K` | `>=19.0 tok/s` |
| `8K-16K` | `>=18.0 tok/s` |
| `16K-32K` | `>=14.0 tok/s` |
| 完成率 | `100%` |

## 10. 最近一轮具体执行清单

按顺序执行：

1. 新增 `CORR-K0~K3` 测试，重点补齐正式默认 `BLOCK_M=32 + scalar=true`。
2. 本地运行静态检查和 `git diff --check`。
3. 同步到远端，检查 `.venv`、源码导入路径、GPU 和端口。
4. 运行 UA2D 定向数值测试。
5. 启动 4B，运行 `DEC4B-D0~D2`。
6. 汇总三档吞吐、TTFT/TPOT、完成率和波动。
7. 更新 `docs/progress.md`，记录已实际完成且测试过的源码改动及结果。
8. 向用户汇报 4B 结果，并询问是否升级到 27B 精度和 profile 验证。

当前立即开始的工作包仅包含第 1 至第 6 项，不直接启动 27B。
