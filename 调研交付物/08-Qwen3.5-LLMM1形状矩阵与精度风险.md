# Qwen3.5 LLMM1 形状矩阵与精度风险

## 结论

当前 gfx936 的 `rocm_unquantized_gemm_impl()` 会让所有满足 `n=1`、`k<=8192`、无 bias、
`m%4==0` 的 BF16 Linear 进入 `LLMM1`。该 gate 覆盖范围很广，不只是 attention；它覆盖
GDN 输入/输出投影、full-attention QKV/O、MLP gate-up 和 LM head。

4B 全局关闭 LLMM1 后，三档 TPOT 均改善约 2%，且生成文本发生变化。结合 LLMM1 kernel
内部使用 reduced-type `__hmul2/__hfma2` 累加，这一路径既可能更慢，也可能是排行榜精度扣分
的来源之一。后续必须按实际 `(m,n,k)` 做算子速度和误差对照，不能继续使用宽泛 gate。

## 4B decode 形状（TP=1）

模型配置：hidden `2560`、intermediate `9216`、32 层，其中 24 层 linear attention、
8 层 full attention。

| 位置 | `(m,n,k)` | 每 token 次数 | 当前 LLMM1 |
| --- | --- | ---: | --- |
| GDN qkvz | `(12288,1,2560)` | 24 | 是 |
| GDN b/a | `(64,1,2560)` | 24 | 是 |
| GDN out | `(2560,1,4096)` | 24 | 是 |
| Full-attention qkv+gate | `(10240,1,2560)` | 8 | 是 |
| Full-attention o | `(2560,1,4096)` | 8 | 是 |
| MLP gate-up | `(18432,1,2560)` | 32 | 是 |
| MLP down | `(2560,1,9216)` | 32 | 否，`k>8192` |
| LM head | `(248320,1,2560)` | 1 | 是 |

## 27B decode 形状（TP=1）

模型配置：hidden `5120`、intermediate `17408`、64 层，其中 48 层 linear attention、
16 层 full attention。

| 位置 | `(m,n,k)` | 每 token 次数 | 当前 LLMM1 |
| --- | --- | ---: | --- |
| GDN qkvz | `(16384,1,5120)` | 48 | 是 |
| GDN b/a | `(96,1,5120)` | 48 | 是 |
| GDN out | `(5120,1,6144)` | 48 | 是 |
| Full-attention qkv+gate | `(14336,1,5120)` | 16 | 是 |
| Full-attention o | `(5120,1,6144)` | 16 | 是 |
| MLP gate-up | `(34816,1,5120)` | 64 | 是 |
| MLP down | `(5120,1,17408)` | 64 | 否，`k>8192` |
| LM head | `(248320,1,5120)` | 1 | 是 |

## 下一步实验

共享存储恢复后，在同一个常驻 Python 进程内逐形状比较 `LLMM1` 和 `F.linear`：

1. 每个 backend 先 warmup 5 次，再交替运行 4 轮，每轮 10 次；大词表头每轮 3 次。
2. 记录 median/P99 latency、max/P99 absolute error、max/P99 relative error和重复确定性。
3. 仅保留速度提升至少 5%、误差满足 BF16 Linear 标准的形状。
4. 若所有形状均无收益，gfx936 默认禁用 LLMM1；若禁用能恢复 27B 精度且吞吐损失小于
   1%，优先作为提交候选。

## 2026-07-10 算子实测

`ALT-C-MICRO2_20260710_1530` 使用 gfx936、BF16、`n=1`，交替测量 LLMM1 和
`F.linear`。正数表示 LLMM1 更快，负数表示应回退 `F.linear`。

| 模型 | 形状 `(m,k)` | 位置 | LLMM1 相对收益 |
| --- | --- | --- | ---: |
| 4B | `(12288,2560)` | GDN qkvz | `+17.15%` |
| 4B | `(64,2560)` | GDN b/a | `+26.53%` |
| 4B | `(2560,4096)` | GDN/full-attention out | `-8.99%` |
| 4B | `(10240,2560)` | Full-attention qkv+gate | `-23.92%` |
| 4B | `(18432,2560)` | MLP gate-up | `-19.42%` |
| 4B | `(248320,2560)` | LM head | `+38.71%` |
| 27B | `(16384,5120)` | GDN qkvz | `+35.91%` |
| 27B | `(96,5120)` | GDN b/a | `+39.85%` |
| 27B | `(5120,6144)` | GDN/full-attention out | `-23.51%` |
| 27B | `(14336,5120)` | Full-attention qkv+gate | `+51.31%` |
| 27B | `(34816,5120)` | MLP gate-up | `+38.12%` |
| 27B | `(248320,5120)` | LM head | `-24.32%` |

所有形状均非 bitwise equal。max absolute diff 为 `0.001953125~0.0078125`，P99
relative diff 约 `0.07~0.18`。因此选择性 dispatch 可以提升速度，但仍需通过生成文本和
27B 精度门禁。

基于逐层调用次数粗算，相对当前宽泛 LLMM1 gate，形状过滤预计每个 decode token 可减少：

- 4B 约 `0.86 ms`。
- 27B 约 `1.71 ms`。

当前候选只在 gfx936 上过滤，保留上述实测收益 `>=5%` 的形状；其它 gfx9 平台维持原行为，
并提供环境变量回退到旧的宽泛 gate。

## 2026-07-10 4B 端到端首轮

`ALT-C1_20260710_1545` 在 gfx936 上显式开启形状过滤，服务保持 `TRITON_ATTN`、
block size 528、单并发和原锁定 batch 参数。三档各 10 条均成功：

| 档位 | output tok/s | P99 TTFT | P99 TPOT | 相对旧 D0 TPOT |
| --- | ---: | ---: | ---: | ---: |
| 4K-8K | `72.90` | `531.94 ms` | `11.95 ms` | `-8.59%` |
| 8K-16K | `37.53` | `10287.63 ms` | `12.89 ms` | `-5.42%` |
| 16K-32K | `30.44` | `2283.19 ms` | `13.29 ms` | `-7.90%` |

三档输入 token 总数均与旧 D0 一致。输出 token 总数分别为 `2571/1522/1114`，旧 D0 为
`2617/1706/1114`；逐样本生成文本完全一致数分别为 `6/10、7/10、9/10`。因此 TPOT
显示出稳定的 `5%~9%` 改善，但 output throughput 会受到生成长度变化影响，中档首轮不能
作为独立性能证据。

下一门禁是过滤关闭的同口径新鲜 baseline，并比较完成率、输入/输出 token、逐样本文本、
TTFT 和 TPOT。只有 A/B 重复后收益稳定，且正式精度评测无回退，才考虑将过滤默认开启。

### 新鲜 baseline 对照

`ALT-C0_20260710_1605` 关闭过滤、其余参数与候选相同。三档均 10/10：

| 档位 | baseline output | 候选 output | baseline P99 TPOT | 候选 P99 TPOT | TPOT 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4K-8K | `66.01` | `72.90` | `13.19 ms` | `11.95 ms` | `-9.35%` |
| 8K-16K | `40.62` | `37.53` | `13.78 ms` | `12.89 ms` | `-6.45%` |
| 16K-32K | `29.36` | `30.44` | `14.52 ms` | `13.29 ms` | `-8.47%` |

短档和长档 output throughput 分别提升 `10.44%`、`3.67%`。中档候选生成 token 数从
`1706` 降到 `1522`，output throughput 下降 `7.61%`；但包含输入 token 的 total
throughput 仍提升 `3.42%`，且 TPOT 改善 `6.45%`。由于正式评分使用 output
throughput，中档输出变化既是性能不确定项，也是精度风险，不能仅凭 TPOT 固化候选。

已基于官方 `run_accuracy.sh` 的实验副本适配 4B 的端口、served model 名和 tokenizer
路径，保持四类数据集、OpenCompass 配置和评分逻辑不变。四类各 1 条烟测通过，正在运行
过滤关闭的完整 4B 精度 baseline；完成候选对照后再决定是否升级 27B。
