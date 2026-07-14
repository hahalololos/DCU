# Qwen3.5-27B 最新源码快速性能画像

时间：2026-07-14 14:20 CST
源码：`vllm_cscc@a5cbd489d07bd4497a1ab949f96d3b7471b2b176`（UA2D 实验 8、GEMV V4）
结论：端到端的绝对首瓶颈是 GEMM，尤其 Decode GEMV 与仍由 Tensile 执行的 MLP down / LM head；长上下文 Prefill 的次级首瓶颈是 GEMM + UA2D。CPU、调度空隙、稳态存储和当前单并发 KV 容量均不是首要吞吐限制。

## 成绩与目标

榜单快照时间为 2026-07-14 11:10。添财程序员队当前第 48 名，三档吞吐（16--32K / 4--8K / 8--16K）为 `12.51 / 19.19 / 17.07 tok/s`，总分 `85.0408`，SLA 与精度扣分均为 0。第 20 名为 `87.6316`，相差 `2.5908` 分。

这轮 profile 的目的是为下一次源码优化定优先级，并非以 profiler 注入后的耗时替代比赛吞吐结论。

## 方法与有效性

- 本地 Git SHA 与远端关键源码 SHA256 已核验一致；远端实际为 editable `vllm_cscc`，GPU 为 BW/gfx936。
- 27B 模型使用 `/root/models/Qwen3.5-27B` 本地副本；三档热态吞吐各 10 请求，均 `10/10` 成功。
- 8--16K 与 16--32K 各选 P50 请求，使用完全相同输入做 `output=1` 和 `output=65` 的配对 trace；输入相同，65-token 结果以 1-token 结果为前缀。
- ROCprof 只用于资源、缓存和访存分类。计数器采集会重放应用，不能用其 kernel 时间判断快慢。
- 4--8K 没有重新采集配对 trace；下文的跨三档加权份额对短档采用历史 trace 结合本轮 UA2D E8/E5 micro 加速比的**估算**。中、长档为当前源码实测。

## 端到端结果

| 输入档 | output tok/s | req/s | P99 TTFT | P99 TPOT | 输入 / 输出 token |
|---|---:|---:|---:|---:|---:|
| 4--8K | 18.5903 | 0.07220 | 1.979 s | 47.657 ms | 62,196 / 2,575 |
| 8--16K | 14.0314 | 0.07638 | 7.811 s | 48.528 ms | 134,349 / 1,837 |
| 16--32K | 9.5440 | 0.07829 | 7.136 s | 49.619 ms | 212,553 / 1,219 |

中档输出 token 数和此前服务 A/B 的样本不完全相同，不能仅以 output tok/s 横向断言回归；但 47.7--49.6 ms 的 P99 TPOT 与配对 Decode trace 一致地表明 Decode 仍主导短、中上下文。

按端到端 wall 时间拆分，4--8K 为 Prefill `11.49%` / Decode `88.51%`，8--16K 为 `32.43%` / `67.57%`，16--32K 为 `52.63%` / `47.37%`。上下文变长后 Prefill 迅速成为同等重要的优化面。

## Trace 热点

### Prefill

| 档位 | kernel 总时长 | GPU gap | GEMM | UA2D | GDN |
|---|---:|---:|---:|---:|---:|
| 8--16K P50 | 3840.60 ms | 38.61 ms（0.995%） | 59.52% | 26.13% | 8.94% |
| 16--32K P50 | 6614.48 ms | 60.81 ms（0.911%） | 52.70% | 34.58% | 7.93% |

Prefill 不是 CPU launch gap 主导：gap 约 1%。长上下文 UA2D 占比从 26.13% 升至 34.58%，但 GEMM 仍为最大单类时间。

### Decode

| 档位 | kernel/token | gap/token | GEMM | UA3D 主核 | GDN |
|---|---:|---:|---:|---:|---:|
| 8--16K P50 | 47.7499 ms | 1.6321 ms | 43.2151 ms（90.50%） | 2.0035 ms | 0.8490 ms |
| 16--32K P50 | 48.8252 ms | 1.6571 ms | 43.2749 ms（88.63%） | 2.9897 ms | 0.8485 ms |

Decode 的两个大核组为：

- `qwen35_gemv_wave2_lds_kernel`：`27.241 ms/token`、`128` 次/token；
- Tensile / `F.linear` 主核：`14.60--14.61 ms/token`、`129` 次/token。

其余主要项远小于上述两类：UA3D `2.00--2.99 ms/token`、GDN `0.85 ms/token`、旧 LLMM1 小形状约 `0.39 ms/token`。GPU launch gap 只有 `1.63--1.66 ms/token`，即使完全消除也只提供约 2.5% 的加权端到端理论上限。

## 专用 GEMV 与未覆盖 Linear

三随机种子下，V4 相对旧 LLMM1 均 bitwise 一致，并已获得：

| 形状 | V4 / LLMM1 | 相对旧路径每 token 节省 |
|---|---:|---:|
| GDN qkvz `(16384,1,5120)` | 约 `1.246x` | `1.66 ms` |
| Attention qkv+gate `(14336,1,5120)` | 约 `1.253x` | `0.50 ms` |
| MLP gate-up `(34816,1,5120)` | 约 `1.226x` | `4.26 ms` |

V4 的 3 个形状均使用 256 线程、24 VGPR、10,752 B LDS，且 LDS bank conflict 为 0；L2 hit 仅 `14.66%--15.51%`。按只读 BF16 权重字节估算，三形状已经分别达到约 `1.196 / 1.192 / 1.208 TB/s` 的有效权重带宽。因此 V4 继续优化应以权重流式读取、load/compute pipeline 和 wave/occupancy 平衡为中心；激活向量复用本身空间很小，已经失败的四行复用 V5 不应重复。

仍由 `F.linear` 执行的 Decode 大项是：MLP down `(5120,1,17408)` 约 `9.60 ms/token`、LM head `(248320,1,5120)` 约 `1.90 ms/token`、Attention out 约 `0.80 ms/token`，合计约 `12.3 ms/token`。此前 MLP down 和 LM head 的简单专用原型已回退；后续应探索不同并行/分块结构，而不是复用旧 LLMM1 组织。

## UA2D、显存与系统路径

UA2D E8 的 8K / 16K / 32K micro 分别为 `1.137x / 1.143x / 1.149x`，且与 E5 bitwise 一致。ROCprof 中 E8 主核（V3）与 E5 参考核均达到 256 VGPR、16 KiB LDS；E8 消除了 scratch（364 B -> 0），原始 FETCH 计数下降 `20.16%`、WRITE 下降 `75.02%`、L2 hit 增加 `6.84` 个百分点，但 LDS bank conflict 上升 `47.00%`。这说明 E8 的物理块遍历优化已经有效，但下一步可在不改变 tile 遍历和 online-softmax 顺序的前提下，定向压低 LDS bank conflict 与 VGPR live range。

- UA3D 专用路径在 8K / 16K / 32K micro 为 `0.2925 / 0.2884 / 0.3044 ms`，且与通用路径 bitwise 一致；服务 trace 的累计时间虽随上下文上升，优先级仍低于 GEMM 和 UA2D。
- 启动到 health 约 170 秒，其中权重读取 `9.86 s`、模型加载 `10.98 s`、engine profile/KV/warmup `67.89 s`、图捕获 `8 s`。本地模型盘不是稳态吞吐瓶颈。
- 可用 KV 为 `6.75 GiB`（`27,440 tokens`），32768 token 的理论最大并发约 `3.11x`。稳态显存约 `61.3--62.7 GiB`；当前测例单并发，KV 是容量/稳定性风险而非第一性能限制。
- `vmstat` 显示 CPU idle 约 88%、iowait 约 0%；基准活跃时 GPU 常见 96--100% 利用率。CPU、HTTP/tokenizer、块存储没有饱和证据。

## 加权判断与下一步

按比赛三档权重、当前中长档 trace 与短档修正估算：Decode GEMM 约占端到端 `57.53%`，Prefill GEMM `19.36%`，UA2D `9.96%`，GDN `4.03%`，UA3D `2.61%`，GPU gap `2.51%`；全部 GEMM 合计约 `76.9%`。

以 Amdahl 上限估计，Decode GEMM 再快 `1.1x` 可使端到端约 `+5.23%`，再快 `1.2x` 约 `+9.59%`；Prefill GEMM `1.1x` 约 `+1.76%`；UA2D `1.2x` 约 `+1.66%`。这是优化方向筛选，不是提交得分承诺。

建议按以下顺序推进：

1. **P0 Decode GEMM**：继续 V4 的权重带宽/流水线优化，并为 MLP down、LM head 建立全新结构的 micro 门禁；先以每 token 节省与 bitwise 正确性筛选，再进行服务 A/B。
2. **P1 Prefill GEMM**：针对三类大 Tensile GEMM 进行长上下文专用的分块与访存实验；16--32K 中其仍占 `52.70%` 的 kernel 时间。
3. **P2 UA2D**：在保持 E8 语义和计算顺序下，测量并减少 LDS bank conflict / VGPR 压力；不可回退到已淘汰的 E5 遍历。
4. **P3 UA3D**：只做低成本 KV load、分段合并或 launch 合并探索；门槛应高于其约 2--3% 的端到端份额。
5. 暂缓 GPU 调度 gap、GDN、小 memcpy、稳态存储路径与单并发 KV 扩容，它们的可获益上限明显低于前三项。

## 产物

- 精简原始结果（不含大型 trace）：`testdata/profile_results/qwen35_27b_latest_20260714/`。
- 硬件计数器汇总：`rocprof/summary.json`；汇总器已支持新 `ua2d_e8.csv` / `gemv_v4.csv` 和 `qwen35_gemv_wave2_lds_kernel` 命名。
- 配对 trace 汇总：`summaries/`；吞吐结果：`baseline/`；micro：`micro/`。
