# Qwen3.5-4B 当前版本性能 Profile 报告

时间：2026-07-11

源码：`vllm_cscc` 提交 `39c465438d157d10445de2871098cb195d5ae8cf`

## 结论

当前最值得投入的优化点是 Qwen3.5 专用 `kernel_unified_attention_2d`，不是继续扩大
LLMM1 shape allowlist。

理由：

- UA2D 在 4–8K、8–16K、16–32K prefill 的纯 GPU kernel 时间占比分别为
  `28.4%/37.9%/58.1%`，上下文越长占比越高。
- gfx936 计数器显示 UA2D 的 L2 命中约 `97.3%`，HBM 读取带宽仅约 `20 GB/s`，但单次
  使用 `224 VGPR`、`32 KB LDS`，并出现约 `1.577e9` 的 LDS bank-conflict 计数；瓶颈是
  kernel 内部寄存器/LDS/矩阵计算和数据布局，而不是 HBM 带宽。
- 线上最新版本唯一明显回退的是 16–32K，UA2D 正好是该档最大的单一热点。
- Decode 虽然主要由 GEMM 占据，但当前 LLMM1 和 hipBLAS/Tensile 路径已经是经过筛选的
  正收益组合；此前盲目恢复 S1 形状已产生端到端负收益。

## 基线复核

关闭 profiler、当前默认 UA2D/shape filter 开启，三档各 10 条：

| 档位 | 完成 | output tok/s | P99 TPOT | 备注 |
| --- | ---: | ---: | ---: | --- |
| 4–8K | 10/10 | 70.52 | 11.88 ms | 相对历史热基线约 -3.4% |
| 8–16K hot2 | 10/10 | 50.43 | 12.49 ms | 与历史热基线一致 |
| 16–32K | 10/10 | 30.49 | 13.24 ms | 与历史热基线一致 |

实验目录：`testdata/experiments/PROFILE4B-BASE_20260711_1800`。

## Torch Profiler 方法

- 三档分别选择可自然生成超过 64 token 的第 7 条样本。
- 对同一输入分别记录 `max_tokens=1` 和 `max_tokens=64`。
- Prefill 使用 hot2 trace；稳定 decode 用 `decode64 - prefill1` 得到约 63 token 的净设备时间。
- 只统计 Chrome trace 中 `cat=kernel` 的 GPU 活动，排除外层 annotation 的重复累计。
- Trace 开启 shape 记录，关闭 stack 和 memory 记录。

Trace 目录：`testdata/experiments/PROFILE4B-TORCH_20260711_1829/traces`，共 7 份，约
`9–50 MB/份`。

## Prefill 热点

| 档位 | 纯 kernel 时间 | UA2D 时间 | UA2D 占比 |
| --- | ---: | ---: | ---: |
| 4–8K | 355.23 ms | 100.93 ms | 28.4% |
| 8–16K | 647.34 ms | 245.11 ms | 37.9% |
| 16–32K | 2162.42 ms | 1255.78 ms | 58.1% |

其余 prefill 时间主要由大 batch GEMM 和 GDN prefill kernels 组成。UA2D 是三个档位共同的
最大单一 kernel，也是唯一随上下文长度快速扩大占比的热点。

若 UA2D 本身加速 10%，按 Amdahl 上限估算，三档纯 kernel 时间可分别改善约
`2.7%/3.6%/5.6%`；加速 20% 时约为 `5.0%/6.7%/10.7%`。

## 稳态 Decode 热点

稳定 decode 的纯 kernel 时间约为：

| 档位 | ms/token |
| --- | ---: |
| 4–8K | 10.77 |
| 8–16K | 11.10 |
| 16–32K | 12.25 |

前三类热点：

| 热点 | 4–8K | 8–16K | 16–32K | 特征 |
| --- | ---: | ---: | ---: | --- |
| 当前 LLMM1 | 28.9% | 28.1% | 25.3% | 约 3.11 ms/token，49 次/token |
| 其他三类 GEMM kernel 合计 | 46.0% | 44.6% | 40.3% | MLP、QKV、O projection |
| UA3D full-attention decode | 6.3% | 9.1% | 18.1% | 随 KV 长度增长 |

LLMM1 中，LM head `(248320,2560)` 单次约 `1.266 ms/token`；其余 GDN qkvz/b-a 共约
`1.85 ms/token`。这说明 LLMM1 确实是 decode 热点，但不是恢复已过滤形状的理由；应只调优
当前已有正收益的三个允许形状。

## rocprof 硬件计数器

微基准与原始结果位于：

- `testdata/profile_hotspots_4b.py`
- `testdata/rocprof_hotspots_metrics.txt`
- `testdata/experiments/PROFILE4B-ROCPROF_20260711_1915`

| Kernel | 未注入计数器延迟 | VGPR | LDS | L2 hit | Fetch | 主要判断 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| UA2D，q=4096/kv=22258 | 51.6 ms | 224 | 32 KB | 97.3% | 约 1021 MiB | LDS/寄存器/计算受限 |
| UA3D，q=1/kv=22258 | 0.299 ms | 248 | 16.5 KB | 1.0% | 约 87 MiB | KV 读取和低命中受限 |
| LLMM1 LM head | 1.278 ms | 28 | 1 KB | 3.6% | 约 1213 MiB | 接近 HBM 带宽受限 |

补充计数器：

- UA2D：`SQ_WAVES=8208`、`SQ_LDS_BANK_CONFLICT=1,576,828,928`。
- UA3D：`SQ_WAVES=256`、`SQ_LDS_BANK_CONFLICT=3,430,400`。
- LM head：`SQ_WAVES=310400`、`SQ_LDS_BANK_CONFLICT=1,055,360`。

因此 LM head 再做纯计算调优空间有限；若不量化权重，主要只能优化加载/调度细节。当前已有
精度扣分，不建议优先进入权重量化。

## 推荐实施顺序

1. UA2D 单 kernel 内部 full-prefix/causal-diagonal 分段：保持一个 kernel 和同一 softmax state，
   对完全可见的 prefix tiles 移除逐元素 causal mask，仅对最后的 diagonal/partial tiles执行
   通用 mask。不要恢复已淘汰的多 kernel + state merge 路线。
2. 同时用 rocprof 观察 LDS bank conflict、VGPR 和延迟，目标不是继续盲扫常量，而是减少
   `acc[BLOCK_M,HEAD_SIZE]` 周边的 LDS/寄存器压力及冲突。
3. UA2D 获得端到端正收益后，再扫描 UA3D 的 segment 数和 KV load 布局；它是修复长上下文
   TPOT 的第二优先级。
4. Decode GEMM 只调优当前允许的 LLMM1 形状；不继续 S2/S3，除非针对精确形状的 micro 和
   端到端都证明正收益。

## 不建议方向

- 继续按模块逐个恢复 LLMM1 shape filter 中的形状。
- 重新实现多 kernel prefix/suffix + state merge；此前 micro 已证明严重退化。
- 优先优化 LM head 算术路径；它已表现为高带宽、低 VGPR 的典型流式 GEMV。
- 通过修改 scheduler、上下文长度、采样或输出行为获得表面吞吐提升。
