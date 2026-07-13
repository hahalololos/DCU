# hipprof 结果分析

## 1. 数据范围

本次分析基于 `results/hipprof/hipprof_output_20260705_134354`：

- `vllm_profile.hipkernel.csv`：GPU kernel 聚合。
- `vllm_profile.hiptrace.csv`：HIP API 聚合。
- `vllm_profile.db`：按时间线拆分三段主测负载。
- `results/baseline/baseline_throughtput.txt`：本地 baseline 日志。

用户提交测评平台 baseline 成绩为：

| 长度档 | 平台实际吞吐率 |
|---|---:|
| 4k-8k | 12.94 |
| 8k-16k | 10.06 |
| 16k-32k | 5.77 |

本地日志里的 throughput 为 12.19、8.79、5.38，和平台结果同趋势但绝对值偏低。后续优化判断应以平台三档权重为准：`4k-8k 20%`、`8k-16k 50%`、`16k-32k 30%`。

## 2. 全局 GPU Kernel 热点

排除 CSV 末尾 `Total` 汇总行后，GPU kernel 总时长约 `5002.814s`。

| 类别 | 调用次数 | 总时长(s) | 占比 | 结论 |
|---|---:|---:|---:|---|
| rocBLAS GEMM / Linear | 13,424,066 | 2589.50 | 51.76% | 全局第一大类，尤其影响 4k-8k 和 8k-16k |
| unified attention | 583,488 | 2130.72 | 42.59% | 长上下文主要瓶颈，尤其影响 16k-32k |
| Triton fused pointwise/reduction | 16,451,055 | 137.75 | 2.75% | 有调用密度，但时长占比低 |
| GDN/linear attention 相关 | 4,419,349 | 107.24 | 2.14% | Qwen3.5 的 48 个 GDN 层不是当前主瓶颈 |
| torch native | 974,685 | 29.31 | 0.59% | 非核心 |
| KV cache 写入/清零 | 584,691 | 3.62 | 0.07% | KV 写入不是主瓶颈 |
| sampling | 36,455 | 0.75 | 0.01% | 采样不可作为第一轮优化目标 |

Top kernel：

| Kernel | 调用次数 | 总时长(s) | 平均耗时(ms) | 占比 |
|---|---:|---:|---:|---:|
| `kernel_unified_attention_2d` | 9,936 | 1947.11 | 195.96 | 38.92% |
| `Cijk_...MT64x32x32...` GEMM | 4,014,304 | 1566.36 | 0.390 | 31.31% |
| `Cijk_...MT32x16x4...` GEMM | 4,625,137 | 524.97 | 0.114 | 10.49% |
| `kernel_unified_attention_3d` | 573,552 | 183.62 | 0.320 | 3.67% |
| `Cijk_...MT128x32x32...` GEMM | 573,472 | 129.52 | 0.226 | 2.59% |

## 3. 三档主测分段

从 DB 的 `BeginNs/EndNs` 找大空档后，profile 中可识别出三段主负载。其 wall time 与 4k-8k、8k-16k、16k-32k 主测时长递增关系一致，以下以 `like` 标注表示由时间线推断。

| 分段 | wall(s) | kernel(s) | 第一热点 | 第二热点 | 主要结论 |
|---|---:|---:|---|---|---|
| 4k-8k-like | 1223.19 | 1087.09 | GEMM 78.74% | attention 14.29% | 短档主要是 Linear/GEMM 吞吐问题 |
| 8k-16k-like | 1689.04 | 1640.40 | GEMM 58.87% | attention 35.58% | 评分权重最高，必须同时优化 GEMM 和 attention |
| 16k-32k-like | 2269.65 | 2249.37 | attention 61.87% | GEMM 33.99% | 长档瓶颈明显转向 prefill attention |

更细的 attention 拆分：

| 分段 | `kernel_unified_attention_2d` | `kernel_unified_attention_3d` | 解读 |
|---|---:|---:|---|
| 4k-8k-like | 120.52s / 1,664 calls | 34.86s / 204,176 calls | prefill attention 仍较小，decode attention 更小 |
| 8k-16k-like | 517.81s / 3,088 calls | 65.92s / 218,384 calls | prefill attention 已成为核心瓶颈之一 |
| 16k-32k-like | 1308.76s / 4,992 calls | 82.84s / 150,880 calls | prefill attention 压倒性主导 |

源码对应关系：

- `vllm/v1/attention/ops/triton_unified_attention.py` 中 `unified_attention()` 在 `max_seqlen_q > 1` 时会走 `kernel_unified_attention_2d`。
- decode 单 token 场景更可能走 `kernel_unified_attention_3d`。
- 启动脚本固定 `--max-num-batched-tokens 4096`，长 prompt 会被切成多个 prefill chunk，因此 `kernel_unified_attention_2d` 随上下文长度迅速放大。

## 4. HIP API 与 Copy 结论

全局 `hiptrace.csv` 中：

| HIP API | 调用次数 | 总时长(s) | 占比 |
|---|---:|---:|---:|
| `hipEventSynchronize` | 73,144 | 4612.86 | 42.38% |
| `hipMemcpyAsync` | 834,587 | 4588.43 | 42.15% |
| `hipMemcpyWithStream` | 4,309 | 1450.15 | 13.32% |
| `hipGraphLaunch` | 36,685 | 177.68 | 1.63% |

但 DB 中 HIPCOPY 设备侧 copy 在三段主测内只有 2-4 秒量级：

| 分段 | copy calls | copy duration(s) | copy bytes |
|---|---:|---:|---:|
| 4k-8k-like | 301,312 | 3.52 | 200.58 GB |
| 8k-16k-like | 328,402 | 2.52 | 416.21 GB |
| 16k-32k-like | 240,262 | 2.38 | 672.15 GB |

因此，HIP API 层的 `hipMemcpyAsync` 和 `hipEventSynchronize` 大占比更像同步等待、异步 API 计时、profiler accounting 或启动/模型加载阶段影响；它不是主测阶段的真实设备 copy 瓶颈。第一轮不应把“减少 memcpy”当作主线，除非能在时间线里定位到具体 decode/prefill 阻塞点。

## 5. 对瓶颈的判断

1. 16k-32k 慢，主要慢在 prefill attention。  
   `kernel_unified_attention_2d` 在 16k-32k-like 分段占单段总 kernel 时长 `58.18%`，是最明确的 P0。

2. 8k-16k 是综合战场。  
   该档评分权重 50%，profile 中 GEMM `58.87%`、attention `35.58%`。只优化 attention 可能拉长档明显，但 8k-16k 收益未必最大；只优化 GEMM 又会错过 16k-32k。

3. 4k-8k 主要由 GEMM/Linear 主导。  
   该档 GEMM 占 `78.74%`，attention 仅 `14.29%`。如果后续改 attention 后 4k-8k 不涨，这是符合 profile 的。

4. Qwen3.5 的 GDN/linear attention 层不是本轮第一热点。  
   虽然模型有 48 个 linear attention/GDN 层，但 profile 中 GDN/causal conv/chunk 相关总占比只有 `2.14%`。

5. KV cache 写入、sampling、RMSNorm/pointwise 不是第一轮收益区。  
   KV 写入只有 `0.07%`，sampling `0.01%`。这些方向可以做尾部优化，但不应先投入。

## 6. 第一轮优化建议

| 优先级 | 方向 | 目标指标 | 依据 | 风险 |
|---|---|---|---|---|
| P0 | 优化/替换 `kernel_unified_attention_2d` prefill path | 16k-32k throughput、TTFT P99；兼顾 8k-16k | 长档单段 58.18% kernel time 来自 2D unified attention | kernel 正确性和边界条件风险高 |
| P0 | 评估 ROCm/AITER unified attention 默认启用或后端选择 | 8k-16k、16k-32k throughput | 当前源码默认 `VLLM_ROCM_USE_AITER=False`，profile 走 vLLM Triton unified attention | 依赖平台是否安装 AITER；需 fallback |
| P0/P1 | GEMM/Linear 后端优化，优先不改权重的 unquantized AITER/Triton GEMM | 4k-8k、8k-16k throughput、TPOT P99 | 全局 GEMM 51.76%，4k-8k 单段 78.74% | 后端兼容性、数值一致性 |
| P1 | KV cache FP8/低精度读取实验 | 16k-32k attention、显存 | attention 读 KV 是长上下文核心成本，赛题允许 KV cache 量化 | 精度风险，需要 OpenCompass 全量验证 |
| P1 | decode 小 batch 线性层/pointwise 融合与 CUDA/HIP graph 路径检查 | TPOT P99 | TPOT 稳定在 70ms 左右，GEMM 调用密度极高 | 收益可能小于 prefill attention |
| P2 | GDN recurrent/chunk kernel 微调 | TPOT、少量 throughput | 总占比 2.14% | 低收益 |
| P2 | KV block 管理、KV 写入、sampling | 尾部延迟 | profile 占比极低 | 不适合作为第一轮主线 |

## 7. 下一步实验清单

1. 在比赛卡上确认日志中的实际 attention backend。  
   重点看是否为 `TRITON_ATTN` / `ROCM_ATTN` / `ROCM_AITER_UNIFIED_ATTN`，以及是否能在不改启动脚本的情况下通过源码默认值切换。

2. 单独跑 8k-16k 与 16k-32k 的短样本 profile。  
   当前 profile 是长时间混合捕获，已足够判断主热点，但做 kernel 改动后需要更短、更快的 A/B。

3. 第一批实现建议按以下顺序：
   - 先做 attention backend/`kernel_unified_attention_2d` 的可逆实验。
   - 再做 GEMM backend 选择或 unquantized AITER/Triton GEMM 实验。
   - 最后再试 KV cache FP8，必须绑定精度回归。

4. 每次实验至少记录：
   - 三档 throughput。
   - TTFT P99、TPOT P99。
   - OpenCompass 四类精度。
   - `hipkernel.csv` 中 `kernel_unified_attention_2d`、GEMM 总时长是否下降。

