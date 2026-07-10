# hipprof HTML trace 分析

## 1. HTML 的定位

`results/hipprof/html/vllm_profile_*.html` 是 Chrome trace / DTK trace viewer 页面。它适合看时间线关系：

- Runtime API 轨道：`hipEventSynchronize`、`hipMemcpyAsync`、`hipGraphLaunch` 等。
- Memory Copy 轨道：真实设备 copy / memset。
- Compute 轨道：GPU kernel，例如 `kernel_unified_attention_2d`、`kernel_unified_attention_3d`、`Cijk_...` GEMM。

它不适合作为全局统计主来源。原因是 HTML 被拆成 157 个大文件，总量约 36.5GB，逐事件全量解析很重；全局占比仍应以 `hipkernel.csv` 和 `vllm_profile.db` 为主，HTML 用来验证“热点发生在时间线哪里、前后连着什么”。

## 2. 三档测试对应的 HTML 文件

以 HTML trace 的 `ts` 秒为坐标，结合 DB 里识别出的大空档，三段主测边界约为：

| 区间 | HTML `ts` 起点 | HTML `ts` 终点 | 对应 HTML 文件 |
|---|---:|---:|---|
| 4k-8k 主测 | 2008.698 | 3231.889 | 约 `vllm_profile_12.html` 到 `vllm_profile_59.html` |
| 8k-16k 主测 | 3262.621 | 4951.659 | 约 `vllm_profile_59.html` 到 `vllm_profile_112.html` |
| 16k-32k 主测 | 4985.558 | 7255.204 | 约 `vllm_profile_112.html` 到 `vllm_profile_157.html` |

边界文件说明：

- `vllm_profile_12.html`：跨过 4k-8k 的开始点。
- `vllm_profile_59.html`：同时包含 4k-8k 结束和 8k-16k 开始附近的事件。
- `vllm_profile_112.html`：同时包含 8k-16k 结束和 16k-32k 开始附近的事件。
- `vllm_profile_157.html`：包含 16k-32k 末尾和收尾事件。

## 3. 代表文件观察

### 4k-8k：GEMM 主导

代表文件 `vllm_profile_16.html`，`ts=2186.8-2211.1`：

| 指标 | 数值 |
|---|---:|
| Compute 总时长 | 23.84s |
| GEMM 占比 | 约 69.3% |
| unified attention 占比 | 约 23.5% |
| `kernel_unified_attention_2d` | 5.07s / 96 calls |
| `kernel_unified_attention_3d` | 0.54s / 3520 calls |

同一档里很多文件没有 `kernel_unified_attention_2d`，例如 `vllm_profile_21.html`、`vllm_profile_41.html`、`vllm_profile_46.html`，这些窗口几乎就是 GEMM 密集 decode。HTML 直观看到 4k-8k 的常态是大量 `Cijk_...` GEMM 小条块铺满 Compute 轨道。

### 8k-16k：GEMM + prefill attention 混合

代表文件 `vllm_profile_80.html`，`ts=3939.0-3968.9`：

| 指标 | 数值 |
|---|---:|
| Compute 总时长 | 29.32s |
| GEMM | 18.63s，约 63.5% |
| unified attention | 9.07s，约 30.9% |
| `kernel_unified_attention_2d` | 7.58s / 48 calls |
| `kernel_unified_attention_3d` | 1.49s / 4272 calls |
| Memory Copy 轨道 | 0.047s |

代表文件 `vllm_profile_106.html`，`ts=4747.0-4790.0`：

| 指标 | 数值 |
|---|---:|
| Compute 总时长 | 42.59s |
| `kernel_unified_attention_2d` | 24.68s / 176 calls |
| unified attention 总占比 | 约 59.5% |
| GEMM 占比 | 约 34.8% |

这说明 8k-16k 内部不是单一形态：decode 密集窗口仍然 GEMM 主导，但遇到长 prompt prefill chunk 时，`kernel_unified_attention_2d` 会突然变成最大长条。

### 16k-32k：`kernel_unified_attention_2d` 主导

代表文件 `vllm_profile_120.html`，`ts=5457.1-5531.6`：

| 指标 | 数值 |
|---|---:|
| Compute 总时长 | 74.15s |
| `kernel_unified_attention_2d` | 56.78s / 213 calls |
| unified attention 总占比 | 约 77.7% |
| GEMM 占比 | 约 18.5% |
| Memory Copy 轨道 | 0.061s |

代表文件 `vllm_profile_130.html`，`ts=5920.7-5969.3`：

| 指标 | 数值 |
|---|---:|
| Compute 总时长 | 48.01s |
| `kernel_unified_attention_2d` | 26.38s / 96 calls |
| unified attention 总占比 | 约 59.1% |
| GEMM 占比 | 约 36.8% |

代表文件 `vllm_profile_157.html`，`ts=7221.4-7296.0`：

| 指标 | 数值 |
|---|---:|
| Compute 总时长 | 33.57s |
| `kernel_unified_attention_2d` | 22.92s / 96 calls |
| unified attention 总占比 | 约 70.4% |
| GEMM 占比 | 约 25.5% |

HTML 中这些文件的 Compute 轨道会出现明显的大块 `kernel_unified_attention_2d` 长条；这就是 16k-32k throughput 掉到最低的主要视觉证据。

## 4. Runtime API 轨道如何解读

HTML 中 Runtime API 轨道常见两个大项：

- `hipEventSynchronize`
- `hipMemcpyAsync`

例如：

| 文件 | Runtime 现象 | Compute 现象 | Copy 轨道 |
|---|---|---|---:|
| `vllm_profile_80.html` | `hipEventSynchronize` 27.04s，`hipMemcpyAsync` 26.87s | Compute 29.32s，UA2D + GEMM 混合 | 0.047s |
| `vllm_profile_120.html` | `hipEventSynchronize` 68.10s，`hipMemcpyAsync` 67.93s | Compute 74.15s，UA2D 主导 | 0.061s |
| `vllm_profile_130.html` | `hipEventSynchronize` 45.39s，`hipMemcpyAsync` 45.23s | Compute 48.01s，UA2D 主导 | 0.052s |

这说明 Runtime API 上的长同步不是独立 CPU 计算瓶颈，也不是实际 copy 很重。它多数是在等待下方 Compute 轨道上的长 kernel 完成。真正设备侧 Memory Copy 轨道在这些代表文件里只有几十毫秒量级。

HTML 里的长同步与长 kernel 对齐很明显：

- `vllm_profile_120.html` 中 `hipEventSynchronize` 可见 8.77s、6.97s、5.16s 级别长条。
- 同一文件 Compute 轨道中 `kernel_unified_attention_2d` 总计 56.78s，是最主要被等待对象。

因此，不能看到 `hipEventSynchronize` / `hipMemcpyAsync` 占 Runtime 时间很高就先做 memcpy 优化；应先看 Compute 轨道。

## 5. HTML 进一步确认的结论

1. `kernel_unified_attention_2d` 是长上下文 prefill/chunked prefill 的核心瓶颈。  
   它调用次数不如 GEMM 多，但单次耗时大。在 16k-32k 的代表 HTML 中，它经常占 Compute 总时间的 60%-78%。

2. `kernel_unified_attention_3d` 更像 decode attention。  
   它调用次数很高，例如几千次，但总时长远低于 `2d`。这说明当前最慢的不是单 token decode attention，而是长 prompt prefill attention。

3. GEMM 仍是 4k-8k 和一部分 8k-16k 的主线。  
   没有 `kernel_unified_attention_2d` 的窗口里，Compute 轨道几乎由 `Cijk_...` GEMM 填满。

4. Memory Copy 轨道不是主测阶段主瓶颈。  
   HTML 代表窗口中 copy 总时长通常只有 0.03-0.06s，而 Compute 是几十秒。

5. Runtime API 轨道的长等待是“症状”，不是第一优化靶点。  
   要减少 `hipEventSynchronize` 的长等待，根本上要缩短下方 Compute 轨道的 attention/GEMM kernel。

## 6. 后续看 HTML 的推荐方式

优先打开这些文件：

- 看 4k-8k GEMM：`vllm_profile_16.html`、`vllm_profile_21.html`、`vllm_profile_41.html`
- 看 8k-16k 混合形态：`vllm_profile_80.html`、`vllm_profile_91.html`、`vllm_profile_106.html`
- 看 16k-32k attention：`vllm_profile_120.html`、`vllm_profile_130.html`、`vllm_profile_157.html`

在页面里搜索：

- `kernel_unified_attention_2d`
- `kernel_unified_attention_3d`
- `Cijk_`
- `hipEventSynchronize`

看法顺序：

1. 先定位 Compute 轨道的大块长条。
2. 再向上看 Runtime API 是否有对应长同步等待。
3. 最后看 Memory Copy 轨道是否真的有长 copy。当前 baseline 里，答案基本是否定的。

