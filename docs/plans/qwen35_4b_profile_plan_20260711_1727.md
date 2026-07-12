# Qwen3.5-4B 性能 Profile 执行计划（2026-07-11 17:27）

## 目标

停止按 GEMM 形状盲目扩展优化。基于当前已推送的 `vllm_cscc` 版本
`39c465438d157d10445de2871098cb195d5ae8cf`，分离 4B 的 prefill 与稳定 decode，
定位三档吞吐量中累计耗时最高、可实现收益上限最大的优化点。

## 固定配置

- 模型：Qwen3.5-4B。
- UA2D fastpath：开启。
- gfx936 LLMM1 shape filter：开启。
- 比赛锁定参数、scheduler、数据集和统计口径均不修改。
- 三档独立采集：`4-8K`、`8-16K`、`16-32K`。
- Profile 结果不用于吞吐量比较；端到端吞吐量只使用关闭 profiler 的热基准。

## 执行顺序

1. 检查 GPU、端口和队友进程，等待共享设备空闲。
2. 关闭 profiler 启动当前源码，三档各跑 10 条，确认热基线偏差不超过约 5%。
3. 使用 Torch Profiler 重新启动服务，开启 shape 记录，关闭 stack/memory 记录。
4. 每档采集 prefill：原始输入、输出 1 token。
5. 每档采集 decode：原始输入、输出 128 token，并跳过最初若干 engine iteration。
6. 汇总 GPU kernel 的累计时间、调用次数、平均耗时和输入形状，按功能分类。
7. 按三档累计占比和 Amdahl 收益上限选择 Top 3 热点。
8. 若远端可获得硬件计数器工具，再对 Top 3 做带宽、occupancy、L2、VGPR/LDS 分析；
   若当前镜像不提供，则以 Torch trace 的时间线、launch gap 和 kernel 统计完成第一轮决策。

## 选择标准

优先级按以下因素综合排序：

```text
优化价值 = 三档加权累计时间占比 × 可实现加速比例 × 跨档复用程度 / 风险
```

重点候选依次由 profile 决定，而不是预先指定：

- UA2D/chunked prefill 的 full tile、尾块、mask 和 merge。
- full-attention PagedAttention decode 的 KV 读取和 reduction。
- GDN/linear-attention state update。
- QKV/O/MLP/LM head 的高频 GEMM 形状。
- RMSNorm、RoPE、activation、residual 等高频小 kernel。
- CPU launch gap、同步和 memcpy。

## 门禁

- 不影响队友已有服务或 benchmark。
- 冷编译、异常输出长度或 GPU 并发导致的样本不得进入结论。
- 只有 profile 证明为累计热点的 GEMM 形状，才允许重新进入 LLMM1 调优。
- 每轮 profile 和结论追加到 `docs/progress.md`。
