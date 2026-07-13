# Qwen3.5-27B 全链路 Profile 执行计划

时间：2026-07-13 19:55 CST

## 目标

以 `vllm_cscc` 提交 `0adf0494e9bdc93a2fe4f9dd96a68f41e1869a73` 为基线，
分离 27B 三档请求的 Prefill、稳定 Decode 和 GPU launch gap，输出带加权端到端收益
估算的 Top 3 优化方向，并对 Top 3 采集 rocprof 硬件计数器。

## 存储约束

远端共享目录 `/public/home/acoh0h1o0p` 已无可用空间。原始 benchmark、Torch trace 和
rocprof 输出统一写入容器本地 `/root/profile27b/`；本地仓库只保存计划、汇总报告和必要的
小型结果文件，不向共享目录写入大 trace。

## 执行阶段

1. 同步本地权威源码和测试工具，确认 Python、vLLM、GPU、端口和模型副本。
2. 关闭 profiler，三档各跑 10 条热基线，保存逐请求输入/输出长度、TTFT、TPOT 和文本。
3. 每档从完成且输出不少于 65 token 的请求中选择输入长度最接近 P50/P90 的两条。
4. 对每条请求采集 `output=1` 与 `output=65` 配对 trace；相同配置重复两次。
5. 以 `trace65 - trace1` 除以 64 估算稳定 Decode，每档汇总 P50/P90 的 kernel 分类和
   GPU 空隙。
6. 使用比赛 `20%/50%/30%` 权重估算端到端收益，选择 Top 3 热点。
7. 对 Top 3 使用精确 27B 形状做无计数器 micro 和 rocprof 三组计数器采集。
8. 将完整结论写入 `调研交付物/`，并在 `docs/progress.md` 追加简要记录。

## 门禁

- 基线三档均须 `10/10` 完成；GPU 并发、冷编译、异常输出长度的数据作废。
- 配对 trace 的相同 prompt、temperature 和前 65 token 必须一致。
- 只将理论加权端到端收益不低于 2% 的热点列入实现候选。
- rocprof 注入时间仅用于硬件瓶颈分类，不用于性能对比。
- 不重复投入已经淘汰的 UA2D 编译参数、TILE64、LLMM1 rows_per_block 和 GDN 小 kernel
  调参，除非新 profile 给出新的端到端收益证据。
