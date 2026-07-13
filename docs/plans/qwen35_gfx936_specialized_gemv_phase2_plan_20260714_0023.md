# Qwen3.5-27B gfx936 专用 GEMV 第二阶段实施计划

时间：2026-07-14 00:23 CST
生产基线：`vllm_cscc@1290f03`

## 目标

- 将现有 gate-up 专用 GEMV 扩展为固定 `K=5120/N=1` 的三形状框架。
- 第一阶段优化 GDN qkvz `(16384,1,5120)`，第二阶段评估 Attention qkv+gate
  `(14336,1,5120)`；gate-up `(34816,1,5120)` 保持回归基线。
- 允许非 bitwise 数值结果，但必须通过算子误差、端到端精度系数和最终得分净收益门禁。

## 实施

1. 保持 `LLMM1Gfx936(weight, activation, variant)` 接口，增加 V3/V4 activation LDS
   复用结构，并允许三个精确 M。
2. 扩展 micro 工具，以三随机种子、正反序七轮比较 F.linear、LLMM1 和 V1--V4。
3. GDN 至少缩短 8% 且节省 `0.70 ms/token` 才接入；Attention 至少缩短 10% 且
   节省 `0.20 ms/token` 才接入。
4. 增加独立回滚开关，未命中或关闭时回退当前 LLMM1/F.linear 路径。
5. 远端增量构建 `_rocm_C`，完成三档 27B A/B、完整精度与 SLA 门禁。

## 验收

- 所有请求完成，无 VM fault/OOM；任一档吞吐稳定回退不超过 1%。
- 算子输出有限且确定，相对 F.linear 满足 `rtol=1e-2/atol=1.5e-2`。
- 按比赛精度系数计算后的最终得分相对基线至少增加 `0.30`。
- 每轮源码优化与测试结果按时间戳追加至 `docs/progress.md`。
