# Qwen3.5-27B Decode GEMV Top 20 冲刺计划

时间：2026-07-13 22:39 CST
生产基线：`vllm_cscc@0adf049`

## 目标

- 首先重写 gfx936 BF16、`N=1/M=34816/K=5120` 的 MLP gate-up GEMV。
- micro 至少提升 `15%`、目标 `25%--35%`；27B 短档或中档端到端至少提升 `3%`。
- gate-up 通过门禁后再处理 `(5120,1,17408)` MLP down；否则立即止损。
- 预留最后 10 小时完成三档、精度、SLA 和榜单提交验证。

## 实施

1. 增加仅供实验的 gfx936 专用内部算子，比较：
   - 单 wave/输出行、4 行/workgroup、无 LDS 跨 wave 归约；
   - 双 wave/输出行、2 行/workgroup、仅保存两个 FP32 partial。
2. 使用 128-bit 连续 BF16 权重与 activation 加载、FP32 累加和 wave shuffle 归约。
3. 胜出版本仅在 gfx936、BF16、精确 gate-up 形状下接入；其他路径保持不变。
4. 远端按同一常驻进程交替测试，记录 median、P99、数值差异和确定性。

## 门禁

- micro 改善 `<10%`：不接入生产，停止该方向。
- micro 改善 `>=15%`：执行 27B 短、中档严格 A/B；加权预测收益需 `>=4%`。
- 三档完成率必须为 100%，其他档位和 TTFT/TPOT 不得稳定回退超过 `1%`。
- 四类精度任一相对 baseline 下降超过 `1%`，回退候选。

## 验证

- 算子：随机、零值、极值、NaN/Inf、重复确定性及 `F.linear` 对照。
- Dispatch：非目标形状、FP16、4B 和其他架构不得误命中。
- 端到端：27B 三档各 10 条，比较吞吐、TTFT P99、TPOT P99、输出长度和文本。
- 每个源码优化及结果按时间戳追加至 `docs/progress.md`。
