# Qwen3.5-27B UA2D 编译参数调优计划

时间：2026-07-13 16:29 CST

## 目标

以 `vllm_cscc` 提交 `0adf049` 的 v2b/WARPS2 为基线，仅调整 gfx936 Triton 编译参数，
争取获得一个可独立提交的中档或长档增益，不改变默认路径和计算顺序。

## 实现

- experiment 6/7/8：`waves_per_eu=1/2/4`。
- experiment 9/10/11：在上述参数上增加
  `matrix_instr_nonkdim=16,kpack=2`。
- 所有新实验仅作用于 Qwen3.5-27B `block_size=784`，继续使用 v2b、两个 warp、一个 stage。
- 默认 experiment 5 保持不变，4B 和通用 Attention 路径保持原配置。

## 门禁

- 27B 合成 8K/16K/32K micro；16K 或 32K 至少提升 5%，其他长度回退不超过 1%。
- GQA=6、`783/784/785`、partial query、`4095/4096` 和重复确定性测试通过。
- 胜出候选再跑 27B 中档、长档各 10 条正反向 A/B；任一档 output throughput 至少提升
  1.5%，P99 TPOT 回退不超过 1%，输出完全一致。
- 未通过门禁则删除实验接入并保留 experiment 5。
