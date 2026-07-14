# Qwen3.5-27B Top 20 冲刺实施计划

时间：2026-07-14 19:05 CST  
起始源码：`vllm_cscc@0038e9f`（合并 shiyi 输出投影 GEMV 候选）

## 目标

- 先复现当前候选约 86 分的三档吞吐、SLA 和精度状态；此结果取代
  `a5cbd48` 的旧 profile 作为所有后续收益判断基线。
- 最终安全目标为总分 `>=88.0`、无 SLA/精度扣分；三档参考线为
  16--32K `>=14.5`、4--8K `>=20.5`、8--16K `>=18.7 tok/s`。

## 实施顺序

1. 同步本地权威源码、重建 `_rocm_C`，验证输出投影 `(5120,1,6144)` 专用 GEMV；
   使用三随机种子、交替顺序 micro 和三档各十条服务 A/B 确认候选有效。
2. 仅对 V4 三个 Decode GEMV 形状探索 load/compute 软件流水、独立累加器和可用的
   gfx936 BF16 packed dot；不重做已淘汰的 MLP-down、LM-head、四行复用和 SwiGLU 融合。
3. 记录中长档真实 Prefill GEMM 形状，筛选现有 Tensile、AITER、hipBLASLt/TunableOp；
   TunableOp 仅用于发现算法，最终候选不得依赖外部 CSV 或运行期调优缓存。
4. 若前两项不足，再在 UA2D E8 的单 kernel、保序路径中做地址/活跃范围/LDS layout
   优化；不回退到 TILE64、BLOCK_M16、多 kernel 拆分或已淘汰的编译参数扫描。

## 门禁

- 所有 micro 都覆盖数值、有限值、重复确定性和 P99；候选必须有可折算的累计节省。
- Decode GEMV 进入服务 A/B 需三个主形状累计节省 `>=0.75 ms/token`；Prefill GEMM
  需热点加权 micro `>=8%`；UA2D 需中长档 micro `>=8%` 且与 E8 bitwise 一致。
- 服务快筛使用热态 `B-C-C-B`；三档各十条最终确认，完成率 100%，任一档吞吐回退不超过
  1%，TTFT/TPOT P99 无异常。若文本发生变化，运行四类 OpenCompass 精度门禁。
- 每个胜出候选独立提交；实验结果简记至 `docs/progress.md`，远端临时源码立即回传本地。
