# Qwen3.5-27B gfx936 Gate-up GEMV + SwiGLU 融合计划

基线：`vllm_cscc@0e3450b`。

1. 将 Decode 的 `(34816,1,5120)` Gate-up V4 GEMV 与 SwiGLU 融合，直接输出
   `(1,17408)`，然后复用现有 MLP down Tensile 路径。
2. 仅命中 gfx936、BF16、TP=1、无量化、无 bias、单 token 精确形状；
   `VLLM_ROCM_GFX936_SPECIALIZED_GEMV=3` 启用，其余自动回退。
3. 保留 GEMV 输出、SiLU 输出和最终乘法的 BF16 舍入顺序，目标与
   `V4 + SiluAndMul` bitwise 一致。
4. micro 要求三种子无回退、中位加速至少 `1.02x`；服务 A/B 要求加权
   output throughput 至少改善 `0.4%`。未达门禁则清理候选。
5. 不重做历史基线，不进行榜单成绩闭环。

## 实施结果

- 候选模式 3 完成 `_rocm_C` 增量构建、算子注册、独立
  `torch.compile`、AOT 保存和全部 CUDAGraph 捕获，定向回归 `40 passed`。
- 三随机种子 micro 中，当前 `V4 + SiluAndMul` 约 `0.3074 ms`，融合核约
  `0.2952 ms`，加速稳定在 `1.041x`；与当前路径 bitwise 一致，且相对
  `F.linear + SiluAndMul` 无超差元素。
- 热态模式 2 -> 模式 3 服务 A/B：短、中、长档 output throughput 分别改善
  `+0.1771%`、`+0.1531%`、`+0.0609%`，加权改善仅 `+0.13023%`；三档均
  `10/10` 成功，文本和输出长度全部一致。
- 服务收益低于 `+0.4%` 门槛，因此按计划淘汰模式 3，不改默认值、不跑完整精度；
  候选生产源码和测试接口均已清理，`vllm_cscc` 回到 `0e3450b`。
- 下一方向转为 UA2D 实验 8 思路上的 LDS 布局、bank 映射与 VGPR live-range
  结构重写，先以独立实验 kernel 验证，再决定是否进入服务 A/B。
