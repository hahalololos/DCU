# UA3D Decode Attention 分段并行优化计划

## 目标

针对 Qwen3.5-4B/27B、单请求长上下文 decode 路径，优化现有 UA3D 分段
softmax，在不改变模型语义、调度参数和评测脚本的前提下降低 TPOT，并量化其对
Top20 吞吐差距的贡献。

## 已知事实

- 当前实现已经固定使用 16 段，并由第二个 Triton kernel 合并局部 max、exp sum 和
  accumulator。
- 4B 初筛：4K 在 4 段后基本饱和，8K 在 8 段后基本饱和；16K 下 16 段仍最快。
- Qwen3.5 KV cache block size 为 528/544/784，而 decode tile 为 16。通用 UA3D 对每个
  lane 重复执行 block 除法、取模和 block-table load，是 gfx936 专用优化候选。

## 实施步骤

1. 为 gfx936、BF16、head size 256、Qwen3.5 4B/27B decode 增加严格 guard。
2. 在 UA3D kernel 内以 tile 起点标量计算 logical block；仅在 tile 跨 block 时加载第二个
   block-table 项，通用路径保持不变。
3. 覆盖 4B/27B、block 边界、4K/8K/16K/32K、segments 4/8/16 的数值和性能测试。
4. 胜出后执行 4B 三档端到端 A/B；要求完成率和 token 数一致，P99 TPOT 不回退超过 1%。
5. 若 4B 长档收益达到 2%，再执行 27B 端到端验证。

## 回滚条件

- 专用与通用 UA3D 超出既有 attention 容差；
- 任一 block 边界出现越界或非确定性；
- 4B 长档吞吐收益低于 1%，或 P99 TPOT 回退超过 1%。
