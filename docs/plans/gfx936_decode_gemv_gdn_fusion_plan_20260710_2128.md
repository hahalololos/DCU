# gfx936 Decode GEMV/GDN 融合优化计划（2026-07-10 21:28 CST）

## 1. 目标与当前基线

目标是在不改变模型权重、推理语义、scheduler 和评测参数的前提下，针对 gfx936 上
Qwen3.5 单 token decode 的高频 GEMV/GDN 路径减少 kernel launch、临时张量和无效访存，
推动排行榜进入前 20。

当前可复现实验基线为：

- Qwen3.5-4B：32 层，其中 24 层 GDN；每个 decode token 每个 GDN 层执行
  `in_proj_qkvz (12288x2560)`、`in_proj_ba (64x2560)`、causal conv、packed recurrent
  GDN、RMSNorm-gate 和 `out_proj (2560x4096)`。
- Qwen3.5-27B：64 层，其中 48 层 GDN；对应输入 GEMV 为
  `in_proj_qkvz (16384x5120)` 和 `in_proj_ba (96x5120)`。
- ALT-C 形状过滤候选保留实测更快的 LLMM1 形状，并把退化形状回退到 `F.linear`；4B
  三档 P99 TPOT 首轮改善约 `6%~9%`，但完整精度 A/B 尚未闭环。
- 当前 packed recurrent GDN kernel 已融合 Q/K L2Norm、`a/b` gating、state update 和
  output；尚未融合前置输入投影。

## 2. 实施顺序

### F0：固定 ALT-C 正确基线

1. 完成过滤关闭/开启的 4B 四类精度同口径 A/B。
2. 候选三档至少补一轮热服务复测，记录输出长度、逐样本文本、TPOT 和吞吐。
3. 只有完成率 100%、精度不回退且性能可重复时，才把 ALT-C 作为融合实验基线。

### F1：双输入权重 GEMV 单 launch 融合

第一版不把 BA 点积直接塞进 recurrent state kernel，避免 `V` 分块导致 BA 点积被重复计算。
改为新增 gfx936 专用双权重 LLMM1：

```text
hidden_states
  + in_proj_qkvz.weight
  + in_proj_ba.weight
  -> 一次 HIP kernel launch
  -> contiguous [qkvz | ba] output
  -> view/split 后进入现有 conv + packed recurrent GDN
```

约束：

- 仅 `n=1`、BF16/FP16、TP=1、无 bias、未量化、gfx936、Qwen3.5 实际形状命中。
- 两个权重的 `K`、dtype、device 必须一致，两个 `M` 都能整除 `rows_per_block`。
- 使用与现有 LLMM1 相同的每行累加顺序；F1 输出应与两次独立 LLMM1 bitwise 相同。
- 默认通过实验环境变量关闭；任何 guard 不满足均回退现有两次 Linear。
- 不创建持久化融合权重，不改 checkpoint、权重数值或模型结构。

预期收益来自每个 GDN 层减少一次 HIP launch 和一个独立输出分配：4B 每 token 最多减少
24 次 launch，27B 最多减少 48 次 launch。

### F2：依据新 profile 决定是否继续向 GDN kernel 内融合

F1 通过后采集稳态 decode profile。只有 BA GEMV、conv、packed recurrent 或
RMSNorm-gate 的 launch/访存累计占比仍显著时，才评估：

- 将 BA 输出直接写入可复用 workspace，减少 allocator 开销；
- 将 causal conv 的 QKV 输出布局与 packed recurrent 消费布局对齐；
- 将 recurrent output 与 RMSNorm-gate 合并，前提是寄存器/访存模型证明可获益。

不采用在每个 `V` tile 内重复计算 BA GEMV的方案；不在无 profile 证据时重写整个 GDN。

## 3. 正确性与性能门禁

### 算子测试

- 4B/27B 实际 `(M1,M2,K)`，BF16 和 FP16。
- fused 与两次独立 LLMM1 bitwise 对照；同时与 `F.linear` 记录 BF16 误差。
- 覆盖非法 dtype、`n>1`、K 不一致、M 非整除时的明确回退/报错。
- 至少重复 20 次检查确定性、NaN、Inf、VM fault。

### 端到端门禁

- 先 4B 三档，每档 10 条；候选收益达到门槛后至少重复两轮。
- 完成率 100%，无 VM fault/OOM/服务重启。
- 相对 ALT-C 基线：P99 TPOT 至少一个档位改善 `>=2%`，三档 output throughput 加权
  收益可重复，任一档不回退超过 `1%`。
- 逐样本输出与 ALT-C 基线完全一致；若 LLMM1 本身导致非 bitwise 差异，只允许融合前后
  完全一致，且必须通过完整精度评测。
- 4B 通过后再验证 27B 小样本、三档和四类精度；最终候选精度扣分必须为 0。

## 4. 构建、版本与回滚

- 修改 `csrc/rocm/` 后按增量编译指南构建 `_rocm_C` 并安装到源码树。
- 每次远端测试前同步本地源码并核对 SHA256、Python/vLLM/editable location 和扩展导入。
- 实验编号使用 `DEC-GDN-F0/F1/F2_YYYYMMDD_HHMM`。
- 每轮源码优化和测试结果写入 `docs/progress.md`。
- F1 使用独立环境变量开关；淘汰时删除实验实现，不让默认路径残留无收益复杂度。

