# Qwen3.5 Gated DeltaNet 推理优化实施方案

生成时间：2026-07-12 19:53

适用范围：`vllm_cscc`、Qwen3.5-4B/Qwen3.5-27B、单卡 gfx936、单请求、4K–32K 长上下文。

## 1. 目标

围绕 Qwen3.5 的 Gated DeltaNet（GDN）路径寻找新的结构性端到端收益，优先改善评分权重最高的 `8K–16K` 档位，同时兼顾 `4K–8K` 和 `16K–32K`。

阶段目标：

1. 建立可重复的 GDN prefill/decode 分项性能画像。
2. 找到至少一个 GDN 子路径 `>= 8%` 的稳定收益候选。
3. 候选在 4B `8K–16K` 端到端吞吐改善至少 `2%`，且 TTFT/TPOT 不回退。
4. 通过 4B 门禁后适配 27B 精确形状，以 27B 结果作为最终依据。
5. 不新增 SLA 或精度扣分，不改变模型结构、权重、推理语义和调度参数。

## 2. 合规边界

允许：

- 修改 vLLM Python、Triton、HIP/C++ 源码。
- 优化 GDN tensor layout、运行期中间张量布局和 kernel 访存。
- 减少不必要的 `split/rearrange/cat/copy`。
- 修改 Triton kernel 的 tile、chunk、wave、stage 和 launch grid。
- 融合 causal convolution、gating、recurrent state update 等等价计算。
- 在不改变有效计算和结果语义的前提下做图编译和小算子融合。
- 研究推理期非持久化 recurrent state 低精度，但必须单独做合规及精度门禁。

禁止：

- 修改或生成持久化模型权重、量化权重或权重缓存。
- 修改 tokenizer、chat template、模型层数、head 数或 GDN 数学结构。
- 跳过 GDN layer、head、token 或 state update。
- 使用投机解码、MTP、draft model、early exit。
- 修改比赛锁定的 scheduler、batch、上下文或 benchmark 参数。
- 根据测试样本内容选择 kernel 或输出路径。

## 3. 模型形状

### 3.1 Qwen3.5-4B

| 参数 | 值 |
| --- | ---: |
| GDN 层数 | 24 |
| hidden size | 2560 |
| key heads | 16 |
| value heads | 32 |
| key/value head dim | 128 |
| key dim | 2048 |
| value dim | 4096 |
| packed QKV dim | 8192 |
| conv kernel width | 4 |
| recurrent state | `[32,128,128]` |
| FP32 state 大小 | 2 MiB/层，48 MiB/序列 |

### 3.2 Qwen3.5-27B

| 参数 | 值 |
| --- | ---: |
| GDN 层数 | 48 |
| hidden size | 5120 |
| key heads | 16 |
| value heads | 48 |
| key/value head dim | 128 |
| key dim | 2048 |
| value dim | 6144 |
| packed QKV dim | 10240 |
| conv kernel width | 4 |
| recurrent state | `[48,128,128]` |
| FP32 state 大小 | 3 MiB/层，144 MiB/序列 |

## 4. 当前实现状态

当前已经具备：

- recurrent state 布局为 `[N, HV, V, K]`，无需重复做 `[K,V] -> [V,K]`。
- `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=True`，packed decode 默认开启。
- decode fast path 已将 `a/b/A_log/dt_bias/QK L2Norm` 合入 packed recurrent kernel。
- prefill kernel warmup 已覆盖 BT=16/32/64，避免首次推理 autotune OOM。
- prefill 默认 chunk size 基本为 64。
- `qkvz + b/a` 双 GEMV 融合已淘汰：4B/27B micro 仅约 `1.09x/1.06x`，4B 端到端仅 `0.1%–0.6%`。

主要源码入口：

- `vllm_cscc/vllm/model_executor/models/qwen3_next.py`
- `vllm_cscc/vllm/model_executor/models/qwen3_5.py`
- `vllm_cscc/vllm/model_executor/layers/fla/ops/chunk.py`
- `vllm_cscc/vllm/model_executor/layers/fla/ops/chunk_o.py`
- `vllm_cscc/vllm/model_executor/layers/fla/ops/solve_tril.py`
- `vllm_cscc/vllm/model_executor/layers/fla/ops/fused_recurrent.py`
- `vllm_cscc/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`
- `vllm_cscc/vllm/model_executor/layers/mamba/ops/causal_conv1d.py`

## 5. 总体实施顺序

```text
G0 分项 profile
  ↓
G1 Qwen3.5 layout 与 b/a copy 条件优化
  ↓
G2 gfx936 prefill chunk kernel 调优
  ↓
G3 packed recurrent decode 调优
  ↓
G4 causal-conv + recurrent 融合
  ↓
G5 output/graph/固定开销优化
  ↓
G6 可选 recurrent state 低精度研究
```

每个阶段均设置独立提交、开关、micro benchmark、正确性测试和端到端门禁。未达到门槛时回滚该阶段，不与后续候选混合。

## 6. 阶段 G0：GDN 分项 Profile

### 6.1 目的

回答以下问题：

1. 4B 与 27B 的 GDN prefill/decode 各占端到端多少时间。
2. prefill 的主要热点是投影、causal conv、`solve_tril`、state update 还是 output kernel。
3. decode 的主要热点是投影、causal conv、packed recurrent、norm 还是 output projection。
4. `split/rearrange/cat` 是否物化并产生显著 GPU kernel 或内存流量。
5. GDN custom op 前后是否存在 CPU/GPU idle gap 或 graph break。

### 6.2 Profile 分项

Prefill：

```text
in_proj_qkvz
in_proj_ba
fix_query_key_value_ordering
split/rearrange/cat
causal_conv1d_fn
decay/cumsum
solve_tril
chunk state update
chunk_fwd_kernel_o
RMSNormGated
out_proj
```

Decode：

```text
in_proj_qkvz
in_proj_ba
split/rearrange/cat
causal_conv1d_update
fused_recurrent_gated_delta_rule_packed_decode
RMSNormGated
out_proj
CPU/kernel gap
```

### 6.3 测试形状

Prefill：

```text
T = 4096, 8192, 12288, 16384, 24576, 32768
4B: H=16, HV=32, K=V=128
27B: H=16, HV=48, K=V=128
```

Decode：

```text
B = 1
T = 1
4B/27B 精确形状
重复 1000 次，排除首次编译和 cache cold start
```

### 6.4 采集指标

- kernel latency 与占比。
- 每层、全 24/48 层累计时间。
- VGPR、SGPR、LDS。
- L2 hit、HBM read/write。
- LDS bank conflict。
- waves/CU、occupancy。
- MFMA/VALU/LDS busy。
- kernel 间隔和 CPU launch 时间。
- 临时张量大小和显存峰值。

### 6.5 决策规则

- `split/rearrange/cat >= GDN prefill 5%`：优先进入 G1。
- `solve_tril >= GDN prefill 15%`：G2 重点调 BT 和三角求解。
- `chunk_fwd_kernel_o` 最大：G2 重点调 BK/BV/waves/stages。
- packed recurrent 最大：G3 升为 P0。
- causal conv + recurrent 之间有明显空洞/流量：进入 G4。
- 投影 GEMM 仍占绝对多数：停止 GDN core 重写，回到精确 shape GEMM backend 调优。

## 7. 阶段 G1：Qwen3.5 Layout 与 B/A Copy 条件优化

### 7.1 执行前源码审计修正

执行前审计确认，当前 `Qwen3_5GatedDeltaNet.forward` 已覆盖父类实现：

```text
in_proj_qkvz
→ mixed_qkvz.split([qkv_size, z_size])
→ mixed_qkv 直接作为连续 view 进入 GDN core
→ z 仅 reshape
```

因此计划最初假设的 Qwen3-Next `split/rearrange/cat` 大物化并不存在于 Qwen3.5，不能据此实现专用 interleaved kernel。当前仍存在：

- `ba.chunk(2)` 后的 `b.contiguous()` 与 `a.contiguous()`。
- `z.reshape`、output reshape 等 view/潜在物化。
- 不同 prefill/decode kernel 对非紧凑 token stride 的支持差异。

### 7.2 条件方案

只有 G0 证明 layout/copy 占 GDN 路径 `>=5%` 时才实施：

- 验证 packed recurrent、prefill gating 和相关 Triton kernel 是否已支持 `b/a` 的非紧凑 token stride。
- 若支持，删除不必要的 `b.contiguous()`/`a.contiguous()`。
- 若部分 kernel 不支持，为 kernel 显式传入 token stride，而不是复制整张量。
- 检查 `z`、norm 前后 reshape 是否只是 view。
- 不修改权重内容、格式或加载语义。
- 未通过所有路径正确性测试时保持原实现。

### 7.3 覆盖路径

```text
prefill
packed non-spec decode
普通 non-spec decode fallback
spec metadata fallback（只做正确性，不做性能候选）
4B / 27B
```

### 7.4 正确性测试

- 4B/27B。
- prefill T=1/15/16/31/32/63/64/65/4096。
- decode 单 token连续更新 128 步。
- state 初始为零和非零。
- 输出、conv state、recurrent state 分别对照。
- 重复运行确定性。
- guard/fallback 测试。

### 7.5 保留门槛

- layout/copy micro 改善 `>=20%`。
- GDN prefill 或 decode 路径改善 `>=3%`。
- 4B 中档 TTFT、吞吐或 TPOT改善 `>=1%`。
- TPOT 不回退超过 `1%`。
- 无精度与完成率回退。

若 layout/copy 占比低于 5% 或端到端收益低于 1%，立即停止 G1，直接进入 G2/G3/G4 中由 G0 选出的主热点。

## 8. 阶段 G2：gfx936 Prefill Chunk Kernel 调优

### 8.1 参数空间

```text
BT = 16, 32, 64, 128
BK = 32, 64, 128
BV = 32, 64, 128
num_warps = 2, 4, 8
num_stages = 1, 2, 3, 4
```

优先缩减组合：

1. 固定 K=V=128。
2. 先扫 BT。
3. 对每个 BT 只保留前两名 BK/BV。
4. 再扫 wave/stage。
5. 分别为 4B HV=32 和 27B HV=48 建配置。

### 8.2 关注点

- BT 增大能提高 GEMM 效率，但会增加 BT×BT 中间矩阵、VGPR、LDS 和三角求解成本。
- gfx936 wave/寄存器特征可能使 BT=32 优于通用默认 BT=64。
- `chunk_fwd_kernel_o` 当前 autotune 配置主要来自通用 FLA，需增加 gfx936 精确配置。
- 不直接扩大所有 autotune 配置，避免首次编译/调优耗时和显存爆炸。
- 最终配置应通过离线 micro 固化，而非正式评测时动态调优。

### 8.3 `solve_tril` 专项

仅当其占 GDN prefill `>=15%` 时实施：

- 针对 BT=32/64 的固定尺寸实现。
- 研究 wave 内 forward substitution。
- 16×16 或 32×32 分块。
- 避免完整 BT×BT 中间结果物化。
- 检查 LDS padding/swizzle 和 bank conflict。
- 缩短 FP32 中间量 live range。

### 8.4 保留门槛

- 主要子 kernel 改善 `>=15%`。
- 完整 GDN prefill 改善 `>=8%`。
- 4B 中档端到端吞吐改善 `>=2%`。
- 27B 精确形状 micro 不回退。

## 9. 阶段 G3：Packed Recurrent Decode 调优

### 9.1 当前配置

```text
BK = 128
BV = 32
num_warps = 1
num_stages = 3
grid = (ceil(V/BV), B*HV)
```

batch=1 时：

- 4B program 数：`4 × 32 = 128`。
- 27B program 数：`4 × 48 = 192`。

### 9.2 扫描空间

```text
BV = 16, 32, 64
num_warps = 1, 2, 4
num_stages = 1, 2, 3
```

可选布局实验：

- 每个 program 处理一个或多个 value-head tile。
- 共享 Q/K head 的 value heads 是否可减少 Q/K 重复读取。
- HV=48 时按 3 个 value heads/1 key head 组织 program。
- state `[HV,V,K]` 的 V/K 遍历顺序。

### 9.3 指标

- 单层 latency。
- 48 层累计 latency。
- state read/write 实际字节数。
- Q/K 重复读取量。
- VGPR、occupancy、L2 hit、HBM 带宽。
- 相对完整 TPOT 的贡献。

### 9.4 保留门槛

- packed recurrent kernel 改善 `>=15%`。
- 完整 GDN decode 改善 `>=8%`。
- 4B TPOT 改善 `>=1%`。
- 27B 精确形状无回退。

## 10. 阶段 G4：Causal Conv + Recurrent 融合

### 10.1 当前路径

```text
causal_conv1d_update
→ 写回 mixed_qkv
→ packed recurrent 重新读取 mixed_qkv
→ 更新 recurrent state
```

### 10.2 候选方案

单 kernel 内完成：

```text
读取 QKV projection
→ 更新 width=4 conv state
→ SiLU
→ Q/K L2Norm
→ gating/delta recurrent update
→ 写回 recurrent state 与 output
```

### 10.3 风险

- 融合后 VGPR 过高。
- conv channel 数较大。
- 16 key heads 与 32/48 value heads 的映射复杂。
- state update 为 FP32，可能增加寄存器和转换压力。
- 少一个 launch 不一定抵消 occupancy 回退。

### 10.4 实施限制

只支持严格专用形状：

```text
B=1
decode token=1
BF16
conv width=4
K=V=128
H=16
HV=32 or 48
TP=1
```

### 10.5 保留门槛

- fused kernel 相对两 kernel 总时间改善 `>=10%`。
- VGPR 不出现导致 occupancy 大幅下降的跳变。
- GDN decode 改善 `>=5%`。
- 端到端 TPOT 改善 `>=1%`。

## 11. 阶段 G5：Output、Graph 与固定开销

检查：

- `core_attn_out = torch.zeros(...)` 的必要性和实际清零成本。
- custom op 是否造成 graph break。
- norm 前后 reshape/rearrange 是否物化。
- GDN 非核心路径是否被 HIP Graph 捕获。
- 是否可直接写入预分配 model-runner buffer。
- `RMSNormGated` 是否已为最优 fused kernel。
- output projection 前是否存在额外 copy/contiguous。

注意：当前源码注释明确指出 `core_attn_out` 不能简单改为 `empty`。任何替换必须先确认关联正确性约束，并使用 poisoned-memory 测试证明 kernel 完整覆盖输出。

保留门槛：

- 48 层累计固定开销下降 `>=10%`，或模型 TPOT 改善 `>=1%`。
- 不引入 graph capture 不稳定或首次运行长尾。

## 12. 阶段 G6：Recurrent State 低精度（可选）

仅在以下条件同时满足时进入：

1. 27B profile 证明 recurrent state 流量是主要瓶颈。
2. G1–G5 没有足够收益。
3. 再次确认比赛允许推理期非持久化 GDN state 量化。

候选：

- BF16 state + FP32 accumulator。
- 分块 scale 的 BF16/FP8 state。
- prefill 最终 state 压缩，decode 内解压并更新。
- 仅量化部分 state tile 的混合精度。

风险：

- state 跨 token 递归累积，误差可能随序列增长。
- 检索、聚合和摘要任务可能对 state 精度敏感。
- 可能产生精度扣分，抵消吞吐收益。

门槛：

- state 流量至少下降 40%。
- GDN decode 至少改善 15%。
- 四类精度均不越过 1%下降线。
- 无新增 SLA 或完成率问题。

## 13. Micro Benchmark 设计

建议新增但暂不在本计划中直接实现：

```text
testdata/profile_gdn_4b_27b.py
testdata/benchmark_gdn_prefill.py
testdata/benchmark_gdn_decode.py
```

要求：

- 可独立选择 4B/27B 形状。
- 可选择 prefill/decode。
- 可扫描 BT/BK/BV/warps/stages。
- 输出 median、P90、P99。
- 支持 warmup 和重复次数配置。
- 对照 reference 输出、conv state、recurrent state。
- 支持 rocprof 只采目标 kernel。
- 不写死比赛样本或答案。

## 14. 正确性测试矩阵

### Prefill

```text
T = 1, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 4096
initial_state = zero / random
4B / 27B
BF16
```

### Decode

```text
连续 decode steps = 1, 2, 16, 128, 512
initial_state = zero / random
conv state = zero / random
state index = contiguous / non-contiguous
4B / 27B
```

比较：

- 单步输出。
- 最终 conv state。
- 最终 recurrent state。
- 重复运行确定性。
- 原路径与专用路径误差。
- greedy token 结果。

## 15. 端到端验证

每轮远端操作先执行：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
```

检查：

```bash
which python
which vllm
python -m pip show vllm
```

4B 快速门禁：

```bash
cd testdata
./start_vllm_4b.sh
./run_throughput_4b.sh 16-32K 10
```

候选进入完整门禁后运行三档，并至少重复三轮热测试。测试前确认 GPU、端口和队友进程，确保基线/候选处于同一独占资源条件。

只有满足以下条件才进入 27B：

- GDN 子路径收益达到阶段门槛。
- 4B `8K–16K` output throughput 改善 `>=2%`。
- 三档 TTFT/TPOT 无明显回退。
- 完成率 100%。
- 精度无新增扣分风险。

27B 只测试已通过 4B 门禁的最终候选，必要时先将模型复制到 `/tmp`，避免共享存储读取干扰。

## 16. 精度门禁

涉及以下变化时必须跑完整四类精度：

- kernel 计算顺序变化。
- chunk size 变化导致累加顺序变化。
- recurrent state 布局或精度变化。
- causal-conv/recurrent 融合。
- Q/K L2Norm 实现变化。

重点关注：

- `gov_report`：当前对数值变化较敏感。
- RULER retrieval/aggregation：检验长上下文 state 保真。
- 连续多 token decode：检验 recurrent error accumulation。

保留标准：

- 最优目标：四类任务均 `Δ <= 1%`。
- 若任一任务越过 1%，必须证明总榜单净收益足以覆盖精度系数下降，否则淘汰。

## 17. 构建策略

- 纯 Python/Triton 改动：重启服务即可；如需 editable 安装，遵循仓库 guide。
- 涉及 `csrc/`、HIP/C++、CMake：必须重建扩展。
- 不得用 `VLLM_USE_PRECOMPILED=1` 覆盖已有自定义源码树 `.so`。
- 构建前必读 `竞赛环境vllm增量编译guide.md`。

## 18. 提交与回滚

每个候选独立提交：

```text
G0: profile/benchmark only
G1: conditional Qwen3.5 layout/b-a copy cleanup
G2: gfx936 chunk tuning
G3: packed recurrent tuning
G4: conv+recurrent fusion
G5: graph/output cleanup
G6: state low precision
```

要求：

- 默认关闭或严格 guard，直到端到端验证完成。
- 每个候选有独立环境变量或 dispatch guard，避免混合实验。
- 淘汰后完整删除源码、测试和环境变量。
- 远端临时改动立即同步回本地。
- 每次源码优化及测试结果按时间戳记录到 `docs/progress.md`。

## 19. 停止条件

出现以下任一情况时停止该方向：

- micro 收益小于 5%。
- GDN 子路径收益小于 3%。
- 端到端收益小于 1%。
- TPOT 或 TTFT 回退超过 2%。
- 完成率下降。
- 产生难以控制的精度扣分。
- 27B 精确形状无收益或回退。
- 需要修改模型权重、调度参数或推理语义才能获得收益。

## 20. 首轮具体执行建议

第一轮先完成：

1. 建立 GDN 分项 profiler/micro benchmark，确认 4B/27B 的真实热点。
2. 用 layout micro 验证 Qwen3.5 当前 `mixed_qkv/z/b/a` 路径的实际成本。

根据 G0 结果选择：

- layout/copy 占比 `>=5%`：进入条件 G1。
- prefill core 主导：进入 BT/BK/BV 调优。
- decode recurrent 主导：进入 BV/warps/stages 调优。
- causal conv 与 recurrent 流量主导：进入融合。
- 投影 GEMM 主导：停止 GDN core 优化，转向 27B 精确 GEMM backend。
