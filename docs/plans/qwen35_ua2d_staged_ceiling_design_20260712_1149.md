# Qwen3.5 UA2D Prefill Attention 分阶段深度优化设计

时间：2026-07-12 11:49

基线源码：`vllm_cscc` 分支 `haha`，提交 `c274c6b`

## 1. 背景与判断

当前榜单提交已包含 Qwen3.5 专用
`kernel_qwen35_unified_attention_2d`。相对上一版，正式 27B 榜单的
`16K-32K` 吞吐由 `9.68` 提升至 `11.03 tok/s`，约提升 `13.95%`；4B 代表形状
`q=4096/kv=22258` 的单 kernel 延迟由约 `52.0 ms` 降至约 `44.8 ms`，约提升
`14.1%`。

当前实现已经完成以下中层技术优化，而非只有参数搜索：

- 新增 Qwen3.5/gfx936 专用 UA2D kernel，删除不命中的通用特性分支；
- 将完全可见的 full-prefix 与 diagonal/partial 区域分段处理，减少 causal mask；
- 固化标量 block-table 地址计算；
- 为 27B `GQA=6` 增加 padded-row 正确性保护；
- 固化当前已验证最优配置 `TILE_SIZE=32`、`BLOCK_M=32`、`warps=4`、
  `stages=1`。

但当前 kernel 仍沿用通用 Triton UA2D 的基本计算骨架：一个 program 处理一个
query block 与一个 KV head，逐 tile 扫描 paged KV，维护完整
`FP32 acc[32,256]`，并依赖 Triton 生成 wave、MFMA、LDS 和寄存器布局。因此当前版本
接近已有结构下的参数局部最优，但尚不能视为 gfx936 的结构或硬件上限。

本设计采用分阶段路线：先用无需加载模型的 27B 合成形状验证结构余量，再用 4B 完成
端到端和精度门禁；每轮结构优化最多支付一次完整 27B 服务启动成本。只有证据证明
Triton 受限时，才升级到 gfx936 专用 HIP/MFMA kernel。

## 2. 目标与边界

### 2.1 性能目标

第一阶段候选必须满足：

- 27B `16K-32K` 代表形状 kernel 中位延迟相对 `c274c6b` 至少改善 `8%`；
- 27B `8K-16K` 代表形状不得出现超过 `2%` 的稳定回退；
- 收益必须覆盖多个真实长度，不能只命中单一特殊形状；
- 通过 4B 门禁后，目标是在正式 27B 长档兑现至少约 `2%-3%` 的 request
  throughput 收益。

HIP/MFMA 原型由于构建和维护成本更高，进入服务验证前要求合成 27B kernel 至少改善
约 `12%`，并能推算出至少约 `3%` 的长档 request throughput 兑现空间。

### 2.2 精度目标

采用平衡型精度策略：

- 允许改变浮点归约顺序；
- BF16 输入和输出，online softmax 与 accumulator 保持 FP32；
- kernel 输出满足现有误差容限，重复执行结果确定；
- 4B 四类完整精度门禁通过；
- 正式平台目标仍为精度扣分 `0`。

不得依靠输出长度变化、截断、跳 token、模型结构变化或测试集特化获得表面吞吐收益。

### 2.3 范围边界

本轮允许修改：

- Qwen3.5/gfx936 专用 Triton UA2D kernel；
- 专用 HIP/MFMA extension kernel 及其严格 dispatch；
- kernel correctness、microbenchmark 和 profile 辅助代码；
- 默认关闭或只用于开发验证的候选开关。

本轮不修改：

- 模型权重、tokenizer、chat template、模型结构和推理语义；
- scheduler、锁定参数和比赛脚本；
- KV Cache 外部配置或持久化布局；
- 通用 UA2D/UA3D 的默认语义；
- 已证明严重回退的 prefix/suffix 多 kernel 加 LSE merge 路线。

## 3. 总体架构

保留 `c274c6b` 的专用 UA2D 作为不可变基线。所有实验使用独立 kernel 名称和严格
dispatch guard，未命中或候选失败时回退当前实现，不在稳定 kernel 中累积实验分支。

候选按以下顺序推进：

1. `V2-A`：边界感知的 paged KV 地址路径；
2. `V2-B`：block-aligned 的物理 KV block 遍历；
3. Triton 资源布局与 ROCm 编译参数原型；
4. gfx936 专用 HIP/MFMA UA2D。

任一阶段达到晋级门槛即可进入下一层端到端验证，不要求完成后续更高风险路线。失败候选
完整回滚，只保留实验记录。

## 4. Triton V2 设计

### 4.1 V2-A：边界感知地址路径

当前 kernel 以全局 `TILE_SIZE=32` 扫描 KV。每个 tile 都加载当前 physical block，且
第二个 physical block 的加载条件基于整个 `max_visible_key`，导致很多并未跨 block 边界
的 tile 也加载下一个 block-table 项。

V2-A 保持现有全局 tile、QK/PV、online softmax 和 mask 逻辑，只调整地址计算：

- 仅当当前 tile 实际跨越 physical block 边界时加载第二个 block id；
- 非边界 tile 只加载一次 block-table；
- 保持跨 block tile 的两段 physical id 和 cache offset 映射；
- full-prefix 与 diagonal/partial 循环使用同一边界判断规则；
- 不改变 tile 顺序和浮点归约顺序，作为最低风险候选。

V2-A 用于判断 block-table 和地址路径是否仍有可观余量。若收益不足，也可作为 V2-B 和
HIP 设计的地址开销基线。

### 4.2 V2-B：block-aligned 遍历

V2-B 将外层循环改为遍历物理 KV block，每个 block 只解析一次 block-table。对于正式
27B `BLOCK_SIZE=784`：

- 每 block 处理 `24 x 32` 个完整 token tile；
- 剩余 `16` token 使用独立 tail 路径；
- 所有 tile 连续更新同一组 `M/L/acc`；
- 完全可见 block/tile 不构造 causal mask；
- 只有最后 diagonal、序列尾和 16-token tail 使用必要 mask；
- 不生成中间 LSE，不跨 kernel merge。

由于 block-aligned 方案会增加每个物理 block 的 tail 处理和 online softmax 更新次数，必须
通过 microbenchmark 判断 block-table/地址收益能否覆盖半 tile 成本。不得仅凭静态分析将
V2-B 设为默认。

为支持 4B 快速门禁，候选同时覆盖当前 guard 允许的 `BLOCK_SIZE=528/544`，但每种
block size 均作为编译期常量生成独立实例，不加入运行时宽泛分支。

### 4.3 Triton 资源布局原型

只有 V2-A/V2-B 未达到门槛且 profile 仍显示明显资源受限时，才进入本阶段：

- 扫描 gfx936 可用的 `waves_per_eu`、MFMA non-K 维度、k-pack 等编译选项；
- 缩短 `S/P/alpha` 等中间值的 live range；
- 检查 `FP32 acc[32,256]` 是否发生寄存器溢出或不利的 LDS 映射；
- 尝试更适合 27B `GQA=6` 的 query/head 到 wave 映射；
- 保持单 kernel online softmax，不引入跨 kernel partial state。

若收益低于 `5%`，或者 rocprof 证明主要限制来自 Triton 无法控制的 accumulator、wave 或
LDS 布局，停止继续扫描 Triton 常量，升级 HIP/MFMA 原型。

## 5. gfx936 HIP/MFMA 设计边界

HIP 第一版严格限定正式 27B 文本 causal prefill：

- gfx936；
- BF16 query/KV/output；
- `head_dim=256`；
- `24` 个 query heads、`4` 个 KV heads、`GQA=6`；
- `BLOCK_SIZE=784`；
- 不支持 alibi、sinks、softcap、qq-bias、multimodal prefix、sliding window 和 FP8；
- 不支持形态无条件回退当前 Triton 专用 kernel。

核心数据流：

1. 一个工作组处理一个 query block 与一个 KV head；
2. 外层按物理 KV block 遍历，block-table 每 block 解析一次；
3. K/V 分阶段加载至 LDS，评估双缓冲隐藏加载延迟；
4. QK 与 PV 使用 gfx936 MFMA；
5. accumulator 分布到多个 wave，避免单个逻辑 program 持有完整
   `FP32 acc[32,256]`；
6. 在同一 kernel 内维护 online softmax 的 `M/L/acc`；
7. full-prefix 使用无 mask 快路径，diagonal 和 tail 单独处理；
8. epilogue 归一化并按原 layout 写回 BF16 output。

第一版不修改 KV Cache layout，不融合 Q/K RMSNorm、RoPE 或 cache update，不加入额外
workspace。只有独立 UA2D 已证明收益后，才评估更大范围融合。

## 6. 验证漏斗

### 6.1 阶段 A：合成 27B kernel 门禁

不加载模型权重，直接构造正式 UA2D 张量、block table、sequence metadata 和 KV Cache。
核心矩阵：

| 维度 | 覆盖范围 |
| --- | --- |
| query heads / KV heads / GQA | `24 / 4 / 6` |
| head dim | `256` |
| block size | `784` |
| query length | `1`、非整块、小 query、约 `4096` |
| KV/context | 约 `8K/16K/24K/32K`，含 block 边界前后 |
| sequence count | 正式单序列为主，补充多序列正确性 |

候选与 `c274c6b` 基线在同一进程交替运行，统一预热，记录中位数、尾延迟和波动。正确性
覆盖：

- block 边界前后和跨 block tile；
- 16-token tail；
- 非整块 query；
- 27B GQA=6 padded row；
- 重复执行确定性；
- 与当前 kernel 的逐元素误差。

rocprof 至少记录：

- kernel latency；
- VGPR、LDS；
- LDS bank conflict；
- L2 hit/fetch；
- wave 数；
- 工具可提供时记录 occupancy、spill 和相关 MFMA 指标。

晋级条件：长档代表形状至少提升 `8%`，中档回退不超过 `2%`，多个长度方向一致，正确性
和确定性通过。

### 6.2 阶段 B：4B 服务完整 A/B

只有合成 27B 晋级的候选才进入 4B 服务验证：

- 三档各至少两轮热态测试；
- 同一基线/候选口径比较 request throughput、output throughput、TTFT 和 TPOT；
- 完成率必须 `100%`；
- 检查输入 token、输出 token、逐样本文本和输出长度；
- 完成 hotpotqa、gov_report、retrieval、aggregation 四类完整精度门禁；
- 中长档 request throughput 不得出现可重复回退，至少一档应有明确提升。

若 output throughput 上升但 request throughput、TTFT 和 kernel micro 不支持该结论，按
输出长度变化处理，不视为有效性能收益。

### 6.3 阶段 C：每轮一次 27B 完整启动

共享存储加载 27B 通常超过 30 分钟，因此每轮结构优化最多安排一次完整 27B 服务启动。
在此之前，所有候选必须通过合成 27B 和 4B 门禁。

27B 候选服务启动后，在同一进程中连续完成：

1. 编译和服务预热；
2. `8K-16K` 多轮热态测试；
3. `16K-32K` 多轮热态测试；
4. `4K-8K` 回归测试；
5. 必要的逐样本输出和服务日志检查；
6. 保存完整环境、提交哈希、启动日志和 benchmark 结果。

27B 端到端不为现场双启动 A/B 支付第二次模型加载成本。对照证据来自：

- 合成 27B 的严格 kernel A/B；
- 4B 的严格服务 A/B；
- `c274c6b` 的历史 27B 本地结果；
- 当前榜单 `16K-32K=11.03`、`4K-8K=16.72`、`8K-16K=14.75`。

若 27B 结果受外部并发、代理、冷编译或显存残留污染，该轮判为无效，不据此回滚候选，
也不立即支付第二次启动成本；等待下一次资源窗口重新安排。

## 7. 晋级、停止与回滚规则

### 7.1 Triton 晋级规则

- kernel 收益 `>=8%`：进入 4B 和一次性 27B 门禁；
- 收益 `5%-8%` 且资源指标明确改善：允许一轮组合优化；
- 收益 `<5%` 或受 Triton 代码生成限制：停止微调，评估 HIP；
- 任一重要中档形状稳定回退超过 `2%`：不晋级，除非新 dispatch 能严格隔离该形状。

### 7.2 HIP 晋级规则

- 合成 27B kernel 收益至少约 `12%`；
- 无 VM fault、越界、非法地址、非确定性或异常资源占用；
- 满足既有误差容限；
- 推算正式长档 request throughput 至少有约 `3%` 的兑现空间。

### 7.3 停止继续投入 UA2D

满足任一条件即停止本轮 UA2D 深挖并转向 Decode/UA3D/GDN：

- Triton 与 HIP 原型均无法达到 kernel `8%` 收益；
- kernel 有收益但端到端无法稳定兑现 `2%`；
- 收益主要来自输出长度或精度变化；
- 无法维持平台精度零扣分目标；
- 最新 27B profile 显示 UA2D 总占比已不足以支撑投入产出。

### 7.4 回滚与构建安全

- 当前 `c274c6b` 专用 UA2D 始终保留为 fallback；
- 候选使用独立 kernel 名称和严格 guard；
- Triton 失败只删除候选，不修改通用路径；
- HIP 构建失败不得覆盖当前源码树可用扩展；
- HIP/C++ 改动遵循根目录增量编译指南，必要时完成全量构建；
- 默认路径只在正确性、性能、精度和 27B 门禁通过后切换。

## 8. 实验记录与交付

每个候选必须保存：

- 候选编号、源码提交哈希和完整环境变量；
- 合成 27B micro 原始数据与统计结果；
- rocprof 原始输出和指标摘要；
- correctness、误差和确定性结果；
- 4B 三档、精度与输出对照；
- 唯一一次 27B 启动的日志和热态测试结果；
- 保留、组合、升级或淘汰的明确结论。

每次源码优化及测试结果按时间戳简记至 `docs/progress.md`。失败路线同样记录原因，防止
重复进入已淘汰的搜索空间。最终提交物仍仅包含修改后的 `vllm_cscc` 源码。

## 9. 推荐执行顺序

1. 固化 `c274c6b` 合成 27B baseline 与 rocprof；
2. 实现并验证 V2-A；
3. V2-A 未达到门槛时实现 V2-B；
4. 根据资源指标决定是否进行一轮 Triton 布局/编译参数原型；
5. Triton 证据指向编译器或资源布局上限时，编写 HIP/MFMA micro 原型；
6. 通过合成 27B 门禁的唯一候选进入 4B 完整 A/B 和精度测试；
7. 每轮只为最终候选启动一次 27B，连续完成全部热态验证；
8. 根据门禁结果保留、回滚或提交榜单。
