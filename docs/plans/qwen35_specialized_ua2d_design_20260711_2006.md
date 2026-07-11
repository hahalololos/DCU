# Qwen3.5 专用 UA2D 内核设计（2026-07-11 20:06）

## 1. 背景与目标

最新 4B profile 显示，`kernel_unified_attention_2d`（UA2D）在 `4–8K`、
`8–16K`、`16–32K` prefill 的纯 GPU kernel 时间占比分别为
`28.4%`、`37.9%`、`58.1%`。长上下文代表形态 `q=4096/kv=22258` 的
rocprof 结果为 `224 VGPR`、`32 KB LDS`、L2 命中约 `97.3%`，并记录约
`1.577e9` 次 LDS bank conflict。瓶颈主要位于 kernel 内部的寄存器、LDS、矩阵计算和
数据布局，而非 HBM 带宽。

本轮目标是在不修改模型、推理语义、比赛脚本和锁定参数的前提下，新增 Qwen3.5 文本
full-attention 专用 UA2D kernel，降低长 prefix 下的通用控制与 causal mask 开销，并在
满足正确性和性能门禁时提升端到端吞吐率。

## 2. 验收标准

第一阶段以 4B 为快速门禁：

- `16–32K` output throughput 相对当前提交基线提升至少 `5%`；
- `4–8K`、`8–16K` output throughput 回退均不超过 `3%`；
- 三档完成率均为 `100%`，TTFT/TPOT 不出现异常回退；
- kernel 数值、边界形态和回退路径测试通过；
- profile 或 micro 必须证明专用 kernel 本身加速，不能只依赖输出长度随机变化。

4B 通过后才进行 27B 验证。27B 是最终形状依据，至少要求完成率和正确性无回退、吞吐不
回退；若 27B 未证明收益，则不将该路径作为 27B 默认实现。

## 3. 方案选择

采用独立 `kernel_qwen35_unified_attention_2d`，由 `unified_attention()` 在严格 guard
下分流。未命中时无条件使用现有通用 UA2D/UA3D 路径。

不采用以下方案：

- 继续在通用 kernel 中增加宽泛分支：难以有效降低编译后的 VGPR/LDS 压力；
- prefix/suffix 多 kernel 加 LSE merge：此前 UAST-2 micro 仅达到现有 UA2D 的
  `0.146x–0.282x`，且增加 launch、临时缓冲与状态合并成本；
- 修改 KV cache 布局或 scheduler：超出本轮范围且风险高。

## 4. 专用路径 guard

专用 kernel 只覆盖已观测到的 Qwen3.5 文本 causal prefill：

| 条件 | 要求 |
| --- | --- |
| 平台 | ROCm，gfx936 |
| dtype | query、KV cache、output 均为 BF16 |
| head size | `256` |
| KV heads | `4` |
| 4B | query heads `16`，GQA ratio `4` |
| 27B | query heads `24`，GQA ratio `6` |
| attention | causal，`max_seqlen_q > 1`，进入 2D 路径 |
| 不支持特性 | alibi、sinks、softcap、qq-bias、mm-prefix、sliding-window、FP8 |
| KV cache | 保持现有 paged cache 和实际 block size，不改变管理语义 |

guard 只依据调用参数和平台能力，不读取模型名称，不缓存测试集信息。任何条件不满足均回退
现有通用实现。

## 5. 内核结构与数据流

### 5.1 调度和索引

- 保持当前二维 grid：第一维为全局 query block，第二维为 KV head；
- 保持 `query_start_len_ptr` 和 `find_seq_idx` 的多序列映射；
- 保持 Qwen3.5 已验证有效的 scalar block-table 加载方式；
- 保留 GQA ratio 不能整除 power-of-two `BLOCK_M` 时的 padded-row mask，防止 27B 相邻
  query block 重叠写回。

### 5.2 单 kernel 内部分段

每个 query block 只维护一组在线 softmax 状态 `M/L/acc`，KV tile 循环分为两段：

1. 完全可见 prefix：key tile 的最大位置不超过该 query block 中最早有效 query 的绝对
   位置。该段无需构造或应用逐元素 causal mask，只保留 KV 尾界和 query 有效行 mask。
2. diagonal/partial：从第一个不能被全部 query 行看见的 tile 开始，到当前 query block
   的最大可见 key 位置为止。该段使用原有 causal mask 和尾块 mask。

两段连续更新相同的 `M/L/acc`，不写临时 LSE、不启动第二个 kernel，也不进行跨 kernel
merge。最终统一执行 `acc / L` 并按现有 stride 写回 output。

### 5.3 专用化内容

专用 kernel 不携带以下通用逻辑：

- alibi slope 和 alibi sqrt；
- attention sinks；
- softcap；
- query-query bias；
- multimodal prefix ranges；
- sliding-window；
- FP8 scale、clamp 和输出转换。

Q/K/V 均按 BF16 计算，softmax 累积状态保持 FP32，数值语义与现有 BF16 Triton 路径一致。

### 5.4 资源压力控制

第一版沿用已验证的 `TILE_SIZE=64`、`BLOCK_M=32`、scalar block table 作为基准，但参数
必须通过远端 micro/profile 重新确认。重点比较：

- `TILE_SIZE=32/64`；
- `BLOCK_M=16/32`，其中 27B 必须验证 padded-row 正确性；
- `num_warps=4/8`；
- ROCm Triton 可用的 stage 配置。

选择依据依次为正确性、单 kernel 延迟、VGPR/LDS/bank conflict、4B 端到端吞吐；不得只按
单一计数器或单次 micro 结果固化参数。

## 6. 错误处理与回退

- Python 分流 guard 是唯一入口；不支持形态不启动专用 kernel；
- 专用 kernel 不包含运行时静默降级或近似计算；
- 若出现编译失败、数值越界、VM fault、非确定性或性能门禁失败，回滚专用分流，保留当前
  已验证的通用 fastpath；
- 调试统计如确有需要必须由环境变量控制、默认关闭，并在固化前删除或保留为无输出状态。

## 7. 测试设计

### 7.1 本地静态门禁

- Python 语法编译；
- `git diff --check`；
- guard 的纯 Python 单元测试，确认命中和回退边界。

### 7.2 远端 kernel 正确性

在加载规定环境后运行定向 attention 测试，专用输出与现有通用 UA2D 对照。覆盖：

- 4B `(16, 4, 256)` 与 27B `(24, 4, 256)`；
- 单序列和多序列；
- query 长度 `1`、小于一个 query block、非整块、`4096`；
- context 为零、短 prefix、长 prefix；
- KV tile 跨 cache block、末尾 partial tile；
- 27B GQA=6 对 `BLOCK_M` padding 的相邻 block 写回安全；
- 所有不支持特性和 dtype 均验证回退通用路径。

### 7.3 性能验证

1. 用 profile 报告的实际 4B 代表形态对 generic/specialized 做交替 micro，包含预热和多轮
   统计；
2. 对最优候选采集 rocprof，比较延迟、VGPR、LDS 和 bank conflict；
3. 关闭 profiler，使用官方 4B 三档脚本各跑 10 条可重复 A/B；
4. 4B 满足验收标准后，再按需启动 27B 三档短测；
5. 所有端到端结论同时检查完成率、输入长度、输出 token、TTFT、TPOT 和服务日志，排除
   冷编译、并发或代理错误。

## 8. 交付与记录

- 源码改动仅位于 `vllm_cscc`；
- 设计和实施计划位于 `docs/plans/`；
- 每轮 vLLM 源码实验、构建和测试结果按时间戳追加到 `docs/progress.md`；
- 只有 4B 和必要的 27B 证据证明正确且有收益时，才保留并提交专用 kernel。
