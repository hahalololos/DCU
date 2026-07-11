# 工作进度

仅记录 vLLM 源码改动、验证结果和结论；实验原始数据见远端
`testdata/experiments/`，实施细节见 `docs/plans/`。

## 2026-07-11

### LLMM1 形状过滤榜单实测

- 第4次提交启用 LLMM1 shape filter：榜单三档吞吐为 `16-32K=9.68`、`4-8K=16.65`、
  `8-16K=15.06`，总分 `77.8753`，精度扣分 `1.1859`，SLA扣分为0。
- 相对第3次提交（`10.05/15.84/14.22`），长/短/中档吞吐分别变化
  `-3.68%/+5.11%/+5.91%`；总分 `76.6248 -> 77.8753`，净增 `1.2505`；精度扣分
  `1.1669 -> 1.1859`，增加 `0.0190`。排名 `56 -> 57` 受其他队伍提交影响，不作为
  本优化自身回退依据。
- 线上结果确认 shape filter 对短、中档及最终总分有正收益，但长档存在负收益且精度代价
  增加；与本地4B发现的 gov_report 约 `-1.0618%` 方向一致。提交 `39c4654` 已将过滤默认
  开启；下一步按 Linear 形状/模块分组拆分过滤，争取保留短中档收益并修复长档与精度回退。

### 4B LLMM1 形状过滤继续验证

- 因队友8001服务阻塞27B，改用本工作区8002、`gpu-memory-utilization=0.45` 继续4B筛选。
  将官方4B目录原样复制到 `/tmp/qwen35_4b_model_20260711`，8.8 GB复制约70秒；源目录与
  本地副本逐文件 SHA256 完全一致。本地模型权重加载为 `1.78 s`，F0/F1使用独立编译缓存。
- F0当前默认（UA2D fastpath/scalar on、LLMM1 shape filter off）实验目录为
  `testdata/experiments/GAIN4B-F0-CURRENT_20260711_1349`。三档均10/10；热基线：4-8K
  output `67.61 tok/s`、P99 TPOT `12.93 ms`；8-16K output `49.89 tok/s`、P99 TPOT
  `13.55 ms`；16-32K output `29.56 tok/s`、P99 TPOT `14.30 ms`。
- F1仅开启 shape filter，实验目录为
  `testdata/experiments/GAIN4B-F1-SHAPEFILTER_20260711_1421`。有效短档10/10：output
  `72.98 tok/s`，相对F0约 `+7.94%`；P99 TPOT `11.81 ms`，约 `-8.66%`。生成token
  `2617 -> 2571`，仍需正式精度门禁。
- F1补测有效热轮：8-16K output `50.60 tok/s`、P99 TPOT `12.50 ms`，相对F0分别约
  `+1.42%/-7.75%`；16-32K output `30.42 tok/s`、P99 TPOT `13.30 ms`，分别约
  `+2.91%/-6.99%`。此前队友benchmark并发期间的长档P99 TPOT `22.89 ms`已明确作废。
- 三档逐样本文本F0/F1完全一致为 `6/10、7/10、9/10`；输入长度全部一致，长档输出长度
  `10/10`一致。使用同一109条数据、同一4B精度脚本副本完成严格A/B：F0为
  `hotpotqa=67.3853、gov_report=33.2009、retrieval=100、aggregation=96.67`；F1为
  `67.3853、32.8483、100、96.67`。gov_report相对下降 `1.0618%`，越过1%免扣阈值，
  对应任务系数将为0.97、四类平均精度系数约0.9925；因此shape filter精度门禁失败，保持
  默认关闭，不升级27B。当前转入UA2D fastpath off/on对照。

### UA2D 清理与 27B 准备

- 固化 UA2D 最优参数并删除阶段 0 形态日志、宽泛 guard、调试开关和无效参数扫描代码，
  净减少约 264 行；本地 `py_compile`、`git diff --check` 通过。
- 27B 定向矩阵收窄为保留的 B32 vector/scalar 路径；远端回归通过，无 NaN/Inf、
  VM fault 或随机失败。
- 新增执行计划 `docs/plans/qwen35_27b_gain_execution_plan_20260711_1250.md`：
  在共享 GPU 空闲后依次完成 27B 基线、LLMM1 A/B、UA2D A/B 与 profile。

### GDN 双 GEMV 融合淘汰

- 实现并构建 `LLMM1Fused2`，仅接入 Qwen3.5 未量化 TP=1、单 token、gfx936 的
  `qkvz + b/a` 路径；远端定向测试 `26 passed`，4B/27B BF16/FP16 micro 全部
  bitwise 一致。
- micro 最优收益：4B 约 `1.09x`、27B 约 `1.06x`；4B 两轮端到端 P99 TPOT 仅
  改善约 `0.1%--0.6%`，未达门槛。
- 已完整回滚 F1 源码；当前工作树仅保留已提交 ALT-C `49aa9d2`。

## 2026-07-10

### LLMM1 形状过滤候选（ALT-C）

- 基于 4B/27B 实际 decode Linear micro，仅保留 LLMM1 收益 `>=5%` 的形状，其他形状
  回退 `F.linear`；新增默认关闭开关
  `VLLM_ROCM_GFX936_LLMM1_SHAPE_FILTER` 和定向测试。
- 4B 初筛相对新鲜 baseline 的 P99 TPOT：`4-8K/-9.35%`、`8-16K/-6.45%`、
  `16-32K/-8.47%`；但生成 token 数存在差异，尚未完成正式精度门禁，不作为有效收益。
- 候选已独立提交 `49aa9d2`，默认关闭，等待完整精度及重复 A/B 验证。

### Attention 候选淘汰

- UAST-2（paged prefix + contiguous suffix + LSE merge）数值/确定性通过
  （UAST-2 `4 passed`、UA2D 回归 `10 passed`），但 4B `q=4096` micro 仅为现有
  UA2D 的 `0.146x--0.282x`，已删除全部源码、测试和开关。
- UA3D block-table 标量化数值测试 `2 passed`；4B/27B micro 最大收益不足 `1%`，
  已淘汰。

### Decode/Linear 初筛

- D0（Skinny/LLMM1 on）4B 三档 output throughput 为
  `67.54/41.84/29.44 tok/s`，P99 TPOT 为 `13.08/13.63/14.43 ms`。
- 关闭 Skinny/LLMM1 的收益未达 `3%`，且文本/生成 token 有差异；HipBLASLt preference
  候选吞吐下降 `32%--48%`、TPOT 增加约 `94%--103%`，均未保留。
- 新增 Qwen3.5-27B UA2D 正确性矩阵与长 query 测试：`CORR-K0~K3 8 passed`、
  长 query `2 passed`，输出逐元素一致。

## 2026-07-09

### Qwen3.5 UA2D fastpath

- 为 Qwen3.5 BF16、head=256、GQA、causal 的实际形状加入严格 UA2D fastpath；未命中时
  回退原路径。初始配置为 `TILE=64/BLOCK_M=16/warps=4/stages=1`。
- 4B 三档各 50 条完整对照：output throughput 分别提升
  `+9.30%/+26.39%/+72.90%`；合并 P99 TPOT `32.53 -> 16.90 ms`，150/150 成功。
- 后续 block-table 标量化与实际形态 micro 表明 B32 更优：4B `q=4096/seq=32768`
  kernel B32 scalar 相对 B16 vector 提升约 `109.82%`，输出 max abs diff 为 `0`。
- B32 scalar 的 4B 端到端三档 output throughput 为 `67.71/41.58/29.50 tok/s`；
  长档相对 B16 baseline 约 `+32%`，P99 TPOT 基本持平。已将默认 `BLOCK_M` 固化为
  `32`、标量化默认设为开启。

### 27B 启动尝试

- 27B UA2D fastpath F0 因共享存储读取 safetensors 持续异常慢，未得到吞吐数据；
  已停止本工作区服务并释放 GPU/端口。后续仅在共享 GPU 空闲时继续。

### gfx936 Skinny GEMM 稳定性修复

- 在 `skinny_gemms.cu` 增加 gfx936 支持，并将不支持的 `v_dot2c_f32_f16` 替换为
  half2/float2 fallback；全量构建成功，源码树 `_rocm_C.abi3.so` 已更新。
- gfx936 `wvSplitK` 在预热会触发 HSA VM fault 或数值不达标；dispatch 收窄为仅允许
  合规 `LLMM1`，否则回退 linear。fallback 测试最初 `3 passed`，后续扩展为 `18 passed`。
- 4B `16-32K` 修复后 `2672.11 tok/s`、P99 TPOT `14.55 ms`，与关闭 Skinny 的
  `2682.23 tok/s`、`14.03 ms` 基本持平；该改动仅作为稳定性修复。

## 2026-07-08

- 新增 gfx936 平台识别 `on_gfx936()`；远端确认为 `gfx936/BW`，不属于既有 `_ON_GFX9`。
- 恢复 HIP 下 `_rocm_C` 构建入口；确认 `env.sh` 下源码优先导入要求源码树存在匹配的
  `_rocm_C.abi3.so`。
