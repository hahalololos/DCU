# 工作进度

仅记录 vLLM 源码改动、验证结果和结论；实验原始数据见远端
`testdata/experiments/`，实施细节见 `docs/plans/`。

## 当前有效状态（2026-07-11）

- 已保留：Qwen3.5 UA2D fastpath（`BLOCK_M=32`、`TILE_SIZE_PREFILL=64`、
  `num_warps=4`、`num_stages=1`）和 block-table 标量化；保留总开关
  `VLLM_ROCM_QWEN_UA2D_FASTPATH` 及标量化回退开关。
- 已保留：gfx936 Skinny GEMM 稳定性修复，gfx936 仅对合规 N=1 形状使用
  `LLMM1`，其余回退 `F.linear`。
- 待验证候选：`49aa9d2` 的 LLMM1 形状过滤，默认关闭；需完成同口径 4B 精度、
  重复性能门禁后决定是否保留。
- 已淘汰：GDN 双 GEMV 融合（`LLMM1Fused2`）、UA3D block-table 标量化、UAST-2、
  UA2D 参数扫描/调试开关；对应源码已移除。
- 远端已同步当前关键文件并通过 `py_compile`、源码优先导入；UA2D 定向回归
  `6 passed`、Skinny GEMM fallback `18 passed`。

## 2026-07-11

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
- F1中档首轮P99 TPOT `12.30 ms`，但存在首次长度编译造成的TTFT尾部，需补热轮。F1长档
  期间队友8001服务及50条benchmark启动，显存总占用89%，P99 TPOT异常为 `22.89 ms`；
  该长档明确作废。已停止本工作区8002服务，不影响队友测试；下一窗口补中档热轮和长档。

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
