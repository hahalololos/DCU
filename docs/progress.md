# 工作进度

仅记录 vLLM 源码改动、验证结果和结论；实验原始数据见远端
`testdata/experiments/`，实施细节见 `docs/plans/`。

## 2026-07-12

### Qwen3.5 UA2D 失效开关清理

- 删除已不影响专用 kernel 执行路径的
  `VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE` 环境变量、dispatch 临时状态及真假测试参数化。
  专用 Qwen3.5 UA2D 仍固定使用已验证的标量 block-table 地址计算；通用 UA2D 明确保持
  `SCALAR_BLOCK_TABLE=False`，计算行为、guard 和 `TILE32/BLOCK_M32/warps4/stages1`
  均未改变。
- 新增环境变量注册表删除契约；修改前直接导入检查按预期失败，删除后通过。三份改动文件
  `py_compile`、`git diff --check` 通过，生产代码无旧变量或 `scalar_block_table_2d` 残留。
  本地完整 pytest 仍受缺少 `tblib` 限制，GPU 定向回归待远端执行。
- vLLM 源码提交：`515ec50`。

### Qwen3.5 专用 UA2D 最终取舍

- 完成无外部并发的 TILE32 4B 热态复测。相对
  `UA2D-SPECIAL-BASE-HOT2_20260711_2214`：短档 request throughput 约 `-1.85%`、
  P99 TTFT `-2.44%`、P99 TPOT `+0.27%`；中档分别约 `+4.65%/-6.34%/-0.15%`；
  长档有效轮 request throughput 约 `+4.8%`、P99 TTFT约 `-6.9%`、P99 TPOT基本持平。
  三档均 `10/10` 完成；短中档目录为
  `UA2D-SPECIAL-CAND-SHORT-HOT1_20260712_0015`、
  `UA2D-SPECIAL-CAND-MID-HOT4_20260712_0012`，长档目录为
  `UA2D-SPECIAL-CAND-CLEANREPEAT_20260711_2343`。
- 同一109条4B精度重放中，TILE32相对F0基线：hotpotqa `67.3853 -> 67.3853`、
  gov_report `33.2009 -> 32.5359`（约 `-2.00%`）、retrieval `100 -> 100`、
  aggregation `96.67 -> 96.67`。TILE64将gov_report改善至 `32.7807`（约 `-1.27%`），
  但热点micro约 `49.74 ms`，慢于TILE32约 `44.8 ms`。
- 用户明确选择吞吐优先并接受上述小幅局部精度回退，因此最终保留
  `TILE_SIZE=32/BLOCK_M=32/warps=4/stages=1`。热点 `q=4096/kv=22258` 相对原
  fastpath约 `52.0 ms` 的单kernel加速约 `14.1%`。最终源码恢复至提交
  `882f326` 的已验证状态，远端定向Qwen3.5/27B测试 `5 passed`，本地与远端关键文件
  SHA256一致。

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
  对应任务系数将为0.97、四类平均精度系数约0.9925；因此shape filter未通过本地精度门禁，
  当时保持默认关闭且不升级27B；后续为验证榜单净收益由提交 `39c4654` 另行默认启用。
- S1按形状拆分：仅将4B full-attention output projection `(M=2560,K=4096)` 恢复到
  LLMM1，其余过滤规则不变；远端定向回归 `18 passed`。实验目录为
  `testdata/experiments/GAIN4B-S1-ATTNOUT_20260711_1702`。
- 相对完整 shape filter 的F1热基线，S1短档 output throughput `-3.36%`、P99 TPOT
  `+4.41%`；长档分别 `-0.64%/+2.87%`。8-16K补热轮后 request throughput因输出更短
  为 `+4.16%`，但 output throughput `-6.51%`、P99 TPOT `+3.35%`，TTFT基本不变。
  该形状无性能收益，未进入完整精度测试，候选源码与测试已清理，本地和远端均恢复
  已推送版本。

### 4B Profile 驱动优化

- 暂停继续尝试 S2/S3 GEMM 形状，改为在当前提交 `39c4654` 上分离采集三档 prefill 与
  稳态 decode；执行计划见 `docs/plans/qwen35_4b_profile_plan_20260711_1727.md`。
- 已确认本地与远端关键源码 SHA256 一致；远端 Torch Profiler 可直接使用，DTK 另提供
  未加入 `PATH` 的 `rocprof/rocprofv2`，可在确定 Top kernels 后采集硬件计数器。
- 远端已有队友的 4B profiler 服务占用8001和45%显存，其 benchmark 进程长期无GPU活动
  但仍未退出。按4B可共享测试规则尝试在8002启动本工作区同配置服务；第二个 EngineCore
  在模型加载和重新编译后进入设备等待态，未产生基准数据。已仅停止本工作区8002进程，
  队友服务保持不动，GPU显存恢复到原45%。队友释放后完成正式采集。
- 无 profiler 热基线三档 output throughput 为 `70.52/50.43/30.49 tok/s`，P99 TPOT 为
  `11.88/12.49/13.24 ms`，三档均10/10；中长档与历史热基线一致，短档偏差约 `-3.4%`。
- Torch Profiler 使用同一代表输入的 `output=1/64` 分离 prefill 和63个稳定decode token。
  UA2D 在短/中/长档 prefill 纯kernel时间占比为 `28.4%/37.9%/58.1%`，是随上下文增长的
  最大单一热点。
- 稳态decode纯kernel时间为 `10.77/11.10/12.25 ms/token`。当前允许形状的LLMM1占
  `28.9%/28.1%/25.3%`，其他GEMM合计约 `46.0%/44.6%/40.3%`；UA3D占比随上下文从
  `6.3%` 增至 `18.1%`。
- rocprof确认：长上下文UA2D为 `224 VGPR/32KB LDS/L2 hit 97.3%`，LDS bank-conflict计数
  约 `1.577e9`，不是HBM带宽瓶颈；UA3D为 `248 VGPR/L2 hit约1%`，受KV读取影响；LM head
  LLMM1约 `1.278 ms`、L2 hit `3.6%`、单次读取约1213MiB，接近带宽受限。
- 结论：下一源码优化优先做UA2D单kernel内部 full-prefix/diagonal 分段并降低LDS/VGPR压力；
  第二优先级为UA3D segment/KV load。停止S2/S3盲目恢复形状。完整报告见
  `调研交付物/qwen35_4b_profile_20260711.md`。

### Qwen3.5 专用 UA2D 实现与 micro 初筛（21:35）

- 在隔离分支 `perf/qwen35-specialized-ua2d` 新增独立
  `kernel_qwen35_unified_attention_2d`，保留 paged KV、多序列映射和单 kernel 在线
  softmax；KV 循环拆为无需逐元素 causal mask 的 full-prefix 与保留 mask 的
  diagonal/partial 两段。实现提交为 `5bf55cb`。
- 专用路径 guard 收紧为仅 gfx936、Q/K/V/output 全 BF16、head size 256、4 KV heads、
  4B `(16,4,4)` 或 27B `(24,4,6)` causal prefill，且关闭 alibi、sinks、softcap、
  qq-bias、mm-prefix、sliding-window 和 FP8；测试提交为 `882f326`。
- 远端因 pytest 环境缺少 `tblib`，未修改共享虚拟环境；改为直接加载同一测试模块并调用
  测试函数。4B/27B、scalar on/off、cache-block 边界、非整块 query、GQA=6 padding、
  `4095/4096` 长 query 共 12 组全部通过，重复输出 bitwise 一致，specialized 与 generic
  在 `atol=1.5e-2/rtol=1e-2` 内一致。
- 最新 profile 代表形态 `4B q=4096/kv=22258`：当前正式 fastpath 基线两轮为
  `52.392/51.956 ms`；专用 kernel 最优 `TILE=32/BLOCK_M=32/warps=4/stages=1` 为
  `44.867/44.749 ms`，平均单 kernel 加速约 `14.1%`。`BLOCK_M=16` 与 8 warps 分别约
  `99.61/94.58 ms`，均已淘汰。
- 计划进行 4B 三档端到端 A/B 时，8001 被队友的 4B profiler 服务占用，配置包含
  `max-num-batched-tokens=8192`、FP8 KV 和 45% 显存，不能作为本轮对照；此前同卡再启动
  第二个 45% 服务会卡住，因此未干扰队友。待 GPU 释放后使用官方 4B 脚本完成正式门禁。
- 21:38 检测到 GPU 短暂释放后尝试启动本工作区 baseline；模型已加载，但队友 profiler
  服务在 KV cache 初始化期间重新占用 45% 显存，本工作区因 `No available memory for the
  cache blocks` 自动退出。未停止或修改队友进程，远端源码已立即恢复为专用 kernel 候选。
- 后续窗口完成原 UA2D fastpath 的 4B 热基线，实验目录为
  `testdata/experiments/UA2D-SPECIAL-BASE-HOT2_20260711_2214`。短/中/长档均 10/10，output
  throughput 为 `73.83/50.73/30.57 tok/s`，P99 TPOT 为
  `11.80/12.46/13.19 ms`；输入/输出 token 分别为
  `62196/2571、134349/1522、212553/1114`。首轮中档包含一次冷编译长尾，已排除并使用
  同服务热轮作为正式基线。切换候选时队友 profiler 再次启动，候选三档等待资源释放。
- 候选后续获得一次三档窗口，预热轮目录为
  `testdata/experiments/UA2D-SPECIAL-CAND-WARM1_20260711_2244`，热轮目录为
  `testdata/experiments/UA2D-SPECIAL-CAND-HOT2_20260711_2249`。热轮短/中/长 output 为
  `73.68/44.78/26.57 tok/s`，中长档 P99 TPOT 异常升至约 `19.8 ms`，未通过门禁。
- 系统化排查发现候选窗口附近队友 profiler 服务反复启动；随后在双方进程均退出时重启
  候选，模型加载显存由正常约 `8.71 GiB` 变为 `17.43 GiB`，KV cache 可用显存为
  `-1.91 GiB` 并自动退出，直接证明存在并发占用。由于专用 UA2D 不命中单 token decode，
  热轮 TPOT 回退不能直接归因于该 kernel，候选数据视为受污染，等待独占窗口复测。
- 逐样本比较 baseline/candidate 热轮：三档文本完全一致均为 `7/10`，输出长度一致分别为
  `8/10、7/10、8/10`；候选中档逐样本 TTFT 多数下降约 `5%--8%`，说明 prefill 路径确有
  收益，但 TILE32 改变累积顺序导致生成存在数值敏感性，后续还需结合精度门禁决定是否
  保留 TILE32，或退回数值更保守的 TILE64 专用路径。
- 两次尝试在 GPU 显示空闲后进行无并发复测，均在本工作区服务初始化或 benchmark warmup
  期间被队友 profiler 服务重新占用；一次出现 KV cache 可用显存 `-1.91 GiB`，另一次
  中档 warmup 长时间停在首请求。均已只停止本工作区进程，未产生有效结果。确认当前无法
  通过短暂空闲窗口完成可靠 A/B，需要与队友协调独占测试时段。

### 4B UA2D fastpath 贡献复核

- PRA26新作业恢复后，将远端残留的临时 `envs.py` 同步为已推送提交 `39c4654`，在
  LLMM1 shape filter 开启的当前提交口径下，仅设置
  `VLLM_ROCM_QWEN_UA2D_FASTPATH=0`。共享模型读取阻塞后改用官方4B的容器本地同字节副本；
  实验目录为 `testdata/experiments/GAIN4B-U0-UA2DOFF_20260711_1639`，三档均完成10/10。
- UA2D off 的短/中/长档 output throughput 为 `65.05/19.57/12.87 tok/s`，request
  throughput 为 `0.2475/0.1339/0.1173 req/s`，平均 TTFT 为
  `0.877/5.640/7.032 s`，P99 TTFT 为 `1.187/25.743/7.720 s`。
- 相对相同 LLMM1 配置的 UA2D on 热基线，fastpath 使短/中/长档 output throughput 分别
  提升 `+12.20%/+158.56%/+136.26%`，request throughput 提升
  `+14.68%/+148.36%/+132.86%`，平均 TTFT 降低 `47.37%/80.30%/69.57%`，P99 TTFT
  降低 `43.47%/94.56%/70.35%`。P99 TPOT 仅变化 `-1.10%/-0.08%/+0.39%`，说明主要
  收益来自 prefill，而非 decode。
- 两组输入长度完全一致；逐样本文本一致 `7/10、8/10、7/10`，输出长度一致
  `8/10、8/10、7/10`，因此 output tok/s 同时受生成长度影响，但 request throughput 与
  TTFT 仍确认 UA2D fastpath 是中长档的决定性正收益。结论：保留当前 UA2D fastpath，
  不再投入 fastpath off 路线；下一步继续拆分 LLMM1 过滤形状。

### UA2D 清理

- 固化 UA2D 最优参数并删除阶段 0 形态日志、宽泛 guard、调试开关和无效参数扫描代码，
  净减少约 264 行；本地 `py_compile`、`git diff --check` 通过。

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
