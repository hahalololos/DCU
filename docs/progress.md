# 工作进度

仅记录 vLLM 源码改动、验证结果和结论；实验原始数据见远端
`testdata/experiments/`，实施细节见 `docs/plans/`。

## 2026-07-13

### Qwen3.5-27B 三档全链路 Profile 与优化优先级

- 以生产提交 `0adf049` 完成 27B 三档各 10 条热基线：4–8K/8–16K/16–32K 的
  output throughput 分别为 `16.4348/12.9019/8.9538 tok/s`，P99 TTFT 为
  `2030.27/4815.13/7491.73 ms`，P99 TPOT 为 `54.61/55.51/56.56 ms`；三档均
  `10/10` 完成、0 失败。中档首轮受冷编译污染，正式值取两次热态中位数，两次偏差
  仅 `0.024%`。
- 从每档选择输入长度 P50/P90 样本，以独立服务采集两轮 `output=1/65` 配对 trace，
  共 24 份。所有配对输入长度和生成前缀一致，输出严格为 1/65 token；最大轮间波动为
  Prefill `0.216%`、Decode `0.210%`。同进程多轮复用 ROCm Torch Profiler 曾在第 4 次
  start/stop 触发 HSA VM fault，改为每个样本独立服务后稳定完成。
- 三档 Prefill 的 GEMM/UA2D 占比分别为 `67.15%/17.17%`、`56.39%/30.02%`、
  `49.90%/38.04%`；稳定 Decode GEMM 约 `49.98--50.00 ms/token`。按比赛
  `20%/50%/30%` 权重，Decode GEMM、UA2D、GDN、UA3D、GPU idle gap 的端到端占比
  约为 `62.57%/10.34%/3.64%/2.69%/2.36%`。
- Decode 精确热点为 LLMM1 `34.333 ms/token、176 次/token` 和其他
  `F.linear`/Tensile `14.642 ms/token、129 次/token`。无计数器 micro 表明 LLMM1
  MLP gate-up 64 次累计约 `23.28 ms/token`，下一步优先做其 kernel 结构优化；随后处理
  MLP down 与 LM head。
- 完成 Top 3 rocprof：UA2D 为 `256 VGPR/16 KiB LDS`，LDS bank conflict 约
  `9.88e8`；UA3D 主核为 `252 VGPR/16.5 KiB LDS`，L2 hit 约 `1.37%`；LLMM1 MLP
  gate-up L2 hit 约 `17.87%`。风险调整后顺序为 Decode GEMM、UA2D LDS/VGPR 结构、
  UA3D KV load/segments；GDN 与 launch gap 暂缓，UA3D merge 仅约 `0.082 ms/token`。
- 完整报告见 `调研交付物/qwen35_27b_profile_20260713.md`；本地小型数据位于
  `testdata/profile_results/qwen35_27b_20260713/`，远端原始 trace 与 rocprof 位于
  `/root/profile27b/`。本轮未修改 `vllm_cscc` 生产源码。

### 27B LLMM1 rows_per_block 形状调优淘汰

- 以 `0adf049` 为生产基线，在同一常驻进程内对四个当前允许的 27B Decode GEMV
  形状扫描 `rows_per_block=2/4/8/16`；每个后端预热 5 次，20 次调用一组、正反序交替
  测量 7 轮。原始结果位于远端
  `testdata/experiments/LLMM1-ROWS-27B_20260713_1710/results.txt`。
- 相对当前 row4，row8 在 GDN qkvz `(16384,5120)`、attention qkv+gate
  `(14336,5120)`、MLP gate-up `(34816,5120)` 的 median 仅改善
  `1.71%/1.38%/3.36%`；GDN b/a `(96,5120)` 的最优仍为 row4，row8 反而回退
  `15.00%`。row2 四形状均回退约 `5.3%--10.1%`，row16 回退约
  `57.6%--63.1%`。
- 所有 row2/4/8/16 输出均与 row4 bitwise 一致，P99 与 median 趋势一致。若忽略单形状
  门禁并为前三个大形状选择 row8，按每 token `48/48/16/64` 次调用折算，四类 LLMM1
  合计仅从 `34.836` 降至 `33.902 ms/token`，理论节省 `0.934 ms/token`，即该子集的
  `2.68%`。
- 三个候选形状均未达到单形状至少 `5%` 的 micro 门槛，且小形状存在明显回退，因此不修改
  `vllm_cscc` 生产 dispatch、不进入 27B 端到端 A/B。保留
  `testdata/profile_hotspots_4b.py llmm1_rows` 作为后续 kernel 结构变化后的复测工具。
- 远端已确认 editable vLLM 指向本项目源码，安装 `tblib 3.2.2` 后相关定向 pytest
  `13 passed`；本地 profile 脚本 `py_compile` 已通过。

### 27B UA2D Triton 编译参数候选淘汰

- 以提交 `0adf049` 的 v2b/TILE32/BLOCK_M32/WARPS2 为基线，仅针对 27B
  `block_size=784` 扫描 AMD Triton 编译参数：`waves_per_eu=1/2/4`，以及叠加
  `matrix_instr_nonkdim=16,kpack=2` 的组合；实验编号为 6--11，默认 experiment 5
  在整个测试期间保持不变。
- 27B 合成 UA2D 在 8K/16K/32K 的 experiment 5 基线为
  `18.4355/40.8380/86.4326 ms`。仅 `waves_per_eu=1` 的 experiment 6 为
  `18.3481/40.8298/86.4340 ms`，相对改善约 `0.47%/0.02%/0%`，输出与基线
  bitwise 一致，但未达到 5% micro 门槛。
- `waves_per_eu=2 + matrix_instr_nonkdim=16 + kpack=2` 的 experiment 10 为
  `18.1329/40.3133/85.7508 ms`，改善约 `1.64%/1.28%/0.79%`，同时相对 experiment 5
  出现最大绝对差 `2.44e-4--4.88e-4`，速度和数值均未通过门禁。
- `waves_per_eu=2` 单独使用稳定回退约 `1.9%--3.4%`；`waves_per_eu=4` 及其 MFMA
  组合回退约 `61%--117%`。因此没有候选进入 27B 服务端到端测试，实验 6--11 的生产
  dispatch、名称和测试已全部删除，vLLM 源码恢复到 `0adf049` 的已验证状态。
- 保留 `testdata/profile_hotspots_4b.py` 的 `--ua2d-experiment` 参数，并将 `--verify`
  对照更新为当前 experiment 5，便于后续复现实验；本地 `py_compile`、轻量契约测试和
  `git diff --check` 通过。完整实验计划见
  `docs/plans/qwen35_27b_ua2d_compiler_tuning_plan_20260713_1629.md`。

### Top 20 冲刺首轮 27B 热点复核与本地模型副本

- 参考早期 27B hipprof 报告确认：短档 GEMM 占约 `78.74%`，中档 GEMM/attention
  分别约 `58.87%/35.58%`，长档 attention 占约 `61.87%`；结合当前版本已完成的
  UA2D/UA3D 优化，先复核 Decode Linear 和当前 Attention micro。
- 扩展 `testdata/profile_hotspots_4b.py`：修正 27B attention block size 为 `784`，并加入
  4B/27B 实际 Linear 形状扫描。27B 上现有 LLMM1 shape filter 方向正确：GDN qkvz、
  GDN b/a、full-attention qkv+gate、MLP gate-up 相对 `F.linear` 分别约
  `1.38x/2.21x/1.53x/1.38x`；attention out 和 LM head 的 LLMM1 分别仅约
  `0.72x/0.75x`，应继续回退 `F.linear`。
- 未覆盖的 27B MLP down `(M=5120,N=1,K=17408)` 的 `F.linear` 基线约
  `0.151 ms`。隔离 Triton dot 原型最佳约 `0.524 ms`，仅为基线 `0.287x`，按门禁淘汰，
  未接入生产源码。
- 当前 27B micro 中，UA3D 在 8K/16K/32K 约 `0.294/0.298/0.308 ms`，scalar
  block-table 与通用路径 bitwise 一致；UA2D 分别约 `21.93/48.52/101.82 ms`，仍是长档
  主热点。32K 下 UA2D 内置 variant 0/1/2/3 约 `106.07/105.37/101.77/106.07 ms`，当前
  默认 v2b（variant 2）继续胜出约 `4%`。
- 将共享存储 27B 模型 `/public/home/acoh0h1o0p/models/Qwen3.5-27B` 使用 rsync 复制至
  容器本地 `/root/models/Qwen3.5-27B`：源模型约 `52G/25` 个文件，本地复制完成后同为
  `52G/25` 个文件。用户明确表示无需等待 SHA256 校验；后续 27B 启动和吞吐测试默认使用
  本地副本，`testdata/start_vllm.sh` 与 `run_throughput.sh` 已更新默认路径。

### 27B UA2D v2b TILE64 候选门禁与淘汰

- 在现有 v2b 算法上新增仅供实验的 27B TILE64 variant，将 `block_size=784` 的 full-tile
  循环从每物理块24次降至12次。27B micro相对同轮TILE32在8K/16K/32K从约
  `21.93/48.48/110.70 ms` 降至 `20.66/45.50/95.64 ms`，分别改善约
  `5.8%/6.2%/13.6%`。
- 27B GQA=6的两组partial-query/cache-block边界测试、variant名称和kernel选择测试通过，
  直接加载测试模块执行结果为 `4 passed`；常规pytest仍受远端缺少`tblib`限制。
- 使用本地27B副本完成中、长档10条正向和反向A/B。热态中档baseline/candidate的
  request throughput约 `0.07048 -> 0.07104 req/s`（`+0.80%`），output throughput
  `12.2850 -> 12.3260 tok/s`（`+0.33%`），P99 TPOT基本持平；未达到中档门槛。
- 长档baseline两次稳定约 `0.06787/0.06790 req/s`、`8.6670/8.6706 tok/s`；候选为
  `0.07395 req/s`（约`+8.9%`），但生成token由`1277`降至`1096`，使output throughput
  降至`8.1051 tok/s`（约`-6.5%`）。输入token均为`212553`，P99 TPOT基本持平。
- 结论：TILE64虽明显缩短prefill并提高request throughput，但改变数值归约顺序后生成轨迹
  变化，且榜单output-throughput口径稳定回退；候选按门禁淘汰，实验variant、选择逻辑和
  正式测试扩展已删除，默认v2b TILE32保持不变。实验数据保存在远端
  `testdata/experiments/TOP20-UA2D-E2-27B`与`TOP20-UA2D-E4-27B`。

### 27B UA2D v2b WARPS2 固化

- 对当前v2b 32K形状采集rocprof：`arch_vgpr=256`、LDS `16384 B`、
  `SQ_LDS_BANK_CONFLICT=1,759,305,952`、L2命中约`97%`，确认kernel受寄存器占用和LDS冲突
  限制。保持TILE32、BLOCK_M32和全部计算顺序不变，仅针对27B `block_size=784` 将workgroup
  从4 warps降至2 warps；4B和其他形状不变，原WARPS4保留为experiment 2回滚路径。
- 27B micro中，WARPS4的8K/16K/32K约`21.89/48.47/101.77 ms`，WARPS2约
  `18.39/40.83/86.58 ms`，分别改善约`16.0%/15.8%/14.9%`。三档WARPS2与WARPS4输出均
  bitwise一致，max abs diff为`0`；GQA=6、`783/784/785`及partial-query边界直接测试
  `4 passed`。
- 使用本地27B副本完成三档10条端到端A/B。短档baseline/candidate output throughput
  `16.3837 -> 16.4272 tok/s`（`+0.27%`），P99 TTFT `2098.92 -> 2029.12 ms`
  （`-3.33%`）；中档热态baseline/candidate `12.2850 -> 12.4682 tok/s`（`+1.49%`），
  P99 TTFT `8127.45 -> 7918.78 ms`（`-2.57%`）；长档`8.6670 -> 8.9573 tok/s`
  （`+3.35%`），P99 TTFT `8019.58 -> 7487.38 ms`（`-6.64%`）。三档P99 TPOT基本持平。
- 三档均`10/10`完成，输入token分别为`62196/134349/212553`，输出token分别为
  `2573/1743/1277`，baseline/candidate逐档完全一致；输入长度、输出长度和逐样本文本均
  `10/10`完全一致。因此将experiment 5（v2b-warps2）设为默认，experiment 2继续提供
  WARPS4回滚。实验数据保存在远端`testdata/experiments/TOP20-UA2D-E5-27B`。
- `/root/models/Qwen3.5-27B`本地副本使权重加载稳定约`9.98--11.87 s`，后续27B测试继续
  使用本地路径。

### UA3D Decode Attention 分段与标量 block-table 初筛

- 审计确认当前 Triton UA3D 已采用固定 16 段并行 softmax，并用第二个 kernel 合并局部
  maximum、exp sum 和 accumulator；因此优化重点从“新增分段”调整为分段数适配及
  Qwen3.5/gfx936 专用访存路径。
- 扩展 `testdata/profile_hotspots_4b.py`，支持 4B/27B 精确 GQA 形状及可配置 segments。
  4B 初筛结果：context=4096 时 segments=4/8/16/32 分别约
  `0.2949/0.2929/0.2950/0.2971 ms`；context=8192 时 segments=4/8 分别约
  `0.4210/0.2941 ms`；context=16384 时 segments=4/8/16 分别约
  `0.8096/0.4232/0.2845 ms`。说明最佳段数随长度增加，16K 仍需要默认 16 段。
- 实现严格限定于 gfx936、BF16、head size 256、Qwen3.5-4B/27B、无额外 bias/window 的
  decode UA3D 标量 block-table 候选：每个 tile 标量计算 logical block，常规情况下只加载
  一个物理 block id，仅跨 528/544/784 token cache block 边界时加载第二项；通用路径保持
  原向量 div/mod 实现。新增 decode-only/模型形状 guard 单元测试。
- 本地 `py_compile` 与 `git diff --check` 通过。远端扫描期间 `scnet-computer` 跳板失联，
  且登录节点出现 SSH host key 变化提示；未自动接受新指纹。候选尚未完成 GPU 数值、
  block 边界与端到端验证，当前不得视为胜出优化。
- 按用户要求先行用于榜单提交，vLLM 源码提交 `99d5cc7` 已推送至 `origin/haha`。该提交
  仍属于未完成 GPU 验证的实验候选；榜单结果返回后必须结合三档吞吐、SLA、精度扣分
  决定保留或回滚。
- 容器恢复后完成 GPU 验证。4B/27B 在 context=527/528/529 的 cache-block 边界及
  8K/16K 长度下，专用路径与通用 UA3D 均 bitwise 一致，max abs diff 为 `0`。
  context=32K 重复反向 A/B 中，4B 通用/候选稳定约 `0.456/0.303 ms`，27B 约
  `0.461/0.301 ms`，UA3D 两-kernel总时间改善约 `33%--35%`；8K/16K 基本持平或有
  `0.5%--1.3%` micro 噪声级回退。
- 4B 长档端到端 baseline/candidate 均 `10/10` 完成，输入 token `212553`、输出 token
  `1096` 完全一致。output throughput `31.82 -> 32.45 tok/s`（`+1.98%`），P99 TPOT
  `13.21 -> 12.59 ms`（`-4.69%`），P99 TTFT `2086.88 -> 2088.49 ms`（基本持平）。
  实验目录为 `UA3D-SCALAR-BASE-LONG_20260713_0954` 和
  `UA3D-SCALAR-CAND-LONG_20260713_1001`。
- 第7次榜单提交相对第6次：长/短/中档吞吐由 `12.08/16.84/14.97` 提升至
  `12.24/16.96/15.05`，分别为 `+1.32%/+0.71%/+0.53%`；最终得分
  `81.1202 -> 81.3964`（`+0.2762`），排名 `65 -> 61`，SLA 与精度扣分仍为0。
  榜单和本地 A/B 同向，决定保留候选，并增加默认开启、可显式关闭的
  `VLLM_ROCM_QWEN_UA3D_SCALAR_BLOCK_TABLE` 回滚开关。

### GDN causal-conv + recurrent 融合候选门禁与淘汰

- 实现隔离原型 `testdata/profile_gdn_fused_decode.py`：主 Triton kernel 直接从 raw QKV、
  width=4 旧 conv state 和卷积权重计算 SiLU、Q/K L2Norm、gating 与 recurrent update，
  第二个轻量 kernel 仅滑动 conv state。不能在原 recurrent grid 内同时更新 conv state，
  否则不同 value tile 会并发读写同一 Q/K state，存在跨 program 竞争。
- 4B/27B 精确形状、HV=32/48、BF16/FP32 recurrent state、有/无 conv bias 的输出、
  recurrent state 和 conv state 均与原两-kernel路径 bitwise 一致。远端 pytest 收集受缺少
  `tblib` 限制，改为直接加载同一测试模块调用 8 组参数化测试，结果 `8 passed`。
- 原型 micro 相对 `causal_conv1d_update + packed recurrent` 两段总时间常见改善约
  `29%--32%`。生产 wrapper 使用实际 BF16 recurrent state 时，BV16 的 4B/27B 改善约
  `25.28%/23.72%`，BV32 改善约 `25.82%/28.27%`；同时确认默认
  `mamba_ssm_cache_dtype=auto` 下 recurrent state 实际跟随 conv cache 为 BF16，而非最初
  假设的 FP32。
- 4B 正式热态基线为中档 `53.4238 tok/s、P99 TPOT 12.3121 ms`，长档
  `31.8838 tok/s、13.0693 ms`。BV16 候选中档为 `53.1766 tok/s、12.4251 ms`
  （吞吐 `-0.4627%`、P99 TPOT `+0.9176%`），长档为
  `31.7834 tok/s、13.1628 ms`（`-0.3149%/+0.7152%`）；两档均 `10/10`，输入和
  输出 token 总数与基线一致。
- BV32 中档为 `45.8477 tok/s、P99 TPOT 12.5466 ms`（吞吐 `-14.1810%`、P99 TPOT
  `+1.9041%`），且出现一次约 5 秒 TTFT 长尾；长档为
  `31.6346 tok/s、13.3241 ms`（`-0.7817%/+1.9496%`），两档同样 `10/10`。
  实验目录为 `GDN-FUSED-BF16-MID-HOT2_20260713_0114`、
  `GDN-FUSED-BF16-LONG_20260713_0114`、`GDN-FUSED-BV32-MID_20260713_0125` 和
  `GDN-FUSED-BV32-LONG_20260713_0125`。
- 融合路径使启动时可用 KV cache 由约 `17.14 GiB` 增至 `17.47 GiB`，但 micro 收益未
  传导到端到端，BV16/BV32 均未达到 TPOT `+1%` 门槛且存在稳定回退。已按方案完整删除
  生产环境变量、导出、两个 kernel、wrapper、模型接入和新增正式测试，仅保留隔离 profile
  工具与实验数据；本地 `vllm_cscc` 子仓库恢复干净并同步至 `scnet-docker-1`。

### GDN prefill state-update 候选端到端门禁

- 将 4B 模型从共享存储复制到新独占容器本地
  `/root/models/Qwen3.5-4B`；源/目标均为 15 个文件，`config.json` 与全部
  safetensors 逐文件 SHA256 一致。权重加载由共享目录约 `38.56 s` 降至本地目录约
  `1.62 s`，后续 4B 服务统一优先使用该副本。
- 在 `scnet-docker-1` 复现 gfx936 state-update 扫描：4B 的
  `BV16/warps1/stages1` 中位数约 `0.5362 ms`，相对原配置集代表项
  `BV64/warps4/stages2` 的 `0.6272 ms` 快约 `14.5%`；27B 扫描中
  `BV16/warps1/stages1` 与 `BV32/warps2/stages1` 分别约 `0.6563/0.6686 ms`，均明显
  快于原配置集代表项。4B/27B 的 `h`、`v_new` 对参考配置均 bitwise 一致，max abs diff
  为 `0`。
- 新容器候选 prefill-core 热态结果：4B T=8192/16384 中位数约
  `3.003/5.975 ms`，27B T=8192 约 `3.866 ms`，与旧容器初筛方向一致。
- 完成 4B 端到端严格 A/B，服务参数、数据顺序和请求参数一致；候选热态实验目录为
  `GDN-CAND-MID-HOT2_20260713_0000`、`GDN-CAND-LONG_20260712_2352`，baseline 为
  `GDN-BASE-MID-HOT2_20260713_0022`、`GDN-BASE-LONG_20260713_0015`。中档候选/基线
  output throughput 均约 `53.42 tok/s`（精确变化 `-0.0061%`），P99 TTFT 改善
  `0.12%`，P99 TPOT 回退 `0.16%`；长档 output throughput 仅改善 `0.063%`，P99 TTFT
  改善 `0.30%`，P99 TPOT 回退 `0.06%`。
- 两档均 `10/10` 完成，输入/输出 token 总数一致，逐样本文本与输出长度均 `10/10`
  完全一致。首轮中档各自都出现一次相同位置的运行期编译长尾，因此以完成长档后再次执行
  的双方 HOT2 作为正式中档对照。
- 结论：单 state-update kernel 收益未传导到端到端，未达到方案要求的中档 `+2%` 门槛；
  候选按门禁淘汰，不进行 27B 端到端测试。`chunk_delta_h.py` 已恢复 Git 基线，本地子仓库
  工作树干净，并以 SHA256 `849765cb...b827e31` 同步远端。

## 2026-07-12

### GDN G0 路径拆分与首轮采样

- 审计 Qwen3.5 GDN 实际前向后确认：`mixed_qkvz` 已按连续的 `mixed_qkv + z`
  输出，原先设想的 interleaved QKVZ 大重排并不存在；仅有 `ba.chunk(2)` 后两次很小的
  contiguous copy。4B/27B、T=4096 的 layout micro 中位数均约 `0.102 ms`，按层数粗算
  分别约 `2.46/4.91 ms` 每请求，未达到 5% 候选门槛，停止该方向。
- 新增 `testdata/profile_gdn_4b_27b.py`，覆盖精确 4B/27B shape 的 layout、投影、
  prefill conv/core、decode conv/core/pipeline，支持事件分位数、峰值显存与 Torch Profiler
  trace；新增 `testdata/summarize_gpu_trace.py` 汇总 trace 的 GPU kernel 占比。工具本地
  `py_compile` 通过，并已同步远端。
- batch=1 decode 首轮结果：4B conv/core/pipeline 中位数约
  `0.163/0.184/0.322 ms`；27B conv/core/pipeline 中位数约
  `0.172/0.181/0.321 ms`。单层时延对 head 数不敏感，27B 的主要放大项是 GDN 层数翻倍。
- 4B、T=4096 的 prefill core 单层中位数约 `1.546 ms`，24 层粗算约 `37.1 ms`。
  Torch trace 中 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 占 `40.86%`、
  `chunk_fwd_kernel_o` 占 `26.59%`、`recompute_w_u_fwd_kernel` 占 `12.11%`、
  solve-tril merge 占 `9.01%`，prefill chunk/state-update 已成为首要候选。
- 27B 同形状 prefill core 单层中位数约 `2.095 ms`，48 层粗算约 `100.5 ms`；trace 中
  state-update 占 `47.17%`、output 占 `20.36%`、recompute 占 `13.67%`、solve-tril merge
  占 `9.86%`。4B/27B 交叉验证均指向 state-update 为第一热点，且 27B 优先级更高。
- packed recurrent 当前固定 `BV32/warps1/stages3`；micro 工具已支持不改生产源码直接扫描
  `BV16/32/64/128 × warps1/2/4 × stages1/2/3`；prefill state-update 也已加入
  `BV16/32/64 × warps1/2/4 × stages1/2/3/4` 的隔离扫描入口。远端共享缓存写入曾令
  冷编译长时间 D 状态，改用容器本地 `/tmp` Triton/TorchInductor cache 后完成 4B trace。
- packed decode 扫描中，4B 最优 `BV128/warps2/stages1` 为 `0.1358 ms`，相对同轮
  `BV32/warps1/stages3` 的 `0.1433 ms` 快约 `5.2%`；27B 最优
  `BV128/warps2/stages3` 为 `0.1405 ms`，相对默认等价配置的 `0.1488 ms` 快约 `5.6%`。
  两模型交叉验证均未达到 8% GDN 子路径门槛，暂列次优，不先改生产 dispatch。
- prefill state-update 扫描中，4B 最优 `BV16/warps1/stages1` 为 `0.5070 ms`，相对现有
  可选配置中最佳约 `0.6270 ms` 快约 `19.1%`；27B 最优 `BV32/warps2/stages1` 为
  `0.6710 ms`，相对现有最佳约 `0.9355 ms` 快约 `28.3%`。已将两配置仅加入 gfx936 的
  state-update autotune 候选集；其他平台配置集不变，所有形状仍由原 key 独立择优，计算
  路径不变。
- 缩放稳定输入后，4B 候选 `BV16/warps1/stages1` 与 27B 候选
  `BV32/warps2/stages1` 相对 `BV32/warps2/stages2` 的 `h`、`v_new` 均 bitwise 一致，
  max abs diff 为 `0`。新鲜 baseline/candidate prefill-core A/B：4B T=4096 基本持平
  `1.586 -> 1.598 ms`，T=8192 改善约 `2.0%`（`3.052 -> 2.989 ms`），T=16384 改善约
  `2.4%`（`6.126 -> 5.976 ms`）；27B T=4096 改善约 `10.1%`
  （`2.088 -> 1.877 ms`），T=8192 改善约 `8.6%`（`4.222 -> 3.857 ms`）。
- 候选源码已恢复并同步远端。准备进入 4B `16-32K` 端到端门禁时，检测到队友正在使用
  端口 8001 和约 40% 显存运行 4B 吞吐，且其脚本结束会匹配停止 4B 服务；未并发启动或
  影响队友进程，等待资源释放后再测。

### Qwen3.5 UA2D 失效开关清理

- 删除已不影响专用 kernel 执行路径的
  `VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE` 环境变量、dispatch 临时状态及真假测试参数化。
  专用 Qwen3.5 UA2D 仍固定使用已验证的标量 block-table 地址计算；通用 UA2D 明确保持
  `SCALAR_BLOCK_TABLE=False`，计算行为、guard 和 `TILE32/BLOCK_M32/warps4/stages1`
  均未改变。
- 新增环境变量注册表删除契约；修改前直接导入检查按预期失败，删除后通过。代码审查发现
  直接删除注册会令严格环境校验拒绝旧变量，随后将旧名称加入独立废弃变量集合：软校验无
  warning、硬校验不抛异常，同时不恢复运行时开关能力。三份改动文件 `py_compile`、
  `git diff --check` 通过，attention dispatch 无旧变量或 `scalar_block_table_2d` 残留。
  本地完整 pytest 仍受缺少 `tblib` 限制，GPU 定向回归待远端执行。
- vLLM 源码提交：`515ec50`、`c274c6b`。

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
## 2026-07-13

### Qwen3.5-27B Decode gate-up 专用 GEMV

- 为 gfx936、BF16、TP=1、单 token 的 Qwen3.5-27B gate-up 精确形状
  `(M=34816, N=1, K=5120)` 实现专用 GEMV，采用 128-bit BF16 加载、FP32 累加和
  wave shuffle；比较单 wave/行与双 wave/行后，将双 wave/行接入严格形状 dispatch。
- 远端 micro：现有 LLMM1 `0.362081 ms`，单 wave `0.325762 ms`，双 wave
  `0.312014 ms`；最终方案相对 LLMM1 提升 `16.05%`（`1.16046x`）。三个随机种子
  `0/1/17` 重复执行均与 `F.linear` bitwise 一致，max abs diff 为 `0`。
- dispatch 定向测试 `20 passed`。27B 端到端三档各 `10/10` 成功：4--8K 吞吐
  `16.3992 -> 17.4188 tok/s`（`+6.22%`），8--16K 为
  `12.4463 -> 13.0498 tok/s`（`+4.85%`），16--32K 候选为
  `9.0495 tok/s`，相对 profile 基线约 `+1.07%`；短、中档 P99 TPOT 分别改善
  `6.51%/6.40%`。
- 完整精度门禁通过：HotpotQA `77.96`、GovReport `33.51`、
  retrieval_multi_point `100.00`、aggregation_keyword_aggregation `100.00`。
- 长档收益较小；未继续重写 MLP down GEMV，因为现有 Tensile 有效带宽已接近专用
  kernel，比赛剩余时间内获得显著端到端收益的成功率较低。

## 2026-07-14

### Qwen3.5-27B gfx936 专用 GEMV 第二阶段（02:00）

- 基于 `vllm_cscc@1290f03` 将 `LLMM1Gfx936` 扩展到固定
  `K=5120/N=1/M={14336,16384,34816}`，保留 V1/V2，并新增 activation LDS 复用的
  V3（单 wave/行）和 V4（双 wave/行）。三随机种子、七轮 micro 均由 V4 胜出：GDN
  qkvz 相对 LLMM1 约快 `24.7%`，Attention qkv+gate 约快 `25.3%`，gate-up 相对原 V2
  再快约 `5%`。
- 将 `VLLM_ROCM_GFX936_SPECIALIZED_GEMV` 固化为三态整数：`0` 全部关闭，`1` 仅原
  gate-up V2（精确生产基线），`2` 三形状 V4（默认）；新增环境变量说明和 dispatch
  回归。远端增量构建 `_rocm_C` 成功，最终定向测试 `34 passed`，editable import 指向
  `/public/home/acoh0h1o0p/DCU/vllm_cscc`。
- 首轮参考候选使用了旧远端启动口径，日志解析为 `max_seq_len=262144`，结果仅保存在
  `testdata/experiments/GEMV-PH2-MODE2_20260714_0058`，不作为正式 A/B。同步本地权威
  脚本后，两组均锁定 `--max-model-len 32768`，正式目录分别为
  `GEMV-PH2-MODE1_20260714_0115` 与 `GEMV-PH2-MODE2-LOCKED_20260714_0128`。
- 正式模式 1 -> 模式 2 三档结果：4--8K output `17.4710 -> 18.4950 tok/s`
  （`+5.86%`）、P99 TPOT `50.998 -> 47.832 ms`；8--16K output
  `13.0807 -> 13.8643 tok/s`（`+5.99%`）、P99 TPOT `51.900 -> 48.732 ms`；
  16--32K output `9.0622 -> 9.2999 tok/s`（`+2.62%`）、P99 TPOT
  `52.932 -> 49.769 ms`。三档均 `10/10`，TTFT 基本不变。按历史无精度扣分榜单反解
  官方基线后，预计吞吐得分净增约 `+2.10`，超过 `+0.30` 门槛。
- 输出文本一致率为 `7/10、6/10、8/10`，因此完成完整精度门禁。模式 2 得分为
  HotpotQA `77.96`、GovReport `33.38`、retrieval `100`、aggregation `100`；相对模式 1
  已验证基线 `77.96/33.51/100/100`，GovReport 相对下降约 `0.39%`，四类精度系数均为
  `1.00`。精度输出固化于候选实验目录的 `accuracy/`。
- 三形状、随机种子 `0/1/17` 的 `torch.randn` 算子门禁全部有限且重复 bitwise 确定；
  相对 F.linear 在 `rtol=1e-2/atol=1.5e-2` 下均为 `0` 个不匹配元素。最大绝对差可达
  `0.5`，但仅出现在大幅值 BF16 元素，全部满足相对误差门限。最终保留模式 2 默认路径。

### Qwen3.5-27B MLP down 专用 GEMV 淘汰（02:35）

- 基于 `vllm_cscc@0e3450b` 为精确形状 `(M=5120,N=1,K=17408)` 实现五个 gfx936 BF16
  实验候选：直接读取 activation 的单/双/四 wave 每行，以及完整约 34 KiB activation LDS
  staging 的单/双 wave 每行；所有候选均使用 128-bit BF16 load、FP32 累加和 wave shuffle。
- 远端 `_rocm_C` 增量构建成功。随机种子 `0/1/17`、每候选七轮、每轮 100 次 micro 中，
  `F.linear` 中位时间为 `0.15017/0.15023/0.15033 ms`；最快单 wave 直接读取候选为
  `0.15809/0.15814/0.15815 ms`，相对基线约慢 `5.0%`。双/四 wave 约慢 `14%--15%`，
  两个 LDS 候选约慢 `102%--122%`。
- 五个候选输出均有限、重复 bitwise 确定，并与 `F.linear` bitwise 一致；失败原因是现有
  Tensile 对该高 K 流式 GEMV 已接近更优带宽效率，而额外 split-K 或 LDS staging 只增加
  同步与占用成本。候选未达到每种子 `1.08x`、综合 `1.10x` 门槛，已删除全部候选源码和
  micro 接口，不接入模式 3、不进行端到端测试；下一步按计划转入保序 UA2D 实验 6--8。

### Qwen3.5-27B 保序 UA2D 资源实验淘汰（02:55）

- 在保持当前实验 5的 `TILE_SIZE=32`、`BLOCK_M=32`、2 warps、KV tile 遍历和 online
  softmax 更新顺序不变的前提下，实现实验 6（延迟 V load）、实验 7（完整物理块与 causal
  边界块拆环）和实验 8（两者组合）。候选使用独立 V3 Triton kernel，未修改生产 V2-B。
- 27B `q=4096` micro 中，实验 6在 8K 无收益，实验 7约快 `6%`，实验 8约快 `8%`；三者
  相对实验 5均 bitwise 一致。随后对实验 5/8进行同进程、交替顺序、三轮复测：8K 为
  `18.39395 -> 17.05341 ms`（`1.0786x`），16K 为 `40.89944 -> 36.73121 ms`
  （`1.1135x`），32K 为 `86.53095 -> 76.28092 ms`（`1.1344x`）。
- 实验 8按 `20%/50%/30%` 加权 speedup 约 `1.1128x`，加权 kernel 时间下降约 `10.11%`；
  但 8K 未达到逐档 `1.10x`，加权结果也未达到 `1.20x` 服务门禁。按预定规则不进行服务
  A/B、rocprof 或精度测试，实验 6--8源码、测试与临时 micro 接口已清理，默认继续保持
  实验 5。远端原始复测输出保存在 `/tmp/ua2d_v3_ab_20260714.txt`。

### Qwen3.5-27B Gate-up GEMV + SwiGLU 融合淘汰（04:20）

- 基于 `vllm_cscc@0e3450b` 实现候选模式 3，将单 token、BF16、TP=1 的精确
  `(34816,1,5120)` gate-up V4 GEMV 与 SwiGLU 融合，直接产生 `(1,17408)`；
  `_rocm_C` 增量构建和注册成功，定向回归 `40 passed`，并通过独立
  `torch.compile`、AOT 保存及全部 CUDAGraph 捕获。
- 三随机种子 micro 中，当前 `V4 + SiluAndMul` 约 `0.3074 ms`，融合核约
  `0.2952 ms`，加速为 `1.04095x/1.04118x/1.04143x`。融合输出与当前路径
  全部 bitwise 一致、重复确定且有限；相对 `F.linear + SiluAndMul` 为 `0` 个超差元素。
- 热态模式 2 -> 模式 3 服务 A/B：4--8K output `18.45013 -> 18.48280`
  （`+0.1771%`），8--16K 为 `14.34401 -> 14.36597`（`+0.1531%`），16--32K
  为 `9.29050 -> 9.29616`（`+0.0609%`）；加权相对吞吐仅提升 `+0.13023%`。
  三档 P99 TPOT 分别改善 `0.2445%/0.2212%/0.1634%`，TTFT 变化很小，均
  `10/10` 成功且文本、输出长度全部一致。
- 因未达到服务加权吞吐 `+0.4%` 门槛，候选按计划淘汰，不进行完整精度测试；
  已清理模式 3 的生产源码、测试接口和环境变量说明，默认继续保持模式 2，源码回到
  干净的 `0e3450b`。下一步转向 UA2D 实验 8 基础上的 LDS 布局、bank 映射和
  VGPR live-range 结构重写。

### Qwen3.5-27B UA2D 实验 8 快速收益门禁（09:43）

- 基于 `vllm_cscc@0e3450b` 恢复独立实验 8 Triton kernel：保持实验 5的
  `TILE32/BLOCK_M32/warps2`、KV tile 顺序和 online-softmax 更新顺序，将完整可见物理
  block 与 causal 边界 block 拆环，并将 V load 延迟至 QK 和 softmax 概率计算之后；
  实验 5默认路径未修改。
- 本地 `py_compile`、`git diff --check` 通过；远端选择/环境变量测试 `15 passed`。
  4B/27B、GQA=4/6、cache-block 边界和 `4095/4096` 长 query GPU 定向回归
  `32 passed`。
- 27B `q=4096`、实验 5/8同进程正反序交替七轮 micro：8K 为
  `18.378306 -> 16.131500 ms`（`1.139281x`），16K 为
  `40.827657 -> 35.717259 ms`（`1.143079x`），32K 为
  `86.492228 -> 75.403604 ms`（`1.147057x`）；三档 P99 同向改善，输出均与实验 5
  bitwise 一致，max abs diff 为 `0`。
- 三档均超过快速冲刺 micro 门禁，进入 27B 中、长档端到端服务 A/B；实验 8仍保持非默认，
  未完成服务验证前不得作为榜单候选。

### Qwen3.5-27B UA2D 实验 8 正式保留与推送（11:20）

- 完成实验 5 -> 实验 8 的 27B 三档正式服务 A/B，三档均 `10/10` 成功：4--8K output
  throughput `18.527398 -> 18.599803 tok/s`（`+0.3908%`），8--16K
  `14.392519 -> 14.568286 tok/s`（`+1.2212%`），16--32K
  `9.307250 -> 9.549492 tok/s`（`+2.6027%`）。P99 TTFT 分别改善
  `1.9568%/3.7379%/4.7500%`，P99 TPOT 三档均无回退。
- 输入、输出 token 数一致；短档生成文本完全一致，27B micro 与 4B/27B 正确性矩阵已证明
  实验 5/8 在覆盖形态下 bitwise 一致、max abs diff 为 `0`。中档首轮曾受 TTFT 长尾污染，
  热态复测恢复正收益，最终采用热态复测结果；短档另补反向顺序复核。
- 按当前榜单反解官方基线后，候选预测三档吞吐为
  `19.3553/17.2076/12.6920 tok/s`，预计总分 `84.9218 -> 85.3590`，净增约
  `+0.4372`，超过快速冲刺 `+0.30` 提交门槛。将
  `VLLM_ROCM_QWEN_UA2D_EXPERIMENT` 默认值由 `5` 固化为 `8`。
- 默认路径远端回归：未设置实验环境变量时确认值为 `8` 且选择 V3 kernel；选择/环境测试
  `15 passed`，27B `query_len=4096`、`24` query heads、cache block `784` 的 GPU
  边界用例 `1 passed`。优化已独立提交并推送至 `haha` 分支：`a5cbd48 perf: optimize
  qwen35 ua2d block traversal`；未混入尚未验证的 GEMV V5。

### Qwen3.5-27B gfx936 GEMV V5 淘汰（11:10）

- 实现 512 threads、8 waves、两 wave/行、四行/workgroup 的 V5 kernel，一次 LDS
  activation staging 供四行复用；`_rocm_C` Ninja 增量构建、安装和算子导入成功，mode 3
  dispatch 定向回归 `13 passed`。
- 三形状、随机种子 `0/1/17`、每候选七轮且每轮 100 次的 micro 中，V5 相对当前 V4
  全部回退：GDN qkvz 约慢 `5.9%--7.8%`，Attention qkv+gate 约慢
  `7.7%--10.2%`，MLP gate-up 约慢 `2.1%--2.2%`。按真实调用次数累计，V5 每 token
  比 V4 多耗约 `1.00--1.08 ms`，远低于“每形状快至少 2%、累计节省至少
  0.50 ms/token”的服务门槛。
- 所有输出均有限，并与 `F.linear` 的 BF16 输出 max abs diff 为 `0`；失败原因是 512
  threads 降低占用率且四行复用节省的 activation 读取不足以抵消同步/调度开销。未进行服务
  A/B，已删除 V5 kernel、mode 3 dispatch、测试和说明，恢复提交 `a5cbd48` 的 V4 生产源码。
  原始 micro 保存于 `testdata/experiments/GEMV-V5-MICRO_20260714_110840/results.txt`。

### Qwen3.5-27B LM Head 探针淘汰（11:25）

- 在恢复 V4 的同一扩展构建上探测 `(M=248320,N=1,K=5120)`：`F.linear`
  `1.904024 ms`，现有 LLMM1 `2.522613 ms`，仅为 `0.7548x`，即回退约 `32.5%`。
- 未达到三种子均 `1.08x` 的接入门槛，且现有 LLMM1 组织不适合该超大 M 形状；不新增
  dispatch、不进行服务 A/B，LM Head 路线停止。

### Qwen3.5-27B UA2D 实验 8 四类精度门禁（12:10）

- 使用默认环境（未设置 `VLLM_ROCM_QWEN_UA2D_EXPERIMENT`，即实验 8）完成四类、共 109
  条样本的 OpenCompass 门禁；服务端口 8001 已在测试后停止，GPU 显存恢复至 2 MiB。
- 结果：HotpotQA `77.96`、GovReport `33.38`、retrieval_multi_point `100.00`、
  aggregation_keyword_aggregation `100.00`。与已验证模式 2 基线
  `77.96/33.51/100.00/100.00` 相比仅 GovReport 下降约 `0.39%`，四类精度系数均为
  `1.00`，无精度扣分。
- 产物目录：`testdata/experiments/UA2D-E8-ACCURACY_20260714_115333`；准确率输出位于
  `accuracy/output/local_accuracy_qwen35/20260714_115357`。
