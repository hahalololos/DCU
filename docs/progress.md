# 工作进度

## 2026-07-10 DCU 根仓库初始化

- 初始化 DCU 根目录 Git 仓库并配置 `origin` 为 `https://github.com/HaHas8468/DCU.git`；`mcp-ssh`、`vllm_cscc` 保持各自独立仓库，仅作为 Git 链接纳入根仓库；忽略本地 `.codegraph/` 索引缓存。

## 2026-07-10 19:13 UAST-2 Attention 结构重写计划

- 新增 `docs/plans/qwen35_uast2_attention_structural_rewrite_plan_20260710_1913.md`；停止 Attention 参数扫描，主线改为 paged prefix、contiguous current chunk 与 online-softmax state merge。
- 核心实现约束：当前 `TRITON_ATTN` 为 NHD KV Cache，旧 prefix prefill 不能直接复用，且现有 contiguous kernel 不输出 LSE；计划先补 partial-state 接口、固定 workspace、严格 guard 和回退，再做 4B 正确性/性能/精度门禁。
- 本轮仅完成计划与源码路径核对，未修改 vLLM 源码、未启动远端服务、未授权 27B。

## 2026-07-10 18:17 mcp-ssh 生命周期与 ControlMaster 自愈

- SSH/SCP 改为独立进程组，取消/超时/shutdown 清理完整进程树；deadline 贯穿解析、建链、执行、清理。
- ControlMaster 新增跨实例原子锁、健康 socket 接管、generation token 与 mux 降级安全重试/禁止重放规则。
- 新增 stdio/TERM/INT/HUP shutdown 与 SIGCONT 恢复复检；单元/契约 `25 passed`。隔离目录下真实 `scnet-docker` 的热连接、主动关闭 master、取消长命令、双实例、STOP/CONT 全通过；Docker daemon 权限不足，三级容器用例留 CI 执行。

## 2026-07-10 16:10 ALT-C LLMM1 形状过滤候选首轮

- 完成 gfx936 LLMM1 实际形状 micro：4B/27B 各 6 个 decode Linear 形状存在明显性能反转；
  候选仅保留实测 LLMM1 收益 `>=5%` 的形状，其余回退 `F.linear`，并新增默认关闭的
  `VLLM_ROCM_GFX936_LLMM1_SHAPE_FILTER` 回滚开关和定向测试。
- `ALT-C1_20260710_1545` 显式开启过滤，4B 三档均 10/10。相对旧 D0，P99 TPOT：
  `4-8K 13.08→11.95 ms`、`8-16K 13.63→12.89 ms`、
  `16-32K 14.43→13.29 ms`，分别约 `-8.6%/-5.4%/-7.9%`。
- 候选 output throughput 为 `72.90/37.53/30.44 tok/s`。中档生成 token 数
  `1706→1522`，不能直接用 throughput 横比；逐样本文本相对旧 D0 完全一致数为
  `6/10、7/10、9/10`，需以新鲜 A/B 和正式精度口径判断。
- `ALT-C0_20260710_1605` 过滤关闭的新鲜 baseline 三档均 10/10；候选相对该 baseline
  的 P99 TPOT 分别 `-9.35%/-6.45%/-8.47%`。长档 output throughput `+3.67%`，
  短档 `+10.44%`，中档因生成 token 数减少 `10.79%` 而为 `-7.61%`。
- 已使用官方 `run_accuracy.sh` 的远端实验副本完成 4B 四类各 1 条烟测，原脚本未修改；
  当前正在跑过滤关闭的四类完整 4B 精度 baseline，随后与候选同口径比较。

## 2026-07-10 14:55 ALT-A 初筛与 ALT-B UA3D 淘汰

- 核对最新榜单：本队第 `51` 名，三档为 `15.84/14.22/10.05 tok/s`，最终得分
  `76.6248`、精度扣分 `1.1669`；当前第 `20` 名为 `86.2912`，三档
  `19.26/17.12/15.12 tok/s`，无 SLA/精度扣分。
- 将 27B UA2D `CORR-K0~K3` 和 4095/4096 长 query 正确性测试单独提交并推送为
  `d89192ccdce87f4a9f314d7c274f0edaf16f1337`；本地 `py_compile`、
  `git diff --check` 通过，远端对应测试文件 SHA256 与本地一致。
- `ALT-A1_20260710_1445` 使用
  `--attention-config '{"use_prefill_decode_attention":true}'` 启动 4B，日志确认模型、
  528 block size、源码路径正确，并命中 `ROCM_ATTN`；服务健康检查为 HTTP 200。
- 首轮 `16-32K 10` 因 benchmark 客户端 10/10 返回 503 作废；EngineCore 未崩溃，随后
  手工 20.6K prompt 成功。显式设置 loopback `NO_PROXY` 后单样本诊断成功：TTFT
  `4894.50 ms`、TPOT `87.75 ms`，初步明显差于当前 `TRITON_ATTN` 基线。
- 统一 10 条复跑开始前，队友在同卡启动 8001 profiler 服务；为避免干扰，立即停止本工作区
  bench 和 8002 服务，未操作队友进程，本轮不形成正式性能结论。
- `ALT-B` 实现默认关闭的 UA3D block-table 标量化，只改变物理 cache 地址计算；新增
  4B/27B 测试覆盖 528/784 block 边界及约 4K/8K/16K/32K sequence，scalar off/on
  各重复 5 次。远端 `2 passed`，所有输出逐元素完全一致。
- `ALT-B-MICRO1_20260710_1450`：4B 在 4K/8K/16K/32K 的 kernel 收益分别约
  `-0.23%/+0.16%/+0.25%/+0.99%`；27B 分别约
  `+0.01%/-0.05%/+0.25%/+0.87%`，max abs diff 均为 `0`。未达到 `>=1%`
  保留门槛，已淘汰并从本地、远端源码移除实验代码；`vllm_cscc` 工作树恢复到
  `d89192c` 干净状态。
- 修复 `mcp-ssh` 后台任务固定使用 `sh -c`、与正常 SSH Bash 语义不一致的问题，改为
  使用远端用户 `$SHELL -c`；`13 passed`，提交并推送为 `fc2432e`。当前已启动的 MCP
  进程需重启后才会加载该修复，本轮继续使用显式 `bash -lc` 兼容。
- 完成 ALT-C 实际 decode Linear 形状推导，结果写入
  `调研交付物/08-Qwen3.5-LLMM1形状矩阵与精度风险.md`。当前宽泛 gate 会覆盖 GDN、
  full-attention、MLP gate-up 和 LM head；4B/27B 仅 MLP down 因 `k>8192` 回退 linear。
- 两次 ALT-C micro 尝试均受共享存储异常慢读影响：第二次连 `import torch` 都超过 60 秒；
  已停止并清理本工作区进程，未得到可采信时延，未修改 vLLM 默认 dispatch。

## 2026-07-10 14:05 CORR-K0~K3 与 4B Decode/GEMM 初筛

- 按 `docs/plans/qwen35_next_optimization_plan_20260710_1322.md` 执行当前工作包，未启动
  27B 服务；实验目录统一使用时间戳，结果保存在远端 `testdata/experiments/`。
- 扩展 `tests/kernels/attention/test_triton_unified_attention.py`，新增 Qwen3.5-27B
  实际 attention 形态 `(heads=24,kv_heads=4,gqa=6,head=256,block=784)` 的
  `CORR-K0~K3` 参数矩阵，覆盖 `BLOCK_M=16/32` 与 scalar block-table off/on。
- 小边界用例覆盖 query length `1/2/3/5/6/15/16/17/31/32`，以及 KV length
  `783/784/785/1567/1568/1569`；每组重复运行 5 次并与 PyTorch paged-attention
  reference 对照。远端结果为 `8 passed`，无 NaN、Inf、VM fault 或随机漂移。
- 新增长 query 定向测试，在不构造巨大 dense reference 的情况下，将正式默认
  `BLOCK_M=32 + scalar=true` 与 generic UA2D 对比，覆盖 query length `4095/4096`；
  远端结果为 `2 passed`，重复 fastpath 输出逐元素完全一致。
- 本地 `py_compile`、`git diff --check` 通过；远端新增测试文件 SHA256 与本地一致，
  远端 `py_compile` 通过。远端 pytest 根 `conftest.py` 依赖的 `tblib` 未安装，本轮定向
  测试只使用 pytest 内置 fixture，因此通过 `--confcutdir=tests/kernels/attention` 避免加载
  无关根 conftest，未修改共享 `.venv`。
- 首次 `mcp-ssh` 最终目标连接因中间主机 host key 尚未建立而失败；通过同一
  `mcp-ssh` 先连接显式目标 `scnet-computer` 后恢复，未修改 SSH 配置或 `mcp-ssh` 源码。
- bench 客户端使用 `/tmp/qwen35_4b_tokenizer`；从官方 4B 模型目录逐文件原样复制
  tokenizer/config/chat-template 文件，并逐文件验证 SHA256 一致，不修改服务模型、权重、
  tokenizer 内容或原始比赛脚本。
- `DEC4B-D0_20260710_1338` 为当前配置：Skinny/LLMM1 on、HipBLASLt preference off。
  三档均 10/10：`4-8K` output `67.54 tok/s`、P99 TTFT `529.32 ms`、P99 TPOT
  `13.08 ms`；`8-16K` output `41.84 tok/s`、P99 TTFT `6875.74 ms`、P99 TPOT
  `13.63 ms`；`16-32K` output `29.44 tok/s`、P99 TTFT `2282.74 ms`、P99 TPOT
  `14.43 ms`。与此前 B32 scalar 热服务结果基本一致。
- `DEC4B-D1_20260710_1347` 仅关闭 Skinny/LLMM1。相对 D0：`4-8K` output
  `+1.42%`、P99 TPOT `-2.26%`；`8-16K` output `-2.41%`、P99 TPOT `-2.01%`；
  `16-32K` output `+0.91%`、P99 TPOT `-2.15%`。三档均 10/10，但短、中档生成
  token 数和文本发生变化；收益未达到 `>=3%` 门槛，不固化为默认配置。该输出差异将作为
  后续 27B 精度归因时检查 LLMM1 的依据。
- `DEC4B-D2_20260710_1357` 在 D1 基础上开启
  `TORCH_BLAS_PREFER_HIPBLASLT=1`。三档均 10/10，但相对 D0 output throughput 分别
  `-48.28%/-39.59%/-32.62%`，P99 TPOT 分别增加约
  `103.43%/99.09%/93.62%`；该候选明确淘汰，不升级 27B，也不做重复轮。
- 本轮没有 4B 性能候选达到升级门槛；P0 正确性门禁已通过。测试结束后已停止本工作区
  8002 服务，确认 8001/8002 无 vLLM/bench 进程，未触碰队友服务。

## 2026-07-10 4B UA2D block-table 标量化初筛

- 按新增测试规范优先使用 4B 筛选；未直接占用 27B 测试资源。
- 在 `triton_unified_attention.py` 中加入可回滚的实验路径：当 Qwen3.5 UA2D
  `block_size >= TILE_SIZE` 时，每个 tile 只标量读取 1--2 个 block-table 项，并复用
  计算出的 cache block offset，避免每个 lane 重复执行非 2 次幂除法/取模。
- 新增 `VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE`，初筛时默认 `False`；同时为
  GQA group size 不能整除 `BLOCK_M` 的情况补充有效行掩码，避免相邻 Q block
  重叠写输出。后者主要针对 27B GQA=6，本轮未升级到 27B 验证。
- 远端 `py_compile`、源码导入检查通过；4B `(heads=16,kv_heads=4,head=256,
  block=528)` 两个定向数值测试通过，覆盖 tile 仅无效 lane 越界和 tile 实际跨入
  第二个 KV block 两种边界。
- 使用 8002、`gpu-memory-utilization=0.45`、`TILE=64/BLOCK_M=16/WARPS=4/
  STAGES=0` 跑 `16-32K 10`。关闭标量化：10/10，total `4287.25 tok/s`、
  output `22.35 tok/s`、P99 TTFT `3617.03 ms`、P99 TPOT `14.40 ms`；结果在
  `testdata/test_4b_scalar_block_table_20260710/off/`。
- 开启标量化：10/10，total `4388.68 tok/s`、output `22.88 tok/s`、P99 TTFT
  `3503.51 ms`、P99 TPOT `14.25 ms`；结果在
  `testdata/test_4b_scalar_block_table_20260710/on/`。相对关闭组 total/output
  throughput 均约 `+2.37%`，P99 TTFT `-3.14%`，P99 TPOT `-1.04%`。
- 重复轮因共享存储上的 bench tokenizer 加载超过 300 秒且尚未发出请求而作废；已只
  清理本工作区残留 bench 进程。为避免后续 bench 客户端重复卡住，将 4B tokenizer 和
  config 文件按字节原样复制到容器本地 `/tmp/qwen35_4b_tokenizer`；逐文件 SHA256 一致，
  仅用于客户端 tokenization，不修改服务、模型或原始测试脚本。
- 4B 实际形态 microbenchmark 使用 `q_len=4096/seq_len=32768/block=528`：
  B16 vector `165.005 ms`、B16 scalar `156.209 ms`、B32 scalar `78.640 ms`、
  B32 vector `80.933 ms`；B32 scalar 相对 B16 vector kernel speedup `109.82%`，
  四组输出 max abs diff 均为 `0`。
- B32 scalar 首轮端到端测试与队友 benchmark 重叠，TPOT P99 异常到 `61.94 ms`，
  整轮作废。等待队友 bench 结束并确认 HCU `0%` 后，使用本地同字节 tokenizer 重跑
  `16-32K 10`：10/10，total `5658.78 tok/s`、output `29.50 tok/s`、P99 TTFT
  `2283.88 ms`、P99 TPOT `14.39 ms`；相对 B16 vector baseline total/output 均约
  `+32.0%`，P99 TTFT `-36.85%`，TPOT 基本持平。结果在
  `testdata/test_4b_scalar_block_table_20260710/b32_scalar_idle_localtok/`。
- 同一 B32 scalar 热服务补跑：`8-16K 10` 为 total `3316.23 tok/s`、output
  `41.58 tok/s`、P99 TTFT `7124.35 ms`、P99 TPOT `13.60 ms`；`4-8K 10` 为
  total `1676.82 tok/s`、output `67.71 tok/s`、P99 TTFT `532.25 ms`、P99 TPOT
  `13.04 ms`。两轮均 10/10，前后无其他 bench、HCU 回到 `0%`。
- 4B 已达到升级 27B 的门槛；将默认 `BLOCK_M` 从 `16` 固化为 `32`，并把
  `VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE` 默认设为 `True`，仍可通过环境变量
  回退。
- 测试结束后已停止本工作区 8002 服务；未主动停止或修改队友服务。

## 2026-07-09 UA2D Qwen3.5 27B fastpath 迁移启动验证

- 新增 `docs/plans/qwen35_27b_ua2d_fastpath_next_plan.md`，明确 27B
  fastpath 命中确认、F0/F1 对照、参数扫矩阵和决策门槛。
- 已将 `triton_unified_attention.py`、`envs.py`、`docs/progress.md`、新计划文档同步到
  远端 `/public/home/xdzs2026_c203/haha`。
- 远端 `py_compile` 通过；基础检查确认 `python` 和 `vllm` 来自
  `/public/home/xdzs2026_c203/haha/.venv`，`pip show vllm` 的 location 为
  `/public/home/xdzs2026_c203/haha/vllm_cscc`；导入确认 `envs.py` 和
  `triton_unified_attention.py` 均来自本工作区源码。
- 远端默认环境确认：`VLLM_ROCM_QWEN_UA2D_FASTPATH=True`、`TILE=64`、
  `BLOCK_M=16`、`NUM_WARPS=4`、`NUM_STAGES=0`。
- 尝试执行 editable install 时，`prepare_metadata_for_build_editable` 子进程进入
  长时间 `D` 状态；由于 `env.sh` 已让源码树优先导入，已停止该残留安装进程并改用源码
  导入检查继续验证。
- 27B F0 启动验证使用官方 `testdata/start_vllm.sh` 参数，显式设置
  `MODEL_DIR=/public/home/xdzs2026_c203/models/Qwen3.5-27B`、
  `VLLM_ROCM_QWEN_UA2D_FASTPATH=0`、`VLLM_ROCM_UA2D_DEBUG_SHAPES=1`。
- 启动日志确认模型为 `Qwen3_5ForConditionalGeneration`，`Using max model len
  262144`，chunked prefill 开启，attention block size 自动设为 `784`，
  后端为 `TRITON_ATTN`，GDN prefill 使用 Triton/FLA kernel。
- F0 服务加载 27B safetensors 时长时间停在 `2/11` shard，运行超过 14 分钟仍未监听
  8001；`/proc/<pid>/io` 显示读速很低，`hy-smi` 显示 VRAM 一度占用约 `80%`。
  为避免长期占用共享 GPU，已停止该服务；停止后确认 8001/8002 无 vLLM/bench 进程，
  GPU VRAM 回到 `0%`。
- 随后用 Python 只读方式测试第 3 个 safetensors shard 前 `512 MiB`，读取约
  `1769 MiB/s`；据此重试 F0 启动。第二次启动仍在 safetensors `2/11` 后进入慢读，
  `/proc/<pid>/io` 约 60 秒只增加约 `100 MiB`，未监听 8001。已再次停止服务并确认
  GPU/端口释放。
- 本轮未得到 27B throughput 数据；下一步需等共享模型存储恢复正常，或准备合规的本地模型
  副本后，再继续 F0/F1 短测对照。

## 2026-07-09 UA2D Qwen3.5 4B fastpath 全量对照

- 远端源码关键文件与本地一致；`py_compile` 和导入检查通过，确认默认
  `VLLM_ROCM_QWEN_UA2D_FASTPATH=True`、`TILE=64`、`BLOCK_M=16`、
  `NUM_WARPS=4`、`NUM_STAGES=0`。
- 使用 4B 全量三档吞吐数据各 50 条，共 150 条；端口使用本工作区 8002，
  启动参数保持 `--gpu-memory-utilization 0.45`，未修改原始比赛脚本。
- 对照方法：同一套 4B 服务参数下分别运行
  `VLLM_ROCM_QWEN_UA2D_FASTPATH=0` 和显式 fastpath 开启
  `TILE=64,BLOCK_M=16,WARPS=4,STAGES=0`；结果保存在远端
  `testdata/test_4b_full_compare_20260709/`。
- `4-8K`：baseline total `1616.70 tok/s`、output `60.21 tok/s`、P99 TTFT
  `1246.60 ms`、P99 TPOT `13.16 ms`；fastpath total `1741.60 tok/s`、
  output `65.81 tok/s`、P99 TTFT `1011.61 ms`、P99 TPOT `13.11 ms`。
  Total throughput `+7.73%`，output throughput `+9.30%`。
- `8-16K`：baseline total `2062.21 tok/s`、output `37.29 tok/s`、P99 TTFT
  `8237.20 ms`、P99 TPOT `13.84 ms`；fastpath total `2636.92 tok/s`、
  output `47.13 tok/s`、P99 TTFT `9284.58 ms`、P99 TPOT `13.92 ms`。
  Total throughput `+27.87%`，output throughput `+26.39%`；P99 TTFT 有尾部抖动，
  但仍低于本轮 baseline 的 `1.5x`。
- `16-32K`：baseline total `2147.12 tok/s`、output `15.86 tok/s`、P99 TTFT
  `8500.49 ms`、P99 TPOT `34.04 ms`；fastpath total `3679.21 tok/s`、
  output `27.42 tok/s`、P99 TTFT `3770.86 ms`、P99 TPOT `17.95 ms`。
  Total throughput `+71.36%`，output throughput `+72.90%`。
- 按赛题档位权重 `20%/50%/30%` 粗算：total throughput 加权提升
  `+36.89%`，output throughput 加权提升 `+36.93%`。结论是最新 UA2D
  fastpath 在 4B 全量 smoke 口径下确有实质提升，主要收益来自中长上下文。
- 按三档 150 条请求合并计算 TPOT P99：baseline `32.53 ms`，fastpath
  `16.90 ms`，下降约 `48.04%`；150/150 请求均成功，无单 token 请求跳过。
- 注意：baseline 轮开始时 GPU 空闲；fastpath 轮期间组员 8001 服务重新启动并占用约
  44%-45% 显存但 HCU 显示空闲。本工作区 8002 测试后已停止，未触碰 8001。

## 2026-07-09 UA2D Qwen3.5 4B fastpath 调参突破

- 在 `triton_unified_attention.py` 中新增严格 4B 实际形态 fastpath：
  `bf16/head_size=256/heads=16/kv_heads=4/gqa=4/block_size=528或544/causal`
  且无 alibi、sinks、softcap、qq_bias、mm_prefix、fp8 output 时命中。
- 新增/扩展实验环境变量：`VLLM_ROCM_QWEN_UA2D_NUM_WARPS`、
  `VLLM_ROCM_QWEN_UA2D_NUM_STAGES`；`NUM_STAGES=0` 表示自动选择，
  `TILE=64` 时自动使用 `num_stages=1` 避免 gfx936 shared memory OOR。
- 后端 A/B 中显式 `TRITON_ATTN` 重启在共享环境下曾长时间卡在 checkpoint 1/2，
  已停止并沿用上一轮 8002 正式基线 `2677.77 tok/s`、P99 TPOT `14.25 ms`。
- `BLOCK_M=16,TILE=32,WARPS=4` 命中 fastpath 但略退化：10/10 成功，
  Total token throughput `2670.34 tok/s`，P99 TTFT `6986.51 ms`，P99 TPOT
  `14.47 ms`。
- `BLOCK_M=16,TILE=64,WARPS=4` 初次使用默认 Triton stages 触发
  shared memory OOR：required `67584`，hardware limit `65536`；加入
  `num_stages=1` 后 smoke 通过。
- `BLOCK_M=16,TILE=64,WARPS=4,NUM_STAGES=1` 正式 `16-32K 10` 结果：
  10/10 成功；Benchmark duration `49.93s`，Total token throughput
  `4279.11 tok/s`，Output token throughput `22.31 tok/s`，P99 TTFT
  `3619.35 ms`，P99 TPOT `14.52 ms`。相对 0.45 基线 total throughput
  约 `+59.8%`，P99 TTFT 约 `-48.2%`。
- 已将默认 fastpath 设为开启，默认 tile 设为 `64`，stages 默认 `0` 自动选择；
  guard 很窄，未命中的形态仍走原 UA2D 参数。
- 测试后已停止本工作区 8002 服务，未触碰组员 8001 服务。

## 2026-07-09 4B 0.45 显存独立端口正式复测

- 为避免使用组员 `/public/home/xdzs2026_c203/scnet` 的 8001 服务，在本工作区
  `/public/home/xdzs2026_c203/haha` 单独启动 4B 服务到 8002，启动参数使用
  `--gpu-memory-utilization 0.45`。
- 使用等价 `vllm bench serve` 命令显式指定 `--port 8002` 跑
  `16-32K 10`，不调用硬编码 8001 的原脚本；结果保存在
  `testdata/test_4b/16-32K_throughput_8002_10/result.json`。
- 10/10 成功；Benchmark duration `79.79s`，Total token throughput
  `2677.77 tok/s`，Output token throughput `13.77 tok/s`，P99 TTFT
  `6982.63 ms`，P99 TPOT `14.25 ms`。
- 对比上一轮本工作区 0.95 结果 `2676.25 tok/s`、P99 TPOT `14.29 ms`，
  主体吞吐基本持平；说明把 4B 启动脚本从 0.95 改到 0.45 后，在当前 16-32K
  单并发 smoke 口径下未引入明显性能回退。测试期间同卡存在组员 8001 服务，
  因此该结论主要用于验证 0.45 配置可用与本工作区改动效果，不作为独占 GPU
  极限性能结论。

## 2026-07-09 4B 启动显存占用调整

- 按用户要求将远端 `testdata/start_vllm_4b.sh` 的 `--gpu-memory-utilization`
  从 `0.95` 改为 `0.45`，方便与组员共享 GPU。
- 已同步回本地 `testdata/start_vllm_4b.sh`；原始比赛脚本 `testdata/start_vllm.sh`
  未修改。
- 停止了此前由本工作区启动的旧 `0.95` 4B 服务，确认 8001 无旧 vLLM 服务占用。
- 重新测试：组员 8001 服务以 `0.45` 运行时显存约 44%；本工作区在 8002 以
  `0.45` 临时启动第二个 4B 服务成功，启动日志显示可用 KV cache memory
  `17.47 GiB`、KV cache size `143,088 tokens`，说明两个 `0.45` 服务可共存。
- 8002 单样本 `16-32K` 短测成功，1/1 请求完成；TTFT `67452.63 ms`，
  mean TPOT `16.72 ms`。该短测无 warmup、且与组员服务共跑，只用于验证共存和可用性，
  不作为正式吞吐对比。
- 测试后已停止本工作区 8002 临时服务，保留组员 8001 服务。

## 2026-07-09 UA2D 阶段 0 形态统计开关

- 新增 UA2D debug/实验环境变量：`VLLM_ROCM_UA2D_DEBUG_SHAPES`、
  `VLLM_ROCM_QWEN_UA2D_FASTPATH`、`VLLM_ROCM_QWEN_UA2D_TILE`、
  `VLLM_ROCM_QWEN_UA2D_BLOCK_M`。
- 在 `triton_unified_attention.py` 中加入 Qwen3.5 UA2D candidate guard 与去重形态日志；
  默认关闭，不改变现有推理路径。
- 本地 `python3 -m py_compile` 和 `git diff --check` 通过。
- 尝试通过 `mcp-ssh` 同步到 `scnet-docker` 时连接失败，尚未做远端 editable install
  和 4B smoke。
- 远端恢复连通后，已同步 `envs.py`、`triton_unified_attention.py`、文档到
  `/public/home/xdzs2026_c203/haha`；远端 `py_compile`、editable install、基础
  `which python/which vllm/pip show vllm` 检查通过。
- 远端导入确认 `triton_unified_attention.py` 来自 `haha/vllm_cscc`，debug/fastpath
  环境变量可读取；Qwen3.5 预期形态 guard 自检返回 `True`。
- 当前 8001 端口和 45% GPU 显存被 `/public/home/xdzs2026_c203/scnet` 下的 4B 服务占用，
  未启动本工作区 4B smoke，避免影响队友。

## 2026-07-09 UA2D 优化实施方案

- 新增 `docs/plans/qwen35_kernel_unified_attention_2d_implementation_plan.md`。
- 明确下一轮从 `kernel_unified_attention_2d` 入手，按后端 A/B、Qwen3.5 专用
  fast path、prefix/diagonal 拆分、contiguous current-chunk prefill 四阶段推进。

## 2026-07-09 P0A gfx936 skinny GEMM 修复后 4B 对比

- 停止 27B 测试计划，分析 4B `16-32K 10` 修复前后结果。
- 修复前禁用 skinny 为 `2682.23 tok/s`、P99 TPOT `14.03 ms`；修复后为
  `2672.11 tok/s`、P99 TPOT `14.55 ms`。结论是稳定性修复，不是 4B 吞吐优化。

## 2026-07-09 P0A gfx936 skinny GEMM 服务 smoke

- 远端启动 4B 服务并运行 `./run_throughput_4b.sh 16-32K 10`。
- 10/10 成功，无 VM fault；Total token throughput `2672.11 tok/s`，
  Output token throughput `13.75 tok/s`，P99 TTFT `6971.80 ms`，P99 TPOT `14.55 ms`。

## 2026-07-09 P0A gfx936 skinny GEMM dispatch 收窄

- 修改 `vllm_cscc/vllm/model_executor/layers/utils.py`，gfx936 只允许尝试
  `LLMM1`，不再默认进入普通 `wvSplitK`；不满足条件时回退
  `torch.nn.functional.linear`。
- dispatch smoke 通过；gfx936 的 `n=1` 命中 `LLMM1`，`n=4` 回退 linear。

## 2026-07-09 P0A gfx936 skinny GEMM micro 验证

- 远端运行一次性 micro smoke，分别测试 `LLMM1` 和 `wvSplitK`。
- `LLMM1` 在 bf16/f16 和 Qwen-ish 形状通过；`wvSplitK-bf16-n1` 未 VM fault
  但数值不达标，继续禁止 gfx936 默认启用普通 `wvSplitK`。

## 2026-07-09 P0A gfx936 skinny GEMM 全量构建

- 修改 `vllm_cscc/csrc/rocm/skinny_gemms.cu`，将 `__gfx936__` 加入
  `__HIP__GFX9__`，不加入 `__HIP__MI3XX__`；并为 gfx936 禁用不支持的
  `v_dot2c_f32_f16` 内联汇编，改为 half2/float2 累积 fallback。
- 第一次远端构建因 `v_dot2c_f32_f16` 不支持 gfx936 失败；追加 fallback 后
  `./build_vllm_wheel_install.sh` 通过，并已复制新 `_rocm_C.abi3.so` 到源码优先
  import 路径。

## 2026-07-08 A1b skinny GEMM 对照实验

- 设置 `VLLM_ROCM_USE_SKINNY_GEMM=0` 后运行 4B `16-32K 10` smoke。
- 4B 服务正常启动，10/10 成功；Total token throughput `2682.23 tok/s`，
  Output token throughput `14.00 tok/s`，P99 TTFT `6972.19 ms`，P99 TPOT `14.03 ms`。

## 2026-07-08 A1b skinny GEMM 失败诊断

- 允许 gfx936 进入 skinny GEMM 后启动 4B 服务验证。
- 初始化/预热阶段触发 HSA VM fault，日志指向 `wvSplitK_hf` /
  `wvSplitK_hf_sml`；结论是 gfx936 不能默认进入普通 `wvSplitK`。

## 2026-07-08 A1b _rocm_C 构建与导入验证

- 恢复 HIP 下 `vllm._rocm_C` 构建入口，远端全量构建并复制 `.so` 到
  `vllm_cscc/vllm/_rocm_C.abi3.so`。
- `wvSplitK`、`LLMM1`、`wvSplitKrc` 均可注册；确认 `env.sh` 下运行时优先
  导入源码树，需要源码树内存在新 `.so`。

## 2026-07-08 A1b fallback 单测

- 新增 `vllm_cscc/tests/rocm/test_skinny_gemm_fallback.py`，覆盖 `_rocm_C` /
  op 缺失时回退 linear，以及 gfx936 gate。
- 本地 `py_compile` 和 `git diff --check` 通过；远端 pytest 结果
  `3 passed`。

## 2026-07-08 gfx936 平台识别

- 修改 `vllm_cscc/vllm/platforms/rocm.py`，新增 `_ON_GFX936` 和
  `on_gfx936()`，保持 `_ON_GFX9` 原列表不变。
- 远端确认当前 DCU 为 `gfx936/BW`，vLLM 识别为 `on_gfx936=True`、
  `on_gfx9=False`。
