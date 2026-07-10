# Qwen3.5-27B UA2D Fastpath 下一步优化方案

生成日期：2026-07-09

## 1. 目标

把 4B 上已经验证有效的 UA2D fastpath 推进到正式 27B 评测口径，确认
`heads=24/kv_heads=4/gqa=6/head_dim=256` 形态是否稳定命中，并筛选 27B
默认参数。

成功标准：

| 指标 | 标准 |
| --- | --- |
| 完成率 | 三档短测均 100% |
| `8k-16k` | output throughput 提升或至少不回退 |
| `16k-32k` | output throughput 明显提升 |
| TTFT/TPOT | P99 不超过 fastpath-off baseline 的 `1.5x` |
| 合规 | 不修改原始比赛脚本、模型、tokenizer、chat template、采样语义 |

## 2. 当前状态

当前 `triton_unified_attention.py` 的 Qwen3.5 fastpath guard 已覆盖：

| 模型 | 形态 |
| --- | --- |
| Qwen3.5-4B | `(heads=16, kv_heads=4, gqa=4, head_dim=256)` |
| Qwen3.5-27B | `(heads=24, kv_heads=4, gqa=6, head_dim=256)` |

允许的 UA2D `block_size` 已包含 `528/544/784`。4B 全量三档对照显示当前默认
`TILE=64/BLOCK_M=16/WARPS=4/STAGES=0` 对中长上下文有明显收益，因此下一步不先写
新 kernel，而是先做 27B 命中确认、A/B 和参数扫。

## 3. 实验矩阵

### 3.1 命中与 baseline 对照

先跑两组 27B 三档短测：

| 组别 | 环境变量 | 目的 |
| --- | --- | --- |
| F0 | `VLLM_ROCM_QWEN_UA2D_FASTPATH=0`、`VLLM_ROCM_UA2D_DEBUG_SHAPES=1` | fastpath-off baseline |
| F1 | `FASTPATH=1,TILE=64,BLOCK_M=16,WARPS=4,STAGES=0,DEBUG_SHAPES=1` | 当前默认 fastpath |

三档命令使用官方脚本口径：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd testdata
./run_throughput.sh 4k-8k 10
./run_throughput.sh 8k-16k 10
./run_throughput.sh 16k-32k 10
```

验收时必须从日志确认 `qwen_fastpath=True`。如果 27B 因实际 `block_size` 不在白名单而
未命中，只允许补充观测到的 Qwen3.5 block size；如果形态不是 Qwen3.5 文本 full
attention，则暂停 fastpath 扩展并重新 profile。

### 3.2 27B 参数扫

先在 `16k-32k 10` 上筛选：

| 组别 | `TILE` | `BLOCK_M` | `WARPS` | `STAGES` |
| --- | ---: | ---: | ---: | ---: |
| A | 64 | 16 | 4 | 0 |
| B | 64 | 16 | 8 | 0 |
| C | 64 | 32 | 4 | 0 |
| D | 32 | 16 | 4 | 0 |

如果 C 最优，再追加：

| 组别 | `TILE` | `BLOCK_M` | `WARPS` | `STAGES` |
| --- | ---: | ---: | ---: | ---: |
| E | 64 | 32 | 8 | 0 |

最优前两组补跑 `8k-16k 10` 和 `4k-8k 10`。若加权 output throughput 差距小于
`3%`，选择更保守的 A 组作为默认。

## 4. 执行约束

- 本地用于阅读、编辑源码和维护 git 历史；远端只用于安装、运行和测试。
- 测试前必须同步本地源码到 `/public/home/xdzs2026_c203/haha`。
- 远端执行项目前必须：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
```

- Python-only 改动使用 editable install；涉及 `csrc/`、HIP/C++、CMake 或编译扩展时才运行
  `./build_vllm_wheel_install.sh`。
- 启动 27B 前先检查 GPU 和 8001 端口占用；共享环境中不得影响队友服务。
- 不修改 `testdata/start_vllm.sh` 和 `testdata/run_throughput.sh`。如需独立结果目录，
  使用等价 `vllm bench serve` 命令或测试后移动结果文件。

## 5. 决策门槛

| 结果 | 下一步 |
| --- | --- |
| 加权 output throughput 提升 `>= 10%` 且 `16k-32k` 提升 `>= 15%` | 固化最优 UA2D 参数 |
| 收益不足但 profile 仍显示 UA2D 是长档主热点 | 进入 prefix full tiles / diagonal tiles 拆分 |
| UA2D 不再是主热点 | 转向 GDN/GEMM profile，不优先做 KV cache 量化 |
| 任一组出现 VM fault、完成率下降或 P99 超出 `1.5x` | 回退到 `VLLM_ROCM_QWEN_UA2D_FASTPATH=0` |

## 6. 记录要求

每次完成源码改动并测试后更新 `docs/progress.md`，记录：

- 实验组别、环境变量、远端安装方式。
- 三档 total/output throughput、TTFT P99、TPOT P99、完成率。
- UA2D shape 日志中的 `qwen_candidate`、`qwen_fastpath`、`block_size`。
- 是否影响队友服务、GPU/端口占用情况。
- 结论：固化、继续调参、进入结构性拆分或回滚。
