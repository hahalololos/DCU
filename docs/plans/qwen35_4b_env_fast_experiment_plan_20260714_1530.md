# Qwen3.5-4B 环境变量快速实验计划

## 目标与边界

仅使用 Qwen3.5-4B 快筛 `HIP_FORCE_DEV_KERNARG` 和 PyTorch TunableOp，不测试 27B、
AITER 或全局 hipBLASLt，不修改 vLLM 源码、模型、评测脚本和锁定参数。4B 结果只用于
筛选环境变量，不能直接作为榜单收益结论。

## 实验矩阵

| 编号 | 配置 | 目的 |
|---|---|---|
| `ENV4B-B0` | 当前环境，候选变量全部 unset | 权威基线 |
| `ENV4B-H1` | `HIP_FORCE_DEV_KERNARG=1` | 测试 kernel 参数访问优化 |
| `ENV4B-TUNE` | TunableOp 在线调优一次 | 生成 4B 实际 GEMM lookup CSV |
| `ENV4B-T1` | 使用 CSV，关闭在线调优 | 隔离 TunableOp 收益 |
| `ENV4B-HT1` | H1 与 T1 组合 | 仅在二者单独通过时测试交互收益 |

## 环境准备

1. 以本地源码和测试工具为权威，同步到 `scnet-docker-1` 的
   `/public/home/acoh0h1o0p/DCU`。
2. 每次远端操作加载 `/opt/dtk/env.sh`、`/opt/hyhal/env.sh`，并设置本项目
   `PYTHONPATH`。
3. 确认 `which python`、`which vllm`、`python -m pip show vllm`、实际导入路径与
   `_rocm_C` 均指向本项目；环境变量实验不重建。
4. 使用 `/root/models/Qwen3.5-4B`、端口 8001、`testdata/start_vllm_4b.sh` 和
   `testdata/run_throughput_4b.sh`。测试前确认 GPU 空闲且端口无未知进程，否则停止。
5. 固定当前 UA2D E8、UA3D scalar、LLMM1 shape filter、Skinny GEMM 和 packed
   recurrent 默认值。B0 中 unset `HIP_FORCE_DEV_KERNARG`、全部 TunableOp 变量、
   AITER 和全局 hipBLASLt 变量。

## 快筛顺序与产物

- H1 顺序：`B0-H1-H1-B0`；T1 顺序：`B0-T1-T1-B0`。
- 每次重新启动服务，三档各跑 5 条。
- 每轮结束立即将 `test_4b` 移到
  `testdata/experiments/ENV4B-<配置>-<轮次>_<时间戳>/`，同时保留完整环境和启动日志。
- 保存三档 JSON、完成率、output throughput、TTFT/TPOT P50/P95/P99，以及逐样本
  输入 token、输出 token 和文本一致性。

## TunableOp 调优

调优阶段 unset `HIP_FORCE_DEV_KERNARG`，设置：

```bash
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=1
PYTORCH_TUNABLEOP_ROCBLAS_ENABLED=1
PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=1
PYTORCH_TUNABLEOP_NUMERICAL_CHECK=1
PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=10
PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=2
PYTORCH_TUNABLEOP_ROTATING_BUFFER_SIZE=0
PYTORCH_TUNABLEOP_FILENAME=<实验目录>/tunable_%d.csv
```

服务就绪后仅发送一条 8--16K、64 输出 token 的调优请求，该请求不计入性能结果。调优
总时限 15 分钟；超时、OOM、VM fault，或未生成带 gfx936 validator/GEMM 条目的 CSV，
即淘汰 T1。

T1 正式服务设置 `PYTORCH_TUNABLEOP_ENABLED=1`、
`PYTORCH_TUNABLEOP_TUNING=0` 并复用生成的 CSV。未命中形状回退默认实现，不允许正式请求
触发在线调优。

## 门禁

加权吞吐变化按 `20% × 短档 + 50% × 中档 + 30% × 长档` 计算。

候选进入最终确认必须同时满足：

- 三档均 `5/5` 完成，两轮加权吞吐同向；H1 至少 `+0.5%`，T1/HT1 至少 `+1.0%`。
- 任一档 output throughput 不得回退超过 `1%`。
- TTFT/TPOT P99 不得比相邻 B0 中位数恶化超过 `5%`。
- 输入 token、输出 token 和逐样本文本与 B0 完全一致；出现差异立即淘汰，不追加精度测试。
- 无 NaN、OOM、VM fault、服务重启或代理响应。

若 H1、T1 均通过，执行 `B0-HT1-HT1-B0`；否则直接选择单项最佳候选。唯一胜出配置再
执行 `B0-W-W-B0` 最终确认，三档各 10 条，要求：

- 三档均 `10/10` 完成；
- 两轮加权吞吐均至少 `+0.7%` 且同向；
- 任一档回退不超过 `1%`；
- TTFT/TPOT P99 相对相邻 B0 回退不超过 `3%`；
- 相邻同配置结果波动小于 `2%`。

## 收尾

实验完成后仅向 `docs/progress.md` 追加简记；TunableOp CSV 保留在远端时间戳实验目录，
不提交、不修改源码默认值。本轮明确排除 AITER、`TORCH_BLAS_PREFER_HIPBLASLT=1`、
NCCL/HSA/GPU queue 变量，不启动 27B、不跑排行榜。
