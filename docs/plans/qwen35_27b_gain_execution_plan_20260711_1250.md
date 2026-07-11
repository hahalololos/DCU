# Qwen3.5-27B 增益验证与优化执行计划

生成时间：2026-07-11 12:50 CST

## 目标

在不改变模型、权重、tokenizer、采样语义及锁定调度参数的前提下，先验证当前正收益优化在
27B 上的实际贡献，再依据最新 profile 选择大 K Decode GEMV 或 27B UA2D 专用化主线。

## 执行顺序

1. 恢复并核验 `scnet-login/scnet-docker`、PRA26 作业、GPU、端口和远端源码状态。
2. 将本地 `origin/haha` 最新 `vllm_cscc` 同步到远端，核验关键文件 SHA256、Python、vLLM
   和源码优先导入路径。
3. 使用官方 Qwen3.5-27B 模型和锁定参数跑当前配置三档短基线，记录完成率、output
   throughput、P99 TTFT/TPOT 和生成结果。
4. 仅开启 `VLLM_ROCM_GFX936_LLMM1_SHAPE_FILTER=1`，重复同口径三档 A/B；收益达标后跑
   生成文本对照及四类精度门禁。
5. 在通过精度的 GEMM 配置上比较 `VLLM_ROCM_QWEN_UA2D_FASTPATH=0/1`，确认 27B
   `heads=24/kv_heads=4/gqa=6/head=256/block=784` 的端到端贡献。
6. 对最优正确配置重新采集 prefill/decode profile：若 Decode Linear 占主导，优先开发
   `(M=5120,N=1,K=17408)` MLP down 专用 gfx936 GEMV；若 UA2D 在长档仍占 GPU 时间
   `40%+`，进入 `block_size=784/GQA=6` 专用化。

## 门禁

- 任一实验完成率必须为 100%，不得出现 VM fault、NaN/Inf 或服务崩溃。
- 候选至少一个档位可重复提升 `>=3%`，其它档位不得稳定回退超过 `1%`。
- 任何生成文本或长度变化必须进入官方精度 A/B；精度系数下降则不得固化。
- 27B 启动前确认 GPU、8001/8002 和队友进程；不得停止或修改非本工作区服务。
- 每次源码优化及测试结果写入 `docs/progress.md`。

## 当前前置状态

- 本地 `vllm_cscc` 已推送 `origin/haha`，HEAD 为 `2295425`。
- UA2D 已固化 B32/TILE64/WARPS4/STAGES1，保留 fastpath/scalar 回退开关。
- LLMM1 形状过滤代码存在但默认关闭。
- 上次 PRA26 作业已达到生命周期终点；必须先确认用户是否已在平台重新启动。
