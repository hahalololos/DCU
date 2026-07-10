# DCU Qwen3.5 vLLM 推理优化实验计划

生成日期：2026-07-08

## 1. 目标

在不修改比赛原始启动与评测脚本、不改变模型权重/结构/tokenizer/chat template/采样语义的前提下，基于当前 profiling 结果，按低风险到高收益的顺序推进 vLLM 0.18.1 + Qwen3.5-27B + DCU 单卡推理优化。

初赛优化目标：

| 指标 | 要求 |
| --- | --- |
| 主目标 | 最大化输出吞吐量 Output Tokens/s |
| 硬约束 | 各档 TTFT P99 <= baseline * 1.5 |
| 硬约束 | 全局 TPOT P99 <= baseline * 1.5 |
| 精度约束 | OpenCompass 四类任务精度尽量不下降，低精度改动必须回归 |
| 重点档位 | 优先 `8k-16k`，其次 `16k-32k`，最后 `4k-8k` |

## 2. 当前测试结论

已有 `hipprof` 与 HTML trace 指向的热点如下：

| 档位 | 主瓶颈 | 依据 |
| --- | --- | --- |
| `4k-8k` | GEMM / Linear | GEMM 占比约 78.74% |
| `8k-16k` | GEMM + prefill attention | GEMM 约 58.87%，attention 约 35.58% |
| `16k-32k` | `kernel_unified_attention_2d` prefill attention | attention 约 61.87%，2D unified attention 是最大长条 |

全局画像：

| 类别 | 占比 | 第一轮判断 |
| --- | ---: | --- |
| rocBLAS GEMM / Linear | 约 51.76% | P0/P1 |
| unified attention | 约 42.59% | P0 |
| GDN / linear attention | 约 2.14% | 暂缓 |
| KV cache 写入/清零 | 约 0.07% | 暂缓 |
| sampling | 约 0.01% | 暂缓 |
| device-side copy | 主测窗口很小 | 暂缓 |

核心判断：第一轮不要优先做 sampling、KV 写入、memcpy 或 GDN 微调；先打 attention prefill 与 GEMM。

## 3. 合规边界

允许方向：

| 方向 | 说明 |
| --- | --- |
| ROCm/DCU 后端选择 | 不改任务定义，只改变内部 kernel/backend |
| GEMM/Linear fast path | 保持 BF16 权重和标准计算语义 |
| attention kernel/backend 优化 | 保持标准 causal attention 语义 |
| KV cache 运行期低精度 | 赛题允许，但必须做精度回归 |
| 环境变量优化 | 必须写入说明文档，且不能依赖评测期联网 |

禁止或暂不碰：

| 方向 | 原因 |
| --- | --- |
| 修改权重、持久化量化权重、权重重排缓存 | 赛题禁止 |
| 投机解码、MTP、draft model | 赛题禁止 |
| 跳层、剪枝、token pruning、early exit | 赛题禁止 |
| 修改 tokenizer/chat template/采样参数 | 改变输出语义 |
| 修改原始比赛脚本 | 项目约定禁止 |
| 直接改 `max-num-batched-tokens` 等锁定参数 | 规则风险高 |
| 内部强制 `max_model_len=32768` | 可能有收益，但需先确认规则边界 |

## 4. 实验总流程

每个实验必须遵循同一套闭环：

1. 本地修改源码。
2. 同步到远端 `/public/home/xdzs2026_c203/haha`。
3. 远端激活环境：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
```

4. Python-only 改动执行 editable install：

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e ./vllm_cscc --no-build-isolation --no-deps
```

5. 涉及 `csrc/`、HIP/C++、CMake、扩展时执行：

```bash
./build_vllm_wheel_install.sh
```

6. 先跑 4B smoke：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
cd testdata
./start_vllm_4b.sh
./run_throughput_4b.sh 16-32K 10
```

7. 再跑三档吞吐短测或全量测试。
8. 若涉及数值路径、低精度、attention kernel 或 sampling/logits 路径，跑 OpenCompass 精度回归。
9. 记录 throughput、TTFT P99、TPOT P99、完成率、显存峰值和关键 kernel profile。

## 5. P0 实验 A：启用 gfx936 ROCm fast path

### 背景

本地源码中 `vllm/platforms/rocm.py` 的 `_ON_GFX9` 只包含：

```text
gfx90a, gfx942, gfx950
```

未包含比赛 DCU 的 `gfx936`。这可能导致以下优化路径被误关：

| 路径 | 影响 |
| --- | --- |
| `on_gfx9()` 判断 | 影响 ROCm fast path 选择 |
| `rocm_unquantized_gemm_impl` | 影响 skinny GEMM / `wvSplitK` / `LLMM1` |
| AITER capability 判断 | 影响 AITER 自动发现与启用 |
| ROCm custom paged attention | 影响部分 paged attention 路径 |

### 方案

先做最小补丁：将 `gfx936` 纳入 gfx9-family 判断，但不要一次性打开所有高风险后端。

建议分两步：

| 步骤 | 改动 | 目标 |
| --- | --- | --- |
| A1 | 只让 `on_gfx9()` 对 `gfx936` 返回 true | 观察 GEMM fast path 是否启用 |
| A2 | 如 A1 稳定，再验证 AITER 是否可 import/可运行 | 避免 AITER 依赖或架构不兼容直接影响 baseline |

### 重点源码

| 文件 | 关注点 |
| --- | --- |
| `vllm_cscc/vllm/platforms/rocm.py` | `_ON_GFX9`、`_capability_from_gcn_arch()`、backend 选择 |
| `vllm_cscc/vllm/model_executor/layers/utils.py` | `rocm_unquantized_gemm_impl()` |
| `vllm_cscc/vllm/_aiter_ops.py` | `is_aiter_found_and_supported()` |

### 验收标准

| 指标 | 期望 |
| --- | --- |
| 服务启动 | 不报 arch/backend/import 错误 |
| `4k-8k` | GEMM 总时长下降或吞吐提升 |
| `8k-16k` | 吞吐提升，TTFT/TPOT P99 不恶化超过 10% |
| 精度 | BF16 GEMM fast path 原则上应无明显下降，仍建议抽样验证 |

### 回滚

恢复 `_ON_GFX9` 判断，不再将 `gfx936` 视作 gfx9-family。

## 6. P0 实验 B：attention backend A/B

### 背景

当前最大热点是 `kernel_unified_attention_2d`。源码中可选后端包括：

| 后端 | 说明 |
| --- | --- |
| `TRITON_ATTN` | 当前默认 unified attention 路径 |
| `ROCM_ATTN` | prefill/decode split 路径 |
| `ROCM_AITER_UNIFIED_ATTN` | AITER Triton unified attention |
| `ROCM_AITER_FA` | AITER FlashAttention/PA 三路径调度 |

外部资料显示 AITER 是 ROCm/vLLM 的主优化方向，但官方硬件支持主要列出 `gfx942/gfx950`，未明确覆盖 `gfx936`。因此必须按可回退实验处理。

### 实验矩阵

| 实验 | 环境/选择 | 目标 |
| --- | --- | --- |
| B0 | 默认 `TRITON_ATTN` | 复现 baseline |
| B1 | `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` | 替换 2D unified attention |
| B2 | `VLLM_ROCM_USE_AITER=1` + `ROCM_AITER_FA` | 验证 AITER prefill/extend/decode 三路径 |
| B3 | `ROCM_ATTN` | 验证 split prefill/decode 是否优于 Triton |

若不能改启动脚本参数，则优先通过源码默认后端选择或环境变量注入方式做实验，并在环境变量说明中登记。

### 重点源码

| 文件 | 关注点 |
| --- | --- |
| `vllm_cscc/vllm/platforms/rocm.py` | `_get_backend_priorities()`、`get_attn_backend_cls()` |
| `vllm_cscc/vllm/v1/attention/backends/registry.py` | 后端枚举 |
| `vllm_cscc/vllm/v1/attention/backends/rocm_aiter_unified_attn.py` | AITER unified attention |
| `vllm_cscc/vllm/v1/attention/backends/rocm_aiter_fa.py` | AITER FA |
| `vllm_cscc/vllm/v1/attention/backends/rocm_attn.py` | ROCm split attention |
| `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py` | 当前热点 kernel |

### 验收标准

| 指标 | 期望 |
| --- | --- |
| `16k-32k` | `kernel_unified_attention_2d` 总时长明显下降，吞吐提升 |
| `8k-16k` | 吞吐提升或至少不回退 |
| TTFT P99 | 不超过 baseline * 1.5，最好不恶化超过 10% |
| TPOT P99 | 不超过 baseline * 1.5，最好不恶化超过 10% |
| 完成率 | 不低于 99% |

### 回滚

禁用 AITER 相关环境变量或恢复默认 backend priority，回到 `TRITON_ATTN`。

## 7. P0/P1 实验 C：Qwen3.5 专用 `kernel_unified_attention_2d` fast path

### 背景

若 AITER 不可用或收益不稳定，则直接针对当前最大热点做小范围专用优化。比赛模型 full attention 层具有稳定形态：

| 参数 | 值 |
| --- | --- |
| `head_dim` | 256 |
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 4 |
| GQA ratio | 6 |
| full attention 层数 | 约 16 |
| 并发 | 1 |
| prefill chunk | 受 `max-num-batched-tokens=4096` 影响 |

### 方案

新增或分支化一个严格条件触发的 fast path：

| 条件 | 说明 |
| --- | --- |
| dtype | bf16 |
| head size | 256 |
| `num_seqs` | 1 |
| attention | causal decoder |
| 特性 | no alibi、no sinks、no softcap、no sliding window |
| block size | 初始只支持 16 |

候选优化：

| 优化点 | 目标 |
| --- | --- |
| 去掉多请求 `find_seq_idx` 通用逻辑 | 降低 kernel 内控制开销 |
| 固化 `HEAD_SIZE=256` 和 GQA ratio 6 | 减少模板/分支复杂度 |
| A/B `TILE_SIZE_PREFILL` | 找到 gfx936 上更优 tile |
| 针对单请求 prefill chunk 重写 launch grid | 降低空块和无效分支 |

### 验收标准

| 指标 | 期望 |
| --- | --- |
| microbenchmark | 单 kernel 时间下降 |
| `16k-32k` | TTFT 与吞吐改善 |
| `8k-16k` | 不回退，最好同步提升 |
| 精度 | OpenCompass 四类任务无明显下降 |

### 回滚

fast path 必须由严格条件保护；关闭条件或删除分支即可回到原 generic kernel。

## 8. P1 实验 D：GEMM / Linear 后端调优

### 背景

GEMM 是全局第一大类热点，也是 `4k-8k` 和 `8k-16k` 的关键。当前重点不是持久化量化权重，而是让 BF16 Linear 走更适合 DCU 的 kernel。

### 方案

| 实验 | 内容 | 风险 |
| --- | --- | --- |
| D1 | `gfx936` 启用 `on_gfx9()` 后观察 skinny GEMM | 中 |
| D2 | A/B `TORCH_BLAS_PREFER_HIPBLASLT=1` | 低 |
| D3 | 若 AITER 可用，验证 `VLLM_ROCM_USE_AITER_TRITON_GEMM` | 中 |
| D4 | 针对 decode 小 batch Linear 形状做 microbenchmark | 中 |

### 重点源码

| 文件 | 关注点 |
| --- | --- |
| `vllm_cscc/vllm/model_executor/layers/utils.py` | BF16 unquantized GEMM dispatch |
| `vllm_cscc/csrc/rocm/skinny_gemms.cu` | ROCm skinny GEMM kernel |
| `vllm_cscc/vllm/envs.py` | ROCm GEMM 相关环境变量 |

### 验收标准

| 指标 | 期望 |
| --- | --- |
| `4k-8k` | 吞吐提升最明显 |
| `8k-16k` | 吞吐提升 |
| TPOT P99 | 不恶化 |
| 精度 | BF16 路径不应有明显下降 |

## 9. P2 实验 E：KV cache FP8 / 低精度 cache

### 背景

赛题允许推理过程中的 KV cache 量化，ROCm 文档也将 FP8 KV-cache 列为支持方向。当前 profile 显示 decode attention 不是第一热点，但 KV 低精度可能改善显存占用和部分长上下文带宽压力。

### 方案

| 步骤 | 内容 |
| --- | --- |
| E1 | 确认当前 `kv_cache_dtype`、cache shape、scale 路径 |
| E2 | 短测 FP8 KV cache 启动和服务稳定性 |
| E3 | 跑 OpenCompass 四类精度 |
| E4 | 若精度下降 <= 1%，再跑三档吞吐 |

### 风险

| 风险 | 控制 |
| --- | --- |
| 精度下降 | 必须跑 OpenCompass |
| TTFT 变差 | 量化/反量化可能增加 prefill 开销 |
| kernel 不兼容 gfx936 | 必须先做 4B smoke |

## 10. 暂缓方向

| 方向 | 暂缓原因 |
| --- | --- |
| sampling 快路径 | profile 占比约 0.01% |
| KV cache 写入优化 | profile 占比约 0.07% |
| GDN/linear attention | profile 占比约 2.14% |
| memcpy 优化 | HTML trace 中真实 device copy 很小 |
| batch scheduler 大改 | 规则风险高，且当前主要瓶颈在 kernel |
| prefix caching | 合规风险高，单请求收益有限 |

## 11. 每轮实验记录模板

| 字段 | 内容 |
| --- | --- |
| 实验编号 | A1 / B1 / C1 等 |
| git commit | 本地提交或 diff 摘要 |
| 远端安装方式 | editable install / rebuild wheel |
| 启动环境变量 | 完整列出 |
| attention backend | 启动日志确认 |
| block size | 启动日志确认 |
| `4k-8k` throughput | tok/s |
| `8k-16k` throughput | tok/s |
| `16k-32k` throughput | tok/s |
| TTFT P99 | 三档分别记录 |
| TPOT P99 | 全局记录 |
| 完成率 | bench 输出 |
| 显存峰值 | `hy-smi` / 日志 |
| profile 摘要 | GEMM、`kernel_unified_attention_2d`、`kernel_unified_attention_3d` |
| 精度结果 | QA / 摘要 / 检索 / 聚合 |
| 结论 | 保留 / 回滚 / 继续拆分 |

## 12. 推荐执行顺序

| 顺序 | 实验 | 通过后进入 |
| ---: | --- | --- |
| 1 | A1：`gfx936` 纳入 gfx9-family，验证 GEMM fast path | D1 |
| 2 | D2：`TORCH_BLAS_PREFER_HIPBLASLT=1` A/B | 保留收益项 |
| 3 | B1：AITER unified attention smoke + 短测 | B2 |
| 4 | B2：AITER FA smoke + 短测 | 三档全量 |
| 5 | C1：Qwen3.5 专用 2D attention fast path microbenchmark | C2 端到端 |
| 6 | E1/E2：FP8 KV cache smoke | OpenCompass |

## 13. 参考资料

| 资料 | 链接 |
| --- | --- |
| ROCm vLLM V1 performance optimization | https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html |
| vLLM ROCm attention backend blog | https://vllm.ai/blog/2026-02-27-rocm-attention-backend |
| ROCm AITER | https://github.com/ROCm/aiter |
| vLLM PagedAttention design | https://docs.vllm.ai/en/latest/design/paged_attention/ |
| PagedAttention paper | https://arxiv.org/abs/2309.06180 |
| FlashAttention paper | https://arxiv.org/abs/2205.14135 |

