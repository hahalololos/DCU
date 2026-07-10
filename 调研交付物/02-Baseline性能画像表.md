# Baseline 性能画像表

当前本地无 vLLM、OpenCompass、HIP/DTK、DCU 设备，不能直接复现官方 baseline。本文件先固化采集口径、命令模板和数据表结构，等拿到官方容器或测评机器后直接填数。

## 1. 评测口径

| 项目 | 固定口径 |
| --- | --- |
| 服务框架 | vLLM 0.18.1 |
| 服务命令基线 | `vllm serve Qwen/Qwen3.5-27B --max-model-len 32768` |
| 模型权重 | 官方 Qwen3.5-27B bf16 原始权重 |
| tokenizer/chat template | 官方 tokenizer 与官方 chat template |
| 并发数 | 初赛固定为 1 |
| 性能工具 | `vllm bench serve` |
| 精度工具 | OpenCompass |
| 吞吐口径 | Output Tokens / Second，不计 prompt tokens |
| TTFT | 客户端 HTTP 请求发出到收到首个生成 token，包含 tokenizer 与 prefill |
| TPOT | 每请求首 token 到末 token 的时间差 / (`output_tokens - 1`)，输出 token 数为 1 的请求不统计 |
| TTFT P99 | 按请求维度、各长度档位独立统计 |
| TPOT P99 | 全部请求汇总后全局统计 |

## 2. 本机状态

| 命令 | 本机结果 |
| --- | --- |
| `python --version` | Python 3.14.6 |
| `vllm` | 未安装或未在 PATH |
| `opencompass` | 未安装或未在 PATH |
| `hipcc` | 未安装或未在 PATH |
| `rocm-smi` | 未安装或未在 PATH |

结论：本机只能准备模板，baseline 数字必须在官方 DCU 容器中采集。

## 3. Baseline 采集流程

### 3.1 环境确认

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import vllm; print(vllm.__version__)"
python -c "import transformers; print(transformers.__version__)"
rocm-smi
hipcc --version
```

要求与赛题一致：

| 组件 | 赛题版本 |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | 2.10.0 |
| vLLM | 0.18.1 |
| transformers | 5.5.0 |

### 3.2 启动 baseline 服务

比赛方已提供官方样例脚本：

```bash
cd testdata
bash start_vllm.sh
```

脚本当前等价于：

```bash
vllm serve Qwen/Qwen3.5-27B \
  --served-model-name Qwen3.5-27B \
  --port 8001 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

注意：赛题技术方案中的 baseline 示例包含 `--max-model-len 32768`，但 `testdata/start_vllm.sh` 未显式设置该参数。正式实验前应向组委会或平台实际命令确认是否需要补齐；若平台最终锁定启动参数，以平台命令为准。

需要保存启动日志中的以下信息：

| 日志项 | 用途 |
| --- | --- |
| GPU/DCU 设备型号与显存 | 判断显存预算和单卡资源 |
| attention backend | 确认默认 Attention kernel |
| KV cache size / num blocks | 计算可容纳 token 和碎片风险 |
| max concurrency for max_model_len | 校验并发固定为 1 下资源余量 |
| dtype/cache dtype | 判断 bf16、fp8 或其他 cache 类型 |
| 是否启用 prefix caching/spec decode | 合规核查 |

### 3.3 性能评测命令模板

比赛方已提供官方样例脚本：

```bash
cd testdata
bash run_throughput.sh all
bash run_throughput.sh 8-16K 10
```

脚本会生成：

| 档位 | 数据集 | 结果目录 |
| --- | --- | --- |
| 4-8K | `testdata/4-8K_throughput.jsonl` | `testdata/test/4-8K_throughput/result.json` |
| 8-16K | `testdata/8-16K_throughput.jsonl` | `testdata/test/8-16K_throughput/result.json` |
| 16-32K | `testdata/16-32K_throughput.jsonl` | `testdata/test/16-32K_throughput/result.json` |

核心参数如下：

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8001 \
  --endpoint /v1/chat/completions \
  --model Qwen/Qwen3.5-27B \
  --tokenizer "$MODEL_DIR" \
  --dataset-name custom \
  --dataset-path <档位 jsonl> \
  --num-prompts <数据集行数或命令行参数> \
  --no-oversample \
  --max-concurrency 1 \
  --request-rate 1 \
  --temperature 0 \
  --disable-shuffle \
  --custom-output-len 1024 \
  --num-warmups 2 \
  --save-detailed \
  --extra-body '{"temperature":0.0}' \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result \
  --result-dir <结果目录> \
  --result-filename result.json
```

### 3.4 精度评测命令模板

比赛方已提供官方样例脚本：

```bash
cd testdata
bash run_accuracy.sh all
bash run_accuracy.sh hotpotqa 5
```

脚本会动态生成 `accuracy_debug/bench.py`，使用 OpenCompass 的 `OpenAISDK` 访问：

```text
http://127.0.0.1:8001/v1
model = Qwen3.5-27B
temperature = 0
max_seq_len = 32768
```

数据集与输出长度：

| 数据集 | 类型 | 来源/口径 | max_out_len |
| --- | --- | --- | ---: |
| `hotpotqa.jsonl` | QA | LongBench HotpotQA 配置 | 1024 |
| `gov_report.jsonl` | 摘要 | LongBench GovReport 配置 | 1024 |
| `retrieval_multi_point.jsonl` | 检索 | CustomDataset + AccEvaluator | 96 |
| `aggregation_keyword_aggregation.jsonl` | 聚合 | CustomDataset + AccEvaluator | 128 |

需要记录：

| 字段 | 要求 |
| --- | --- |
| API base | `http://127.0.0.1:8001/v1` |
| model name | `Qwen3.5-27B` |
| temperature | 0.0 |
| max out len | 按数据集配置 |
| tokenizer | 官方 tokenizer |

## 4. Baseline 性能画像表

| 长度档位 | 任务类型 | 请求数 | 平均输入 token | 平均输出 token | Throughput(tok/s) | TTFT P50(ms) | TTFT P90(ms) | TTFT P99(ms) | TPOT P50(ms) | TPOT P90(ms) | TPOT P99(ms) | 显存峰值 | 完成率 | 精度 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4k-8k | QA | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 4k-8k | 摘要 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 4k-8k | 检索 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 4k-8k | 聚合 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 8k-16k | QA | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 8k-16k | 摘要 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 8k-16k | 检索 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 8k-16k | 聚合 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 16k-32k | QA | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 16k-32k | 摘要 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 16k-32k | 检索 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| 16k-32k | 聚合 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

## 5. SLA 计算表

| 档位 | Baseline TTFT P99 | 参赛 TTFT P99 上限 | Baseline TPOT P99 全局 | 参赛 TPOT P99 上限 | 是否通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| 4k-8k | 待实测 | `Baseline * 1.5` | 待实测 | `Baseline * 1.5` | 待判定 |
| 8k-16k | 待实测 | `Baseline * 1.5` | 待实测 | `Baseline * 1.5` | 待判定 |
| 16k-32k | 待实测 | `Baseline * 1.5` | 待实测 | `Baseline * 1.5` | 待判定 |

## 6. 慢点归因字段

每轮 benchmark 后必须补充：

| 归因项 | 采集方式 | 判断目标 |
| --- | --- | --- |
| TTFT 是否随输入长度近似线性/超线性增长 | 三档 TTFT P50/P99 对比 | 判断 prefill/tokenizer/调度瓶颈 |
| TPOT 是否随上下文长度增长 | 三档 TPOT P50/P99 对比 | 判断 decode KV 读取带宽瓶颈 |
| P99 与 P50 差距 | P99/P50 比值 | 判断尾延迟稳定性 |
| 显存峰值与碎片 | `rocm-smi`、vLLM 日志、profiling | 判断 KV block 与临时 buffer 压力 |
| kernel 时间分布 | rocprof/平台 profiler | 判断 Attention/Linear/RMSNorm 等热点 |
| 完成率 | bench 输出 | 低于 99% 有熔断风险 |

## 7. 数据来源要求

性能结论必须附以下至少一种来源：

1. `vllm bench serve --save-result --save-detailed` 结果文件。
2. vLLM 服务启动日志。
3. OpenCompass 输出目录中的评测结果。
4. `rocm-smi` 或平台监控日志。
5. rocprof/DTK profiler 输出。

没有数据来源的结论只能写成“假设”或“待验证”。
