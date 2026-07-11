# SCNet选手测试调试文档

### 评测规范说明

选手对 vLLM 内核、调度、算子融合、编译参数等优化后，**必须能正常重新编译生成 dist 下 whl 包并安装运行**，评测脚本以此作为校验标准。

## 1\. 评测脚本

### testdata 目录文件清单

#### 吞吐测试数据集

- `4-8K_throughput.jsonl`

- `8-16K_throughput.jsonl`

- `16-32K_throughput.jsonl`

#### 精度测试数据集

- `hotpotqa.jsonl`（问答任务）

- `gov_report.jsonl`（长文摘要任务）

- `retrieval_multi_point.jsonl`（多点检索任务）

- `aggregation_keyword_aggregation.jsonl`（关键词聚合任务）

#### 运行脚本

- `start_vllm.sh`：启动 vLLM 推理服务

- `run_throughput.sh`：吞吐性能测试脚本

- `run_accuracy.sh`：模型精度评测脚本

## 2\. 启动 vLLM 推理服务

### 操作步骤（家目录 testdata 下执行）

```bash
./start_vllm.sh
```

1. 服务启动耗时约 10 分钟，终端不可关闭；后续所有测试需保持该终端运行

2. 服务默认地址：`http://127.0.0.1:8001`

3. 查看 DCU 显卡状态：

```bash
hy-smi
```

### 验证服务可用性（新开终端执行 curl）

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen3.5-27B",
"messages": [
{"role": "user", "content": "你好,简单回复一句话。"}
],
"temperature": 0.0,
"max_tokens": 64
}'
```

正常返回对话内容即代表部署成功；容器 / 服务关闭后需重新执行启动脚本。

## 3\. 吞吐性能测试

新开终端进入`~/testdata`目录执行脚本：

```bash
# 全部数据集完整测试
./run_throughput.sh

# 全部数据集仅取前10条数据调试
./run_throughput.sh all 10

# 指定数据集取前N条示例
./run_throughput.sh 4-8K 10
./run_throughput.sh 8-16K 20
./run_throughput.sh 16-32K 15
```

### 评测核心指标

1. 整体吞吐量

2. 首 token 延迟 TTFT P99

3. 单 token 时延 TPOT P99

> 本地调试结果仅作基线参考，最终得分以评测机正式运行结果为准。
> 
> 

## 4\. 模型精度测试

新开终端进入`~/testdata`目录执行脚本：

```bash
# 全量精度测试
./run_accuracy.sh

# 指定数据集截取前N条调试示例
./run_accuracy.sh hotpotqa 10
./run_accuracy.sh gov_report 10
./run_accuracy.sh retrieval_multi_point 10
./run_accuracy.sh aggregation_keyword_aggregation 10
```

### 实时查看评测日志

```bash
tail -f accuracy_debug/opencompass_run.log
```

### 精度评测规则

1. **OpenCompass 评测 LongBench 数据集**

    - hotpotqa：问答任务，评价指标 F1

    - gov\_report：长文本摘要任务，评价指标 ROUGE

2. **自定义后处理评测**
retrieval\_multi\_point、aggregation\_keyword\_aggregation：脚本单独解析输出与标准答案比对，日志内 0 值指标无需参考，终端会输出最终 100 分制准确率。

### 评测输出目录

精度结果输出路径：`testdata/accuracy_debug/output/local_accuracy_qwen35/`
终端会直接打印汇总精度分数。

## 5\. 提交前自检清单

提交代码与优化方案前，务必确认以下流程全部正常运行：

1. 修改 vLLM 源码后，可正常编译生成 whl 安装包

2. 离线安装 whl 包后，vLLM 服务正常启动无报错

3. curl 单条推理请求能够正常返回模型输出

4. 吞吐脚本 `run_throughput.sh` 完整运行无异常

5. 精度脚本 `run_accuracy.sh` 完整运行无异常
