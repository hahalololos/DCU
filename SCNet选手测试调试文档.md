# SCNet选手测试调试文档

## 1\. 平台登录

超算平台地址：[https://www\.scnet\.cn](https://www.scnet.cn)
登录方式：账号密码登录 / 短信登录 / 扫码登录
平台定位：打造国家先进算力底座，数字中国算力高速路

## 2\. 进入容器服务控制台

1. 登录平台后，点击页面右上角「控制台」

2. 在控制台菜单中选择「容器服务」

### 平台活动提示

超算互联网・智 "惠" 开发季
活动时间：2026\.6\.15 \~ 2026\.7\.15
优惠：全新 Token Plan 低至 99 元，新用户专享 1000Tokens，千万卡时放送

## 3\. 镜像克隆操作

依次点击：**镜像管理 → 镜像仓库 → 核心节点分区一**

1. 搜索镜像：`qwen3.5-dtk26.04:0509`

2. 将目标镜像克隆至「我的镜像」

## 4\. 创建容器实例

1. 菜单路径：**容器实例 → 返回旧版**

2. 点击「创建容器」

### 创建配置要求

- 分区队列：核心节点分区一 `hx1hdexclu08`

- 开发工具：SSH

- 镜像来源：我的镜像，选中上一步克隆的镜像

- 配置完成后点击右下角「创建」

## 5\. 连接容器 SSH

容器创建完成后，点击 SSH 入口进入容器终端

## 6\. 源码安装 vLLM（家目录执行）

> 重要提醒：容器停止后，非家目录数据会丢失，所有代码、文件、模型统一存放至用户家目录；重启容器后需重新编译、安装 vLLM
> 
> 

```bash
# 拉取指定版本vllm源码
git clone -b v0.18.1 --depth 1 http://developer.sourcefind.cn/codes/OpenDAS/vllm_cscc.git
cd vllm_cscc

# 编译whl安装包
# 首次编译约10分钟，后续增量编译约2分钟
python setup.py bdist_wheel

# 进入编译产物目录，离线安装
cd dist
pip install vllm-*.whl --no-deps
```

### 评测规范说明

选手对 vLLM 内核、调度、算子融合、编译参数等优化后，**必须能正常重新编译生成 dist 下 whl 包并安装运行**，评测脚本以此作为校验标准。

## 7\. 下载 Qwen\-3\.5\-27B 模型

回到用户家目录执行以下命令：

```bash
# 安装模型下载工具
pip install modelscope

# 模型下载至家目录
modelscope download --model Qwen/Qwen3.5-27B --local_dir ./Qwen3.5-27B
```

> 说明：模型存放在家目录持久保存，启动 vLLM 前复制到 /root 可大幅缩短模型加载耗时。
> 
> 

## 8\. 下载测试数据集与评测脚本

### 1\. 下载压缩包

```bash
curl -f -C -o testdata.tar.gz https://zzefile.scnet.cn:65011/efile/s/d/c2N5MTE1OTkxMDU1OQ==/a927e65672549b46
```

### 2\. 解压文件

```bash
mkdir -p ./testdata
tar -xzf testdata.tar.gz -C ./testdata --strip-components=1
```

### 3\. 赋予脚本执行权限

```bash
cd ./testdata
chmod +x *.sh
```

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

## 9\. 启动 vLLM 推理服务

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

## 10\. 吞吐性能测试

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

## 11\. 模型精度测试

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

## 12\. 提交前自检清单

提交代码与优化方案前，务必确认以下流程全部正常运行：

1. 修改 vLLM 源码后，可正常编译生成 whl 安装包

2. 离线安装 whl 包后，vLLM 服务正常启动无报错

3. curl 单条推理请求能够正常返回模型输出

4. 吞吐脚本 `run_throughput.sh` 完整运行无异常

5. 精度脚本 `run_accuracy.sh` 完整运行无异常
