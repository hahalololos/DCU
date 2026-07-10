# AGENTS.md

赛题文档：`智能计算创新设计赛-基于国产加速卡的千问大模型推理服务优化-技术方案.md`。

## 文档与仓库

- 文档尽量使用中文；计划放 `docs/plans/`（文件名含时间戳），调研放 `调研交付物/`。
- `docs/progress.md` 必须按时间戳简要记录**vLLM 源码优化工作与测试结果**。
- `DCU/`、`vllm_cscc/`、`mcp-ssh/` 是独立仓库，分别管理各自的远端与提交历史。

## 环境与分工

- 远端通过 `mcp-ssh` 的 `scnet-docker` 访问；默认目录为
  `/public/home/xdzs2026_c203/haha`。发现 `mcp-ssh` 问题可修改本地
  `./mcp-ssh/` 源码。
- 本地仅用于阅读、编辑和 Git；远端仅用于构建、运行、测试。测试前先同步本地源码；若在远端临时修改，立即同步回本地。
- 远端是共享 root 容器，禁止把本项目 vLLM 安装到全局 Python。每次远端执行项目前必须：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
```

- 路径：`.venv`、`vllm_cscc/`、`env.sh`、`.cache/` 均在上述目录下。`env.sh` 必须加载 DTK/HYHAL，并让 `vllm_cscc` 位于 `PYTHONPATH` 最前；否则可能导入全局 vLLM 或报 `No HIP GPUs are available`。
- 启动 vLLM 前检查共享 GPU 与端口；不得影响队友。4B 可共享测试。

## PRA26 容器状态提醒

- Agent 不得自动创建、重启、停止或删除 PRA26 容器。
- 远端操作前若发现 `scnet-docker` 不可达，应先通过 `scnet-login` 检查当前用户是否存在 RUNNING 作业。
- 确认容器未启动、作业已结束，或疑似达到 4 小时限制时，立即停止远端构建与测试，提醒
  用户在平台页面手动启动或重启容器。
- 用户完成手动重启后，计算节点名和容器 IP 可能变化；Agent 应先刷新 `scnet-computer`、`scnet-docker` 的 SSH 配置并验证连接，再继续原任务。

## 比赛边界

提交物仅为修改后的 `vllm_cscc` 源码。不得：

- 改原始比赛脚本（用户明确要求除外）、模型权重/tokenizer/chat template/结构，或推理语义；
- 截断输入/输出、跳样本/层、token pruning、early-exit，或缓存测试集、答案、中间结果、可复用量化/压缩权重；
- 使用投机解码、draft model、外挂/自训练预测器、多头预测；
- 调整或绕开锁定参数、scheduler 行为、bench 统计及解析（含 `max-model-len`、`max-num-seqs`、`max-num-batched-tokens`）。

优先优化：KV Cache/显存管理、非持久化 KV 或激活动态量化、Attention/PagedAttention、Decode Linear/Attention 融合与内存路径，以及不改变有效计算和语义的 HIP/C++/Triton kernel。

## 构建

涉及构建时，先参阅根目录 `竞赛环境vllm增量编译guide.md`。该指南规定远端 `.venv + env.sh + 指定比赛 wheel` 的基线安装、CMake/Ninja 增量编译及验证流程；纯 Python/Triton 改动无需 CMake，已有 editable 安装时重启相关进程即可生效。注意：在已编译自定义扩展后，不要重复执行带 `VLLM_USE_PRECOMPILED=1` 的 editable 安装，以免 wheel 覆盖源码树中的 `.so`。

仅 Python 改动可 editable install：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e ./vllm_cscc --no-build-isolation --no-deps
```

涉及 `csrc/`、HIP/C++、CMake 或扩展时必须重建。首次建立基线、构建状态不可用或需要全量构建时执行：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
./build_vllm_wheel_install.sh
```

已有成功的 CMake 基线后，日常局部 C++/HIP 改动可按上述指南选择对应扩展目标进行 Ninja 增量编译与安装；不得用增量构建绕过必要的全量重建或功能验证。

构建后检查：

```bash
which python; which vllm; python -m pip show vllm
```

两者应位于 `.venv`，且 editable location 为 `.../vllm_cscc`。由于 `env.sh` 优先源码树，wheel 中已有扩展但源码树没有 `.so` 时会报 `No module named vllm._rocm_C`。

## 测试

- 默认用 4B 筛选；先完成可重复的 4B 对照，确认完成率、正确性、时延无回退且有明显收益，才按需验证 27B。27B 共享存储启动通常约半小时，非必要不测。
- 4B 模型：`/public/home/xdzs2026_c203/models/Qwen3.5-4B`；27B 模型：`/public/home/xdzs2026_c203/models/Qwen3.5-27B`。

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
cd testdata
./start_vllm_4b.sh
./run_throughput_4b.sh 16-32K 10
```

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
