# AGENTS.md

必读赛题文档：`智能计算创新设计赛-基于国产加速卡的千问大模型推理服务优化-技术方案.md`。

## 文档与仓库

- 文档尽量中文。
- 计划写入 `docs/plans/`（文件名含时间戳），调研写入 `调研交付物/`。
- 每次 vLLM 源码优化及测试结果，按时间戳简记至 `docs/progress.md`。
- `docs/rankings/`有队伍性能排名，目标前20名。
- `DCU/`、`vllm_cscc/`、`mcp-ssh/` 是独立 Git 仓库，各自维护远端与提交历史。

## 远端环境与容器

- 本地仅阅读、编辑、Git；远端仅构建、运行、测试。测试前同步本地源码；远端临时改动必须立即同步回本地。
- 远端统一经 `mcp-ssh` 的 `scnet-docker-1` 访问，工作目录为
  `/public/home/acoh0h1o0p/DCU`。
- 以本地仓库源码和提交历史为权威。每次构建、测试前先将本地相关源码和测试工具同步到远端，不依赖远端
  `git status`、分支或提交记录判断版本。
- `scnet-docker-1` 为本项目独占 root 容器，无需虚拟环境隔离，可以直接使用系统 Python、
  全局安装或构建本项目 vLLM。首次使用或同步后仍须执行 `which python`、`which vllm` 和
  `python -m pip show vllm`，确认实际命令、版本和源码位置；不得误用系统中其他 vLLM 副本。
- 启动 vLLM 前需清理本项目遗留进程并检查 GPU、端口，避免同一任务的旧服务污染结果。
- 每次远端操作先执行：

```bash
cd /public/home/acoh0h1o0p/DCU
source /opt/dtk/env.sh
source /opt/hyhal/env.sh
export PYTHONPATH="$PWD/vllm_cscc:${PYTHONPATH:-}"
```
  上述两个 `/opt` 环境脚本用于加载系统 DTK/HYHAL 运行库，缺少它们时 vendor
  PyTorch 会因找不到 `libgalaxyhip.so.5` 而无法导入。

- 访问 `127.0.0.1`/`localhost` 时须设置大小写 `NO_PROXY` 或使用 `curl --noproxy '*'`；若出现 502/503，先检查响应是否来自 Squid/代理，避免误判为 vLLM 故障。
- 除非用户明确要求，不得创建、重启、停止或删除 PRA26 容器。`scnet-docker-1` 不可达时，
  先用 `scnet-login-1` 检查 `squeue` ；若容器未启动、作业结束或疑似达到时限，停止远端
  工作并请用户在平台手动处理。用户处理后，刷新 `scnet-computer-1`/`scnet-docker-1` SSH
  配置并验证连接。

## 比赛边界

提交物仅限修改后的 `vllm_cscc` 源码。除非用户明确要求，不得修改比赛脚本；不得修改模型权重、tokenizer、chat template、模型结构或推理语义。

不得截断输入/输出、跳样本/层、token pruning、early-exit，或缓存测试集/答案/中间结果/可复用量化或压缩权重；不得使用投机解码、draft model、外挂或自训练预测器、多头预测；不得调整或绕过锁定参数、scheduler、bench 统计/解析（含 `max-model-len`、`max-num-seqs`、`max-num-batched-tokens`）。

优先：KV Cache/显存、非持久化 KV 或激活动态量化、Attention/PagedAttention、Decode Linear/Attention 融合与内存路径，以及不改变有效计算和语义的 HIP/C++/Triton kernel。

## 构建与验证

构建前必读根目录 `竞赛环境vllm增量编译guide.md`。纯 Python/Triton 改动无需 CMake，已有 editable 安装时重启相关进程即可；已有自定义扩展后，不得再用带 `VLLM_USE_PRECOMPILED=1` 的 editable 安装覆盖源码树 `.so`。

- 仅 Python 改动可执行：

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/acoh0h1o0p/DCU/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e ./vllm_cscc --no-build-isolation --no-deps
```

  执行前必须确认上述 wheel 文件确实存在。若新容器尚无可复用 wheel 或扩展，按构建指南
  建立新基线，不得引用旧账号目录。

- 涉及 `csrc/`、HIP/C++、CMake 或扩展必须重建；首次基线、构建状态不可用或需全量构建时执行 `./build_vllm_wheel_install.sh`。有成功 CMake 基线后，按指南进行对应 Ninja 增量构建；不得借此绕过必要全量构建或功能验证。
- 构建后执行 `which python; which vllm; python -m pip show vllm`：允许使用系统路径，但
  editable location 或实际导入路径必须指向 `/public/home/acoh0h1o0p/DCU/vllm_cscc`。
  若 wheel 有扩展而源码树无 `.so`，`PYTHONPATH` 的源码优先级仍可能导致
  `No module named vllm._rocm_C`。

## 测试

4B、27B 已完整复制到容器本地 `/root/models/`，后续启动优先使用该副本以缩短加载时间；共享存储路径仅作为复制源和回退。

- 4B（首选）：`/root/models/Qwen3.5-4B`
- 4B（共享存储回退）：`/public/home/acoh0h1o0p/models/Qwen3.5-4B`
- 27B（首选）：`/root/models/Qwen3.5-27B`
- 27B（共享存储回退）：`/public/home/acoh0h1o0p/models/Qwen3.5-27B`

```bash
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
