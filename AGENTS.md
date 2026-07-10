# AGENTS.md

赛题文档 `智能计算创新设计赛-基于国产加速卡的千问大模型推理服务优化-技术方案.md`。

## 文档规范

- 文档尽量使用中文。
- `docs/plans` 下放置计划方案，需要时间戳。
- 调研文档放在 `调研交付物/`。
- 必须维护 `docs/progress.md`，按时间戳简要记录，包含完成的工作和结果，尽量少字表述清楚即可。

## git仓库规范
`DCU/`、`vllm_cscc/`和`mcp-ssh/`分别做了单独的仓库关联和管理。

## 远端环境

- 通过 `mcp-ssh` 连接远端，主机别名为 `scnet-docker`。
- 注：使用 `mcp-ssh` 过程中遇到什么问题、不足可以直接优化本地 `./mcp-ssh/` 里的源码。
- 默认远端工作目录：

```bash
/public/home/xdzs2026_c203/haha
```

## Python 与运行环境

这是共享 root 容器，禁止把本项目的 vLLM 安装到容器全局 Python 环境。

远端执行项目命令前必须使用：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
```

核心路径：

```bash
venv:   /public/home/xdzs2026_c203/haha/.venv
source: /public/home/xdzs2026_c203/haha/vllm_cscc
env:    /public/home/xdzs2026_c203/haha/env.sh
cache:  /public/home/xdzs2026_c203/haha/.cache
```

`env.sh` 必须加载厂商 DTK/HYHAL 环境，并把 `vllm_cscc` 放在
`PYTHONPATH` 最前面；否则可能导入全局 vLLM，或 PyTorch 报
`No HIP GPUs are available`。

## 本地与远端分工

- 本地机器用于阅读、编辑源码和维护 git 历史。
- 远端工作区只用于构建、运行和测试。
- 测试前必须把本地源码同步到远端。
- 尽量避免直接在远端临时改源码；如果不得不改，必须立即同步回本地。
- 启动 vLLM 前先检查共享 GPU 和端口占用，避免影响队友。4B模型可与队友共同运行测试。

## 比赛约束

比赛提交物为修改后的 `vllm_cscc` 源码。

禁止：

- 修改原始比赛测试脚本，除非用户明确要求。
- 修改模型权重、tokenizer、chat template、模型结构或推理语义。
- 通过截断输入、减少输出、跳过样本、跳过层、token pruning、early-exit
  等方式规避真实推理开销。
- 预缓存测试集、答案、中间结果，或生成可复用的量化权重/压缩权重缓存。
- 引入投机解码、draft model、外挂小模型、自训练预测器、多头预测等能力。
- 调整或绕开锁定参数与评测口径，包括 `max-model-len`、`max-num-seqs`、
  `max-num-batched-tokens`、batch scheduler 相关行为、bench 统计口径和结果解析。

允许优先考虑：

- KV Cache 与显存管理优化。
- 推理过程中的非持久化 KV Cache 量化、激活动态量化、kernel 内部临时低精度计算。
- Attention / PagedAttention kernel 或 backend 路径优化。
- Decode 阶段 Linear、Attention、算子融合、kernel launch 与内存拷贝路径优化。
- 不改变模型有效计算结构和输出语义的自定义 HIP/C++/Triton kernel。

## 构建规范

Python-only 改动可以使用 editable install：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e ./vllm_cscc --no-build-isolation --no-deps
```

涉及 `csrc/`、HIP/C++、CMake 或编译扩展时，必须重新构建并安装：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
./build_vllm_wheel_install.sh
```

构建/安装后做基础检查：

```bash
which python
which vllm
python -m pip show vllm
```

期望：`python` 和 `vllm` 位于 `.venv` 下，vLLM editable location 为
`/public/home/xdzs2026_c203/haha/vllm_cscc`。

注意：`env.sh` 会让源码树优先于 site-packages 被导入。若编译扩展已经打入
wheel 但源码树下缺少 `.so`，可能出现 `No module named vllm._rocm_C`。

## 测试规范

- 优化实验默认优先使用 4B 模型筛选。
- 先完成 4B 必要的对照测试，确认结果有明显、可重复的提升且完成率、正确性与时延没有回退后，才考虑是否进行到 27B 模型验证，确有必要则测27B。
- 用于共享存储，27B模型启动很慢是正常现象，通常需要半个小时。所以除非必须尽量不测27B，多用架构一致的4B代替。

4B 模型使用副本脚本：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd testdata
./start_vllm_4b.sh
./run_throughput_4b.sh 16-32K 10
```

模型路径：

```bash
27B: /public/home/xdzs2026_c203/models/Qwen3.5-27B
4B:  /public/home/xdzs2026_c203/models/Qwen3.5-4B
```

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tool** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them, including dynamic-dispatch hops grep can't follow. Name a file or symbol in the query to read its current line-numbered source. If it's listed but deferred, load it by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` prints the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
