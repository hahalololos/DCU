# Qwen3.5 专用 UA2D 快速榜单提交 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 验证当前 `vllm_cscc` 专用 UA2D 候选在 4B 中长上下文的稳定收益，完成 27B 最小正确性门禁，并形成可直接提交榜单的源码状态与验证记录。

**Architecture:** 冻结 `vllm_cscc` 提交 `c274c6b`，不实现新 kernel。远端先建立原 UA2D fastpath 与当前专用 UA2D 的同环境 4B 热态 A/B，再复用现有定向测试验证 27B `(24,4,6)` 路径；所有原始结果保存在远端 `testdata/experiments/`，结论同步到本地 `docs/progress.md`。

**Tech Stack:** Python 3.10、vLLM 0.18.1、PyTorch 2.10.0、Triton、ROCm/DTK、gfx936、Qwen3.5-4B/27B、`vllm bench serve`、Git。

## Global Constraints

- 仅提交 `vllm_cscc` 源码；不得修改比赛脚本、模型、权重、tokenizer、chat template、scheduler、bench 统计或锁定参数。
- 本地只阅读、编辑和 Git；远端 `/public/home/xdzs2026_c203/haha` 只构建、运行和测试。
- 每次远端命令先执行 `cd /public/home/xdzs2026_c203/haha && source .venv/bin/activate && source env.sh`。
- 启动服务前检查 GPU、8001/8002 端口和队友进程；只停止本工作区启动的进程。
- 访问本地服务使用 `curl --noproxy '*'`，并设置 `NO_PROXY=127.0.0.1,localhost` 与 `no_proxy=127.0.0.1,localhost`。
- 当前候选固定为 `c274c6b`，本轮不新增源码优化；若测试发现缺陷，停止快速提交并回到设计阶段。
- 4B 三档均须 `10/10` 完成；中档或长档请求吞吐至少一档可重复提升约 `3%`，另一重要档位不得稳定回退超过约 `2%`，TPOT P99 不得稳定恶化超过约 `2%`。
- 27B 必须通过现有定向正确性与专用 guard 命中验证；完整三档性能不是本轮硬门槛。

---

## File Map

- Read: `竞赛环境vllm增量编译guide.md` — 确认纯 Python/Triton 同步与 editable 环境要求。
- Read: `docs/plans/qwen35_ua2d_leaderboard_submission_design_20260712_1002.md` — 本计划的验收依据。
- Read/execute: `testdata/start_vllm_4b.sh` — 启动 4B 服务，不修改脚本。
- Read/execute: `testdata/run_throughput_4b.sh` — 三档 10 条吞吐门禁，不修改脚本。
- Read/execute: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py` — 4B/27B UA2D 定向正确性测试。
- Modify: `docs/progress.md` — 记录环境、A/B、27B 门禁和最终决策。
- Read only: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py` — 核对当前候选、guard 和提交哈希，不做源码改动。

### Task 1: 建立干净的远端候选环境

**Files:**
- Read: `竞赛环境vllm增量编译guide.md`
- Read: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`

**Interfaces:**
- Consumes: 本地 `vllm_cscc` 提交 `c274c6b`。
- Produces: 源码一致、Python/vLLM 指向正确、GPU 与端口可用的远端测试环境。

- [ ] **Step 1: 本地确认候选和工作树**

Run:

```bash
cd /home/haha/DCU/vllm_cscc
git status --short --branch
git rev-parse HEAD
git log -5 --oneline
```

Expected: HEAD 为 `c274c6b`，没有未提交源码改动；最近提交包含 `515ec50`、`882f326`、`5bf55cb`。

- [ ] **Step 2: 阅读增量编译指南并确认本轮无需 CMake**

Run:

```bash
cd /home/haha/DCU
sed -n '1,260p' 竞赛环境vllm增量编译guide.md
```

Expected: 确认当前候选仅为已有 Python/Triton 路径，不覆盖源码树已有 `_rocm_C`。

- [ ] **Step 3: 检查远端连接、PRA26 作业、GPU 与端口**

通过 `scnet-docker` 执行：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
rocm-smi
ss -ltnp | grep -E ':8001|:8002' || true
ps -ef | grep -E 'vllm|EngineCore|run_throughput' | grep -v grep || true
```

Expected: PRA26 容器可达；有足够空闲显存；计划使用的端口未被队友占用。若容器不可达，使用 `scnet-login` 检查作业；作业未运行则停止本轮并请用户在平台处理。

- [ ] **Step 4: 同步本地源码到远端**

使用项目现有 `mcp-ssh/scnet-docker` 文件同步能力，将 `/home/haha/DCU/vllm_cscc/` 同步到 `/public/home/xdzs2026_c203/haha/vllm_cscc/`，排除 `.git/`、本地构建缓存和无关实验产物。

Expected: 远端当前源码与本地 `c274c6b` 一致；任何远端临时源码改动先同步回本地或明确清理，不能静默覆盖。

- [ ] **Step 5: 核对关键文件和运行环境**

本地执行：

```bash
cd /home/haha/DCU
sha256sum vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py \
  vllm_cscc/vllm/envs.py \
  vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py
```

远端执行相同三个文件的 `sha256sum`，随后执行：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
which python
which vllm
python -m pip show vllm
python -c 'import vllm; print(vllm.__file__)'
```

Expected: 三个 SHA256 逐项一致；`python`、`vllm` 位于 `.venv`；editable location 和导入路径指向 `/public/home/xdzs2026_c203/haha/vllm_cscc`。

### Task 2: 复核专用 UA2D 的 4B/27B 定向正确性

**Files:**
- Test: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Read: `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`

**Interfaces:**
- Consumes: Task 1 的远端候选环境。
- Produces: 当前候选对 4B/27B 形状、边界与回退路径的正确性证据。

- [ ] **Step 1: 定位现有 Qwen3.5 专用测试名称**

Run locally:

```bash
cd /home/haha/DCU/vllm_cscc
rg -n 'qwen35|specialized|GQA|4095|4096' tests/kernels/attention/test_triton_unified_attention.py
```

Expected: 找到提交 `882f326` 中覆盖 4B、27B、GQA=6、cache-block 边界和长 query 的测试。

- [ ] **Step 2: 在远端运行定向测试**

优先使用测试文件中实际测试节点名执行：

```bash
cd /public/home/xdzs2026_c203/haha/vllm_cscc
source ../.venv/bin/activate && source ../env.sh
python -m pytest tests/kernels/attention/test_triton_unified_attention.py -k 'qwen35' -q
```

Expected: 所有收集到的 Qwen3.5 定向测试通过。若仍因共享环境缺少 `tblib` 无法收集，使用此前已验证的直接加载测试模块方式调用相同测试函数，不安装全局依赖，并保存完整输出。

- [ ] **Step 3: 核对 27B guard 条件未被清理提交改变**

Run:

```bash
cd /public/home/xdzs2026_c203/haha/vllm_cscc
source ../.venv/bin/activate && source ../env.sh
rg -n '_is_qwen35|kernel_qwen35|24, 4, 6|num_queries_per_kv' \
  vllm/v1/attention/ops/triton_unified_attention.py
```

Expected: 27B `(24,4,6)`、BF16、head size 256、4 KV heads、gfx936、causal prefill guard 存在；非候选仍回退通用路径。

### Task 3: 完成原 fastpath 与专用 UA2D 的 4B 热态 A/B

**Files:**
- Execute: `testdata/start_vllm_4b.sh`
- Execute: `testdata/run_throughput_4b.sh`
- Read: 远端 `testdata/experiments/*/result.json`

**Interfaces:**
- Consumes: Task 2 已通过的候选。
- Produces: 三档原 fastpath baseline 与专用 UA2D candidate 的同环境性能数据。

- [ ] **Step 1: 选择实验目录和独占端口**

远端设置：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
STAMP=$(date +%Y%m%d_%H%M)
echo "$STAMP"
```

Expected: 记录本轮时间戳；若 8001 被队友占用，停止执行并选择不冲突的本工作区方案，不能修改队友服务。

- [ ] **Step 2: 恢复原 fastpath baseline**

使用 Git 中专用 UA2D 引入前、但包含当前 LLMM1 shape filter 的对照提交建立临时工作树；对照保持 `39c4654` 的 LLMM1 配置，并保留原 UA2D fastpath。不得修改本地主分支或覆盖候选源码。

Run:

```bash
cd /public/home/xdzs2026_c203/haha/vllm_cscc
git worktree remove /public/home/xdzs2026_c203/haha/vllm_cscc_ua2d_baseline --force 2>/dev/null || true
git worktree add --detach /public/home/xdzs2026_c203/haha/vllm_cscc_ua2d_baseline 39c4654
cd /public/home/xdzs2026_c203/haha/vllm_cscc_ua2d_baseline
git rev-parse HEAD
git status --short
```

Expected: HEAD 为 `39c4654`，工作树干净。

- [ ] **Step 3: 启动 baseline 4B 服务并检查健康状态**

Run:

```bash
cd /public/home/xdzs2026_c203/haha/testdata
source ../.venv/bin/activate && source ../env.sh
export PYTHONPATH=/public/home/xdzs2026_c203/haha/vllm_cscc_ua2d_baseline:${PYTHONPATH}
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
MODEL_DIR=/tmp/qwen35_4b_model_20260711 ./start_vllm_4b.sh
```

在另一远端会话检查：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8001/v1/models
```

Expected: 服务正常返回 Qwen3.5-4B 模型信息；启动日志无 OOM、VM fault 或负 KV cache 可用显存。

- [ ] **Step 4: 运行 baseline 三档热态门禁**

Run:

```bash
cd /public/home/xdzs2026_c203/haha/testdata
source ../.venv/bin/activate && source ../env.sh
rm -rf test_4b
./run_throughput_4b.sh 4-8K 10
./run_throughput_4b.sh 8-16K 10
./run_throughput_4b.sh 16-32K 10
mkdir -p experiments/UA2D-SUBMIT-BASE-${STAMP}
cp -a test_4b/. experiments/UA2D-SUBMIT-BASE-${STAMP}/
```

Expected: 三档均 `10/10`；结果保存到独立时间戳目录。首轮若包含 Triton 冷编译，保持服务运行并重跑受影响档位，以第二个热轮作为正式数据。

- [ ] **Step 5: 只停止 baseline 本工作区服务**

记录启动进程 PID，发送正常终止信号并确认显存释放。不得使用会匹配队友进程的宽泛 `pkill`。

Expected: baseline 服务退出；GPU 显存回到启动前状态；8001 释放。

- [ ] **Step 6: 启动当前候选并运行相同三档**

使用 `/public/home/xdzs2026_c203/haha/vllm_cscc`，重复 Step 3–5；结果保存为：

```bash
mkdir -p experiments/UA2D-SUBMIT-CAND-${STAMP}
cp -a test_4b/. experiments/UA2D-SUBMIT-CAND-${STAMP}/
```

Expected: 三档均 `10/10`，无外部并发、冷编译或代理污染。

- [ ] **Step 7: 计算 A/B 并执行门禁判定**

从六个 `result.json` 提取 request throughput、output throughput、P99 TTFT、P99 TPOT、completed、input tokens 和 output tokens，按下式逐档计算：

```text
relative_change = (candidate - baseline) / baseline * 100%
```

Expected: 中档或长档请求吞吐至少一档提升约 `3%`；另一重要档位回退不超过约 `2%`；TPOT P99 无超过约 `2%` 的稳定恶化。未通过则停止，不进入快速榜单提交。

### Task 4: 完成 27B 最小门禁

**Files:**
- Test: `vllm_cscc/tests/kernels/attention/test_triton_unified_attention.py`
- Read/execute: `testdata/start_vllm.sh`
- Read/execute: `testdata/run_throughput.sh`

**Interfaces:**
- Consumes: Task 3 通过的候选。
- Produces: 27B 正确性、guard 命中和可选中档端到端证据。

- [ ] **Step 1: 再次确认 GPU 独占窗口**

Run:

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate && source env.sh
rocm-smi
ss -ltnp | grep -E ':8001|:8002' || true
ps -ef | grep -E 'vllm|EngineCore' | grep -v grep || true
```

Expected: 27B 启动不会影响队友；若共享存储或 GPU 条件不具备，只保留 Task 2 的 27B kernel 定向正确性作为最低门禁。

- [ ] **Step 2: 在可用时启动 27B 并确认专用路径可运行**

Run:

```bash
cd /public/home/xdzs2026_c203/haha/testdata
source ../.venv/bin/activate && source ../env.sh
export PYTHONPATH=/public/home/xdzs2026_c203/haha/vllm_cscc:${PYTHONPATH}
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
./start_vllm.sh
```

Expected: 服务完成模型加载和 KV cache 初始化，无 OOM、VM fault、NaN/Inf 或 Python fallback 异常。若加载接近半小时但仍持续正常读取，不误判为故障；若 PRA26 作业接近生命周期限制，停止并保留已完成的定向测试证据。

- [ ] **Step 3: 资源允许时优先跑 8–16K 10 条**

Run in a second session:

```bash
cd /public/home/xdzs2026_c203/haha/testdata
source ../.venv/bin/activate && source ../env.sh
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
./run_throughput.sh 8-16K 10
```

Expected: `10/10` 完成，SLA 指标无明显异常。此结果用于增强信心，不要求在本轮构造完整 27B A/B。

- [ ] **Step 4: 只停止本工作区 27B 服务并记录日志**

使用启动时记录的 PID 正常终止服务，保存启动日志和可选 8–16K 结果目录。

Expected: GPU 和端口释放，不影响任何队友进程。

### Task 5: 记录结论并形成提交候选

**Files:**
- Modify: `docs/progress.md`
- Read: `docs/rankings/HistRank.md`

**Interfaces:**
- Consumes: Tasks 1–4 的哈希、测试输出、性能结果和异常说明。
- Produces: 可审计的提交决策与榜单提交候选哈希。

- [ ] **Step 1: 在 `docs/progress.md` 添加时间戳记录**

新增标题 `### 专用 UA2D 快速榜单门禁`，并逐项写入候选提交 `c274c6b`、本地与远端三个关键文件 SHA256、baseline/candidate 的完整实验目录、三档完成率、request/output throughput、P99 TTFT、P99 TPOT、逐档相对变化、27B 测试通过数量、guard 核对结果、可选 8–16K 结果和最终提交或回滚决定。所有数字直接引用本轮保存的日志与 `result.json`，不得使用估算值。

- [ ] **Step 2: 检查文档与仓库状态**

Run:

```bash
cd /home/haha/DCU
git diff --check
git status --short
cd vllm_cscc
git status --short --branch
git rev-parse HEAD
```

Expected: DCU 只包含本轮进度文档改动；`vllm_cscc` HEAD 为 `c274c6b` 且工作树干净。

- [ ] **Step 3: 提交进度记录**

Run:

```bash
cd /home/haha/DCU
git add docs/progress.md
git commit -m "docs: record UA2D leaderboard submission gate"
```

Expected: 提交成功，提交内容仅为本轮验证记录。

- [ ] **Step 4: 给出最终提交建议**

若所有硬门禁通过，明确报告：榜单候选为 `vllm_cscc@c274c6b`，可提交源码；同时列出 4B A/B 和 27B 门禁证据。若任一硬门禁失败，明确报告失败指标和对应实验目录，不提交榜单，也不现场放宽门槛。

### Task 6: 榜单结果回填

**Files:**
- Modify: `docs/rankings/HistRank.md`
- Modify: `docs/progress.md`

**Interfaces:**
- Consumes: 用户或平台提供的新榜单结果。
- Produces: 本轮优化的真实 27B 榜单结论和下一优化方向。

- [ ] **Step 1: 获取新榜单三档数据**

记录提交次数、当时排名、16–32K/4–8K/8–16K 吞吐、SLA 扣分、最终得分和精度扣分。

- [ ] **Step 2: 更新历史排名表与进度结论**

在 `docs/rankings/HistRank.md` 增加一行；在 `docs/progress.md` 对比当前 `77.8753`，分别计算三档和总分变化。

- [ ] **Step 3: 根据榜单选择下一主线**

若中长档与总分提升，下一轮进入 UA3D decode KV-load 优化；若长档无改善或回退，下一轮优先拆分 LLMM1 shape filter。SLA 或完成率异常时，先系统化诊断，不继续叠加优化。

- [ ] **Step 4: 提交榜单记录**

Run:

```bash
cd /home/haha/DCU
git add docs/rankings/HistRank.md docs/progress.md
git commit -m "docs: record specialized UA2D leaderboard result"
```

Expected: 提交成功，排名和进度数据与平台结果一致。
