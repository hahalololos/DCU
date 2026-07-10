# DCU gfx936 调研报告

调研日期：2026-07-07

调研对象：海光 DCU / Hygon DCU，重点关注 `gfx936` 架构标识、DTK/ROCm 软件栈、大模型推理与编译适配。

## 一、结论摘要

`gfx936` 是海光 DCU 生态中较新的 ROCm/HIP 编译目标之一，公开资料里常与 Hygon DCU、DTK 25.04/26.04、DAS、vLLM、KTransformers、Paddle DCU 等实践绑定出现。它不是 AMD 官方主线 ROCm 常见的公开产品代号，而更像海光 DTK/ROCm 兼容软件栈中的 DCU 后端架构目标。

从现有资料看，海光 DCU 的路线是 GPGPU + ROCm/HIP 兼容生态，软件层以 DTK/DCU Toolkit、DAS、厂商 PyTorch/Paddle whl、vLLM-DTK 镜像等为核心。对开发者来说，`gfx936` 的关键不是“像 CUDA 一样直接跑”，而是必须严格匹配驱动、DTK、厂商 torch、编译目标、容器环境、`PYTHONPATH`、`ROCM_PATH`、`LD_LIBRARY_PATH`。

## 二、硬件与产品定位

DCU 是海光面向 AI、深度学习、HPC、大数据处理的加速卡。Paddle 官方文档将 DCU 定义为海光推出的 AI/深度学习加速卡，并说明 Paddle ROCm 版可在海光 CPU 与 DCU 上训练和预测。

产业资料显示，海光 DCU 产品线通常被称为“深算系列”：深算一号、深算二号已商用，深算三号在研发或推进中。研报和行业文章普遍认为，海光 DCU 的优势在于国产化、x86 服务器生态、ROCm/HIP 兼容路线和 AI/HPC 场景落地；短板则集中在 CUDA 生态差距、部分算子/通信/图优化成熟度、公开规格透明度不足。

`gfx936` 的公开官方规格资料较少。社区资料中明确出现 “Hygon DCU, gfx936, e.g. BW100/BW150”，但这类信息来自项目 issue 和构建实践，不能等同于厂商正式白皮书规格。

## 三、软件栈

海光 DCU 的核心软件栈大致如下：

| 层级 | 组件 | 说明 |
| --- | --- | --- |
| 驱动/运行时 | DTK Driver、hy-smi、rocm-smi、rocminfo | 设备识别、显存/功耗/温度、ROCm agent 信息 |
| 编译/运行 | DTK、HIP、hycc/hipcc/dcc | ROCm/HIP 兼容编译链 |
| AI 框架 | 厂商 PyTorch、Paddle DCU、TensorFlow DCU | 必须匹配 DTK 版本 |
| 推理框架 | vLLM-DTK、GPUStack DCU、KTransformers | 通常依赖专用镜像或 vendor torch |
| 容器/调度 | Docker、K8s、HAMi DCU sharing | 需要 `/dev/kfd`、`/dev/dri`、`/opt/hyhal` 等挂载 |

GPUStack 文档中，Hygon DCU 推理验证环境包括 Ubuntu 20.04/22.04、DTK 24.04.3、driver `rock-5.7.1-6.2.26-V1.5`，Docker 模式支持 llama-box、vLLM、vox-box；非 Docker 模式中 vLLM 后端未支持。HAMi 文档显示 DCU 共享要求 `dtk driver >= 24.04`、`hy-smi v1.6.0`，支持按显存、算力比例、DCU 类型做资源分配。

## 四、gfx936 开发要点

社区实践中，`gfx936` 编译常见环境变量包括：

```bash
export PYTORCH_ROCM_ARCH=gfx936
export ROCM_PATH=/opt/dtk
export CPUINFER_USE_ROCM=1
```

KTransformers 的 issue 记录显示，`kt-kernel v0.6.3.post1` 可在 Hygon DCU `gfx936`、DTK 26.04、vendor PyTorch 2.5.1、Python 3.10 环境下构建成功。关键注意点有两个：

1. 需要安装 `libhwloc-dev` 或对应系统的 `hwloc-devel`。
2. 构建 Python 扩展时建议使用：

```bash
pip install . --no-build-isolation --no-deps
```

否则 pip build isolation 可能拉取 PyPI 上的通用 torch，覆盖或绕开厂商 ROCm/DCU torch，导致链接错误。

Open3D Paddle 后端的 README 也列出 ROCm release 支持 `gfx906, gfx926, gfx928, gfx936`，并给出 DTK 25.04 容器环境，这进一步说明 `gfx936` 已被部分开源项目作为 DCU ROCm 架构目标纳入构建矩阵。

## 五、大模型推理实践

vLLM 是当前 DCU 推理调研中最常见的框架之一。公开实践文章记录了在海光 K100 AI + DTK 驱动 + 专用 vLLM 镜像中部署 Qwen2.5-14B-Instruct。关键启动参数包括：

```bash
--device=/dev/kfd
--device=/dev/dri
--device=/dev/mkfd
-v /opt/hyhal:/opt/hyhal:ro
--group-add video
--cap-add=SYS_PTRACE
--security-opt seccomp=unconfined
--network=host
```

该实践还指出，海光 DCU + ROCm 环境下部分 vLLM/CUDA Graph 路径可能不稳定，需要加：

```bash
--enforce-eager
```

这会牺牲一部分性能，但提高稳定性。这个结论和 ROCm/vLLM 官方文档中对 Docker、`/dev/kfd`、`/dev/dri`、`group-add video`、`ipc/network` 等要求基本一致，只是 DCU 需要额外挂载海光 HAL/驱动相关路径。

## 六、主要风险与坑点

1. 版本强绑定

   DTK、驱动、vendor torch、vLLM-DTK、DAS 包之间高度耦合。不能随意用 PyPI/官方 AMD ROCm wheel 替代厂商包。

2. 环境变量容易误配

   项目环境中必须确保本地源码路径在 `PYTHONPATH` 前面，否则可能导入全局 vLLM。DTK/HYHAL 的 env 脚本也必须正确 source，否则可能出现 `No HIP GPUs are available`。

3. 构建隔离问题

   Python 扩展编译时，`pip install .` 默认 build isolation 很容易拉错 torch。DCU/ROCm/vendor torch 场景建议显式加 `--no-build-isolation --no-deps`。

4. 算子兼容与性能缺口

   某些 CUDA Graph、FlashAttention、FlashInfer、Triton kernel、多卡通信路径可能需要 DCU 专门适配，不能按 NVIDIA CUDA 路径直接假设可用。

5. 官方公开资料不足

   `gfx936` 的正式硬件规格、对应产品 SKU、显存/带宽/互联参数等公开资料较少。调研中只能确认它作为 DTK/ROCm 架构目标在社区实践中可用，不能确认完整硬件规格。

## 七、对当前 vLLM/DCU 项目的建议

结合当前项目环境，建议按以下方式维护 DCU `gfx936` 环境：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
which python
which vllm
python -m pip show vllm
```

如果是 Python-only 变更，优先走 editable install：

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e ./vllm_cscc --no-build-isolation --no-deps
```

如果涉及 `csrc/`、HIP/C++、CMake、编译扩展，则需要完整 rebuild/install，并确认构建目标包含 `gfx936`。

## 八、参考来源

- [PaddlePaddle：海光 DCU 芯片运行飞桨](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.2/guides/09_hardware_support/rocm_docs/index_cn.html)
- [GPUStack：Running Inference with Hygon DCUs](https://docs.gpustack.ai/0.5/tutorials/running-inference-with-hygon-dcus/)
- [HAMi：Enable Hygon DCU sharing](https://project-hami.io/docs/userguide/hygon-device/enable-hygon-dcu-sharing)
- [KTransformers issue：Hygon DCU gfx936 / DTK 26.04 build works](https://github.com/kvcache-ai/ktransformers/issues/2065)
- [PFCCLab Open3D：ROCm archs include gfx936](https://github.com/PFCCLab/Open3D)
- [AMD ROCm：vLLM inference](https://rocm.docs.amd.com/en/7.12.0-preview/rocm-for-ai/vllm.html)
- [海光 DCU 上部署 Qwen2.5-14B 实践](https://gitcode.csdn.net/69e828220a2f6a37c5a16cae.html)
- [华福证券海光信息研报 PDF](https://pdf.dfcfw.com/pdf/H3_AP202405141633079130_1.pdf?1715720114000.pdf)
- [通信世界：AI 芯片受限，海光信息 DCU 能否担起替代重任](https://www.cww.net.cn/article?id=588769)
