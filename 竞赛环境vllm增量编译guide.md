# 竞赛环境 vLLM 增量编译指南（DCU gfx936 / DTK）

> 适用于本项目的比赛远端环境：BW1000 DCU（gfx936）、DTK/HIP，以及
> `/public/home/xdzs2026_c203/haha/vllm_cscc` 源码树。本指南只用于修改
> `csrc/`、HIP/C++、CMake 或扩展时的编译；纯 Python/Triton 改动不需要 CMake 编译。

## 使用边界

- 本地仅用于阅读、编辑和 Git；**只在远端**构建、运行和测试。编译前先将本地源码同步到远端。
- `DCU/` 与 `vllm_cscc/` 是独立 Git 仓库；本指南只操作 `vllm_cscc/`。
- 不修改比赛脚本、模型、tokenizer、chat template、锁定调度参数或推理语义。
- 远端为共享容器。启动服务前检查 GPU、端口；不要安装或覆盖全局 Python 中的 vLLM。

---

## 环境信息

| 项目 | 值 |
|---|---|
| GPU 架构 | gfx936 |
| DTK/ROCm 路径 | `/opt/dtk`（由 `env.sh` 加载） |
| Python | `/public/home/xdzs2026_c203/haha/.venv/bin/python` (3.10) |
| vLLM 源码 | `/public/home/xdzs2026_c203/haha/vllm_cscc` |
| 预编译 wheel | `/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl` |

---

## 前置准备：进入受控环境

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd vllm_cscc
```

`env.sh` 必须使 `vllm_cscc` 位于 `PYTHONPATH` 最前，并加载 DTK/HYHAL；否则可能导入全局 vLLM，或出现 `No HIP GPUs are available`。

`ccache` 是可选加速项。共享容器中不要直接执行 `sudo apt install`；先检查是否已提供：

```bash
command -v ccache && ccache --version
```

未安装时，删去预设中的三个 `*_COMPILER_LAUNCHER` 配置即可，增量编译仍然有效。

---

## 第一步：用预编译 wheel 做可编辑安装

在已激活的远端 `.venv` 中，显式指定比赛提供的 wheel 后执行：

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e . --no-build-isolation --no-deps
```

**作用**：从指定的比赛预编译 ROCm wheel 提取已有 `.so` 到源码树，跳过全部 C++/HIP 编译，快速建立可运行基线。

**参数说明**：

- `VLLM_USE_PRECOMPILED=1`：启用预编译 wheel 模式，提取 `.so` 并跳过 `build_ext`
- `--no-deps`：跳过运行时依赖安装
- `--no-build-isolation`：使用系统已装的工具链，避免隔离环境安装新版 `setuptools_scm` 导致 `get_version(write_to=...)` API 不兼容

> 注意：当前源码已定义 HIP 专属 `_rocm_C` 扩展，预编译提取逻辑也会尝试提取它；实际 wheel 是否包含该文件以安装输出为准。无论基线是否包含它，只要修改了 `csrc/rocm/`，均须通过后续 CMake 步骤重编译并安装。

安装后确认解释器与 editable 安装均来自项目环境：

```bash
which python
which vllm
python -m pip show vllm
```

前两项应位于 `.venv`，`Editable project location` 应为 `.../vllm_cscc`。

---

## 第二步：创建 CMakeUserPresets.json

仅在**远端** `vllm_cscc` 仓库根目录创建 `CMakeUserPresets.json`。这是机器相关配置，不应提交包含个人路径的版本：

```json
{
    "version": 6,
    "cmakeMinimumRequired": {
        "major": 3,
        "minor": 26,
        "patch": 1
    },
    "configurePresets": [
        {
            "name": "rocm-release",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/cmake-build-rocm",
            "environment": {
                "PYTORCH_ROCM_ARCH": "gfx936"
            },
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "VLLM_TARGET_DEVICE": "rocm",
                "VLLM_PYTHON_EXECUTABLE": "/public/home/xdzs2026_c203/haha/.venv/bin/python",
                "ROCM_PATH": "/opt/dtk",
                "CMAKE_INSTALL_PREFIX": "${sourceDir}",
                "CMAKE_JOB_POOLS": "compile=16",
                "CMAKE_C_COMPILER_LAUNCHER": "ccache",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache",
                "CMAKE_HIP_COMPILER_LAUNCHER": "ccache"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "rocm-release",
            "configurePreset": "rocm-release",
            "jobs": 16
        }
    ]
}
```

**关键变量说明**：

| 变量 | 作用 |
|---|---|
| `VLLM_TARGET_DEVICE: "rocm"` | 让 CMakeLists.txt 走 HIP 路径而非默认 CUDA |
| `PYTORCH_ROCM_ARCH: "gfx936"` | 环境变量，被 `cmake/utils.cmake` 的 `override_gpu_arches` 读取，指定 HIP 编译目标架构 |
| `ROCM_PATH` | DTK 安装路径，用于查找 `libamdhip64.so` 等库 |
| `VLLM_PYTHON_EXECUTABLE` | 项目 `.venv` 的 Python，CMake 据此查找与运行时一致的 PyTorch/头文件；不能使用 `/usr/bin/python` |
| `CMAKE_INSTALL_PREFIX: "${sourceDir}"` | 编译产物安装回源码树，可编辑安装即时生效 |
| `CMAKE_HIP_COMPILER_LAUNCHER: "ccache"` | HIP 编译启用 ccache 缓存 |

---

## 第三步：配置 CMake

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd vllm_cscc
cmake --preset rocm-release
```

确认输出中包含以下关键信息：

- `Target device: rocm`
- `Building PyTorch for GPU arch: gfx936`
- `gfx936` 在 HIP supported arches 列表中
- `_rocm_C` 扩展已启用（`Enabling ... extension` 中未缺少）

> 可能出现 `hipsparselt not found`、`libunwind` 路径冲突等 Warning。不要仅凭该说明忽略它们：若构建失败、动态链接失败或功能验证失败，应保留完整日志并排查环境；构建与导入均正常时再记录为已知 Warning。

---

## 第四步：编译并安装

### Python/Triton 改动：无需 CMake

例如修改 `vllm/**/*.py`、Triton 内核或测试代码时，已有 editable 安装会直接使用源码；重启相关 Python 进程或 vLLM 服务后即可生效，**无需再次执行** `pip install -e`。

> 重要：带 `VLLM_USE_PRECOMPILED=1` 的 `pip install -e` 会再次从 wheel 提取并写入 `.so`。在已完成 C++/HIP 增量编译后重复执行它，可能覆盖你刚安装的自定义扩展。只有首次建立基线、主动回退到比赛 wheel，或 editable 安装失效时才执行第一步的安装命令；之后若要继续 C++/HIP 优化，应直接用本节的 CMake 命令。

### C++/HIP/CMake 改动：首次或全量编译

```bash
cmake --build --preset rocm-release --target install
```

> 首次全量编译会编译全部 4 个扩展的源文件，耗时较长。之后增量编译会跳过未修改的文件。

### 仅编译指定目标

```bash
cmake --build --preset rocm-release --target _rocm_C
cmake --install cmake-build-rocm --component _rocm_C
```

- 第一条：编译 `csrc/rocm/` 下的 3 个源文件，产出 `_rocm_C.abi3.so` 到构建目录
- 第二条：将 `.so` 复制到源码树 `vllm/_rocm_C.abi3.so`，覆盖/新建

---

## 验证

以`_rocm_C`目标为例验证。

```bash
# 检查 .so 是否存在
ls -la vllm/_rocm_C.abi3.so

# 验证导入和算子注册
python -c "import vllm._rocm_C; import torch; print(hasattr(torch.ops._rocm_C, 'wvSplitK'))"
```

输出 `True` 即表示编译成功。

---

## 日常增量编译

修改 `csrc/rocm/` 下的文件后（如 `skinny_gemms.cu`），在已激活并加载 `env.sh` 的远端 shell 中执行：

```bash
cmake --build --preset rocm-release --target _rocm_C && cmake --install cmake-build-rocm --component _rocm_C
```

Ninja 自动检测文件变更，只重编译改动的文件，跳过未变文件，然后重新链接 `.so` 并安装。

**典型输出**（仅改了 `skinny_gemms.cu`）：

```
[1/3] Running hipify on _rocm_C extension source files.
  skinny_gemms.cu -> skinny_gemms.hip [ok]        # 重新 hipify
  attention.cu -> attention.hip [skipped]          # 跳过
[2/3] 编译 skinny_gemms.hip                          # 仅此文件编译
[3/3] Linking HIP shared module _rocm_C.abi3.so      # 重新链接
```

> 注意：hipify 是 CMake 的 `add_custom_target`，每次构建都会运行（即使文件未变），但会自动跳过未变文件的转换。使用 `--target _rocm_C` 只触发 3 个文件的 hipify 扫描；若用 `--target install` 则会扫描全部扩展，耗时更长。改动 `_C`、`_moe_C` 或显存分配器时，应将目标替换为对应名称。

---

## CMake 扩展目标一览（HIP 构建）

CMakeLists.txt 对 HIP 构建定义了 4 个扩展目标，每个目标从 `csrc/` 下多个子目录拉取源文件编译为一个 `.so`：

| 扩展目标 | 产物 | 语言 | 源文件数 | 说明 |
|---|---|---|---|---|
| `cumem_allocator` | `vllm/cumem_allocator.abi3.so` | CXX | 1 | 显存分配器，链接 `libamdhip64.so` |
| `_C` | `vllm/_C.abi3.so` | HIP | 25 | 主扩展，包含 attention、quantization、sampler 等 |
| `_moe_C` | `vllm/_moe_C.abi3.so` | HIP | 3 | MoE 相关算子 |
| `_rocm_C` | `vllm/_rocm_C.abi3.so` | HIP | 3 | ROCm 专属算子（wvSplitK 等），仅 HIP 构建启用 |

### cumem_allocator

| 源文件 | 功能 |
|---|---|
| `csrc/cumem_allocator.cpp` | 显存分配器实现 |

定义位置：`CMakeLists.txt:240-276`

### _C

| 源文件 | 功能 |
|---|---|
| `csrc/mamba/mamba_ssm/selective_scan_fwd.cu` | Mamba SSM 选择性扫描 |
| `csrc/cache_kernels.cu` | KV cache 内核 |
| `csrc/cache_kernels_fused.cu` | KV cache 融合内核 |
| `csrc/attention/paged_attention_v1.cu` | PagedAttention v1 |
| `csrc/attention/paged_attention_v2.cu` | PagedAttention v2 |
| `csrc/attention/merge_attn_states.cu` | 注意力状态合并 |
| `csrc/attention/vertical_slash_index.cu` | 纵向/斜向索引 |
| `csrc/pos_encoding_kernels.cu` | 位置编码 |
| `csrc/activation_kernels.cu` | 激活函数 |
| `csrc/layernorm_kernels.cu` | LayerNorm |
| `csrc/fused_qknorm_rope_kernel.cu` | QK-Norm + RoPE 融合 |
| `csrc/layernorm_quant_kernels.cu` | LayerNorm + 量化融合 |
| `csrc/sampler.cu` | 采样器 |
| `csrc/topk.cu` | Top-K |
| `csrc/cuda_view.cu` | 视图操作 |
| `csrc/quantization/gptq/q_gemm.cu` | GPTQ 量化 GEMM |
| `csrc/quantization/w8a8/int8/scaled_quant.cu` | INT8 量化 |
| `csrc/quantization/w8a8/fp8/common.cu` | FP8 量化通用 |
| `csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu` | 融合 LayerNorm + per-token 量化 |
| `csrc/quantization/gguf/gguf_kernel.cu` | GGUF 内核 |
| `csrc/quantization/activation_kernels.cu` | 量化激活函数 |
| `csrc/cuda_utils_kernels.cu` | CUDA 工具内核 |
| `csrc/custom_all_reduce.cu` | 自定义 AllReduce |
| `csrc/custom_quickreduce.cu` | QuickReduce（仅 HIP） |
| `csrc/torch_bindings.cpp` | PyTorch 算子注册绑定 |

定义位置：`CMakeLists.txt:282-981`

### _moe_C

| 源文件 | 功能 |
|---|---|
| `csrc/moe/torch_bindings.cpp` | PyTorch 算子注册绑定 |
| `csrc/moe/moe_align_sum_kernels.cu` | MoE 对齐求和内核 |
| `csrc/moe/topk_softmax_kernels.cu` | Top-K + Softmax 内核 |

> CUDA 构建额外包含 `moe_wna16.cu`、`grouped_topk_kernels.cu`、`router_gemm.cu` 及 permute 系列内核，HIP 构建不含这些。

定义位置：`CMakeLists.txt:993-1165`

### _rocm_C

| 源文件 | 功能 |
|---|---|
| `csrc/rocm/torch_bindings.cpp` | PyTorch 算子注册绑定（wvSplitK 等） |
| `csrc/rocm/skinny_gemms.cu` | 瘦 GEMM 内核（wvSplitK / LLGemm1 等，decode 阶段 GEMM 加速） |
| `csrc/rocm/attention.cu` | ROCm 专属注意力内核 |

定义位置：`CMakeLists.txt:1167-1185`

> `_rocm_C` 仅在 `VLLM_GPU_LANG STREQUAL "HIP"` 时定义，CUDA 构建不包含此目标。

---

## 完全清理重建

遇到持续编译错误时：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd vllm_cscc
rm -rf cmake-build-rocm
cmake --preset rocm-release
cmake --build --preset rocm-release --target _rocm_C
cmake --install cmake-build-rocm --component _rocm_C
```

---

## 背景原理

| 概念 | 说明 |
|---|---|
| 两套构建系统 | `setup.py` 控制 `pip install`/`bdist_wheel`；`CMakeLists.txt` 控制 `cmake --build`。两者独立，CMakeLists.txt 中 `_rocm_C` 对 HIP 构建无条件启用 |
| hipify | HIP 构建前，`cmake/hipify.py` 将 `.cu` 文件转为 `.hip`，替换 CUDA API 为 HIP 等价物。作为 `add_custom_target` 每次都会运行，但自动跳过未变文件 |
| 预编译 wheel | 从比赛指定路径提取已编译 `.so`，作为增量编译的起点基线；必须与当前 DTK/Python 环境匹配 |
| Ninja 增量 | 跟踪构建目录中的文件依赖，仅重编译变更文件，未变文件跳过。注意：Ninja 只认构建目录中的编译状态，不认源码树中已有的 `.so` |
| ccache | 缓存编译结果，相同输入直接命中缓存跳过编译 |
