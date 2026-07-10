# 竞赛环境 vLLM 增量编译指南

适用于本项目远端 BW1000 DCU（gfx936）环境的 `vllm_cscc`。仅在修改 `csrc/`、HIP/C++、CMake 或扩展时使用 CMake/Ninja；纯 Python/Triton 改动无需编译。

## 规则与环境

- 本地仅编辑和 Git；构建、运行、测试只在远端执行，且先同步源码。
- 远端目录为 `/public/home/xdzs2026_c203/haha`；禁止安装或导入全局 vLLM。
- 每次执行前均加载项目环境：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
cd vllm_cscc
```

`env.sh` 必须加载 DTK/HYHAL，并让 `vllm_cscc` 位于 `PYTHONPATH` 最前。否则可能导入全局 vLLM，或报 `No HIP GPUs are available`。

## 首次建立基线

先用比赛提供的预编译 wheel 建立 editable 安装：

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=/public/home/xdzs2026_c203/haha/vllm_cscc/dist/vllm-0.18.1+das.dtk2604-cp310-cp310-linux_x86_64.whl
export VLLM_USE_PRECOMPILED=1
python -m pip install -e . --no-build-isolation --no-deps
which python; which vllm; python -m pip show vllm
```

`python`、`vllm` 应位于 `.venv`，且 editable location 应为 `.../vllm_cscc`。预编译 wheel 用于提供已有 `.so` 基线；修改 C++/HIP 后仍须重新编译对应扩展。

> 已经增量编译过自定义扩展时，不要重复执行上述安装命令：它会从 wheel 再次提取 `.so`，可能覆盖自定义产物。纯 Python/Triton 改动在已有 editable 安装下只需重启 Python 进程或 vLLM 服务。

## 配置 CMake（首次）

仅在远端 `vllm_cscc` 根目录创建（该文件已被忽略，不提交）`CMakeUserPresets.json`：

```json
{
  "version": 6,
  "configurePresets": [{
    "name": "rocm-release",
    "generator": "Ninja",
    "binaryDir": "${sourceDir}/cmake-build-rocm",
    "environment": {"PYTORCH_ROCM_ARCH": "gfx936"},
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
  }],
  "buildPresets": [{"name": "rocm-release", "configurePreset": "rocm-release", "jobs": 16}]
}
```

若远端没有 `ccache`（用 `command -v ccache` 检查），删除三个 `*_COMPILER_LAUNCHER` 项。随后配置：

```bash
cmake --preset rocm-release
```

确认输出包含 `Target device: rocm` 与 `gfx936`。构建或导入失败时不要忽略 CMake Warning，应保留完整日志排查。

## 构建与验证

首次 C++/HIP 构建、构建状态损坏，或需要全量构建时，按项目规定使用完整构建脚本：

```bash
cd /public/home/xdzs2026_c203/haha
source .venv/bin/activate
source env.sh
./build_vllm_wheel_install.sh
```

日常局部改动只构建所属目标，再安装到源码树：

| 改动位置 | 目标 |
|---|---|
| `csrc/rocm/` | `_rocm_C` |
| 主算子、KV Cache、PagedAttention 等 | `_C` |
| MoE 算子 | `_moe_C` |
| `csrc/cumem_allocator.cpp` | `cumem_allocator` |

例如修改 `csrc/rocm/skinny_gemms.cu`：

```bash
cmake --build --preset rocm-release --target _rocm_C
cmake --install cmake-build-rocm --component _rocm_C
ls -la vllm/_rocm_C.abi3.so
python -c "import vllm._rocm_C; import torch; print(hasattr(torch.ops._rocm_C, 'wvSplitK'))"
```

最后一条输出 `True` 表示该扩展可导入且算子已注册。Ninja 会自动跳过未变文件；`ccache` 命中时可进一步缩短重复编译。

## 清理重建与测试

持续编译错误时再清理构建目录：

```bash
rm -rf cmake-build-rocm
cmake --preset rocm-release
cmake --build --preset rocm-release --target _rocm_C
cmake --install cmake-build-rocm --component _rocm_C
```

若需要全部扩展重新构建，使用前述 `./build_vllm_wheel_install.sh`。

构建成功只代表扩展可用，不代表优化有效。按项目规则先运行可重复的 4B 对照，确认完成率、正确性和时延无回退后，再决定是否验证 27B；结果记录到 `docs/progress.md`。
