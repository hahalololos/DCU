# gfx936 精度能力与量化可行性调研

日期：2026-07-13

## 结论

本赛题当前的 gfx936（容器显示为 C-3000 / BW）应以 **BF16 为主、FP16 为备选**。不建议投入
W8A8、FP8 GEMM 或 INT4 权重路径；它们既不适合现有软件栈，也与赛题禁止持久化权重量化/重排
的边界冲突。若做“量化”，唯一值得排在后续验证队列中的方向是 **推理期、非持久化的 FP8 KV
Cache**，但它不是原生 FP8 Attention：当前卡需在读出时反量化，长上下文带宽收益须大于转换和
精度代价后才可能成立。

## 实机与软件栈证据

2026-07-13 在 `scnet-docker-1` 按项目规定加载 DTK/HYHAL 后检查：

| 项目 | 实测结果 | 含义 |
| --- | --- | --- |
| GPU | `gfx936:sramecc+:xnack-`，80 CU，64 GiB，`Fast F16 Operation: TRUE` | 确认当前目标不是 MI300 的 gfx942。 |
| PyTorch | 2.10.0，HIP 6.3.26093 | BF16、FP16、四种 PyTorch FP8 类型、INT8 均可分配张量。分配/转换不等于矩阵计算支持。 |
| BF16/FP16 GEMM | `torch.matmul` 成功，输出分别为 BF16/FP16 | 这是可用的标准高性能计算路径。 |
| FP8 GEMM | `Float8_e4m3fn`、`Float8_e4m3fnuz`、`Float8_e5m2` 的 `torch.matmul` 均报 `addmm_cuda not implemented` | 当前 PyTorch/DTK 没有可直接使用的 FP8 GEMM。 |
| INT8 GEMM | `torch.int8` 的 `torch.matmul` 报 `addmm_cuda not implemented for Char` | 当前 PyTorch/DTK 没有可直接使用的 INT8 GEMM。 |
| vLLM ROCm 平台 | `current_platform.supports_fp8()` 为 `False` | vLLM 的 gfx936 分支显式不把该卡列入 FP8 支持设备。 |

补充：尝试用当前 `hipcc --offload-arch=gfx936` 编译 FP8 MFMA 和 INT8 MFMA intrinsic，编译器分别
要求 `fp8-insts` 和 `mai-insts` target feature，未生成目标代码。这一探针受 DTK 的 gfx936
离线编译映射限制，不能单独用来断言硬件 ISA；但它与 PyTorch/vLLM 的实际不可用结果一致，足以
否定“现成低精度 GEMM 路径”的工程假设。

## 源码交叉核对

- `vllm/platforms/rocm.py` 的 `supports_fp8()` 只放行 `gfx94`、`gfx95`、`gfx12`，不含
  gfx936；这避免将 MI300 的 FP8 实现误用到本卡。
- 当前源码的 gfx936 Decode Linear 专用内核使用 `bf16` MFMA；已有热点实验也是围绕 BF16
  `LLMM1` 与 attention 内核进行。
- 源码中若干 FP8 kernel 属于 MI3XX/gfx94 等路径，存在于仓库不代表 gfx936 可执行。

## 与赛题边界的关系

赛题禁止在加载前/后或服务初始化时持久化量化、重排压缩和复用量化权重缓存。因此以下方案不应做：

- 离线 W8A8、W4A16/AWQ/GPTQ，或首次加载时将原始 BF16 权重转换后常驻保存；
- 为低比特 GEMM 改变权重布局并跨请求复用；
- 依赖预量化模型/权重文件。

赛题明确允许动态激活量化、KV Cache 量化、kernel 内临时转换与低精度矩阵乘法；它们仍需通过
精度评测和 SLA。

## 建议优先级

1. 继续 BF16 Decode Linear、Paged/Unified Attention 与 KV 内存访问优化；这是当前已验证的
   计算路径，且与 16K--32K 长上下文重点相符。
2. 如需探索量化，只做独立原型：BF16 KV 按块动态缩放到 FP8（不落盘、不复用权重），Attention
   读取时在 kernel 内反量化为 BF16/FP32 累加；测 8K/16K/32K 的 cache 字节数、attention
   时间、端到端 TTFT/TPOT 和 OpenCompass 精度。
3. 原型准入条件：首先证明 FP8 KV 读写+反量化的 microbenchmark 至少节省 5%，然后才进入服务
   A/B；任一精度类别相对 baseline 下降超过 1% 或 TPOT 回退即停止。
4. 暂不做动态 W8A8：即使做到不持久化，逐层动态量化 BF16 权重和激活的转换、scale、布局成本会
   落在每个请求上；且现有 gfx936 栈没有可用的 INT8/FP8 GEMM 后端，预期劣于 BF16 MFMA。

## 外部资料

- AMD 对 CDNA3/MI300 系列公开标注 FP8；MI250X（CDNA2）公开标注 FP8 不支持，说明不能把 AMD
  gfx94 的 FP8 经验外推到同为 gfx9 编号的目标：
  <https://ir.amd.com/news-events/press-releases/detail/1201/amd-accelerates-pace-of-data-center-ai-innovation-and-leadership-with-expanded-amd-instinct-gpu-roadmap>
- vLLM 文档中说明 ROCm 的 FP8 KV Cache 可使用 FP8 E4M3；格式支持也不等同于各架构的 FP8
  GEMM 支持：
  <https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/>
