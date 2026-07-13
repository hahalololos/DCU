# Qwen3.5-27B + vLLM + 国产 DCU 推理加速技术调研报告

调研时间：2026-07-12。

调研范围：单卡、单请求、4K–32K 长上下文；不改变模型结构、权重、推理语义、调度锁定参数，不使用投机解码或持久化权重量化。

## 一、核心结论

结合论文、Qwen/vLLM/AMD 官方资料、上游实现和本项目 4B 实测，建议按以下顺序投入：

1. **继续优化 Qwen3.5 专用 UA2D 长序列 prefill kernel**
   这是当前证据最充分、与评分权重最吻合的方向。重点不是继续盲扫 tile，而是减少 causal mask、VGPR 活跃区间和 LDS bank conflict。
2. **开发 UA3D/Decode Attention 的 Flash-Decoding 式 KV 分段并行**
   这是 16K–32K TPOT 的第一候选。核心是沿 KV 长度增加并行度，再用在线 softmax 合并分段结果。
3. **系统调优 Gated DeltaNet 核心，而不是只优化其投影 GEMM**
   27B 有 48 层 GDN，其固定 recurrent state 约 144 MiB。应重点测试 recurrent kernel、state 布局、prefill chunk size 和 CPU/图编译路径。
4. **把 CUDA/HIP Graph、`torch.compile` 和小算子融合列为中等优先级**
   当前 4B 的 GPU kernel 时间与端到端 TPOT 之间仍有约 1 ms/token 的差距，但必须先确认现有图捕获状态。
5. **FP8 KV Cache 只作为有严格精度门禁的后备路线**
   它理论上能把 full-attention KV 流量减半，但 `head_dim=256` 的 prefill 开销、DCU 原生 FP8 能力和长文本精度都存在明显不确定性。
6. **KV block manager、LM head、继续扩大 LLMM1 shape allowlist 不应是当前主线**
   单请求场景的碎片收益有限；LM head 已接近带宽受限；LLMM1 扩形状已出现端到端负收益和精度扣分。

## 二、模型和工作负载的确定事实

### 2.1 Qwen3.5-27B 的准确结构

27B 官方配置为：

| 参数 | Qwen3.5-27B |
| --- | ---: |
| 层数 | 64 |
| hidden size | 5120 |
| intermediate size | 17408 |
| full / linear attention | 16 / 48 层 |
| full attention Q heads | 24 |
| full attention KV heads | 4 |
| GQA 比例 | 6:1 |
| full attention head dim | 256 |
| GDN key heads | 16 |
| GDN value heads | 48 |
| GDN key/value dim | 128 |
| GDN convolution kernel | 4 |
| recurrent state dtype | FP32 |
| 最大上下文 | 262144 |

权威依据：

- [Qwen3.5 官方说明](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3.5-27B 原始配置](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json)
- [Transformers Qwen3.5 配置实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py)
- [AMD Qwen3.5 Day-0 支持说明](https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-qwen-3-5-on-amd-instinct-gpus.html)

一个重要纠错是：不能套用 35B-A3B 或 Transformers 类默认值。其他型号可能是 2 个 KV heads、32 个 linear value heads，而 **27B 实际是 4 和 48**。

### 2.2 KV Cache 与 GDN state 的物理规模

27B 只有 16 层 full attention 使用传统随长度增长的 KV Cache：

```text
每 token KV
= 16层 × 2(K/V) × 4 KV heads × 256 × 2 bytes
= 64 KiB/token
```

| 上下文 | full-attention KV |
| --- | ---: |
| 8K | 512 MiB |
| 16K | 1 GiB |
| 32K | 2 GiB |

GDN recurrent state 的形状是 `[HV, V, K] = [48,128,128]`，FP32：

```text
单层 state = 48 × 128 × 128 × 4 bytes = 3 MiB
48层合计 = 144 MiB/sequence
```

因此：

- Full attention：显存和 decode 读取量随上下文线性增长。
- GDN：显存不随上下文增长，但每个 decode token 都要更新固定状态。
- 32K 时 full-attention KV 读取压力明显大于 GDN state，但 GDN 有 48 层，kernel 数量、投影和状态更新仍不可忽视。

本地源码已经使用上游推荐的 `[N, HV, V, K]` state 布局，可见 `vllm_cscc/vllm/model_executor/layers/fla/ops/fused_recurrent.py`。

## 三、当前实测瓶颈与论文结论是否一致

### 3.1 Prefill：UA2D 是最强证据热点

4B profile：

| 输入档 | UA2D 占纯 GPU kernel 时间 |
| --- | ---: |
| 4K–8K | 28.4% |
| 8K–16K | 37.9% |
| 16K–32K | 58.1% |

硬件计数器：

- VGPR：224
- LDS：32 KiB
- L2 hit：约 97.3%
- HBM 读取：约 20 GB/s
- LDS bank-conflict：约 `1.577×10^9`

这说明它不是典型的 HBM 带宽瓶颈，而是：

- 大量 FP32 softmax/累加状态延长 VGPR live range；
- `acc[BLOCK_M, 256]` 占用大量寄存器；
- K/V staging、转置或 fragment 布局导致 LDS 冲突；
- causal mask、边界判断和非矩阵运算消耗较高；
- occupancy 和 MFMA/LDS 流水不足。

这与以下资料一致：

- [FlashAttention-2](https://arxiv.org/abs/2307.08691)强调减少非矩阵 FLOPs、causal masking、同步和 shared-memory 读写。
- [FlashAttention-2 官方技术说明](https://hazyresearch.stanford.edu/blog/2023-07-17-flash2)明确把 work partitioning 和 shared-memory 通信视为主要优化点。
- [AMD LDS bank conflict 文档](https://rocm.docs.amd.com/projects/composable_kernel/en/latest/conceptual/ck_tile/hardware/lds_bank_conflicts.html)说明 LDS 按 4-byte bank 映射，不合适的 lane/地址布局会产生串行化。
- [AMD LDS 优化文章](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)展示了通过 padding/swizzle 改变 bank 映射的路径。
- [AMD register pressure 指南](https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html)指出缩短变量生存期、调整 tile 和显式控制中间状态可提升 occupancy。

### 3.2 当前 full-prefix/diagonal 方案方向正确

当前候选 kernel 已经把 KV tiles 分成：

- 对所有 query rows 完全可见的 full-prefix tiles；
- 需要 causal mask 的 diagonal/partial tiles。

源码位于 `vllm_cscc/vllm/v1/attention/ops/triton_unified_attention.py`。

代表形状 `q=4096, kv=22258`：

- 原路径约 52 ms；
- 专用路径约 44.8 ms；
- 单 kernel 加速约 14.1%。

按 Amdahl 定律，仅这一项理论上可缩短三档纯 kernel 时间约：

| 档位 | 估算缩短 |
| --- | ---: |
| 4K–8K | 3.5% |
| 8K–16K | 4.7% |
| 16K–32K | 7.2% |

这和候选测试中多数中档样本 TTFT 下降约 5%–8%基本吻合。现有端到端候选测试受共享 GPU 干扰，不能作为淘汰依据。

但 TILE=32 改变了 softmax 累加顺序，已经出现生成文本差异。因此建议同时保留：

- T32：性能上限版本；
- T64：数值更保守版本；
- generic：正确性回退版本。

最终选择应由完整 OpenCompass 精度，而非只看 `allclose` 决定。

## 四、UA2D 下一步具体优化方向

### 4.1 优先降低 VGPR，而不是继续放大 tile

建议逐项做消融：

1. 缩小 FP32 `acc` 的同时驻留范围。
2. 将 full-prefix 循环拆成更短的软件流水阶段。
3. 避免同时保留 Q、K、V、S、P 和完整 acc 的多个版本。
4. 尝试重新排序 softmax 更新，使旧 `S/P` 更早失活。
5. 对 Q/K 和 P/V 两次 `tl.dot` 分别检查实际 VGPR 峰值。
6. 检查编译器是否把可重算标量长期保存在 VGPR 中。

目标不应只看 latency，而应同时看：

- VGPR：争取从 224 降至 192 或更低；
- LDS bank conflict：至少下降 30%；
- waves/CU 或 occupancy；
- MFMA busy、VALU busy、LDS busy；
- 端到端 TTFT 和文本一致率。

### 4.2 对 LDS 使用 padding/swizzle

应重点检查：

- `HEAD_SIZE=256` 连续维映射是否反复落在相同 bank；
- Q/K fragment 的 LDS 写入和读回是否使用同一种布局；
- `ds_write` 无冲突但 `ds_read` 冲突的非对称情况；
- TILE=32 时，64-lane wave 是否造成两组 lane 对相同 32 banks 访问；
- padding 一列或 XOR/swizzle address 是否减少冲突。

AMD 官方资料能证明“bank conflict 是真实机制”，但具体最佳布局不能直接从 MI300 复制到 gfx936，必须以 gfx936 rocprof 计数器为准。

### 4.3 不建议拆成多个独立 kernel

此前 prefix/suffix + LSE merge 已有严重负收益。原因与 Flash-Decoding 不同：

- Prefill 本身已有大量 query blocks，并行度通常足够；
- 拆 kernel 增加 launch、临时 LSE/output 写回和归约；
- 当前瓶颈主要在单 kernel 内部资源，而不是并行度不足。

因此保留“单 kernel 内 full-prefix/diagonal 两段”是更合理的设计。

## 五、Decode Attention：UA3D 应采用 Flash-Decoding 思路

当前 UA3D 占稳定 decode kernel 时间：

| 档位 | UA3D 占比 |
| --- | ---: |
| 4K–8K | 6.3% |
| 8K–16K | 9.1% |
| 16K–32K | 18.1% |

硬件特征：

- VGPR：248
- L2 hit：约 1%
- 单次 fetch：约 87 MiB
- 延迟随 KV 长度增长

这是典型的长 KV、低 query 并行度、带宽与 occupancy 混合瓶颈。

[PyTorch Flash-Decoding](https://pytorch.org/blog/flash-decoding)和[Stanford CRFM 原始说明](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)给出的核心方法是：

1. 沿 KV 序列切成多个 segments；
2. 每个 segment 独立执行 online softmax；
3. 输出局部 maximum、sum 和 accumulator；
4. 用一个小 reduction 合并所有 segments。

它特别适合：

- batch=1；
- query length=1；
- 上下文较长；
- 仅依靠 query/head 维无法占满 GPU。

建议扫描：

```text
segments = 1, 2, 4, 8, 16
KV length = 4K, 8K, 16K, 24K, 32K
GQA = 4（4B）和 6（27B）
```

分段数应仅由 tensor shape/长度决定，不能依据样本内容。

预期上限：

- UA3D kernel 加速 20%：4B decode 总体约改善 1.0%–3.0%。
- UA3D kernel 加速 30%：约改善 1.5%–4.2%。
- 27B 有 16 层 full attention，可能比 4B 更受益，但 27B 计算量也更大，必须实测。

风险点：

- segment 太多会被 reduction 和临时 buffer 抵消；
- Paged KV 的 block table 会破坏连续读取；
- GQA=6 不是 2 的幂，需要避免 padding wave 空耗；
- 需要严格保持 softmax 合并的 FP32 数值稳定性。

## 六、Gated DeltaNet：第二条值得重点调研的路径

### 6.1 为什么不能忽略 GDN

Qwen3.5-27B 有 48 层 GDN。Gated DeltaNet 论文说明，其 chunkwise 形式把顺序 recurrence 改写成 GEMM-rich 运算，从而利用现代矩阵单元：

- [Gated Delta Networks，ICLR 2025](https://openreview.net/forum?id=r8H7xhYPwz)
- [论文 PDF](https://arxiv.org/pdf/2412.06464)
- [DeltaNet chunkwise parallel 算法](https://arxiv.org/pdf/2406.06484)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
- [vLLM Qwen3-Next 支持说明](https://vllm.ai/blog/2025-09-11-qwen3-next)

AMD 官方确认 Qwen3.5 在 vLLM/ROCm 中使用 Triton 的 `fused_recurrent_gated_delta_rule`。

### 6.2 上游已经证明有效的优化线索

[vLLM Qwen3-Next 性能跟踪](https://github.com/vllm-project/vllm/issues/27225)汇总了：

- GDN state layout `[K,V] → [V,K]`；
- GDN decode kernel block 调整；
- GDN attention 的 `torch.compile`；
- GDN prefill kernel；
- CPU metadata 开销；
- 小 batch linear 性能；
- qkv/z/b/a 投影融合。

几个关键上游结果：

- [GDN `torch.compile` #27152](https://github.com/vllm-project/vllm/pull/27152)：GDN 子路径约 14%，Qwen3-Next 端到端约 4%–5%，但数据来自 NVIDIA。
- [GDN decode #31722](https://github.com/vllm-project/vllm/pull/31722)：调整 kernel block 后，H200/B200 上部分形状约 2× kernel 改善，但测试 batch 明显大于 1。
- [State layout #33291](https://github.com/vllm-project/vllm/pull/33291)：多个 kernel 团队独立认为 `[V,K]` 更利于 prefill/decode。
- [GDN prefill #32846](https://github.com/vllm-project/vllm/pull/32846)：围绕新 state layout 接入专用 prefill kernel。

本地源码已经具备 `[N,HV,V,K]` 布局，并把 GDN core 放在自定义 op 边界内；说明部分上游收益已经进入当前版本，不能简单重复 cherry-pick。

### 6.3 针对 gfx936 建议测试

优先测试：

1. `fused_recurrent_gated_delta_rule` 单 token、batch=1 的实际 kernel 占比。
2. 4B：`HV=32,K=V=128`；27B：`HV=48,K=V=128`。
3. recurrent state 的读取、写回和 cache hit。
4. block size/waves 对 32、48 value heads 的映射。
5. prefill chunk size 32/64/128。
6. Q/K L2 normalization 是否在 kernel 内完成。
7. `rearrange`、`cat`、`zeros` 等 GDN Python/PyTorch 操作是否形成 GPU idle gap。

本地相关路径：

- `vllm_cscc/vllm/model_executor/models/qwen3_next.py`
- `vllm_cscc/vllm/model_executor/layers/fla/ops/fused_recurrent.py`

不过已有 qkvz+b/a 双 GEMV 融合仅带来 0.1%–0.6% 端到端改善，因此不应再次优先投入同类投影融合。

## 七、Decode GEMM 与 LM head

4B 稳态 decode 中：

- 当前 LLMM1：约 25%–29%；
- 其他 GEMM：约 40%–46%；
- GEMM 总计超过 65%。

但不能据此直接推导“大量可优化空间”：

- batch=1 GEMM/GEMV 经常由权重读取带宽主导；
- LM head `(248320,2560)` 单次读取约 1.2 GiB；
- LM head VGPR 低、L2 hit 低，行为已经接近流式读权重；
- 不允许持久化权重量化，无法从根本上减少权重字节数。

建议：

- 只保留精确 shape 的 hipBLAS/LLMM1 autotune；
- 优化权重布局、向量加载、split-K 和 wave 数；
- 每个候选必须同时通过 micro 和端到端；
- 不再通过扩大 allowlist 推断收益；
- 可尝试 RMSNorm/residual/activation 与相邻 projection 的轻量融合，但预期通常只有 1%–3%。

不建议投入“LM head + argmax 融合”作为主线，因为它仍必须读取完整词表权重，节省的 logits 写回量相对 1.2 GiB 权重读取很小。

## 八、KV Cache 量化：潜力高，但当前不是 P0

### 8.1 多信源一致结论

以下论文均证明 KV Cache 是长上下文 decode 的带宽瓶颈：

- [KIVI，ICML 2024](https://arxiv.org/html/2402.02750v2)
- [KVQuant，NeurIPS 2024](https://arxiv.org/html/2401.18079v4)
- [vLLM FP8 KV Cache 文档](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache)
- [vLLM 2026 FP8 KV Cache 实测](https://vllm.ai/blog/2026-04-22-fp8-kvcache)

FP8 能把 27B 32K full-attention KV 从约 2 GiB 降至约 1 GiB，并理论上把 UA3D 的 KV 流量减半。

### 8.2 反证和风险

vLLM 的最新交叉验证给出几个重要反例：

- 单请求 H100 上 FP8 decode 的 break-even 大约在 7K；
- `head_dim=256` 使用高精度两级累加时，长上下文 TTFT 可能恶化约 1.6×；
- 关闭两级累加可恢复速度，但需要重新验证长上下文精度；
- 未校准 scale 在某些模型上会明显损失准确率；
- 某些 hybrid/sliding-window 模型无法有效摊薄量化开销。

对当前赛题还多出两个风险：

1. gfx936 是否具备真正高效的 FP8 dot/convert 路径尚未证实。
2. 赛题精度 1% 就可能开始扣分，且 Qwen3.5 已表现出对数值累加顺序敏感。

因此只有满足以下条件才值得升级为正式候选：

- UA3D 已被证实为 27B 主要 TPOT 瓶颈；
- kernel 能直接消费 FP8 KV，而不是先完整反量化到 BF16；
- 32K micro 至少加速 15%；
- 四类精度全部不越过 1% 门槛；
- TTFT 不显著回退。

## 九、PagedAttention 和 KV block manager 的真实优先级

[PagedAttention/SOSP 2023](https://arxiv.org/abs/2309.06180)证明了分页分配可以大幅减少多请求服务中的 KV 碎片。

但本赛题是并发 1：

- 没有多请求长度差异导致的严重外部碎片；
- last-block waste 上限只有一个 block；
- KV block 复用、抢占和共享前缀价值很低；
- 主要收益来自 attention kernel 的读取布局，而非分配器。

[vAttention](https://arxiv.org/html/2405.04437v2)还给出反证：block size 会显著影响 kernel latency，某些 vLLM decode kernel 使用较大 block 时可慢到 3×。因此不能因为“更大 block 减少 block-table 查询”就推断端到端更快。

建议仅检查：

- 运行期是否反复 allocate/free；
- block table 是否在 CPU 每 token 重建；
- block size 是否造成 UA3D 非合并访问；
- 标量 block-table 优化是否减少实际 load。

现有 block-table 标量化收益不足 1%，说明该方向应维持 P3。

## 十、HIP Graph、编译与 CPU 开销

vLLM V1 采用 piecewise graph：attention 保持 eager，其他 token-wise 运算放入图中。

来源：

- [vLLM torch.compile 设计](https://docs.vllm.ai/en/stable/design/torch_compile)
- [vLLM CUDA Graph 设计](https://docs.vllm.ai/en/stable/design/cuda_graphs)
- [AMD vLLM 优化指南](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html)

AMD 文档显示：

- `TRITON_ATTN`、`ROCM_ATTN`、AITER unified attention 可支持完整图；
- AITER FA 等多路径 backend 对混合 batch 的 graph 支持有限；
- 不同后端和 batch composition 会触发降级。

当前应先测而不是直接改：

```text
端到端 TPOT - 所有纯 GPU kernel 时间
GPU kernel 之间的空洞
CPU launch / metadata / tensor view-copy 时间
是否已捕获 decode 非 attention 路径
是否因动态 shape 每轮 graph break
```

当前 4B 纯 kernel 为 10.77–12.25 ms/token，而端到端 P99 TPOT 为 11.8–13.2 ms，说明总的非 kernel/同步差距约 0.8–1.1 ms/token。它是可见机会，但上限低于 UA2D 主热点。

## 十一、最终优先级矩阵

| 优先级 | 技术路线 | 证据强度 | 预期作用 | 风险 |
| --- | --- | --- | --- | --- |
| P0 | UA2D full-prefix/diagonal 单 kernel | 很强 | 中长档 TTFT、总吞吐 | 数值累加差异 |
| P0 | UA2D VGPR/LDS/swizzle 优化 | 很强 | 三档 prefill，长档最大 | 编译器行为复杂 |
| P0 | UA3D Flash-Decoding KV segmentation | 强 | 16K–32K TPOT | reduction 抵消收益 |
| P0 | 27B 精确形状 micro 与 profile | 必需 | 防止 4B 误导 | 27B 启动成本高 |
| P1 | GDN recurrent kernel/block/chunk | 中强 | 48 层固定开销 | 上游收益多为 NVIDIA |
| P1 | GDN/非 attention 图编译与 CPU gap | 中 | TPOT、固定开销 | ROCm graph 稳定性 |
| P1 | RMSNorm/residual/activation 融合 | 中 | kernel launch、激活访存 | 单项收益较小 |
| P2 | FP8 KV Cache + 直接 FP8 attention | 中 | 长 KV 带宽、显存 | 精度和 TTFT 高风险 |
| P2 | 精确 shape GEMM autotune | 中 | Decode TPOT | 易出现 micro 正、E2E 负 |
| P3 | KV block manager 微调 | 较弱 | 显存稳定性 | 并发 1 收益有限 |
| P3 | LM head 融合 | 较弱 | sampler/launch | 权重带宽不可消除 |

## 十二、推荐实验路线

### 阶段 A：把 UA2D 候选做成可信结果

同时测试 generic、T64、T32：

- 三档各 3 轮热测试；
- 独占 GPU；
- 固定输入、输出上限和进程状态；
- 记录 request throughput、output throughput、TTFT P99、TPOT P99；
- 比较逐样本输出长度和完整文本；
- 运行四类 OpenCompass；
- 采集 VGPR、LDS conflict、waves、MFMA/VALU/LDS busy。

保留门槛建议：

- 8K–16K 或 16K–32K TTFT 至少改善 3%；
- output throughput 至少改善 2%；
- TPOT 不回退超过 1%；
- 完成率 100%；
- 四类精度均不越过 1%下降线。

### 阶段 B：UA3D segmentation

先做 kernel micro：

- 4B、27B 两套 GQA；
- 8K/16K/24K/32K；
- segment 数 1/2/4/8/16；
- 记录临时 buffer 和 reduction 比例。

只有 micro 稳定加速 15%以上才进入端到端。

### 阶段 C：GDN kernel 分解

分别测：

- qkvz/b-a projection；
- causal conv update；
- recurrent state kernel；
- normalization/output gate；
- output projection；
- Python/metadata gap。

若 recurrent core 占比较小，就不要被上游大 batch 的“2× kernel”结果误导。

### 阶段 D：27B 最终验证

27B 只测试已经通过 4B 门禁、且形状机制可迁移的候选：

- UA2D：24 Q heads、4 KV heads、GQA=6；
- UA3D：16 full-attention 层；
- GDN：48 value heads、144 MiB recurrent state；
- GEMM：5120/17408/248320 精确形状。

## 十三、不建议继续投入的方向

- 继续恢复 LLMM1 被过滤形状；
- 多 kernel UA2D prefix/suffix + LSE merge；
- 单纯扩大 attention tile；
- 优先优化 LM head 算术；
- 为并发 1 深度重写 block allocator；
- 未经精度验证直接开启 FP8 KV；
- prefix caching；
- 修改 scheduler、batch 参数或测试统计；
- 投机解码、MTP、权重持久化量化、剪枝或跳计算。

## 十四、多信源交叉验证结果

| 结论 | 相互验证来源 | 判断 |
| --- | --- | --- |
| Qwen3.5 是 3:1 GDN/full attention | Qwen 官方、27B config、Transformers、vLLM、AMD | 已确认 |
| 27B 为 4 KV heads、GQA=6 | 27B config、本地 kernel guard | 已确认 |
| UA2D 瓶颈在片上资源而非 HBM | 本地 rocprof、FA2、AMD VGPR/LDS 文档 | 强支持 |
| full-prefix 减少 mask 是正确方向 | FA2、当前源码、micro 14.1% | 强支持 |
| Decode 长上下文需要沿 KV 增加并行度 | 本地 UA3D、PyTorch/Stanford Flash-Decoding | 强支持 |
| GDN 应使用 chunkwise prefill、recurrent decode | GDN论文、FLA、AMD、vLLM源码 | 已确认 |
| GDN 上游优化可直接获得同等收益 | 上游多为 H200/B200，gfx936 未验证 | 不能确认 |
| FP8 KV 一定加速 | vLLM 有正例，也有 head_dim=256/精度反例 | 条件成立 |
| KV block 越大越快 | vAttention 给出反例 | 不成立 |
| 通用 Qwen3.5 默认配置可代表 27B | 27B config 与默认值冲突 | 不成立 |

总体判断：**最可信的近期突破口仍是 UA2D；最值得并行准备的下一条路线是 UA3D Flash-Decoding；最可能被低估的中期路线是 GDN recurrent core。** 这三项分别覆盖长上下文 prefill、长上下文 decode 和 Qwen3.5 特有的 48 层线性注意力路径，且均不触碰赛题禁止边界。
