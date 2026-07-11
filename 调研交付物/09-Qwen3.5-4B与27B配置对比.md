# Qwen3.5-4B 与 27B 配置对比

调研日期：2026-07-11

## 结论

两个模型同属 Qwen3.5 混合线性/全注意力架构，核心机制一致：`head_dim=256`、4 个
KV heads、每 4 层 1 层 full attention、其余为 linear attention，最大上下文均为
262144。因此 4B 可用于筛选通用调度、KV 管理和注意力路径优化；但 27B 并非仅增加权重，
其层数和主要算子形状均不同，形状相关的 HIP/Triton/GEMM 优化必须按 27B 配置适配。

## 文本骨干配置

| 字段 | Qwen3.5-4B | Qwen3.5-27B |
| --- | ---: | ---: |
| `num_hidden_layers` | 32 | 64 |
| `hidden_size` | 2560 | 5120 |
| `intermediate_size` | 9216 | 17408 |
| `num_attention_heads` | 16 | 24 |
| `num_key_value_heads` | 4 | 4 |
| `head_dim` | 256 | 256 |
| `linear_num_key_heads` | 16 | 16 |
| `linear_num_value_heads` | 32 | 48 |
| `full_attention_interval` | 4 | 4 |
| full / linear attention 层数 | 8 / 24 | 16 / 48 |
| `max_position_embeddings` | 262144 | 262144 |

两者均使用 BF16，词表大小均为 248320；视觉编码器也存在尺度差异（4B：24 层、宽度
1024；27B：27 层、宽度 1152）。

## 对优化与测试的含义

- 4B 结果可作为通用执行路径和显存/KV 压力优化的快速门禁，不能单独证明 27B 的收益。
- 减少 4B 可见 GPU 只能模拟部分显存余量和 KV 分页压力，不能复现 27B 的矩阵尺寸、层数
  与每 token 计算量。
- 27B shape-sensitive 优化至少覆盖 `hidden_size=5120`、`intermediate_size=17408`、
  24 个 attention heads、48 个 linear value heads、64 层；赛方 27B 测试是最终性能依据。

## 来源

- [Qwen3.5-4B 官方 config.json](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)
- [Qwen3.5-27B 官方 config.json](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)
