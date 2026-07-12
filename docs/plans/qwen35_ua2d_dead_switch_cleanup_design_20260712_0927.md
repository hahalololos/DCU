# Qwen3.5 UA2D 失效开关清理设计

时间：2026-07-12 09:27 CST

## 目标

彻底删除已不影响执行路径的
`VLLM_ROCM_QWEN_UA2D_SCALAR_BLOCK_TABLE` 环境变量，避免用户和测试误以为专用
UA2D kernel 仍可在标量与非标量 block-table 实现之间切换。

## 当前问题

Qwen3.5 专用 UA2D kernel 已固定采用经过验证的标量 block-table 地址计算。现有代码仍
声明该环境变量并计算 `scalar_block_table_2d`，但专用 kernel launch 不消费该值；它只在
不会进入的通用 UA2D 分支中作为参数使用，因此开关实际无效。

## 设计

- 从 `vllm/envs.py` 的类型声明和环境变量注册表中删除该变量。
- 删除 `unified_attention()` 中的 `scalar_block_table_2d` 临时状态。
- 通用 UA2D launch 显式使用 `SCALAR_BLOCK_TABLE=False`，保持非 Qwen3.5 路径行为不变。
- 删除测试中对失效变量的设置和真假参数化；保留4B/27B、cache-block 边界、长 query、
  数值一致性及确定性覆盖。
- 外部若仍设置旧变量，vLLM 将像其他未知环境变量一样不读取它，不报错。

## 验证

1. 先添加静态契约测试，要求环境变量注册表不再包含旧变量；修改前该测试必须失败。
2. 删除实现后运行该契约测试及 Qwen3.5 UA2D 定向测试。
3. 本地执行 `py_compile`、`git diff --check`；GPU 测试在远端环境执行。

## 非目标

- 不修改专用 kernel 的计算、guard、参数或默认启用状态。
- 不调整通用 UA2D、3D attention、模型、scheduler 或比赛脚本。
- 不改变推理语义和当前 TILE32 性能取舍。
