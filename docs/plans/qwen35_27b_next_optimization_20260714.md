# Qwen3.5-27B gfx936 下一步优化计划

时间：2026-07-14  
源码基线：`vllm_cscc@0e3450b`

## 路线

1. 为 27B MLP down 精确形状 `(M=5120,N=1,K=17408)` 实现 gfx936 BF16 专用
   GEMV V5--V9，并以现有 `F.linear` 为 micro 基线。
2. 只有每个随机种子至少 `1.08x`、三种子中位数至少 `1.10x` 时，才接入环境变量模式 3
   并进行三档端到端和完整精度测试；否则删除候选生产代码。
3. MLP down 淘汰后转向 UA2D 实验 6--8，只改变 load/控制流调度，保持 TILE32、KV
   遍历与 online-softmax 归约顺序，并要求相对实验 5 bitwise 一致。

实施结果：MLP down 五个候选中最快者仍比 `F.linear` 慢约 `5%`，已按门禁淘汰并清理；
随后已完成 UA2D 实验 6--8。

UA2D 实施结果：组合实验 8在 8K/16K/32K 分别达到
`1.0786x/1.1135x/1.1344x`，但加权仅 `1.1128x`，未达到服务门禁，已淘汰并恢复实验 5。

## MLP down 候选

- V5：1 wave/行，4 行/workgroup，直接读取 activation。
- V6：2 wave/行，2 行/workgroup，直接读取 activation。
- V7：4 wave/行，1 行/workgroup，直接读取 activation。
- V8：V5 加完整 activation LDS staging。
- V9：V6 加完整 activation LDS staging。

统一使用 256 threads、128-bit BF16 load、FP32 累加和 wave shuffle。若 micro 达标，
`VLLM_ROCM_GFX936_SPECIALIZED_GEMV=3` 表示模式 2 加 MLP-down 最优候选；验证完成前默认
仍保持模式 2。

## 门禁

- 算子：随机种子 `0/1/17`、七轮计时、重复确定、有限值，且相对 `F.linear` 满足
  `rtol=1e-2/atol=1.5e-2`。
- 服务：三档各两轮热态 10 条；短中档 P99 TPOT 至少改善 `1.5%`，长档 TPOT 回退不超过
  `1%`，TTFT 回退不超过 `2%`，预测平台得分增加至少 `0.30`。
- 精度：四类任务系数均为 `1.00`。
- 每轮实现与测试结果按时间戳写入 `docs/progress.md`，源码与根仓库文档分别提交。
