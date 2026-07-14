# Qwen3.5-27B 最新源码快速 Profile 计划

时间：2026-07-14 12:45 CST  
目标源码：`vllm_cscc@a5cbd489d07bd4497a1ab949f96d3b7471b2b176`

## 目标

针对最新榜单源码快速更新性能画像，覆盖端到端吞吐与时延、Prefill/Decode 算子热点、
GPU kernel 资源与访存、CPU/调度空隙、KV Cache/显存和模型存储读取，输出可直接指导
下一轮优化的瓶颈优先级报告。

## 当前榜单基准

- 队伍：添财程序员队，当前第 48 名。
- 三档实际吞吐（16–32K / 4–8K / 8–16K）：`12.51 / 19.19 / 17.07 tok/s`。
- 最终得分 `85.0408`，SLA 扣分 `0`，精度扣分 `0`。
- 第 20 名得分 `87.6316`，当前差距 `2.5908`。

## 执行步骤

1. 以本地 Git 和源码为权威，同步 `vllm_cscc`、测试脚本和 profile 工具至
   `scnet-docker-1:/public/home/acoh0h1o0p/DCU`。
2. 加载 DTK/HYHAL，核验 `python`、`vllm`、editable location、GPU、端口、模型副本和
   远端磁盘空间；清理本项目遗留服务。
3. 运行代表性热态吞吐基准，确认最新源码的吞吐、TTFT、TPOT、成功率与榜单方向一致。
4. 复用配对 trace 方法，以相同请求的 `output=1/65` 分离 Prefill 和稳定 Decode；至少覆盖
   比赛高权重的 8–16K，以及长档 16–32K。
5. 汇总 GPU kernel 时间、调用数、launch gap，并对变化最大的 UA2D、新专用 GEMV、UA3D
   采集 micro/rocprof 或现有可用硬件计数器。
6. 检查 KV Cache 容量、显存余量、CPU 利用/调度等待、HTTP/tokenizer 路径、模型本地盘
   读取，区分启动瓶颈和稳态瓶颈。
7. 将结论写入 `调研交付物/`，并在 `docs/progress.md` 追加本轮简记。

## 有效性门禁

- 不修改比赛脚本、锁定参数、模型权重、tokenizer、chat template 或推理语义。
- profile 使用独立热态服务；冷编译、并发占卡、代理响应和异常请求数据作废。
- profiler 注入耗时仅用于热点与硬件瓶颈分类，不直接作为吞吐结论。
- 所有结论区分实测、估算和历史证据；最新源码未重新验证的旧结论不得直接当作当前事实。
