# 实际执行报告

日期：2026-10-23。本机没有 CUDA，未下载模型，也未运行真实 Agent 质量实验。

| 阶段 | 实际状态 | 证据 |
| --- | --- | --- |
| 代码与环境审计 | 完成 | docs/audit.md |
| TE、保护位、固定/自适应策略 | PASS | tests/test_core.py |
| 最小合成运行 | PASS，21/21 完成 | results/synthetic-smoke/tasks.jsonl |
| 恢复运行 | PASS，记录仍为 21 条 | synthetic-smoke-console.json 与 synthetic-resume-console.json 一致 |
| 低预算压力实验 | PASS，21/21 完成，81 次强制回退 | results/synthetic-pressure/tasks.jsonl |
| 单元/机制集成测试 | 14 PASS | results/pytest.txt 和 XML |
| CUDA Llama 数值一致性与真实淘汰 | NOT RUN，1 SKIP | tests/test_gpu_model.py |
| 真实模型代理统计、质量与显存 | NOT RUN | 当前无 CUDA/本地模型 |
| 验证集搜索与冻结测试 | NOT RUN | selection.py 已实现，未产生虚构 frozen 结果 |
| 质量—耗时曲线 | NOT RUN | 合成任务无质量指标，plot.py 明确拒绝生成 |

已验证：TE 真实 gather 到独立较短 tensor；层/head/dtype 与保护位保持；固定延迟不使用尚未产生的 Query；T/R/TR 可拆分；普通自适应触发遵守最小间隔；安全回退显式记录；超预算拒绝；原始日志部分写入恢复与完成样本去重。

压力实验仅针对 Bank 口径上限。合成 wall_seconds 是本机张量机制测量，不是 GPU decode 或 Agent 加速数据。没有自适应优越性、准确率、显存节省比例或统计显著性结论。

上传目标环境后：先通过 GPU 模型一致性测试，确认实际模型动作协议和结构保护；再准备真实固定检索数据，验证集调参，冻结后运行测试集。若采用官方基准，需另接官方工具与 grader。严格进程显存同预算比较仍需加入完整工作区约束，见 metrics-and-limits.md。
