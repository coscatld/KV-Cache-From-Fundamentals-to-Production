url： https://arxiv.org/abs/2512.22087
首次 arXiv 提交：2025-12-26
发表信息：本次仅核实 arXiv 条目。
核查日期：2026-10-23

keyword： #LLM_Agents ， #Context_Management ， #SWE_Agents

# Context as a Tool: Context Management for Long-Horizon SWE-Agents

# 脉络
仅追加历史或被动按阈值压缩，可能使长程代码任务发生上下文膨胀和语义漂移。

# 方法
- 把上下文维护变成可由 Agent 主动调用的工具。
- 工作区区分稳定任务语义、凝练的长期记忆和高保真的近期交互。
- 通过 CAT-GENERATOR 构造含上下文管理动作的轨迹，训练 SWE-Compressor；在 SWE-Bench-Verified 上评估。

# 对切入点的启发
研究可观测任务里程碑能否作为 KV 压缩触发信号，并与单纯长度阈值、固定延迟公平比较。

# 对照与边界
Agent 主动触发上下文维护已有先例。KV 方向的贡献应具体到拟合质量、缓存预算或运行开销，不能只强调“由 Agent 决定”。

# 待读问题
- 精读正文中的更新规则、预算分配、额外计算及消融设置。
- 用相同模型、样本和缓存预算建立对照；将论文结果与新方案推测分开记录。

来源：上方原始论文页面。本条为来源核实与阅读提要，尚未进行完整正文/代码审计或本地复现。
