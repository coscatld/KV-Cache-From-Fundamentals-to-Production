url： https://arxiv.org/abs/2510.00615
首次 arXiv 提交：2025-10-01
发表信息：arXiv 当前版本标注 ICML 2026；本笔记未另核会议条目。
核查日期：2026-10-23

keyword： #LLM_Agents ， #Context_Management ， #Text_Compression

# ACON: Optimizing Context Compression for Long-horizon LLM Agents

# 脉络
长程 Agent 同时面临上下文增长和无关信息干扰，固定文本压缩规则可能丢失关键状态。

# 方法
- 联合处理 observation 与 history 的自然语言压缩。
- 依据 Agent 失败分析，迭代优化自然语言压缩指南，无需微调底层 Agent 模型。
- 进一步将优化后的压缩器蒸馏到小模型，降低压缩开销。

# 对切入点的启发
把失败案例归因到具体遗漏信息，研究这些信息能否反向调整 AM 的拟合 Query、样本权重或保护规则。

# 对照与边界
这里优化的是文本压缩指南，迁移到 KV 层需要额外建立“丢失事实—Query—KV 条目”的可观测联系。失败驱动的规则调整也应仅使用开发或验证数据。

# 待读问题
- 精读正文中的更新规则、预算分配、额外计算及消融设置。
- 用相同模型、样本和缓存预算建立对照；将论文结果与新方案推测分开记录。

来源：上方原始论文页面。本条为来源核实与阅读提要，尚未进行完整正文/代码审计或本地复现。
