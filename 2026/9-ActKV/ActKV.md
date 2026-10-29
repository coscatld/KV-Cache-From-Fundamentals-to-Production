url： https://arxiv.org/abs/2609.31395
首次 arXiv 提交：2026-09-25
来源类型：arXiv 论文；本次未核实正式会议发表。
核查日期：2026-10-24

keyword： #LLM_Agents ， #KVCache_Compaction ， #Proxy_Queries

# ActKV: Efficient LLM Agents through Action-Guided KV Cache Management

# 核心思路
- 以 KV 对 Action 生成的贡献为压缩依据，优先保留对后续动作关键的条目。
- 用模型内在置信度调整缓存预算，并设计页级压缩管理与内核。

# 和任务状态加权 Query 采样的关系
与任务状态方案共同关注 Agent 后续行动。可比较动作重要性/置信度信号，与未完成目标、证据和工具依赖所产生的阶段权重；不能仅因输入名称不同就认定机制不同。

# 建议对照
固定总缓存和 Query 数，检查状态信号是否在 Action-guided 对照之外提供额外收益；分别消融评分与预算自适应。

# 待核实
- 本条依据原始来源的摘要或技术说明整理，尚未完成全文、代码与实验审计。
- 进一步核实评分公式、Query 获取方式、状态/阶段定义及原始对照设置。
- 不把“摘要未提及某机制”作为该工作完全没有该机制的证据。
