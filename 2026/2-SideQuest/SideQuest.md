url： https://arxiv.org/abs/2602.22603
首次 arXiv 提交：2026-02-26
来源类型：arXiv 论文；本次未核实正式会议发表。
核查日期：2026-10-24

keyword： #LLM_Agents ， #KVCache_Compaction ， #Proxy_Queries

# SideQuest: Model-Driven KV Cache Management for Long-Horizon Agentic Reasoning

# 核心思路
- 让推理模型判断上下文 Token 对任务是否仍有用。
- 把 KV 管理作为与主推理并行的辅助任务，避免管理过程的 Token 污染主任务上下文；原文包含模型训练。

# 和任务状态加权 Query 采样的关系
已涉及模型判断信息未来用途，是任务状态感知的重要对照。候选区别是结构化状态到阶段权重、再到 Query 采样的显式链条及其额外成本，而非笼统声称首次引入任务状态。

# 建议对照
比较模型驱动管理与轻量状态映射；计入状态维护、辅助推理、训练和主任务墙钟成本。

# 待核实
- 本条依据原始来源的摘要或技术说明整理，尚未完成全文、代码与实验审计。
- 进一步核实评分公式、Query 获取方式、状态/阶段定义及原始对照设置。
- 不把“摘要未提及某机制”作为该工作完全没有该机制的证据。
