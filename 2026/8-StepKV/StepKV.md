url： https://arxiv.org/abs/2609.22158
首次 arXiv 提交：2026-08-26
来源类型：arXiv 论文；本次未核实正式会议发表。
核查日期：2026-10-24

keyword： #LLM_Agents ， #KVCache_Compaction ， #Proxy_Queries

# StepKV: Step-Aware KV Cache Compression for LLM Agents

# 核心思路
- 为 KV 条目保留其生成 reasoning step 的归属。
- 由轨迹信号估计 step utility，并与 token saliency 结合，进行全局 Token 排序与预算内保留。

# 和任务状态加权 Query 采样的关系
与“历史阶段权重 × 重要性指标”高度相关。当前摘要支持 step utility 与 token saliency 的组合，但不足以断言它完全没有显式任务状态信号；需精读 utility 定义与更新规则。

# 建议对照
明确权重作用位置：本设想先用状态权重选择 AM 拟合 Query；StepKV 摘要描述的是对待保留 KV Token 排序。需控制表示算法，分离 Query 选择与 KV 评分的收益。

# 待核实
- 本条依据原始来源的摘要或技术说明整理，尚未完成全文、代码与实验审计。
- 进一步核实评分公式、Query 获取方式、状态/阶段定义及原始对照设置。
- 不把“摘要未提及某机制”作为该工作完全没有该机制的证据。

日期说明：v1 为 2026-08-26，v2 为 2026-09-22；按页面的 v1 日期归入 8 月，而非从编号推断。
