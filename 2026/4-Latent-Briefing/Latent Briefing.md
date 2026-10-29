url： https://labs.ramp.com/research/latent-briefing-kv-cache/
来源标注发布日期：2026-04-10
来源类型：作者技术文章（Ramp Labs），未核实到独立学术论文或会议发表。
核查日期：2026-10-24

keyword： #LLM_Agents ， #KVCache_Compaction ， #Proxy_Queries

# Latent Briefing: Efficient Memory Sharing for Multi-Agent Systems via KV Cache Compaction

# 核心思路
- 在 worker 模型中处理 orchestrator 轨迹和当前委派任务，用任务提示产生的 Query 对轨迹 KV 评分。
- 在 AM 基础上讨论跨层/头共享位置选择、MAD 阈值及批量拟合，服务多 Agent 记忆共享。

# 和任务状态加权 Query 采样的关系
与“任务驱动的 AM Query”直接相关。显式状态映射必须与直接用当前任务提示提取 Query 的方案比较，而不能只对照均匀或近期采样。

# 建议对照
固定 worker 模型和输入信息，对比任务提示 Query、状态转写 Query、历史 Query 的状态加权子集；记录所有额外编码成本。

# 待核实
- 本条依据原始来源的摘要或技术说明整理，尚未完成全文、代码与实验审计。
- 进一步核实评分公式、Query 获取方式、状态/阶段定义及原始对照设置。
- 不把“摘要未提及某机制”作为该工作完全没有该机制的证据。

方法边界：技术文章的实现描述与 AM 原论文应分开核查；共享掩码是该实现选择，不能据此断言所有逐头 AM 都不能批处理。
