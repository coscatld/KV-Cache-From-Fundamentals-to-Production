url： https://arxiv.org/abs/2609.14872
首次 arXiv 提交：2026-09-14
来源类型：arXiv 论文；本次未核实正式会议发表。
核查日期：2026-10-24

keyword： #LLM_Agents ， #KVCache_Compaction ， #Proxy_Queries

# AgentKV: Phase-Aware KV Eviction for Agentic LLMs

# 核心思路
- 把未来 Query 看成 think、act、tool 等 phase 的混合；不同 phase 可能占据不同 Query 子空间。
- 维护每个 phase 的小 Query buffer，用其并集对 Key 评分，并支持跨轮保留压缩缓存。

# 和任务状态加权 Query 采样的关系
是“按状态/阶段选择 Query”的直接近邻。需要区分交互 phase 标签与当前未完成目标等细粒度状态，证明后者确实改善代表 Query 的选择。

# 建议对照
首先实现等总 Query 数的 phase buffer 基线，再比较状态加权采样，避免收益只是来自更大的 Query buffer。

# 待核实
- 本条依据原始来源的摘要或技术说明整理，尚未完成全文、代码与实验审计。
- 进一步核实评分公式、Query 获取方式、状态/阶段定义及原始对照设置。
- 不把“摘要未提及某机制”作为该工作完全没有该机制的证据。
