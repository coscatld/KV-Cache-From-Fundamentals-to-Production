url： https://arxiv.org/abs/2608.03276
首次 arXiv 提交：2026-08-04
来源类型：arXiv 当前版本自述 EMNLP 2026 Main Conference；本次未单独核实会议条目。
核查日期：2026-10-24

keyword： #LLM_Agents ， #KVCache_Compaction ， #Proxy_Queries

# TaskPress: Query-Agnostic KV Cache Compression via Task-Guided Pruning

# 核心思路
- 用高层 task guide 作为 Prefill 阶段的 meta-query，构造可跨下游 Query 复用的紧凑缓存。
- 结合量化尺度因子识别影响较大的表示离群值，辅助 Token 重要性估计。

# 和任务状态加权 Query 采样的关系
“任务引导压缩”已有直接先例。待测区别是随执行更新的未完成目标/依赖状态，相对于固定高层 task guide，能否产生更有效的 Query 子集。

# 建议对照
比较初始任务 guide、当前状态转写的 guide 和显式阶段加权采样，固定额外前向与 Query 预算。

# 待核实
- 本条依据原始来源的摘要或技术说明整理，尚未完成全文、代码与实验审计。
- 进一步核实评分公式、Query 获取方式、状态/阶段定义及原始对照设置。
- 不把“摘要未提及某机制”作为该工作完全没有该机制的证据。
