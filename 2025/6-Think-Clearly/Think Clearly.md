url： https://arxiv.org/abs/2507.08806
首次 arXiv 提交：2025-06-17
发表信息：[Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1169/)
核查日期：2026-10-23

keyword： #KVCache_Eviction ， #LLM_Reasoning ， #Structure_Aware_Pruning

# Think Clearly: Improving Reasoning via Redundant Token Pruning

# 脉络
推理轨迹中存在冗余片段；如何利用结构信息识别对后续推理贡献较低的内容？

# 方法
- 在中间推理步骤后加入显式结束指令与 end-of-thinking token，利用其注意力信号衡量历史 Token 贡献。
- 采用结构感知裁剪，优先删除贡献较低的推理块，而不只逐 Token 裁剪。
- 移除临时注入的结束指令后继续推理，无需训练。

# 对切入点的启发
测试结构边界 Query 能否补充 AM 的拟合集；区分“片段重要性”与“Query 对未来读取的代表性”。

# 对照与边界
边界信号并非免费且天然存在：原方法含主动注入的指令。对照实验需计入额外前向与潜在生成干预，不能直接把它等同于被动读取原轨迹边界。

# 待读问题
- 精读正文中的更新规则、预算分配、额外计算及消融设置。
- 用相同模型、样本和缓存预算建立对照；将论文结果与新方案推测分开记录。

来源：上方原始论文页面及会议页面。本条为来源核实与阅读提要，尚未进行完整正文/代码审计或本地复现。

日期说明：arXiv 编号为 2507.08806，但页面 Submission history 的 v1 日期为 2025-06-17；目录按该日期标记。
