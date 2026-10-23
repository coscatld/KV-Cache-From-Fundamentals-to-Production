url： https://arxiv.org/abs/2609.04971
首次 arXiv 提交：2026-09-04
发表信息：[ICML 2026 会议页面](https://proceedings.mlr.press/v306/kim26af.html)
核查日期：2026-10-23

keyword： #KVCache_Eviction ， #Proxy_Queries ， #LLM_Reasoning

# BeaconKV: Key-Value Cache Compression Guided by Beacon Queries for Efficient Large Reasoning Model Inference

# 脉络
近期 Query 未必能代表长推理轨迹中的未来读取：部分 Thought Revisiting Tokens 会重新关注很早的计划等信息。

# 方法
- 观察到相关 Query 在表示空间形成少量相似组。
- 维护全局 Query 模式的紧凑代表 beacon queries，用它们辅助判断哪些 KV 以后可能被重新访问，无需保存全部 Query 历史。
- 方法无需训练；其主要对象是推理生成中的 KV 选择，不是 AM 的偏置与 Value 拟合。

# 对切入点的启发
比较“近期 Query”和“代表性历史 Query”用于 AM 拟合时，在相同 Query 数与 KV 预算下是否改善未参与拟合的后续 Query 上的输出及 mass 误差。

# 对照与边界
全局 Query 聚类与代表 Query 已是其核心设计。新方向需要给出面向 AM 目标的选择机制及额外收益；仅换聚类算法不足以建立区别。

# 待读问题
- 精读正文中的更新规则、预算分配、额外计算及消融设置。
- 用相同模型、样本和缓存预算建立对照；将论文结果与新方案推测分开记录。

来源：上方原始论文页面及会议页面。本条为来源核实与阅读提要，尚未进行完整正文/代码审计或本地复现。

日期说明：目录月份按 arXiv v1，不等同于会议公开月份；会议页面单独保留。
