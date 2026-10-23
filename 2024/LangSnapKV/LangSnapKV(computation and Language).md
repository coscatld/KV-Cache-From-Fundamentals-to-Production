url: https://arxiv.org/abs/2404.14469

keyword： #KVCache_Eviction ， #Proxy_Queries

# SnapKV: LLM Knows What You are Looking for Before Generation
首次 arXiv 提交：2024-04-22。此文件原名保留，论文实际题名为 SnapKV。

# 脉络
利用 Prompt 尾部 observation window 的 Query/Attention 信号，估计生成期间需要访问的历史位置。

# 方法
- 根据观察窗口的注意力，逐头识别重要 KV 位置，并结合位置聚集信息进行保留。
- 通过筛选原始 KV 缩短缓存，无需微调模型。

# 对切入点的启发
作为“近期 Query 代理”的直接基线，与 BeaconKV 的全局代表 Query 比较。迁移到在线 Agent 时，需要验证近期窗口能否覆盖远期才需要的信息；该工作本身不是 AM 拟合算法。

来源：[原始论文](https://arxiv.org/abs/2404.14469)。核查日期：2026-10-23；本段是阅读提要，未运行复现。
