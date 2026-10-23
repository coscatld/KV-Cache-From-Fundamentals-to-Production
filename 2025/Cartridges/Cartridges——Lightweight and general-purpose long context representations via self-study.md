url： https://arxiv.org/html/2506.06266?_immersive_translate_auto_translate=1
github： [https://github.com/HazyResearch/cartridgesHazyResearch/cartridges](https://github.com/HazyResearch/cartridges)

keyword： #KVCache_Compaction ， #Self_Study ， #Context_Distillation

# Cartridges：离线训练紧凑 KV
首次 arXiv 提交：2025-06-06。

论文正式题名为 **Cartridges: Lightweight and general-purpose long context representations via self-study**；“Storing long contexts in tiny caches with self-study”是作者项目介绍的标题，两者指向同一工作。

# 脉络
同一语料被反复查询时，可以用离线计算换取后续推理的缓存节省，并在多次请求中摊薄训练成本。

# 方法
- 用 self-study 生成围绕语料的合成对话。
- 通过上下文蒸馏训练较小的 KV 表示；推理时加载训练好的 Cartridge，供后续 Query 读取。
- 直接优化紧凑缓存，与 AM 分解局部 Attention 拟合子问题的路线不同。

# 对切入点的启发
研究离线先验能否减少 AM 在线拟合所需的 Query 数量或优化成本。需要分别记录合成数据生成、离线训练和每次在线使用的开销，以及先验对不同任务的泛化。它提供离线学习表示的参考，不直接验证轻量残差网络能够补偿 AM 误差。

来源：[论文](https://arxiv.org/abs/2506.06266) · [作者介绍](https://hazyresearch.stanford.edu/blog/2025-06-08-cartridges)。核查日期：2026-10-23；本段是阅读提要，未运行复现。
