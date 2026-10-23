url： https://arxiv.org/html/2504.09936?_immersive_translate_auto_translate=1

大背景：大型语言模型（LLMs）的高效推断受限于不断增长的键值（KV）缓存
目标：在严格的内存约束下保持性能，实现单步无损压缩，并为多步压缩提供误差界限
keywords： #KVCache_Quantization , #KVCache_Merging , #KVCache_Eviction , 

传统：选择性地淘汰较不重要的 KV 缓存条目，导致信息丢失和幻觉
最近：基于合并的策略被探索，通过合并本应丢弃的 KV 对来保留更多信息=>引起注意力分布的不一致
我们：KeepKV——一种新型自适应 KV 缓存合并方法
- 选举人票机制：记录合并历史并自适应调整注意力评分
- 零推断-扰动合并方法：补偿缓存合并引起的注意力损失
![[Pasted image 20261020221013.png]]


先前的问题：不可避免地导致注意力不一致和输出扰动。合并后的 KV 对的注意力得分低于合并前原始分数的总和——称之为“注意力下落”


