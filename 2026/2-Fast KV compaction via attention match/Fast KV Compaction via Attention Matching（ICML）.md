url: https://arxiv.org/html/2602.16284?_immersive_translate_auto_translate=1
Github: https://github.com/adamzweiger/compaction

大背景：如何高效管理、压缩和检索上下文信息。
目标：通过一次 KV cache compaction，在显著缩小缓存的同时，尽可能保持模型的下游行为。
Keywords: #KVCache_Compaction, #Attention_matching , #LongContext

# 一、如何提出问题：质疑了之前方法的哪些假设？

## Research Problem：高压缩率下，如何兼顾质量与压缩成本？

如何在避免缓慢、昂贵的端到端训练的前提下，快速完成高压缩比的 KV cache compaction，同时尽可能保持模型的下游行为？

这里有两个需要同时考虑的目标：压缩后的缓存是否好用，以及构造这个缓存需要付出多少成本。“最高可达 50×、部分设置下秒级”应和具体方法变体、数据集一起理解。

**学长分析（转述）：** Cartridges 的端到端优化太慢，促使作者寻找质量与构造速度的平衡。AM 的切口是重排求解流程：用局部注意力行为作为拟合目标，把部分求解转为线性代数问题。

**补充核对：** Cartridges 本身就在构造 compact KV；这里的速度问题主要影响快速更新、即时压缩等使用场景。对于长期复用的上下文，还需要考虑离线成本的摊薄。

本文主要研究一次性、与未来具体问题无关的压缩：先压缩已有上下文，再将结果用于后续问题。这个场景强调压缩结果的可复用性。[原文 §4、§5](https://arxiv.org/html/2602.16284v2#S4)

## Previous Assumption

### 1. 文本摘要：用更短的文字保留后续任务需要的信息

摘要将原文转成较短的文本，再让模型处理摘要。它依赖摘要过程能够保留未来任务所需的信息。

**阅读判断：**“摘要保留主要意思”和“摘要足以回答尚未看到的问题”是不同要求。压缩率越高，越需要检查具体事实、数字及跨段关系的丢失情况。

### 2. KV 筛选与合并：找到更值得保留的条目

这类方法直接处理 KV 条目，应和文本摘要分开理解。

- **H2O：** 依据少量 token 承担较多注意力贡献的观察，保留重要条目及近期条目。[H2O 原论文](https://arxiv.org/abs/2306.14048)
- **SnapKV：** 依据 prompt 末端观察窗口中的注意力模式，选择重要的 KV 位置。[SnapKV 原论文](https://arxiv.org/abs/2404.14469)
- **KVzip：** 利用上下文重建过程估计 KV 重要性，服务于不知道未来具体问题的缓存复用场景。[KVzip 原论文](https://arxiv.org/abs/2505.23416)
- **合并方法：** 将多个条目合成较少的代表条目，进一步利用条目间的冗余。

**阅读判断：**“删掉低重要性条目”隐含了一个需要检验的近似：保留条目的组合能够代替完整缓存的作用。即使 key 选得合理，直接沿用其原始 value，也可能无法代表被删掉条目的贡献。

### 3. 端到端优化：重新学习一个紧凑的 latent KV

**[[Cartridges——Lightweight and general-purpose long context representations via self-study|高质量 latent 压缩必须靠端到端优化]]？**

Cartridges 为一个上下文生成 self-study 数据，再通过端到端优化学习 compact KV。“必须”是本笔记提出的研究疑问；Cartridges 展示的是一条有效路径。

它还强调：离线构造成本可以在同一语料的多次查询中摊薄。因此，“昂贵”的实际影响取决于上下文更新频率和后续复用次数。[Cartridges 原论文](https://arxiv.org/abs/2506.06266)

## 本文的问题转向：能否直接拟合缓存的注意力行为？

**阅读判断：** 本文把问题推进为：选出较少的 key 后，能否重新拟合 attention biases 和 values，让这些条目代替完整缓存的注意力作用，并用更便宜的子问题求解？

这条思路连接了前面两类 KV 方法：保留 key 子集的效率，以及重新拟合紧凑表示的能力。需要关注的是“保留怎样的功能”，以及为实现这种近似付出了哪些限制。

> [!question] 待思考：问题是否真正成立？
> - [ ] 高压缩率下，主要瓶颈是 key 选择不准确，还是保留原始 values 不足以代表被删掉的条目？什么实验能区分两者？
> - [ ] 什么场景最需要快速 compaction？如果同一文档长期不变、被反复查询，Cartridges 的离线成本会如何影响选择？
> - [ ] 把端到端输出目标改成逐层注意力目标，具体减少了哪些计算，又放弃了哪些优化自由度？

# 二、解决问题的方法：依赖什么假设，为什么合理？

## Observation：局部输出和 attention mass 共同决定拼接后的贡献

原笔记中的“attention 相等、softmax 相等”应写成：在参考 queries 上，近似匹配局部 attention output 和未归一化的 attention mass。原缓存与压缩缓存的条目数不同，整个 softmax 向量也有不同的维度。

对某个 query $q$，记原缓存块的局部输出为 $O_K(q)$，未压缩部分的局部输出为 $O_F(q)$，对应的 mass 为：

$$
m_K(q)=\sum_{j=1}^{T}\exp\left(\frac{qK_j^\top}{\sqrt d}\right).
$$

两块拼接后的输出为：

$$
O_{\mathrm{all}}(q)
=
\frac{m_K(q)}{m_K(q)+m_F(q)}O_K(q)
+
\frac{m_F(q)}{m_K(q)+m_F(q)}O_F(q).
$$

因此，对固定的 $q$ 和相同的未压缩部分，如果压缩块同时保留局部输出和 mass，就能保留它在拼接后的贡献。实际算法在有限参考 queries 上近似实现这两个条件。[原文 §2、Appendix A.2](https://arxiv.org/html/2602.16284v2#A1.SS2)

**学长分析（转述）：** KV 块一方面通过 mass 与其他块竞争，另一方面提供局部 attention 的 value 加权输出；可以继续追问这个概括是否遗漏了重要作用。

**补充核对：** 在固定 query 的标准 attention 计算中，上面的分解关系已完整描述块的贡献。后续更直接的切口是有限参考 queries 上的近似是否泛化、误差是否随层传播，以及位置语义是否保持。它们需要与“再找一种 KV 作用”分开分析。

## Method：原手绘图与六步流程

![[0001.jpeg]]

手绘图里的例子说明了为什么只保持局部输出不够：

- 原 values 为 $[1,3,5,7]$，压缩后为 $[2,6]$；假设各条目的 attention logits 都为 $0$，两者局部输出均为 $4$。
- 拼接一个 value 为 $10$、logit 同样为 $0$ 的条目后，原输出为 $26/5=5.2$，压缩后为 $18/3=6$。
- 为两个压缩条目分别加入 $\beta=\log 2$，就能在这个例子中恢复原块的 mass，使拼接输出回到 $5.2$。

这个例子验证了 mass 的作用；真实 queries 下的拟合效果还需要实验检查。

### Step 0：构造参考 queries $Q_{\mathrm{ref}}$

通过 repeat-prefill、self-study 收集模型处理上下文时产生的 queries；on-policy 方式进一步考虑前面层已经压缩后的状态。

**需要成立的假设：** 这些参考 queries 能覆盖未来任务中重要的注意力行为。拟合误差小，首先说明参考样本上的匹配较好。

### Step 1：选择 compact keys $C_k$

将 $C_k$ 限定为原始 keys 的子集，采用 Highest Attention Keys 或 OMP 选择。前者使用注意力重要性分数；后者围绕 mass 拟合残差进行贪心选择。

**阅读判断：** 子集限制降低了搜索成本，也限制了表示能力。随着缓存预算缩小，“原始 key 中是否存在合适的少量代表”会变得更重要。

### Step 2：拟合 attention biases $\beta$

令 $w_j=\exp(\beta_j)$，并记：

$$
A_{ij}=\exp\left(\frac{q_i(C_k)_j^\top}{\sqrt d}\right).
$$

由 mass 匹配得到非负最小二乘问题：

$$
w^\star=\arg\min_{w\geq 0}\|Aw-m\|_2^2,
\qquad
\beta_j=\log w_j.
$$

$w_j$ 可以理解为一个保留 key 所代表的 attention mass 权重；实现中对极小权重做截断后再取对数。

### Step 3：拟合 compact values $C_v$

固定 $C_k$ 和 $\beta$ 后，压缩缓存的局部注意力权重构成固定的设计矩阵：

$$
X=\operatorname{softmax}\left(\frac{Q_{\mathrm{ref}}C_k^\top}{\sqrt d}+\beta\right),
$$

其中 $\beta$ 按 query 行广播。以原缓存的局部输出构成 $Y$，求解：

$$
C_v^\star=\arg\min_{C_v}\|XC_v-Y\|_F^2.
$$

这一步让保留的 values 承担重新拟合后的作用。

**学长分析（转述）：** 先固定 key，再利用最小二乘直接求 value，从而改变整个 pipeline。

**补充核对：**“key 不变”在这里是求解策略：为当前上下文选择 $C_k$，拟合并固定 $\beta$ 后，再条件求解 $C_v$。不同上下文、层和 heads 仍可选择不同 keys；需要研究的是子集限制和分阶段求解的代价。

**实现阅读要点：**“部分子问题可高效求解”应和整个流程的求解方式分开。key 选择包含搜索，NNLS 的实现也可使用迭代求解。[Appendix C.2](https://arxiv.org/html/2602.16284v2#A3.SS2)

### Step 4：为不同 KV heads 分配不同预算

手绘图已经记录了按 head 敏感性分配容量的思路。理解这个模块时，需要检查 head 重要性是否随输入变化，以及多个 head 同时改变容量时，单 head 测试还能否预测整体效果。[原文 §3.4](https://arxiv.org/html/2602.16284v2#S3.SS4)

### Step 5：分块压缩并拼接缓存

分块可以控制每次处理的规模。原文区分从完整 prefill 后的 KV 切块，以及独立处理文本块；还保留原序列的逻辑位置语义。[原文 §3.5、Appendix C.3](https://arxiv.org/html/2602.16284v2#S3.SS5)

**阅读判断：** 分块之间的可拼接性、跨块信息是否已进入 KV、后续 token 的位置编号，是需要分别检查的问题。

## Core Hypothesis

**阅读判断：** 这个方法的有效性依赖一条连续的证据链：

参考 queries 覆盖重要行为 → 少量 keys 与拟合后的 biases、values 能近似这些行为 → 逐层压缩的误差保持可控 → 下游任务效果能够保留。

分块注意力恒等式解释了固定 query 下两个目标的作用。未来 query 的覆盖、层间误差传播和最终任务表现，需要进一步用实验支撑。

> [!question] 待思考：哪些假设最容易失效？
> - [ ] 如果未来问题只追问很少出现的细节，reference queries 会不会漏掉相应行为？能否按“常见信息／稀有细节”分别测量效果？
> - [ ] 前面层压缩会改变后面层的 query 分布。on-policy queries 缓解了多少偏移？连续生成时偏移是否还会累积？
> - [ ] 先拟合 mass、再拟合输出，会不会过早降低某些 key 的权重，使其难以参与后续输出拟合？应怎样检查这个两阶段过程？
> - [ ] 单独测量 head 敏感性时，其他 heads 保持固定；多个 heads 同时压缩时，是否出现明显交互作用？
> - [ ] 独立文本块与从完整 KV 切出的块，在跨块依赖任务上有何差异？位置对齐能修复什么，又有哪些信息需要在 prefill 时形成？

# 三、实验如何设计：比较对象、证据与结论范围

## Baselines：为什么选这些方法？

以下“比较意义”是阅读判断，用来明确每个基线回答的问题。

- **[[Cartridges——Lightweight and general-purpose long context representations via self-study|Cartridges]]：** 端到端优化 compact latent KV。比较意义：检验快速构造缓存能在质量上接近昂贵优化到什么程度。
- **H2O：** 基于注意力分数筛选 KV。比较意义：检查只筛选、直接保留原始 KV 的近似效果。
- **SnapKV（[Li et al., 2024](https://arxiv.org/html/2602.16284?_immersive_translate_auto_translate=1#bib.bib11)）与 PyramidKV（[Cai et al., 2025](https://arxiv.org/html/2602.16284?_immersive_translate_auto_translate=1#bib.bib12)）：** 比较已有的注意力筛选及预算分配策略。
- **KVzip（[Kim et al., 2025a](https://arxiv.org/html/2602.16284?_immersive_translate_auto_translate=1#bib.bib13)）：** 同样面向未来问题未知的缓存复用。比较意义：需要仔细拆开 query 来源、key 选择、head 预算和重新拟合各自带来的收益。
- **KVMerger（[Wang et al., 2024](https://arxiv.org/html/2602.16284?_immersive_translate_auto_translate=1#bib.bib25)）：** 通过合并条目减少 KV。比较意义：检验聚合原有条目与重新拟合其注意力作用之间的差异。
- **Summarization：** 以全文摘要或分块摘要替代原文。比较意义：和文本层面的压缩比较信息保留情况；原文测试了多种摘要提示词，见 [Table 7](https://arxiv.org/html/2602.16284?_immersive_translate_auto_translate=1#A2.T7 "Table 7 ‣ Summarization. ‣ B.3 Baselines ‣ Appendix B Experimental Details ‣ Fast KV Compaction via Attention Matching")。

## 比较条件：基线为适配本文场景做过改动

原文将 H2O 改为一次性压缩版本 H2O+；SnapKV、PyramidKV 使用合成问题适配 question-agnostic 场景。DuoAttention 在所测压缩率下表现较差，被省略在主图之外。具体配置见 [Appendix B.3](https://arxiv.org/html/2602.16284?_immersive_translate_auto_translate=1#A2.SS3.SSS0.Px2 "H2O, SnapKV, PyramidKV, and KVzip. ‣ B.3 Baselines ‣ Appendix B Experimental Details ‣ Fast KV Compaction via Attention Matching")。

**阅读判断：** 统一场景有助于比较。解释结果时，也要记住原方法的适用条件已经变化。对于未放入主图的结果，应继续检查实现、超参数和适用的压缩范围。

## Evaluation Protocol：实验究竟在测什么？

主实验在压缩后的缓存上回答 QuALITY 和 LongHealth 问题。需要记录的样本边界是：QuALITY 中 Cartridges 只在 20 篇文章上评估，其他方法评估 50 篇；LongHealth 有 4 个上下文。[Appendix B.2](https://arxiv.org/html/2602.16284v2#A2.SS2)

**阅读判断：**

- **一次压缩、多次提问**对应缓存复用；应同时考虑构造成本和后续使用成本。
- **多个问题共享上下文**意味着统计不确定性时需要考虑文档层面的相关性。
- **先比较完整缓存与无上下文基线**，才能判断压缩方法保留了多少依赖上下文的能力。
- **选择题正确率与行为重建误差**回答不同问题，应结合阅读。

v2 还包含 QASPER、LongBench v2 和 RULER；其中 LongBench v2 使用 47 个筛选后的上下文。新增实验要和原有主实验分别记录。[原文 §4.2](https://arxiv.org/html/2602.16284v2#S4.SS2)

## 原实验图：怎样读出有效结论？

![[Pasted image 20261020190628.png]]

- 横轴 Compacted Size 表示保留比例：$0.02$ 对应保留 $2\%$，即 $50\times$ compaction。
- 两条虚线分别是 Original Cache 和 No Context，帮助判断相对完整缓存的损失，以及相对无上下文的收益。
- 原图呈现了 AM 在质量上的优势，也显示不同数据集对高压缩率的敏感程度不同。
- Gemma 的压缩比例只针对 global-attention 部分，应和整个 KV cache、整卡显存的节省区分。[原文 §4.3](https://arxiv.org/html/2602.16284v2#S4.SS3)

**作者解释：** 在 LongHealth 的部分极端压缩设置下，Cartridges 优于 AM，作者将其归因于更宽的紧凑表示搜索空间。

**阅读判断：** 这个解释还应与 query 覆盖、子集限制、局部目标和预算分配等候选原因区分。单看最终正确率难以完成因果归因。

## Ablation：哪些模块真正贡献了收益？

原文 Figure 5 的消融中，非均匀 head 预算影响最大；指标使用原缓存生成内容上的 log-perplexity。[原文 §4.6](https://arxiv.org/html/2602.16284v2#S4.SS6)

**学长分析（转述）：** self-study 与 head 自适应敏感度的收益值得重点解释，分析其机制后再寻找切口。

**原文核对（v2）：** Figure 5 的完整方法去除组件实验中，self-study、on-policy 的影响相对较小；Figure 6 的参考 query 策略比较中，self-study 最好，尤其在极端压缩下。解释“提升很大”时，需要说明它相对哪种 query 策略、使用何种预算和指标。[Appendix D、Figure 6](https://arxiv.org/html/2602.16284v2#A4)

**阅读判断：** 整套方法效果更好，和“提升主要由 attention matching 带来”，需要不同强度的证据。消融有助于拆分贡献，还需要关注模块间的交互，以及代理指标与下游任务是否一致。

**补充实验设想（尚未执行）：** 在每种 head 预算方案内固定 $Q_{\mathrm{ref}}$ 和 key 子集，比较保留原始 KV、仅拟合 $\beta$、仅拟合 $C_v$、同时拟合二者这四种配置；再分别对 uniform 和 nonuniform 预算执行。这样可以更清楚地检查重新拟合与预算分配的作用。

## System Efficiency：时间、质量和节省是否对应同一设置？

原笔记记录了分钟级与小时级的差别。继续分析时，需要将每个 AM 变体的质量与它自身的耗时配对，并核对计时硬件及计入的阶段。[原文 §4.4](https://arxiv.org/html/2602.16284v2#S4.SS4)

**阅读判断：** 应分别关注参考 queries 构造、key 选择、数值求解、峰值显存及压缩后的解码开销。缓存条目减少的倍数，需要通过系统测量才能转化成实际服务收益。

> [!question] 待思考：实验结论能支持到什么程度？
> - [ ] 和 KVzip 等方法相比，统一 query 来源、head 预算及 key 子集后，拟合 $\beta$、$C_v$ 的独立收益有多大？上面的补充实验能否隔离这些变量？
> - [ ] Cartridges 的比较是否使用相同文档集合和相同硬件？固定缓存大小、固定构造时间两种约束下，结论各是什么？
> - [ ] LongHealth 按上下文分别统计后，结果是否稳定？LongBench v2 筛选子集上的结论，适用于怎样的问题范围？
> - [ ] 非均匀预算带来的质量收益，能否在实际缓存布局和 attention kernel 中兑现？query 生成、峰值显存和预算预计算应如何计入总成本？

## Discussion 与后续方向

原笔记标出的解释保留为：==🔴端到端优化可以在更宽的紧凑表示空间中搜索==。它是需要继续验证的机制解释。

后续方向可按对应限制整理：

1. **放宽 $C_k$ 子集限制。** 检查直接构造新 keys 能否提高极端压缩率下的表示能力，以及增加了多少求解成本。
2. **集成推理引擎，结合检索与文本摘要。** 测量真实解码、缓存切换和并发服务收益；按任务的信息保留需求选择组合方式。
3. **继续验证在线压缩。** v2 已在 Appendix F.3 给出初步在线实验。[Online Compaction](https://arxiv.org/html/2602.16284v2#A6.SS3) 仍需思考反复压缩时的误差积累、历史细节遗忘及长轨迹效果。

# 四、从学长分析出发：待验证的研究切口

以下均为研究问题或实验设想，尚未执行；候选机制需要用对照实验区分。

## 切口 A（优先）：self-study 的收益来自覆盖，还是拟合条件？

**研究问题：** 固定 query 数量和 KV 预算后，self-study 是否仍优于 repeat-prefill？它改善的是稀有信息覆盖、queries 的有效多样性，还是设计矩阵 $X$ 的数值性质？

**候选假设：** self-study 产生的访问方式覆盖了重复上下文时较少使用的信息；也可能使拟合样本更丰富，减少冗余。两种解释应分别测量。

**最小验证：** 对同一批文档，比较 repeat-prefill 与 self-study 两种 query 来源，分别搭配 uniform 和原有 nonuniform head 预算，形成四组。每个条件下固定总 KV 容量、参考 query 数量、key 选择与拟合算法，并保持评估问题独立于采样任务。

重点记录未见 queries 上的输出误差、普通与稀有细节问题的正确率，以及构造和求解各阶段的耗时。可辅助检查 $X$ 的有效秩、奇异值谱；跨条件比较时明确使用相同的归一化与统计口径。

**结果判断：** 若收益在等样本数下仍存在且主要落在稀有问题上，支持覆盖解释；若与矩阵数值性质同步变化，可进一步检查拟合条件；若收益随 head 预算显著变化，应继续分析两模块的交互。上述相关性还需后续对照来定位原因。

> [!question] 待验证 A
> - [ ] 先做等 query 数量的四组对照，确认 self-study 的收益在什么条件下出现。
> - [ ] 再按等构造时间比较：能否用更便宜的任务或更少的生成达到相近效果？生成完整池后再抽样，主要减少后续计算，不能直接计为 query 构造时间的节省。

## 切口 B：mass 选 key 是否足够，key merging 能否带来独立收益？

**学长分析（转述）：** 可以改变按照 mass 选择 key 的方式，或以选中的 key 为锚点，吸收其他条目的信息，把 merge 接入流程。

**补充核对：** 这个问题主要针对 AM-OMP。它围绕 mass 残差选 key；HighestAttnKeys 则使用聚合注意力分数。分析改进时需要明确基线版本。

**研究问题：** mass 误差相近的 key 子集，在拟合 $C_v$ 后，是否仍有明显不同的输出误差？如果存在这种差异，选择或构造 keys 时考虑输出侧信息就有进一步研究的依据。

### 后续相关工作：ARC-KV

[ARC-KV: Amortizing Anchor Search for Reconstruction-Based KV Cache Compaction](https://arxiv.org/abs/2609.36835) 于 2026-09-29 提交，其 value-aware 锚点选择、受约束的 key merging、随后拟合 $\beta$ 与 $C_v$，与学长描述的近期方向很接近。这里记录为相关工作，尚不能据转述确定就是学长指的那篇论文。[方法 §3.2–3.4](https://arxiv.org/html/2609.36835v1#S3.SS4)

**推导判断：** 若只先合并 values，随后仍使用相同的 $X$、$Y$ 和同一确定的精确最小二乘求解规则，最终解不会因这次初始化而改变。需要说明 merging 改变了 keys、目标或约束中的哪一项。ARC-KV 的 key merging 改变 $C_k$，因此也改变后续拟合的设计矩阵 $X$。

**最小验证：** 固定 query 来源与数量、每个 head 的 KV 预算和求解流程，比较三组：原有 OMP anchors 后拟合；相同 anchors 做 key merging 后重新拟合；引入 value-aware 选择但不做 merging，再拟合。分别测量 mass 误差、未见 queries 的输出误差、下游正确率和总构造成本。前两组检查 merging，第一与第三组检查选择策略。

> [!question] 待验证 B
> - [ ] 先对照阅读 ARC-KV，明确其已解决的部分，以及候选改动在目标、约束、计算成本上的实际差异。
> - [ ] 若 mass 匹配已好而输出仍差，进一步区分 key 子集的表示限制、参考 query 覆盖不足及求解数值问题。

## 切口 C：head 敏感度是固定的模型特征，还是随文档、任务变化？

**机制直觉：** 不同 heads 增加容量后的边际收益不同。把预算分给收益更高的 heads，能够提高相同总容量下的效果。需要验证的是：这个收益排序能在多大范围内复用。

**已有证据：** AM 分析了 head 敏感度和预算的跨输入稳定性；ARC-KV 也比较了固定模型级预算与 uniform 分配。[AM §3.4、Appendix E](https://arxiv.org/html/2602.16284v2#A5)、[ARC-KV Appendix E.3](https://arxiv.org/html/2609.36835v1#A5.SS3)

**研究问题：** 同一文档上，普通理解与稀有细节任务对应的 head 敏感度排序是否变化？换用 self-study queries 后，原有预算是否仍合理？多个 heads 同时压缩时，单 head 测试能否预测整体收益？

**最小验证：** 先在少量文档上，用不同 query 来源或任务分组测量敏感度排序的变化。若变化明显，再在固定总容量下比较原有预算与由独立校准任务得到的预算。校准与测试问题分离，并把校准计算计入成本。

**结果判断：** 排序稳定时，固定预算是有价值的简化；排序变化且重新分配能稳定改善未见问题时，才有依据继续设计更便宜的上下文或任务自适应预算。

> [!question] 待验证 C
> - [ ] 先测敏感度排序是否随 query 分布变化，再决定是否实现动态预算。
> - [ ] 检查预算收益是否依赖少数 heads，以及联合压缩时是否存在单 head 曲线无法解释的交互。

## Question：把原有疑问具体化

- **$Q_{\mathrm{future}}$ 和 $Q_{\mathrm{ref}}$：** 压缩时额外构造的是参考 queries；真实未来 queries 在正常回答过程中逐层产生。未来样本上能否保持效果，对应切口 A。先完整运行未来问题再收集 queries，可作为离线诊断对照，但会用到未来问题的信息，需要与实际部署结果分开。
- **“大文档效果不好”：** 先明确观察到的是上下文变长、信息密度增加，还是保留比例下降后的退化。“端到端优化可以在更宽的紧凑表示空间中搜索”是一个候选解释；还需与 query 覆盖、分块和预算问题区分，不能直接把所有长文档退化归到同一原因。

> [!todo] 下一步顺序
> - [ ] 先做切口 A 的四组诊断，找到值得解释的稳定差异。
> - [ ] 同时读 ARC-KV 的锚点选择与 key merging，对照切口 B 的实际差异。
> - [ ] 有了敏感度排序变化的证据，再推进切口 C 的预算调整。

会议来源：[ICML 2026 正式页面](https://proceedings.mlr.press/v306/zweiger26a.html)，核查日期：2026-10-23。
