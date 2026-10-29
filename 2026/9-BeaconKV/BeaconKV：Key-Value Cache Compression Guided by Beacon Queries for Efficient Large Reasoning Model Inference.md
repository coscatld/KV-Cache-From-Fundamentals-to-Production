url： https://arxiv.org/abs/2609.04971
首次 arXiv 提交：2026-09-04
发表信息：[ICML 2026 会议页面](https://proceedings.mlr.press/v306/kim26af.html)
核查日期：2026-10-23

keyword： #KVCache_Eviction ， #Proxy_Queries ， #LLM_Reasoning

背景：
他们查询分为两种不同类型：局部查询，主要关注邻近的键；全局查询，对应于 TRT，关注推理轨迹中远处的键。
然后他们发现在CoT中会产生思维重访令牌（TRT），并且与对应的Q Cluster into a small number of similarity groups in embedding space。
因此他们通过**持续最远点采样（FPS）** 通过为每一个cluster选择一个代表性查询，叫 **信标查询**。然后利用”Attention-Based KV Scoring with Recent Queries“为每个 KV 条目计算重要性分数，筛选出合适的local and global KV


## TRT
TRT 产生的全局查询，在查询嵌入空间（query embedding space）里并非杂乱无章地随机分布，而是会形成少数几个明显的“相似性簇”（similarity clusters）。

因此现有的**KV 驱逐方法**的假设是错误的，即**近期查询是预测未来注意力的良好代理**
## Geometric Structure of Global Querie
- 用的是 **pre-RoPE 查询**（还没加旋转位置编码的查询向量）。
- 距离用的是 **余弦相似度**：两个向量夹角越大（余弦相似度越低），就认为它们几何上差异越大。
- FPS（最远点采样）的目标就是：每次都选一个和当前已选集合**最不相似**（余弦相似度最小）的查询。
![[Pasted image 20261026162514.png]]


## 具体公式

![[Pasted image 20261026161420.png]]
### 通过持续 FPS 进行观测查询选择
BeaconKV 从已生成的 pre-RoPE 查询中选取具有代表性的查询作为信标查询（beacon queries），用于后续的 KV 评分。传统 FPS 需要访问全部历史查询，内存开销随序列长度增长。为此，我们采用 Continual FPS，一种内存高效的在线算法。

具体而言，每个注意力头不再存储所有历史查询，而是维护一个容量有界的缓冲区。当缓冲区达到最大容量 $B_Q^{max}$ ​ 时，触发一次 FPS 下采样，将其压缩回最小规模 $B_Q^{min}$，仅保留几何上最具区分性的查询：
$$Q_{obs}^{pre}←FPS⁡(Q_{obs}^{pre},B_Q^{min})$$
最终的观测查询集由两部分构成：​
$$Q_{observation}=Q^{pre}_{recent}+Q^{pre}_{beacon}$$

### 基于注意力的 KV 评分与 Beacon 查询
在解码步 t 达到 KV 缓存预算上限时，BeaconKV 使用上述观测查询集对当前 KV 缓存进行重要性评分。首先对查询进行位置对齐：信标查询使用当前步 t 的 RoPE，近期查询保留其原始位置编码。随后计算注意力权重，并以 max 聚合得到每个 KV 位置的分数：

$$\operatorname{Score}[j] = \max_{h \in g} \max_{q \in Q_{\text{obs}}} W[h, q, j]$$

最后结合固定保留的前缀与近期窗口，按分数进行 Top-K 选择，完成 KV 缓存压缩。

## BeaconKV 实验设置

### 模型
- R1-Distill-Qwen-7B
- R1-Distill-Llama-8B
- Qwen3-4B
- Qwen3-14B

### 基准任务
| 领域 | 数据集 | 指标 |
|------|--------|------|
| 数学 | AIME24、MATH-500 | pass@1 |
| 代码 | LiveCodeBench | pass@1 |
| 科学 | GPQA-Diamond | pass@1 |

- AIME24 跑 8 次取平均，其余跑 4 次取平均。
- 最大生成长度：32,768 tokens。
- 采样：Top-p = 0.95，temperature = 0.6。

### KV 缓存预算
- 最大预算 $B_{\text{KV}}^{\max}$：256 / 512 / 1024 / 2048 / 4096
- 最小预算 $B_{\text{KV}}^{\min} = \frac{7}{8} B_{\text{KV}}^{\max}$（达到上限时驱逐约 1/8）
- 始终保留：前缀 + 最近窗口（通常 32 tokens；R-KV 按原文用 8）

### BeaconKV 超参
- Beacon queries：max 32 → min 16（Continual FPS）
- Recent queries：固定 16
- 评分：max 聚合（非平均）

### 对比方法
- Full KV（无压缩）
- RPC
- R-KV
- SnapKV
- Initial+Recent（消融对照）

### 硬件与效率
- 单卡 NVIDIA A100 80GB
- 关注指标：准确率、吞吐量（tokens/s）、解码延迟、峰值显存
- 在 2K 预算下，显存最高可降至约 1/5.8，吞吐量最高提升约 4.3×

### 主要消融
- Recent / Beacon 数量组合（16+16 较优）
- Max vs Mean 聚合
- Continual FPS vs K-Means / 普通 FPS 等

整体看，实验覆盖了不同规模的开源推理模型、数学/代码/科学三类任务，并在多种严格 KV 预算下对比了主流压缩方法，同时报告了准确率与系统效率。

## supplement
### FPS：
从一堆点里挑出一个有代表性的子集。做法是：

1. 先随便选一个点当起点；
2. 每次都选“离已选点最远”的那个点加入集合；
3. 重复直到选够数量。

这样选出的点会尽量均匀地覆盖整个空间，而不是扎堆在一起。


### 基于recent的score of attention

具体而言，令 $q_\tau \in \mathbb{R}^d$ 表示解码步 $\tau$ 处的查询，$k_j \in \mathbb{R}^d$ 表示索引 $j$ 处的缓存键。从 $q_\tau$ 到 $k_j$ 的注意力权重通过缩放点积注意力计算：

$$
w_{\tau,j} = \frac{\exp(q_\tau^\top k_j / \sqrt{d})}{\sum_l \exp(q_\tau^\top k_l / \sqrt{d})}.

$$

近期基于注意力的 KV 评分方法（Li et al., 2024; Song et al., 2025; Cai et al., 2025）在一组观测查询 $Q_{\text{obs}}$ 上聚合上述注意力权重。观测查询定义为来自最近若干解码步的查询集合，窗口大小 $N_{\text{obs}}$ 控制使用多少个近期查询来估计 KV 重要性。形式上，当在解码步 $t$ 触发 KV 驱逐时，观测查询集定义为：

$$
Q_{\text{obs}} := \{ q_\tau \mid \tau \in \mathcal{T}_{\text{obs}} \}, \quad \mathcal{T}_{\text{obs}} = \{ t - N_{\text{obs}} + 1, \dots, t \}.

$$

给定观测查询集 $Q_{\text{obs}}$，缓存键 $k_j$ 的重要性分数 $s_j$ 通过聚合 $Q_{\text{obs}}$ 中查询所诱导的注意力权重计算，可采用取最大值或取平均值的方式：

$$
s_j^{\max} = \max_{\tau \in \mathcal{T}_{\text{obs}}} w_{\tau,j}, \qquad s_j^{\text{mean}} = \frac{1}{N_{\text{obs}}} \sum_{\tau \in \mathcal{T}_{\text{obs}}} w_{\tau,j}.

$$

利用得到的重要性分数 $\{ s_j \}_{j=1}^L$ 和 KV 缓存预算 $B_{\text{KV}}$，通过保留前 $B_{\text{KV}}$ 个 KV 对来执行驱逐。令 $\mathcal{I} \subset \{1, \dots, L\}$ 表示前 $B_{\text{KV}}$ 个分数对应的索引集：

$$
\mathcal{I} = \operatorname{TopK}\big( \{ s_j \}_{j=1}^L, B_{\text{KV}} \big).

$$

压缩后的 KV 缓存则通过沿序列维度索引得到：

$$
\hat{K} \leftarrow K[:, \mathcal{I}, :], \qquad \hat{V} \leftarrow V[:, \mathcal{I}, :].
$$
