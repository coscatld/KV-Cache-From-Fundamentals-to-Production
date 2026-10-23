url: https://arxiv.org/abs/2608.23843

keyword： #low-rank_decomposition ， #KVCache_Eviction ， #KVCache_Quantization 


# 脉络
先前的研究：以往的低秩方法通常从模型权重中导出固定投影空间、从校准激活中构建固定空间，或在较大的缓存区域上构建共享基：可能无法捕捉细节丰富但重要的信息。

观察：将每个注意力头的 KV 缓存划分为固定长度的逻辑页，并观察到单个页内存在显著的秩结构，存在大量低秩冗余。（从 PagedAttention 用于管理 KV 缓存内存的固定大小块中获得启发，并将逻辑页重新解释为压缩单元）

提出贡献：
- 我们提出 PuzzleKV，一种无需训练和校准的方法，它在每一层和每个 KV 头内独立分解已完成的 KV 页，通过页局部低秩表示保留每个 token。
- 我们实现了 PuzzleKV，支持批量页分解、无需重建即可直接对稠密页和分解页执行注意力，以及在解码过程中增量转换页。
- 我们在 Qwen3-8B 和 Llama-3.1-8B-Instruct 上，基于 RULER 和 LongBench 对 PuzzleKV 进行了评估。在约为 Full KV 存储 60% 的情况下，PuzzleKV 在所有评估设置中达到 Full KV 性能的 96% 以上，在 RULER 上相较 Global SVD 取得显著提升，并在 LongBench 上达到有竞争力的性能。结合逐因子 INT4 量化，它仅使用原始存储的 18.7%，同时达到 Full KV 性能的 93% 以上。
---
# method
![[Pasted image 20261022165118.png]]

### 1.Dense Prefill and Page Partition and Page-wise Decomposition
- Prefill： 输入整个 Prompt，模型逐层计算 Q、K、V，并正常完成 Attention。K、V 写入 KV Cache。
- 分页： 对每层、每个 KV Head 的 K、V 沿 Token 维度划分逻辑页。
- SVD 截断： 对符合条件的页面分别压缩 K、V，保存低秩因子。
- Decode： 生成新 Token 时，直接利用压缩页和保留的稠密页计算 Attention。

### 2.Mixed Paged Attention
每个 Query 依次完成：**逐页计算 logits → 全局 Softmax 归一化 → 逐页聚合 Value → 求和输出**。稠密页读取原始 KV，低秩页直接使用因子，无需重建完整 KV。

**① 页面表示。** 对单个 Query $q\in\mathbb R^{1\times d}$，稠密页保存 $K_j\in\mathbb R^{P\times d}$、$V_j\in\mathbb R^{P\times d_v}$；低秩页保存：
$$
K_i\approx A_i^K B_i^K,\qquad V_i\approx A_i^V B_i^V
$$
其中 $A_i^K\in\mathbb R^{P\times r_k}$、$B_i^K\in\mathbb R^{r_k\times d}$；Value 因子形状对应为 $P\times r_v$、$r_v\times d_v$。$P$ 为页长度，$r_k,r_v$ 为各自的秩。

**② 逐页计算 logits。** 稠密页直接与 Key 相乘；低秩页先将 Query 投影到秩空间，再得到该页每个 Token 的分数：
$$
S_p=
\begin{cases}
qK_p^\top/\sqrt d, & p\in\mathcal D\quad\text{（稠密页）}\\[2pt]
\bigl(q(B_p^K)^\top\bigr)(A_p^K)^\top/\sqrt d, & p\in\mathcal C\quad\text{（低秩页）}
\end{cases}
$$
两种路径均得到 $1\times P$ 的 logits，并按 Token 的逻辑位置施加因果掩码。

**③ 全局归一化。** 所有页面共同竞争 Attention 权重，而非每页独立归一化：
$$
[W_1,\ldots,W_m]=\operatorname{softmax}\bigl([S_1,\ldots,S_m]+M\bigr)
$$
$M$ 为掩码，可见位置为 $0$，被屏蔽位置为 $-\infty$。上式表达数学上的全局 Softmax；实现可通过在线 Softmax 合并各页的统计量与部分输出，无需实际拼接全部 logits。

**④ 聚合 Value 并求和。** 使用同一全局 Softmax 得到的各页权重 $W_p$，稠密页直接加权，低秩页先聚合到秩空间，再映射回 Value 维度：
$$
\boxed{
O=\sum_{j\in\mathcal D}W_jV_j
+\sum_{i\in\mathcal C}(W_iA_i^V)B_i^V
}
$$
输出 $O\in\mathbb R^{1\times d_v}$；相对于原始 Full KV，近似误差来自 K、V 的低秩分解。

### 3.Incremental Cache Update
PuzzleKV 将压缩从预填充阶段扩展到整个解码过程，随着生成过程持续推进，混合缓存始终保持最新状态，而非仅压缩一次。每个新的键值对都会追加到最新的稠密页中，并立即参与注意力计算。当某个解码步骤填满该页时，该页便进入稠密局部窗口，而最旧的一页离开窗口时会被分解一次，其因子写入预分配存储，并释放其稠密槽位。已分解的页永远不会被重新访问，因此每一步最多只产生一次页分解，且每个生成的 token 都以压缩形式保留。

| 页面状态  | 判定条件         | 执行操作     |
| ----- | ------------ | -------- |
| 当前稠密页 | 尚未填满         | 继续追加 K、V |
| 近期稠密页 | 已填满，但仍在近期窗口内 | 保持完整 K、V |
| 待压缩页  | 已填满，且离开近期窗口  | 执行截断 SVD |
| 已压缩页  | 已保存低秩因子      | 不再重新分解   |


---
# supplement
### 划分页：
1. 保留局部细节。 不同上下文区域可能具有不同的统计结构，独立分解可以降低共享基底带来的近似误差。
2. 支持增量压缩。 自回归生成时，新 Token 不断进入 KV Cache。一个页填满并满足压缩条件后，就可以独立压缩，不必每次重算整个历史缓存。
3. 兼顾计算效率。 推理时可以直接处理尚未压缩的稠密页，并对已压缩的页使用相应的低秩计算方式。
![[Pasted image 20261022171416.png]]

### 截断 SVD：
假设KV cache的的 Key 矩阵为：
$$K\in\mathbb R^{N\times d} $$
首先进行奇异值分解（SVD）：
$$K=U\Sigma V^\top $$
其中：
- $U$：左奇异向量组成的矩阵。
- $\Sigma$：由奇异值构成的对角矩阵，奇异值按从大到小排列。
- $V$：右奇异向量组成的矩阵。
截断 SVD 只保留前 $r$个奇异值：
$$\boxed{K_r=U_r\Sigma_rV_r^\top} $$
其中 $r<\min(N,d)$。
![[Pasted image 20261022185840.png]]