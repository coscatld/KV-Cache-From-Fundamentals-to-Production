url： https://arxiv.org/abs/2608.12331
keyword： #KVCache_Merging , #KVCache_Eviction , #Attention_matching 


大背景：模型在推理过程中会产生过长CoT序列，不仅增大内存占用也降低了解码吞吐量。

现有方法的局限：
- 针对预填充阶段的长输入提示，但在解码过程中对已生成 token 的高效压缩，即轨迹中途压缩，尽管对于经常产生长中间 token 链的推理工作负载至关重要，却仍未得到充分探索。

- [[Fast KV Compaction via Attention Matching（ICML）| 注意力匹配（Attention Matching, AM）]]通过构造紧凑的键和值来复现注意力输出并保留注意力质量，利用避免梯度下降的闭式解实现大规模一次性压缩。-> AM 面向预填充场景，并假设参考查询来自对固定上下文的自主学习。

- [[Reasoning Path Compression：Compressing Generation Trajectories for Efficient LLM Reasoning|推理路径压缩（Reasoning Path Compression, RPC）]]通过基于近期查询的选择器窗口定期逐出 token 来处理轨迹中途压缩。然而 RPC 依赖简单的逐出策略，正如我们形式化证明的那样，该策略在高压缩比下会系统性地低估注意力质量。

两种方法都将推理轨迹视为扁平 token 序列并施加均匀压缩，忽略了 CoT 推理的层次结构。尽管先前方法沿两个轴变化——压缩原语（逐出 vs. 优化）和查询来源（自主学习 vs. 选择器窗口）——但两者均未利用第三个正交轴：结构感知的预算分配。

# 研究视角：正交轴
```text
### 轴 1：压缩原语（Compression Primitive）

决定怎么压缩。

- Token Eviction：直接删除部分 Token 对应的 KV。
- Attention Matching：构造紧凑的 KV 表示，拟合原始注意力输出。

### 轴 2：查询来源（Query Source）

决定依据什么 Query 来指导压缩。

- Self-study：从已有上下文中自主生成参考查询。
- Selector Window：从近期推理轨迹中选取 Query。
- 其他方案：例如延迟压缩，使用后续实际生成的 Query。

### 轴 3：预算分配（Budget Allocation）

决定不同部分保留多少 KV。

- 均匀预算：不同推理片段采用相同的压缩比例。
- 结构感知预算：重要的推理片段保留更多 KV，冗余片段压缩得更激进。
```

# Thought-Aware Attention Matching (TAM)

![[Pasted image 20261024160008.png]]

## 方法主线

TAM 把 **近期 Query 引导的 AM** 变成结构感知压缩：先按推理步骤划分历史，再根据重要性和长度分配保留预算，保护关键 Token，最后对选出的 Key 统一拟合 Attention mass 与输出。**分段的是选 Key 和预算，全局进行的是偏置与 Value 拟合。**

```text
达到最大缓存长度，或距上次压缩已生成 P 个 Token
      ↓
保留最近 τ 个 Token，提取最近 R 个 Token 的参考 Query
      ↓
对可压缩前缀做推理分段
      ↓
计算 Token 注意力 → 识别关键 Token、计算片段重要性
      ↓
先预留关键 Token 的位置，再分配各段剩余预算
      ↓
每段选择非关键 Key，合并全部选中 Key
      ↓
全局拟合 β（NNLS）与 Value（OLS）
      ↓
拼回近期稠密尾部，继续自回归生成
```

来源：[§4、Algorithm 1](https://arxiv.org/html/2608.12331v1#S4)。以下按单层、单 KV head 说明，各层各头分别处理。

### 1. Selector Window：用哪些 Query？

当前长度为 $L$，取末尾 $R$ 个 Token，通过一次前向与 hooks 提取参考查询：
$$
Q_{\mathrm{ref}}\in\mathbb R^{R\times d}.
$$
它们代表当前推理前沿，用作未来读取的代理，不额外生成 self-study 对话。提取仍有前向计算成本。$R$ 是 **Query 观察窗口**，$\tau$ 是 **原始 KV 保护窗口**，两者作用不同。

来源：[§4.2](https://arxiv.org/html/2608.12331v1#S4.SS2)。

### 2. Thought Segmentation：历史怎么分段？

保留最后 $\tau$ 个 Token 的稠密 KV，只对前面的历史划分 $S_1,\ldots,S_m$，每段长度为 $n_i$。

- **默认规则：** 扫描解码文本中的双换行，把段落边界作为推理步骤边界；合并短于 $\ell_{\min}$ 的连续短段，避免过度切分。
- **替代方案：** 计算近期 Query 对各位置的平均注意力 $\bar\alpha_j$，用相邻差值 $|\bar\alpha_j-\bar\alpha_{j-1}|$ 检测突变；超过梯度中位数的两倍时设边界。

分段是启发式结构识别，不需要先判断每段属于“计划”“计算”还是“结论”。

来源：[§4.3](https://arxiv.org/html/2608.12331v1#S4.SS3)。

### 3. Importance：哪些片段更重要？

先计算参考 Query 的标准 Attention 权重：
$$
\alpha_j(q)=\frac{\exp(qK_j^\top/\sqrt d)}{\sum_{k=1}^{L}\exp(qK_k^\top/\sqrt d)}.
$$
这里在 **包括保护尾部的完整前缀** 上归一化。片段重要性为：
$$
w_i=\frac{1}{R}\sum_{q\in Q_{\mathrm{ref}}}\sum_{j\in S_i}\alpha_j(q).
$$
即“当前 Query 平均向该段分配多少注意力”，不是该段内部独立 Softmax 的结果。由于只对可压缩段求和，$\sum_iw_i\leq1$；剩余权重属于保护尾部。

这里的 $w_i$ 是**归一化注意力权重之和**，与后面 AM 匹配的未归一化指数和 Mass 要区分。

来源：[§4.4，式 5](https://arxiv.org/html/2608.12331v1#S4.SS4)。

### 4. Pivotal Protection：哪些 Token 直接保留？

计算可压缩前缀中各 Token 的平均注意力：
$$
\bar\alpha_j=\frac{1}{R}\sum_{q\in Q_{\mathrm{ref}}}\alpha_j(q),\qquad
\mathcal P=\{j:\bar\alpha_j>c\bar\alpha_{\mathrm{mean}}\}.
$$
默认 $c=3$，$\bar\alpha_{\mathrm{mean}}$ 为可压缩位置平均分数。高分位置被视为推理锚点，例如题目常数、关键中间结果或定义。

论文要求：关键 Token 的原始 Key、Value 直接保留，且其偏置固定为 $\beta_j=0$，保留原始 logits；其他 Key 的偏置才参与拟合。**本轮受到保护不等于永久保护**，下次压缩会重新评分。

来源：[§4.5、§4.7](https://arxiv.org/html/2608.12331v1#S4.SS5)。

### 5. Adaptive Budget：各段保留多少 Key？

按原文式 6 的记号，先从预算 $t$ 中扣除关键 Token：
$$
t'=t-|\mathcal P|.
$$
剩余预算按重要性和段长的几何平均分配：
$$
t_i=\max\left(1,\left\lfloor
 t'\frac{\sqrt{w_i n_i}}{\sum_{k=1}^{m}\sqrt{w_k n_k}}
\right\rfloor\right).
$$
取整余量按最大小数余数补齐。每段在 $S_i\setminus\mathcal P$ 中，选择注意力分数最高的 $t_i$ 个 Key；关键 Token 已单独预留，不重复占段内名额。

- 按 $n_i$ 分配：偏向段长，忽略重要性。
- 按 $w_i$ 分配：偏向注意力集中处，可能低估长段的表示需求。
- 按 $\sqrt{w_i n_i}$ 分配：同时考虑两者。

平方根分配的显式最优解对应附录的误差模型 $\phi(r)=r-1$，不是对任意真实 AM 误差都无条件最优。

来源：[§4.4、Proposition C.1](https://arxiv.org/html/2608.12331v1#S4.SS4)。

### 6. Global AM：选完 Key 后怎么拟合？

将关键 Key 与各段选中的 Key 合并为 $C_k$。针对待拟合的历史块，记原始 Key、Value 为 $K,V$，参考输出为 $Y=\operatorname{Attn}(Q_{\mathrm{ref}},K,V)$。

**① 全局匹配 Mass。** 令
$$
A=\exp(Q_{\mathrm{ref}}C_k^\top/\sqrt d),\qquad
b=\exp(Q_{\mathrm{ref}}K^\top/\sqrt d)\mathbf1.
$$
用 $u=\exp(\beta)$ 把偏置拟合写成 NNLS：
$$
\min_{u\geq0}\|Au-b\|_2^2,\qquad u_j=1\quad(j\in\mathcal P).
$$
关键 Token 的 $u_j=1$ 对应 $\beta_j=0$；非关键位置负责补偿被删 Key 的未归一化注意力质量。

**② 全局匹配输出。** 固定 Key 和偏置，构造
$$
X=\operatorname{softmax}(Q_{\mathrm{ref}}C_k^\top/\sqrt d+\beta),\qquad
\min_{C_v}\|XC_v-Y\|_F^2.
$$
Value 用最小二乘拟合。所有片段共同参与拟合，避免各段独立归一化后失去正确的跨段权重。最后将紧凑表示 $(C_k,\beta,C_v)$ 与保护尾部的稠密 KV 合并。

上式是全局拟合目标的整理；关键 Value 的固定约束需结合下方原文歧义理解。Mass 与输出的匹配针对参考 Query，不自动保证任意未来 Query 的输出保持不变。

来源：[§3.2、§4.7、Algorithm 1](https://arxiv.org/html/2608.12331v1#S4.SS7)。

### 7. Periodic Trigger：什么时候再压缩？

达到最大序列长度，或距上次压缩已新增 $P$ 个 Token 时，再次运行整个流程：重新取 Query、分段、评分、分配预算及拟合。此前的关键 Token 重新竞争保留资格。

- $P$ 小：峰值缓存较低，但压缩开销和重复近似更频繁。
- $P$ 大：压缩成本更容易摊薄，但生成期间缓存增长更多。

**TAM 的自适应主要体现在片段预算，触发规则本身是固定间隔/长度阈值。** 它也不同于“压缩一次后永远冻结旧轮”的在线 Agent 方案。

来源：[§4.6](https://arxiv.org/html/2608.12331v1#S4.SS6)。

### 8. 复现时需要核实的细节

1. **总预算是否包含尾部：** 式 6 使用 $t'=t-|\mathcal P|$，Algorithm 1 最后又拼回 $\tau$ 个尾部 Token；若 $t$ 是紧凑前缀预算，总长度应为 $t+\tau$。正文同时用缓存在 $t$ 与 $t+P$ 间波动描述总量。实现前需统一两种记号，显式区分前缀预算与总缓存预算。
2. **保护 Value 与 OLS：** 文字要求关键 K/V 不变，但 Algorithm 1 只明确固定 $\beta_j=0$，未写出全局 OLS 固定关键 Value 行的约束。严格保护时，应固定 $C_v^{\mathcal P}=V^{\mathcal P}$，只拟合其余行；这是满足文字要求的实现解释，需代码核实。
3. **预算边界：**$\max(1,\cdot)$、关键 Token 预留及段内候选不足可能与总预算冲突。$t'<m$、关键 Token 超预算或零重要性时的处理，不能仅靠取整公式保证。
4. **重复压缩后的分段：** 紧凑 KV 中有拟合后的 Value。如何维护原始文本位置与段标签，供下一次按“更新 Token 序列”重新分段，需进一步核查实现。

以上为阅读时发现的实现待核实项，不作为论文已明确给出的算法步骤。

## 对研究切入点的启发

- **Query 选择：** TAM 仍依赖近期 selector window；可测试代表性历史 Query 能否改善远期回读，而不只降低窗口内拟合误差。
- **时机选择：** 以固定 $P$ 为直接基线，研究何时值得等待更多 Query；在相同峰值预算下计入新增缓存与拟合开销。
- **表示组织：** TAM 已覆盖结构感知预算与分段选 Key。若研究局部 AM，需要进一步比较“全局拟合”和“各段独立拟合”，而不把分段本身当作新增贡献。

核查日期：2026-10-24。依据论文 v1 的 §3–4 与附录 A、C 梳理；未核查实现代码或运行复现。
