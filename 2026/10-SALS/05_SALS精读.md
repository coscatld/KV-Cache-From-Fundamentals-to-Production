# SALS 精读：在低秩空间选 token，而不是把 token 永久合并

核查日期：2026-10-22。本文是论文与官方代码的阅读报告，不是本地复现实验报告。数字来自原文表格；推导、批评与实验建议均另行标明。

## 1. 论文身份与一句话定位

- 题名：**SALS: Sparse Attention in Latent Space for KV Cache Compression**。
- 作者：Junlin Mu、Hantao Huang、Jihang Zhang、Minghui Yu、Tao Wang、Yidong Li。
- 发表：**NeurIPS 2025 主会**，已由[会议正式页面](https://proceedings.neurips.cc/paper_files/paper/2025/hash/00a0ebcad584c59dbc439c2af8793638-Abstract-Conference.html)核实，不只是预印本。
- 主阅读版本：[会议 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/00a0ebcad584c59dbc439c2af8793638-Paper-Conference.pdf)，并交叉检查[arXiv HTML](https://arxiv.org/html/2510.24273v1)。
- [官方仓库](https://github.com/Draconis98/SALS)，本次固定检查提交 `6aaa698b7aff321638e3f309c4e89a5fd09cb9d5`。论文摘要仍写“未来开源”，但检查时仓库已经公开。

**核心机制：历史 K 保留所有 token 的低维表示；当前 query 在不带 RoPE 的低维空间粗筛 token；只对入选 token 重建 K、恢复原位置 RoPE，并计算稀疏注意力。实际实验的 V 采用量化，不是同样的低秩投影。**

这解决的是“特征维度压缩与动态稀疏读取如何共同落地”，不等于把长度为 T 的上下文改写为 t 个永久压缩 token。

## 2. 为什么值得读：RoPE 把两种简单方案都卡住了

设单层 KV 总特征宽度为 D，缓存长度为 S，低秩维度为 r。

### 2.1 先压缩、再旋转：精度较好，但重建很贵

若缓存 pre-RoPE 的 K，并用正交基 U 压缩：

$$
Z=KU,\qquad \widehat K=ZU^\top,\qquad U\in\mathbb R^{D\times r}.
$$

历史每个位置拥有不同 RoPE 旋转。为了与当前 query 正确做位置相关匹配，朴素实现每步都要恢复全部 S 个历史 K。重建量级是 $O(SrD)$，可能比本来以读内存为主的 decode attention 更贵。

### 2.2 先旋转、再压缩：省重建，但子空间更难压

对每个 token 施加不同旋转，并不等于给整个矩阵乘同一个正交矩阵。共享低秩子空间可能因此被分散到更多方向，固定 r 下重建误差增大。

**数学阅读提醒：**RoPE 保持每个向量的二范数，不能把论文“variance amplification”误读为“每个向量能量变大”或“RoPE 对任何数据必然增秩”。不同位置旋转可改变数据集的均值、协方差谱和有效秩，这是经验观察而非无条件定理。若所有位置旋转相同，矩阵奇异值不变；若数据有特殊对称结构，也不保证有效秩增加。

另一个细节：原文式(2)写 $UU^\top\approx I$。当 r<D 时，它不可能在整个空间按算子范数逼近单位阵；合理含义应是**对目标数据主要占据的子空间近似恒等**。复现时应测 $\|K-KUU^\top\|$，不要验证错误的全空间恒等命题。[原文 §3、附录A](https://arxiv.org/html/2510.24273v1#S3)

## 3. 精确理解三个不同预算

| 符号 | 含义 | 控制什么 |
|---|---|---|
| r | K 缓存的 latent 宽度 | 常驻缓存大小、重建精度 |
| r* | 用于粗筛的 latent 宽度，通常小于 r | 每步扫描全部历史的成本 |
| k / N_c | 当前步最终入选 token 数 | 重建与精细 attention 成本 |

不能把 `SALS-25%` 当成“只保留25% token”，也不能把 `1/8 sparsity` 当成“KV缓存只剩1/8”。未入选 token 仍以低维 K、量化 V 形式留在缓存中，下个 query 可以再次选中。

## 4. 离线校准：训练自由，不是成本为零

论文先收集校准样本的 pre-RoPE K，把 KV-head 维度与 head-dim 合并，得到 K∈R^(N×D)。论文写：

$$
C=K^\top K=U\Lambda U^\top,\qquad U_r=U[:,\text{最大的 }r\text{ 个特征值对应列}].
$$

固定总体秩预算 r，所有 head 联合投影的可行集合包含逐头块对角投影。因此，最大化捕获方差时，最优联合投影不劣于逐头投影。这是一个清晰的集合包含论证，但仅保证**校准矩阵的方差目标**，不保证未来注意力输出、检索精度或延迟更优。

### 4.1 论文公式与代码存在均值处理差异

论文用未中心化的 KᵀK 表达“协方差”；代码 `utils/layerwrapper.py` 使用 `torch.cov` 并积累均值，`compression/model_prune.py` 构造恢复偏置：

$$
b=(I-U_rU_r^\top)\mu,\qquad \widehat k=U_rU_r^\top k+b.
$$

所以按论文直接写无偏置 PCA，与官方实现并不完全相同。官方 EVD 升序返回后取尾部列，代码中的“最后 r* 个坐标”对应最大特征值方向；不能机械照抄论文“前 r* 维”而忘记排序。[投影代码](https://github.com/Draconis98/SALS/blob/6aaa698b7aff321638e3f309c4e89a5fd09cb9d5/compression/model_prune.py)、[统计代码](https://github.com/Draconis98/SALS/blob/6aaa698b7aff321638e3f309c4e89a5fd09cb9d5/utils/layerwrapper.py)

### 4.2 校准的真实资源账本

原文采用 C4 的 512 段、每段4096 tokens，总计约210万 calibration tokens；不是对每个用户上下文重新训练一个模型。没有端到端梯度训练，但仍需前向、统计量收集与各层 EVD。

分析上，统计矩阵 O(D²) 空间，稠密 EVD 约 O(D³)；基矩阵占 O(Dr)。官方 `preprocess.py` 在 CPU 分配 `(nsamples,4096,hidden_size)` 的 `inps` 和同尺寸 `outs`。以512、4096、4096、FP16计算，两者合计约32 GiB，仅是隐藏状态缓冲，尚不含模型和其他开销。低成本实验应先减小样本数，再检查校准质量是否饱和。[校准实现](https://github.com/Draconis98/SALS/blob/6aaa698b7aff321638e3f309c4e89a5fd09cb9d5/compression/preprocess.py)

## 5. 一步 decode 的完整推理链

以下用列向量统一记号，D 是联合 KV 特征维度；MHA 情形最直观。GQA 要额外处理 query-head 到 KV-head 的分组映射，不能直接把不同宽度向量相乘。

1. **新增缓存：**将当前 pre-RoPE key 投影为 z_t=U_rᵀk_t；value 进入近期高精度窗口，旧 value 分组量化。
2. **便宜粗筛：**u=U_rᵀq；只用前 r* 个主方向，计算所有历史 token 的代理分数：
   $$
   s_j=u_{1:r^*}^\top z_{j,1:r^*}.
   $$
   这不是最终 attention probability，也没有 RoPE。
3. **选择集合：**取代理分数 top-k，再结合固定 sink/recent 预算。实际集合必须去重、保留正确位置；跳过稀疏层另处理。
4. **仅重建入选 K：**$\widehat k_j=U_rz_j+b$，只针对 j∈C。
5. **施加真实位置旋转：**$q_R=R_tq$，$\widehat k_{R,j}=R_j\widehat k_j$。j 是原始 token 位置，不能改成压缩后序号。
6. **精细注意力：**按各 query head 计算
   $$
   \widehat p_j=\frac{\exp(q_R^\top\widehat k_{R,j}/\sqrt{d_h})}{\sum_{i\in C}\exp(q_R^\top\widehat k_{R,i}/\sqrt{d_h})},\qquad
   \widehat y=\sum_{j\in C}\widehat p_j\widehat v_j.
   $$

这里“exact sparse attention”仅表示：在选定、重建后的集合上计算标准注意力。它**不是相对于完整原始 KV 的精确输出**，因为存在选集误差、K重建误差与V量化误差。[论文 Algorithm 1、§4](https://arxiv.org/html/2510.24273v1#S4)

### 5.1 粗筛有效的证据是什么？

论文定义 overlap score 为选集捕获的完整注意力质量：OS=Σ_(j∈C)p_j，而不是 top-k 集合交并比。作者观察到中间层较高，首两层较差，据此设置稀疏例外层。

**本文推导：**若暂时假设 K、V完全无误，仅删除总质量 ε 的 token，且所有 $\|v_j\|\le M$，则“选集重新归一化输出”与全输出之差至多2Mε。这说明 OS 有意义，但不能据此排除重建/量化误差，也不能用全局平均 OS 掩盖某一个关键 head 的失败。

## 6. 存储、带宽、计算：不能混为一谈

### 6.1 更贴近实际算法的存储估计（本文推导）

令 K latent 每元素 b_K 字节，V每元素量化为 b_v bits，原始FP16基线每个K/V元素2字节。忽略窗口、尺度和例外层时：

$$
M_{\rm SALS}\approx S r b_K+S D\frac{b_v}{8},\qquad
M_{\rm base}\approx4SD.
$$

若 b_K=2，r/D=1/4、V为4bit，则理想比值是1/4；r/D=1/8、V为2bit，则为1/8。但近期窗口、量化scale/zero、投影权重、工作区与例外层都会增加实际占用。

故论文 Table 2 的实际归一化缓存0.28/0.15与名义25%/12.5%不是一回事。摘要“6.4倍”不能简单替换为“8倍”，更不能与1/8 token读取预算再乘出“64倍常驻缓存压缩”。

### 6.2 带宽模型的一个重要不一致

论文 §4.5 写两阶段搬运 $Sr^*+2kr$，把精细阶段近似成K、V都以r维读入。但 §5.1 明确说 V 不低秩，而是量化。因此这不是实际实现的完整字节模型。

更稳妥的估计是：

$$
B\approx Sr^*b_K+krb_K+kD\frac{b_v}{8}+B_{\rm quant\ metadata}+B_{\rm reconstruction/attention\ workspace}.
$$

此外还需读取/复用投影权重并执行约O(krD)的重建、top-k选择和kernel launch。融合kernel能减少中间张量写回，却不能消除全部算术。**预测显存、HBM流量、算子延迟、端到端吞吐必须分别测。**

## 7. 关键实验：只保留可追溯数字

### 7.1 基本配置

- LLaMA2-7B-chat（MHA）、Mistral-7B-v0.2（GQA）；另有LLaMA3.1-8B-Instruct检索实验。
- K比例25%/12.5%，r*=r/2；V对应4bit/2bit。
- LongBench的LLaMA保留16 sink +432 critical +64 recent=512 tokens；Mistral数量加倍。
- 论文跳过稀疏的层为0、1、31。GSM8K/CoQA最近窗口128，剩余上下文按1/4稀疏。
- 设备描述为Xeon Platinum8336C、单张Ampere GPU、128GB RAM，正文未给可唯一识别的GPU型号。

来源：[会议PDF §5.1–5.3](https://proceedings.neurips.cc/paper_files/paper/2025/file/00a0ebcad584c59dbc439c2af8793638-Paper-Conference.pdf)。注意“平均1/8”不表示任意实际长度输入始终保留恰好12.5%。

### 7.2 准确率与流量

| 表格/指标 | 原始基线 | SALS-25% | SALS-12.5% |
|---|---:|---:|---:|
| Table 2 GSM8K strict | 0.2335 | 0.2312 | 0.2176 |
| Table 2 CoQA | 0.5997 | 0.5975 | 0.6070 |
| Table 3 LLaMA LongBench Avg | 32.65 | 32.26 | 31.97 |
| Table 3 Mistral LongBench Avg | 43.12 | 42.79 | 40.99 |
| Table 3 归一化Memory Access | 1.00 | 0.11 | 0.06 |
| Table 5 RULER Avg，长度有歧义 | 81.60 | 80.81 | 75.86 |

不要把CoQA某点上升解释成“压缩提升能力”：无重复准确率实验/置信区间，不足以支持这种因果结论。Mistral在12.5%时比基线低2.13分，也说明“无损”并不普遍。[论文 Tables 2–5](https://arxiv.org/html/2510.24273v1#S5)

### 7.3 速度：5.7倍只对应特定算子配置

| Table 6 配置 | FlashAttention ms | SALS-25% ms | SALS-12.5% ms |
|---|---:|---:|---:|
| batch8，1k | 0.230 | 0.409±0.093 | 0.357±0.051 |
| batch8，4k | 1.630 | 0.530±0.008 | 0.439±0.016 |
| batch16，4k | 3.230 | 0.757±0.026 | 0.565±0.013 |

按表重算：batch16、4k、12.5%为3.230/0.565≈5.72倍；batch8、4k为约3.71倍；batch8、1k反而较慢。

| Table 7 配置 | GPT-fast tokens/s | SALS-25% | SALS-12.5% |
|---|---:|---:|---:|
| batch8，4k | 118 | 154.1±7.1 | 163.5±5.1 |
| batch8，32k | 19.8 | 67.97±0.8 | 89.47±1.4 |
| batch4，64k | 9.2 | 32.92±0.3 | 42.96±0.9 |

4k端到端约1.39倍，32k约4.52倍，不能拿5.72倍当全部模型吞吐。短上下文、小batch和较强GQA可能减少带宽收益，需独立测量。[论文 Tables 6–7](https://arxiv.org/html/2510.24273v1#S5)

## 8. 必须保留的论文与仓库风险

### 8.1 RULER“128k验证”尚有文本冲突

§5.4同一段既写128k上下文、选16k active tokens，又写4k sequence；Table 5标题同样标4k。不能据此断言表内81.60/80.81确实来自128k。本文将其视为**作者宣称有128k评估，但表格对应长度待日志/配置核实**。

### 8.2 速度文字与表格不完全一致

§5.5文字出现“7,46×”及32k“4.57×”，与表格比值不严格相符。正文称算子1000次、端到端50次重复，checklist又概括为10次；“±”被称mean/variance但标注语义不够清楚。不把这些±擅自改称标准差或置信区间。

### 8.3 官方代码不是拿来即跑的无瑕疵复现包

本次仅静态核查固定提交，未装环境、下载模型、运行GPU实验：

1. `eval_pipeline.py`读取未在parser声明的`args.k_bits`、`args.sparse_rank`、`args.method`、`args.withfinetune`；还把声明的`with_overlap`拼作`witho_verlap`。官方示例传入的`--sparse_rank`、`--method`也未在该parser定义。应先修复参数契约，不能把示例命令当已验证可执行。[入口](https://github.com/Draconis98/SALS/blob/6aaa698b7aff321638e3f309c4e89a5fd09cb9d5/eval_pipeline.py)
2. `compression_pipeline.py`同样读取若干未声明参数，需一起审核。
3. Mistral示例跳过`0,1,2`，与论文`0,1,31`不一致；必须明确复现的是哪种配置。[Mistral示例](https://github.com/Draconis98/SALS/blob/6aaa698b7aff321638e3f309c4e89a5fd09cb9d5/example/long_bench_mistral_sparse_run.sh)
4. **最重要：LLaMA准确率路径额外保留两份原始K。**`model_impl/llama.py`在decode中持续拼接`key_states_origin`和`key_states_origin_without_rope`，并在第379行将它们返回`past_key_value`；保留动作不受`with_overlap`开关保护。它们可能服务调试/overlap，但按该路径计算显存时会破坏“只保留压缩K”的假设。应先区分算法最小缓存、调试缓存与生产benchmark缓存，再分析`gpt_fast`路径。此观察不自动否定另一条速度实现。[固定版本实现](https://github.com/Draconis98/SALS/blob/6aaa698b7aff321638e3f309c4e89a5fd09cb9d5/model_impl/llama.py#L140-L160)
5. `requirements.txt`有精确版本约束，不能随意升级后声称忠实复现；还需审计依赖兼容性与第三方安装脚本。[依赖](https://github.com/Draconis98/SALS/blob/6aaa698b7aff321638e3f309c4e89a5fd09cb9d5/requirements.txt)

## 9. 与 Attention Matching 的关系：互补轴，不是同义方法

[Attention Matching原文](https://arxiv.org/html/2602.16284v1)定义的是把T个KV条目替换为t<T个条目，每个条目的head维度d不必改变；通过参考queries拟合attention输出和mass，并引入每个保留条目的bias β。它不是以“特征矩阵本身低秩”为必要前提。

| 维度 | SALS | Attention Matching |
|---|---|---|
| 主要压缩轴 | K的feature width、V的bitwidth | 历史token条目数 |
| 未被当前query选中者 | 仍存在压缩缓存 | 原块被更短表示替换 |
| 主要目标 | 校准K方差、便宜动态选集 | 参考queries上的输出与mass匹配 |
| RoPE | 粗筛不用，重建后按原位置使用 | 原始缓存keys已包含RoPE |
| 付费阶段 | 离线校准；每decode粗筛/恢复 | 上下文compaction；之后使用短缓存 |

**本文判断：**二者可组合，但直接相乘压缩率并不成立。AM的合成/筛选后keys未必保留SALS校准的pre-RoPE低秩结构，β也必须参与代理评分。AM compacted条目不能默认当作有一个原始位置的真实token，再随意重施RoPE。更合理的起点是先验证组合表示是否与位置约定兼容，而不是先承诺“50×乘8×”。

## 10. 低成本可复现实验路线（本文建议，尚未执行）

### 实验A：只验证“RoPE之前更可压”

固定一个可运行的小模型、同一批64段短文本，按层保存少量pre/post-RoPE K。报告r90、r95与固定r的重建误差，并区分中心化/未中心化统计。设置同一旋转施加所有token的对照，防止把正交变换本身误判为增秩。不要拿其他小模型结果直接声称复现了7B表格。

### 实验B：把代理选集与重建误差分开

固定同一批真实query和KV，比较：

- 全量原始attention；
- 原始K上的oracle top-k；
- latent粗筛但使用原始K/V计算选集attention；
- 相同选集+重建K；
- 相同选集+重建K+量化V。

分别记录捕获质量OS、输出相对L2、最差head和最差query、下一token KL。这样才能知道瓶颈是“漏选”还是“重建错”，而不是只看最终QA准确率。

### 实验C：预算小网格

先测r/D∈{1/2,1/4,1/8}，r*/r∈{1/4,1/2,1}，k/S∈{1/4,1/8}。层预算先固定，确认敏感层后才做自适应。区分训练校准集、超参数验证集、最终测试集，禁止在同一测试集上选最优配置后只报最好结果。

### 实验D：真实字节与延迟审计

逐项统计latent K、近期高秩K、量化V、scale/min、debug K、投影权重与scratch buffer；记录allocated与reserved峰值。计时拆成scan、top-k、gather、recover+RoPE、attention，并测完整decode。先batch1、1k/4k寻找盈亏点，再扩展；不要一开始就用最大batch掩盖启动开销。

### 实验E：最低风险的AM连接

先比较“同字节预算下AM”和“SALS”，统一模型、文本、后续query、prefill与校准计费口径。若再做组合，明确标为新方案，先验证β与RoPE语义、质量退化来源，再投入kernel工程。

**建议验收标准：**先要求机制性曲线可复现、无调试缓存污染、预算/长度/层例外记录完整，再讨论论文级速度与新颖性。比起立即做更复杂方法，这些检查更可能发现真正值得研究的失配。

## 11. 总结

SALS最值得借鉴的是“低维代理评分与高维精细计算分工”，而不是一个普适6.4×/5.7×保证。它用后训练校准绕过重训模型，用动态稀疏化绕过全历史重建，但效果依赖层、上下文、query分布、低秩预算以及融合实现。

对后续研究，优先问题应是：**在给定真实字节与延迟预算下，什么可观测量最能预测漏选和输出误差？**若回答清楚，再做layer/head自适应预算、attention-aware投影或与AM结合，才更可能形成可解释、可复现的贡献。
