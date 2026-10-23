# 架构与数学

Config → KVCache（每层独立）→ Page（每 KV head 独立矩阵或 Factors）→ attention。

页 K:[P,D]、V:[P,Dv]。SVD 在 GPU FP32 中执行，A=U_r Σ_r:[P,r]，B=Vh_r:[r,D]，因子转为缓存 dtype。请求 rank 超过 min(P,D) 时截断；不基于数值阈值推断秩。rank_for_budget 按因子实际 dtype 字节计算可用 rank，预算不足返回 0，调用者需自行选择稠密或拒绝预算。

Key logits = (Q Bkᵀ) Akᵀ / √D。所有页 logits 拼接，按绝对位置 mask，再统一稳定 softmax。每页取全局权重 Wp，输出求和 Σp (Wp Av) Bv；稠密页使用 Wp Vp。实现没有重建历史 K/V，但保留 O(Hq Q T) logits/weights。全 mask 行输出 0。

后续融合可使用在线 softmax：新块局部最大值 m_b，m'=max(m,m_b)，l'=exp(m-m')l+Σ exp(s-m')，o'=exp(m-m')o+exp(s-m')V，最终 o/l。当前没有采用这一融合路径，不宣称 FlashAttention 内存复杂度。

页区间是 [start,start+length)。任何与 sink 相交的页保持稠密，近期窗口相交页亦保持稠密；完整历史页才压缩。故稠密窗口可能向页边界多保留至多 P−1 token。最后不完整页不压缩。只在两个因子组合都节省 tensor 字节时转换，原始 dense 引用被释放。输入 clone 防止切片继续持有整块输入 storage。

Triton A：两次矩阵乘法产生 [Q,r]、[Q,P]，需转置小因子；B：两次乘法产生 [Q,rv]、[Q,Dv]。tile 为 16×32×32，mask 覆盖奇数维度，FP32 IEEE dot。该配置尚未调优。
