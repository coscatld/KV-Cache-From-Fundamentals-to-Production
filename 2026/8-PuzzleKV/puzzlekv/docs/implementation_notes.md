# 论文与工程实现

来源：https://arxiv.org/html/2608.23843v1 ，重点 §3.3。已核实论文采用逐层逐 KV head 页分解、sink/recent 稠密区、直接混合 attention 和增量页转换。

本项目采用相同基本分解与直接低秩公式。以下是工程选择：固定 rank、页边界保守保留窗口、两个矩阵都节省空间才压缩、FP32 SVD 后转回 dtype、全局 logits softmax、两阶段 Triton GEMM、随机小型 HF Llama adapter。论文的在线 softmax 融合实现、实验模型和量化结果未在这里复现。

没有删除 token，没有做数值量化；FP16/BF16 是存储 dtype，低秩是特征表示压缩。预算函数以字节选择 rank，不等于整缓存预算分配策略。torch.svd_lowrank 尚未评测，因此未替代确定性 SVD。

HF adapter 在 attach_llama 中替换 Llama self_attn.forward，使用外部逐层 KVCache；调用 use_cache=False 并显式传递绝对 position_ids。它绕开 Transformers Cache API，不声称实现通用 Cache 兼容。用完调用 restore。当前版本适配性必须在目标机 integration 测试中确认。
