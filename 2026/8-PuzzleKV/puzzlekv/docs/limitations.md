# 限制

所有数值与性能 GPU 测试尚待目标 CUDA 机器运行。Triton JIT 编译及 HF 版本兼容也尚未验证。没有实测加速比、显存节省比例和输出误差，禁止据代码推断结果。

仅 batch=1、连续无 padding 的自回归序列。HF adapter 仅适用于 eager Llama 常规 MHA/GQA，无滑窗、beam search、speculative decoding、prefix sharing 或通用 generate 集成；复杂 attention_mask 不受支持，应通过显式位置和连续因果调用使用。随机模型结果不能证明语言质量保持。

未实现融合 Kernel C、量化、页间批量压缩调度、训练反传与生产分配器。逐 token torch.cat 和 Python 页循环可能成为瓶颈。Triton A/B 有额外 launch、转置及临时矩阵开销。SVD 会产生临时 FP32 buffers。benchmark 的批量 SVD 是独立 PyTorch batched 操作，缓存转换仍逐页分解。

tensor_bytes 是活跃 tensor 元素大小；不含 allocator reservation、Tensor 对象或持有输入数据的上游模型。metadata_shallow_bytes 是 Python 对象浅层估计，不是完整进程内存。CUDA 峰值使用 PyTorch allocator 统计，不能覆盖所有驱动或库内部内存。不同方法 benchmark 输出用后立即释放，peak_increment 是同次测量相对起点峰值。

SVD 的最小 Frobenius 重建误差不保证 attention 输出误差最小。低 rank 在随机满秩数据上可能产生很大误差；必须报告实测值，不用质量指标阈值掩盖误差。
