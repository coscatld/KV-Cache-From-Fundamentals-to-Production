# PuzzleKV CUDA 研究原型

对应论文：[PuzzleKV v1](https://arxiv.org/html/2608.23843v1)。项目面向 Linux + NVIDIA CUDA GPU；不使用 CPU 结果代替 GPU 验证。现有上级目录笔记与图片保持原样。

## 安装与运行

```bash
python -m venv .venv
source .venv/bin/activate
# 先安装与你的 CUDA 驱动匹配的 PyTorch CUDA wheel
pip install -e '.[test,model,gpu]'
python scripts/environment.py
pytest -ra --junitxml=results/pytest.xml
python benchmarks/run.py --dtype float16 --length 1024 --page-size 32 --rank 8 --hf
```

不需要下载模型：HF 测试使用随机初始化的小型 Llama，验证两层真实模型调用路径；不代表预训练模型质量。所有 GPU 测试是待执行项。未具备 CUDA 时 pytest 会显示 SKIP / NOT RUN，而非通过。

核心输入 `[Hkv,T,D]`，query `[Hq,Q,D]`；batch=1，每层一个缓存。append 接收已完成 RoPE 的 Key，保留绝对位置。GQA 要求 Hq 为 Hkv 的整数倍。

PyTorch 执行 SVD、缓存管理、稠密 attention 和全局 softmax。Triton 执行低秩 logits 与 Value 聚合的矩阵乘法，使用 FP32 累积。尚无融合混合 attention Kernel C。`attention(..., backend="triton")` 采用 Triton A/B 与 PyTorch 混合调度。

当前交付状态见 [验证报告](docs/validation.md)，数学与范围见 docs。GPU 实测前无加速比、显存节省或误差数值。benchmark 报告 tensor 实际字节数、Python 元数据浅层大小与 CUDA allocator 峰值；它们是不同口径。
