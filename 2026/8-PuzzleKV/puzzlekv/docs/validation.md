# 实际验证记录

日期：2026-10-22。按用户要求以 GPU 为目标，不执行 CPU 数值替代验证。

| 项目 | 状态 | 证据 |
|---|---|---|
| editable 安装 | PASS | 本地 pip install -e . 完成 |
| Python 语法与导入 | PASS | compileall，puzzlekv/各模块 import 完成 |
| HF API 静态签名检查 | PASS | 本机 LlamaAttention.forward 参数符合 adapter 签名 |
| 测试收集 | PASS | 21 个 GPU 用例完成收集 |
| PyTorch GPU 正确性 | NOT RUN | 当前无 CUDA |
| Triton JIT / 数值 parity | NOT RUN | 当前无 CUDA/Triton |
| HF 模型执行 | NOT RUN | 当前无 CUDA |
| GPU benchmark | NOT RUN | 当前无 CUDA |

results/environment-local.json 记录实际环境，results/pytest-local.txt 与 XML 记录 21 SKIP，没有数值测试 PASS。

## GPU 上执行

安装 README 中的依赖后执行 `bash scripts/run_gpu.sh`。脚本先确认 CUDA 和依赖；pytest 失败立即停止，不以 benchmark 替代正确性测试。原始结果写入 results。支持 BF16 时运行对应 benchmark。机器、版本、参数、随机种子写入每份 JSON。

## 待填结果

- Triton 相比 PyTorch 加速比：NOT RUN。将从 GPU JSON 的 triton_speedup_vs_torch_mixed 获取；可能小于 1。
- 实际 KV tensor 字节节省：NOT RUN。用 storage.full_tensor_bytes − storage.tensor_bytes；比例据实测字节计算。
- 进程/显卡总显存节省：NOT RUN。当前 benchmark 报 allocator 峰值，不能替代完整模型显存实验。
- Attention 输出误差：NOT RUN。JSON errors.attention_relative。
- Key / Value / logits 误差：NOT RUN。JSON errors。
- 最终模型输出误差：NOT RUN。JSON hf.logits_relative_error 与 greedy_agreement，仅随机小模型。

当前没有可供归因的性能数据，故未实施基于瓶颈的调优。下一步先在目标 GPU 运行测试，修复 JIT/版本兼容，再对页循环、launch 开销和 SVD 分别 profiling，决定是否实现 Kernel C。
