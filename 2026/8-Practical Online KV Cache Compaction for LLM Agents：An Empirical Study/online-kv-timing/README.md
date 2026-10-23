# Agent 在线 KV Cache 自适应时机实验

统一 Token Eviction，比较即时 Boundary、延迟 1/3/5 轮、T（仅时机）、R（仅比例）、TR。研究假设尚未验证。

## 最小运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,model,plot]'
pytest -ra
online-kv --config configs/synthetic.json --output results/synthetic
```

这条命令运行真实 tensor 的机制性合成实验，不加载语言模型，也不产生任务质量分数。CPU 仅用于控制器与缓存机制测试；真实模型实验面向 CUDA。

## 上传 GPU 后

安装与你的 GPU 驱动匹配的 PyTorch CUDA wheel。使用本地、固定 commit 的标准 Llama 模型，修改 configs/llama.json 的 model/revision 和缓存预算。若需要从 Hub 获取模型，使用 `hf download MODEL_ID --revision COMMIT_SHA --local-dir MODEL_PATH`。不在程序中隐式下载权重。

```bash
pytest -ra tests/test_gpu_model.py
online-kv --config configs/llama.json --output results/validation --split validation
python -m online_kv.selection --config configs/llama.json --output results/search
online-kv --config configs/llama.json --output results/test --split test --frozen results/search/frozen.json
python scripts/plot.py results/test
```

测试失败先修复，不应绕过 CUDA 模型一致性测试。续跑重复相同命令：按任务 ID + 策略去重；不同源码、配置、数据或环境会拒绝复用同一目录。中断任务重新执行，已完成任务不会重复计入。

目前 data/tasks.json 是自建的极小离线检索诊断集，不是 BrowseComp-Plus 或 WideSearch。真实模型驱动 search/answer 动作；检索为固定语料词集合重合度排序，不调用外部 API。质量指标是规范化短答案 exact match。若替换数据集，需保持 dev/validation/test 独立 ID，并实现相应官方 grader，不能用示例结果声称复现论文。

## 文件入口

- src/online_kv/compression.py：实际逐层逐头 KV gather、保护位与预算检查。
- policies.py：统一触发控制器、固定/自适应时机、独立比例规则。
- model.py：最小 Llama Attention adapter，直接读取压缩后 KV，保留绝对 RoPE 位置。
- experiments.py：Agent 循环、计时、结构化日志、恢复运行与汇总。
- selection.py：验证集统一候选搜索、记录全部试验、冻结 T/R 规则。
- scripts/plot.py：从原始日志重建汇总、配对差异与质量—耗时图。

原理、资源口径与真实验证范围见 docs。本项目没有 AM，也没有把删 Prompt 或重新 Prefill 当作 KV 压缩。
