# 阶段 1：代码与环境审计

日期：2026-10-23。

目标目录只有论文笔记与图片，没有现成 Agent、检索服务、TE 或数据评估代码。工作区不是 Git checkout。相关可复用项目是相邻的 PuzzleKV 原型，但其页低秩算法与本任务的 TE、轮次冻结语义不同，因此没有修改或迁移它。

本机 macOS ARM64，Python 3.10、PyTorch 2.13.0，无 CUDA。使用已安装 pytest 的本地解释器运行机制测试；真实模型测试不执行 CPU 替代验证。版本证据见 results 的 manifest。

新增独立项目 online-kv-timing，保留原始笔记。关键数据流：任务 → 固定系统提示 → 模型生成 JSON 动作 → 本地 search → 工具响应 → 闭合轮次 → 记录 post-RoPE Query → Controller → 统一 TE → 后续模型直接读取新 KV tensor。

缓存创建/追加/释放入口在 Bank.append/compact；模型每个 feed 调用追加逐层 KV，原始 segment 被独立 gather 的新 storage 替换后释放。每层每 KV head 可访问，但仅支持标准同维度 Llama。Controller 只能看到已完成轮的观测值。强制回退发生在模型增加 KV 之前；无可压缩历史且活动轮过大则显式记录 infeasible，不删除活动轮。

没有官方检索基准、预训练权重或 GPU，不能本地完成论文级实验。替代方案为合成 tensor 机制检查 + 自建小检索集的 CUDA 模型路径。真实模型版本与官方基准接入由明确配置和 adapter 边界隔离。
