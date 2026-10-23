# 资源口径、限制与结果分析

## 缓存与内存

硬上限 peak_kv_bytes 针对 Bank 拥有的持久 K/V、绝对位置、保护位和压缩时新旧缓存共存部分。实际字节数由 numel × element_size 计算，包含 KV Token 数之外的 dtype 差异。追加时还保守预留 torch.cat 输出与原输入共存空间。没有符合资格的历史可释放时记录 kv_budget / infeasible，活动轮和系统 prefix 不会被删除。

该 cap **不是进程总显存上限**：投影产生的临时 KV、Attention 拼接/重复头工作区、Query 日志、TE FP32 logits 和模型权重不全受 Bank 上限控制。TE 估计 workspace 另记；真实模型记录 PyTorch allocator allocated/reserved 峰值。峰值 KV 是受跟踪对象口径，不能据此声称所有 GPU 上的 KV 临时张量都已纳入。若要做严格进程显存等预算结论，仍需在 GPU profiling 后给 adapter 增加统一工作区预算和峰值验证。

压缩前后字节、Token 数、触发次数、回退次数、等待期间新增量记录在 events。比例是 Token 保留比例，不是显存节省比例；protection floor 会改变实际保留率。临时 Attention 拼接真实读取 compact KV，不重建原始已淘汰条目。

## 时间与任务

wall_seconds 真实包围一个任务生命周期，CUDA 上起止同步。prefill、decode、query_collection、query_statistics、compression、retrieval、tool_total 分开记；tool_total 包含 retrieval，不能重复相加；模块时间也不替代墙钟时间。当前工具串行、无外部计费，工具次数与重复率如实记录，重复率是完全相同搜索字符串比例，非语义重复率。模型加载不含在逐任务墙钟时间，实验启动和调参成本应另计。

质量是本地小检索集的 short-answer exact match，不是官方 benchmark 准确率。synthetic 的 quality/success 为 null，禁止将缓存机制完成率当任务成功率。固定答案/无有效工具调用仍可能得到分数，因此必须查看 actions、失败类型和工具次数，而非只看准确率。

## 真实模型边界

仅标准 Llama eager attention、batch=1、常规 MHA/GQA、固定 head_dim、连续绝对位置，不支持量化/分片权重、滑窗、混合注意力、beam search、训练、padding 或任意自定义 mask。通过最小 self_attn.forward adapter 和 use_cache=False 避开 Transformers Cache 版本接口；这不关闭本项目外部缓存。新增 token 仍是增量前向，历史 KV 来自 Bank，不重新 Prefill 原始文本。默认 local_files_only=True，模型和 tokenizer 必须提前准备并固定 revision。

JSON 行动作协议是本项目最小 Agent，未完全复现原生 chat_template。模型能否遵循动作协议、版本兼容与 numerical parity 必须在 CUDA 上验证。

## 当前结论

只可验证模块测试、实际 tensor 淘汰、因果事件约束、预算拒绝和恢复去重。合成记录不支持自适应优于固定策略、端到端加速或 Agent 质量保持的结论。

真实模型测试、验证集调参、冻结测试评估、质量—耗时图和 GPU 内存峰值尚未运行。plot.py 遇到合成日志会明确拒绝画质量曲线；不会填充虚构得分。项目是可上传执行的实验框架，严格资源公平与论文级最终验收尚待 GPU 和目标 benchmark 完成。
