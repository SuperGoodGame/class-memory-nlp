# Summary Memory 改进汇报说明

## 一句话结论

Summary Memory 的核心改进不是“用摘要替代原文”，而是把长文本转成可检索的结构化长期记忆，再用原文片段补充证据。它相比 Raw RAG 更容易命中关键信息，相比 Full Context 大幅降低 prompt 成本。

在 GPT-5.4 实验中，Summary Memory 达到与 Full Context 相同的 `95.1%` 准确率，但平均 prompt tokens 从 `36708.6` 降到 `6903.6`，只用了约 `18.8%` 的上下文成本，约等于节省 `81.2%` prompt tokens。

## 当前方法做了什么

项目在 `alice_in_wonderland.md` 上比较三种长文本问答策略：

| 方法 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Full Context / No-Memory | 把全文直接放进 prompt | 信息最完整，是效果上限 | token 成本最高，扩展性差 |
| Raw RAG | 检索原始 chunk，再让模型回答 | 成本最低，流程简单 | 容易漏掉跨段信息或细粒度事实 |
| Summary Memory | 检索结构化摘要记忆，并补充原文 chunk | 兼顾全局语义和局部证据 | 需要预先生成摘要，摘要质量会影响效果 |

Summary Memory 的流程：

1. 将原文切成较大的 section。
2. 调用模型生成结构化长期记忆，包含人物、事件、精确事实和 supporting quotes。
3. 将 summary memory 建成本地向量索引。
4. 回答问题时先检索 summary memory，定位相关场景。
5. 再合并 raw chunk 作为 supporting excerpts，保证答案有原文证据。

这个设计的重点是分工：

- Summary memory 负责定位和组织信息。
- Raw chunk 负责提供精确事实和原文证据。
- LLM 负责基于两类上下文生成简短答案。

## 当前评测结果

评测集包含 `61` 个关于《Alice in Wonderland》的细节问题。主要指标包括：

- Accuracy：答案是否包含预期关键词。
- Hit Rate：检索上下文是否命中答案相关关键词。
- Avg Prompt Tokens：平均 prompt 成本。
- Avg Latency：平均请求耗时。

| Model | Method | Accuracy | Hit Rate | Avg Prompt Tokens | Avg Latency |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-4o | Full Context / No-Memory | 91.8% (56/61) | - | 36709.6 | 3.10s |
| GPT-4o | Raw RAG | 82.0% (50/61) | 85.2% (52/61) | 1228.1 | 1.97s |
| GPT-4o | Summary Memory | 85.2% (52/61) | 91.8% (56/61) | 4587.2 | 2.27s |
| GPT-5.4 | Full Context / No-Memory | 95.1% (58/61) | - | 36708.6 | 15.89s |
| GPT-5.4 | Raw RAG | 91.8% (56/61) | 86.9% (53/61) | 1258.9 | 14.05s |
| GPT-5.4 | Summary Memory | 95.1% (58/61) | 98.4% (60/61) | 6903.6 | 13.12s |

## 主要发现

### 1. Summary Memory 比 Raw RAG 更稳

在两个模型上，Summary Memory 都比 Raw RAG 更好：

- GPT-4o：Accuracy 从 `82.0%` 提升到 `85.2%`，Hit Rate 从 `85.2%` 提升到 `91.8%`。
- GPT-5.4：Accuracy 从 `91.8%` 提升到 `95.1%`，Hit Rate 从 `86.9%` 提升到 `98.4%`。

这说明结构化记忆比直接检索原始 chunk 更容易把相关信息找出来。

### 2. 强模型更能利用结构化记忆

在 GPT-5.4 上，Summary Memory 与 Full Context 的准确率相同，都是 `95.1%`。这说明当 backbone model 足够强时，结构化 memory 可以接近全文输入的效果。

### 3. 成本明显低于 Full Context

GPT-5.4 下：

- Full Context 平均 prompt tokens：`36708.6`
- Summary Memory 平均 prompt tokens：`6903.6`

Summary Memory 只用了 Full Context 约 `18.8%` 的 prompt 成本，节省约 `81.2%`。

## 还可以怎么优化

### 优化 1：压缩 summary schema

当前 GPT-5.4 版本的 summary compression ratio 是 `1.118`，说明摘要记忆在这次运行里比原文 section 还长。这个结果对“压缩记忆”的叙事不够理想。

建议把字段压缩成更紧凑的形式：

| 当前字段 | 优化后字段 |
| --- | --- |
| section_summary | summary |
| key_entities | entities |
| key_events | events |
| exact_facts | facts |
| supporting_quotes | quotes |

同时限制每个 section 的事实数量和 quote 长度，例如：

- `summary` 不超过 2 句。
- `facts` 最多 8 条。
- `quotes` 最多 3 条，每条不超过 20 个词。

目标是让 compression ratio 明确低于 `1.0`，最好控制在 `0.3` 到 `0.6`。

### 优化 2：分离检索文本和回答文本

现在检索和回答使用同一份 formatted summary。可以拆成两层：

- Retrieval view：只保留短关键词、实体、事件、事实 ID，用于向量检索。
- Answer view：命中后再加载更完整的 facts 和 quotes，用于回答。

这样可以降低检索噪声，也能减少最终 prompt 中的冗余内容。

### 优化 3：自适应 top-k

当前参数固定为 `summary_k=3`、`raw_k=4`。可以根据问题类型动态调整：

| 问题类型 | 策略 |
| --- | --- |
| 人物、地点、物品 | 少取 raw chunks，优先 summary memory |
| 数字、标签、原文措辞 | 多取 raw chunks，保留精确证据 |
| 检索分数很高 | 降低 top-k，节省 token |
| 检索分数较低 | 增加候选，再 rerank |

目标是在不降低 Accuracy 的情况下继续减少 prompt tokens。

### 优化 4：加入 reranker

当前 summary 检索主要依赖 embedding similarity 和 keyword overlap。可以增加轻量 rerank：

1. 先用向量检索取更多候选，例如 top-9。
2. 用关键词、实体、数字、短语重合度重新排序。
3. 只把排名最高的 top-3 放进最终 prompt。

对细节问答来说，数字、专名、短语匹配通常比纯语义相似更可靠。

### 优化 5：重跑 timeout 样本

GPT-5.4 Summary Memory 日志里有 1 个 timeout，被计为失败。如果只重跑失败样本，Summary Memory 可能从 `58/61` 提升到 `59/61`。这不会改变方法，但能让展示结果更完整。

## PPT 推荐讲法

### Slide 标题

Structured Summary Memory: Better Retrieval with Lower Context Cost

### 讲稿版本

Raw RAG 直接检索原文 chunk，优点是便宜，但问题是 chunk 粒度固定，容易漏掉跨段信息。我们的 Summary Memory 先把长文本整理成结构化长期记忆，保留人物、事件、数字、标签和精确事实。回答时先检索 summary memory 定位相关场景，再加入原文片段作为证据。

实验结果显示，Summary Memory 在 GPT-4o 和 GPT-5.4 上都优于 Raw RAG。特别是在 GPT-5.4 上，它达到和 Full Context 相同的 `95.1%` 准确率，但只使用约 `18.8%` 的 prompt tokens。这说明结构化记忆可以在保持效果的同时显著降低长文本问答的上下文成本。

### 可放在结论页的一句话

Summary Memory turns long context into structured, retrievable memory: it improves retrieval over Raw RAG and approaches Full Context quality at a fraction of the prompt cost.

## 后续工作优先级

1. 先压缩 summary schema，让 compression ratio 明确低于 `1.0`。
2. 重跑 timeout 样本，确认 GPT-5.4 Summary Memory 的上限。
3. 加入 retrieval view / answer view 分离，减少 prompt 冗余。
4. 做 adaptive top-k 和 reranking，进一步优化 token 成本与命中率。
5. 将同样方法迁移到 LoCoMo 或 LongMemEval，验证是否能泛化到真实长对话记忆任务。
