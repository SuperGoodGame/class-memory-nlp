# 技术报告 Design: Structured Summary Memory for Long-Context QA

## 1. 报告定位

这份技术报告不要把当前实验包装成大规模 benchmark。更稳妥的定位是：

> We study whether structured memory can improve the quality-cost tradeoff of long-context question answering. The current AliceQA experiment is a controlled pilot case study, and we extend it with sampled standard long-context QA datasets using a local Gemma 4 model to reduce API cost.

中文表达：

> 我们研究 structured memory 是否能改善长文本问答中的质量-成本折中。AliceQA 是一个小规模 controlled pilot case study；为了提高实验可信度，我们再用标准 long-context QA 数据集的 sampled subset，并使用本地 Gemma 4 模型进行低成本实验。

核心态度：

- 不声称方法已经在所有场景下泛化。
- 不把 61 个手工问题说成大规模 benchmark。
- 强调我们实现了可复现 pipeline，并在标准数据集 sample 上做本地验证。

## 2. Research Questions

建议写 2-3 个清晰问题：

1. Compared with Raw RAG, can Structured Summary Memory retrieve more answer-relevant evidence?
2. Can Structured Summary Memory reduce active context size compared with Full Context or raw chunk retrieval?
3. Under the same local Gemma 4 model, does Summary Memory improve answer quality or quality-cost tradeoff on sampled standard datasets?

中文理解：

1. 相比 Raw RAG，Summary Memory 是否更容易检索到答案相关证据？
2. Summary Memory 是否能减少每个问题需要送入模型的 active context？
3. 在同一个本地 Gemma 4 模型下，Summary Memory 是否能在标准数据集 sample 上带来更好的质量-成本折中？

## 3. Methods

### 3.1 Full Context

Full Context 把整篇文档放进 prompt。它可以作为 AliceQA 里的 upper-bound reference，但不一定适合本地标准数据集实验，因为本地模型上下文窗口和硬件限制更明显。

报告措辞：

> Full Context is used as a reference setting in the controlled AliceQA experiment. For broader local experiments, we focus on retrieval-based methods because local models have limited context windows and hardware resources.

### 3.2 Raw RAG

Raw RAG 是 baseline：

- 文档切成 fixed-size chunks。
- 使用本地 `all-MiniLM-L6-v2` 生成 embedding。
- 使用自定义 local vector store 保存 `records.json` 和 `embeddings.npy`。
- 查询时问题也转成 embedding。
- 用 cosine similarity 检索 top-k chunks。
- 把 top-k raw chunks 放进本地 Gemma 4 prompt。

报告里要强调：

> We do not rely on a heavy RAG framework such as LangChain or ChromaDB. This keeps the retrieval layer minimal and avoids introducing extra framework-level variables.

### 3.3 Structured Summary Memory

Summary Memory 是一条 memory construction and retrieval pipeline：

1. 把长文档切成较大的 section。
2. 对每个 section 生成结构化 JSON memory record。
3. JSON schema 包括：
   - `section_summary`
   - `key_entities`
   - `key_events`
   - `exact_facts`
   - `supporting_quotes`
4. 把 memory records 格式化成可检索文本。
5. 用本地 embedding 建 summary vector index。
6. 查询时先检索 summary memory。
7. 再补回对应 raw chunks 作为 supporting evidence。
8. 最后用本地 Gemma 4 回答。

要解释 preprocessing cost：

> Summary Memory has a one-time preprocessing cost because each section needs to be summarized once. After the memory is built and cached, all questions reuse the same memory index. Therefore, the method reduces repeated active context usage during question answering.

同时要承认 Full Context 也能 cache：

> Full-context prompting can also benefit from prompt caching or KV cache. However, it still keeps a long active context during inference, which can increase memory usage, attention computation, and latency.

## 4. Dataset Design

### 4.1 AliceQA-61

用途：

- Controlled pilot case study。
- 61 个问题来自 `Alice in Wonderland`。
- 问题和 expected keywords 是手工构造的。
- 用来验证 pipeline、debug retrieval、展示方法直观效果。

报告必须承认：

> AliceQA-61 is manually curated and small, so it should not be interpreted as a broad benchmark.

### 4.2 Standard Dataset: LongBench

建议使用 LongBench sampled subsets，不要一开始全量跑。

优先级：

1. `narrativeqa`
   - 和 AliceQA 最接近。
   - 测长故事、人物、事件、剧情信息。
2. `qasper`
   - 科研论文问答。
   - 文本风格和小说不同，可以测试方法是否只适合 narrative text。
3. `multifieldqa_en` optional
   - 多领域长文档问答。
   - 如果时间足够再加。

建议 sample size：

- `narrativeqa`: 50 examples
- `qasper`: 50 examples
- `multifieldqa_en`: 50 examples optional

报告措辞：

> To avoid overfitting the evaluation to our manually curated AliceQA set, we add sampled subsets from LongBench. NarrativeQA tests long narrative understanding, while Qasper tests information-seeking QA over scientific papers.

## 5. Local Model Setup

使用本地 Gemma 4，不使用 GPT-4o/GPT-5.4 跑标准数据集。

报告写法：

> We use a local Gemma 4 instruction model for standard benchmark experiments to avoid expensive remote API calls. Since the model is smaller than commercial frontier models, absolute accuracy is not the only focus. The main comparison is between Raw RAG and Summary Memory under the same local model.

必须记录：

- exact model checkpoint
- model precision
- device
- context limit
- decoding parameters
- sample size

建议 decoding：

- temperature: `0.0` or very low
- max_new_tokens: `128` or `256`
- same prompt template across methods

## 6. Metrics

### 6.1 Retrieval Metrics

Retrieval 是主指标，因为方法核心是 memory representation。

建议指标：

- `Retrieval Hit Rate`: retrieved context 是否包含 expected answer / evidence keyword。
- `Recall@k`: 如果实现方便，报告 `Recall@1`, `Recall@3`, `Recall@5`。
- `Avg Retrieved Context Chars`: 平均检索上下文长度。
- `Retrieval Latency`: 本地检索耗时。

### 6.2 Generation Metrics

用本地 Gemma 4 做 answer generation：

- `Keyword Accuracy`: answer 是否包含 expected answer keywords。
- `Exact Match / F1`: 如果标准数据集已有评测脚本，可以补充。
- `Avg Output Tokens`
- `Avg Total Latency`

如果本地模型表现一般，不要慌，报告可以强调：

> The local model is used to compare strategies under a fixed low-cost setting, not to maximize absolute benchmark performance.

### 6.3 Cost Metrics

报告 active context cost：

- Full Context prompt size。
- Raw RAG retrieved context size。
- Summary Memory retrieved summary + evidence context size。
- Summary preprocessing cost 单独说明，不混进 per-query online cost。

## 7. Experiments

### Experiment 1: AliceQA Controlled Case Study

Methods:

- Full Context
- Raw RAG
- Summary Memory

Purpose:

- Show pipeline works end-to-end.
- Show Summary Memory can match or approach Full Context with smaller active context.
- Show Raw RAG is cheaper but may miss details.

结果表建议：

| Dataset | Model | Method | Accuracy | Retrieval Hit | Avg Context Tokens/Chars | Latency |
|---|---|---|---:|---:|---:|---:|
| AliceQA-61 | GPT-5.4 or Gemma 4 | Full Context |  | N/A |  |  |
| AliceQA-61 | GPT-5.4 or Gemma 4 | Raw RAG |  |  |  |  |
| AliceQA-61 | GPT-5.4 or Gemma 4 | Summary Memory |  |  |  |  |

如果报告想统一低成本实验，可以把 GPT-5.4 放到 previous pilot result，把 Gemma 4 放到 local standard evaluation。

### Experiment 2: Retrieval-Only Local Evaluation

Methods:

- Raw RAG
- Summary Memory

Datasets:

- AliceQA-61
- NarrativeQA sample
- Qasper sample

Purpose:

- 不调用远程 LLM。
- 只看 retrieved context 是否包含答案证据。
- 证明 memory representation 对 retrieval 有帮助。

结果表建议：

| Dataset | Method | Hit Rate | Recall@3 | Avg Context Chars | Retrieval Latency |
|---|---|---:|---:|---:|---:|
| AliceQA-61 | Raw RAG |  |  |  |  |
| AliceQA-61 | Summary Memory |  |  |  |  |
| NarrativeQA-50 | Raw RAG |  |  |  |  |
| NarrativeQA-50 | Summary Memory |  |  |  |  |
| Qasper-50 | Raw RAG |  |  |  |  |
| Qasper-50 | Summary Memory |  |  |  |  |

### Experiment 3: Local Gemma 4 Generation

Methods:

- Raw RAG + Gemma 4
- Summary Memory + Gemma 4

Datasets:

- NarrativeQA sample 50
- Qasper sample 50
- AliceQA-61 if time allows

Purpose:

- 给技术报告提供真实标准数据集 + 本地模型结果。
- 在同一个本地模型下比较 memory strategies。

结果表建议：

| Dataset | Local Model | Method | Answer Accuracy | Retrieval Hit | Avg Context Chars | Avg Latency |
|---|---|---|---:|---:|---:|---:|
| NarrativeQA-50 | Gemma 4 | Raw RAG |  |  |  |  |
| NarrativeQA-50 | Gemma 4 | Summary Memory |  |  |  |  |
| Qasper-50 | Gemma 4 | Raw RAG |  |  |  |  |
| Qasper-50 | Gemma 4 | Summary Memory |  |  |  |  |

## 8. Expected Discussion

应该主动讨论这些限制：

- AliceQA-61 是小规模手工数据集。
- LongBench 只跑 sampled subset，规模小于 full benchmark。
- 本地 Gemma 4 与 frontier model 能力不同，所以绝对准确率可能不高。
- Summary Memory 有一次性 preprocessing cost。
- Full Context 可以用 prompt cache / KV cache，但 active context 仍然更长。
- Summary quality 会影响 downstream retrieval 和 generation。
- Keyword matching 评估可能低估或高估真实语义正确性。

同时强调贡献：

- 实现了 Full Context、Raw RAG、Summary Memory 三种策略。
- 使用本地 embedding 和自定义 local vector store，避免框架变量。
- 设计了 structured JSON memory schema。
- 加入 raw-evidence fallback，避免 summary-only hallucination。
- 从 controlled case study 扩展到标准数据集 sample。
- 使用本地 Gemma 4 降低实验成本。

## 9. Conclusion Template

英文结论可以这样写：

> Our experiments suggest that Structured Summary Memory is a promising low-cost memory strategy for long-context question answering. In the controlled AliceQA case study, it improves the quality-cost tradeoff compared with Raw RAG and approaches Full Context performance with much smaller active context. On sampled LongBench subsets with a local Gemma 4 model, the evaluation further tests whether the retrieval advantage transfers to standard long-context QA settings. However, because the standard experiments are sample-limited and use a local small model, broader evaluation with stronger models is still needed.

中文理解：

> 实验说明 Structured Summary Memory 是一个有潜力的低成本长文本 QA memory 方法。在 AliceQA controlled case study 中，它相比 Raw RAG 提供了更好的质量-成本折中，并且用更小的 active context 接近 Full Context。通过 LongBench 子集和本地 Gemma 4，我们进一步测试这个方法是否能迁移到标准 long-context QA 场景。但由于标准实验是 sample-limited，而且使用本地小模型，后续仍需要更大规模和更强模型验证。

## 10. Code Agent Checklist

让 code agent 完成后，需要能回答这些问题：

- 实际使用的 Gemma 4 checkpoint 是什么？
- LongBench 下载和缓存是否都在 repo-local `.cache/`？
- 每个 dataset sample 了多少条？
- Raw RAG 和 Summary Memory 是否使用同一个 embedding model？
- Raw RAG 和 Summary Memory 是否使用同一个 Gemma 4 prompt 和 decoding setting？
- Summary preprocessing token/context cost 是否单独记录？
- 结果是否保存到 `results/`？
- README 或 run note 是否说明了运行命令？
- 技术报告能否直接引用结果表？
