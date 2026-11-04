# AI6127 课程作业（Amazon 2023 商品数据集）

**NTU AI6127 | 混合检索增强生成系统（Hybrid RAG System）**

- **数据集**：McAuley-Lab Amazon Reviews 2023（`meta_Clothing_Shoes_and_Jewelry` / `meta_Home_and_Kitchen` / `meta_Toys_and_Games`）
- **核心实现**：字段级切块 + 稀疏检索（BM25）+ 稠密检索（bge-m3 / FAISS）+ 父文档合并 + 倒数秩融合（RRF）+ LLM 生成 + Gradio Web 界面

## 项目目录结构

```text
AI6127/
├── dataset.py              # Q1：全量流式读取 + 蓄水池随机采样、过滤、清洗、去重、字段级切块、语料统计
├── corpus.py               # 知识库统一读取（每个进程只加载一次）与 BM25 分词器
├── attributes.py           # 商品结构化属性（品类、人群、年龄段、价格段、材质）与约束校验
├── sparse_bm25.py          # BM25 稀疏检索（稀疏矩阵实现；chunk 合并到商品）
├── build_faiss.py          # bge-m3 稠密向量索引（IndexFlatIP）；--unit chunk|product，--prompt
├── hybrid_retriever.py     # search()：融合方式 / 元数据与时间过滤 / 重排序；父文档合并；分阶段计时
├── reranker.py             # Q3：bge-reranker-v2-m3 交叉编码器重排序
├── benchmark_queries.py    # Q2：五条查询的结果与查询速度
├── eval_retrieval.py       # Q3：检索配置评测（P@5、R@10、MRR、nDCG、时间线实验）
├── llm.py                  # DeepSeek 调用封装（JSON、重试、用量统计、思考模式开关）
├── generator.py            # 证据整理（价格/评分来自 meta）、JSON 解析与引用校验
├── agent.py                # Q4/Q5：多轮澄清式推荐 Agent（状态追踪、决策、检索、证据检查、生成、自检）
├── simulator.py            # Q4/Q5：用户模拟器（只知道隐藏需求，不知道目标商品）
├── run_dialogues.py        # Q4/Q5：多轮对话评测与消融
├── evaluate_decisions.py   # Q4：澄清决策 P/R/F1
├── rag_judge.py            # Q4：faithfulness / answer relevance / context precision
├── build_scenarios.py      # Q4：生成评测场景
├── build_annotation_set.py # Q4：标注记录与标注表（决策 + pooled 相关性）
├── compute_iaa.py          # Q4：标注一致性与金标准
├── app.py                  # Gradio 多轮对话界面
├── report.md               # 作业报告（英文）
├── eval/
│   ├── scenarios/          # 评测场景（隐藏需求、开场白、对话快照）
│   ├── annotation/         # 标注记录、标注表、标注指南、金标准、一致率报告
│   ├── q2_benchmark/       # Q2 五条查询的结果与耗时
│   ├── q3_retrieval/       # Q3 检索评测结果
│   ├── q4_decisions/       # Q4 澄清决策评测结果
│   └── q5_runs/            # Q4/Q5 多轮对话记录与消融汇总
├── data/
│   ├── kb.json             # 商品级知识库：id、text（完整正文）、meta（价格、评分、评论数、店铺、类目）
│   ├── chunks.json         # chunk 级检索单元：chunk_id、parent_id、field（overview/description/specs）、text
│   ├── kb_pyserini.jsonl   # chunk 级语料（Pyserini 格式）
│   ├── corpus_stats.json   # 语料统计（Q1 报告用）
│   ├── faiss_chunk.bin     # chunk 级向量索引（build_faiss.py 生成）
│   ├── faiss_chunk.json    # 索引说明：模型、检索单元、是否加前缀、精度、向量对应的 chunk id
│   └── faiss_product.*     # 商品级索引（--unit product，Q3 消融：不切块）
│                           # 注意：*.bin 超过 GitHub 100 MB 单文件上限，已在 .gitignore 中排除，需本地重建或从共享网盘下载
├── README.md
└── requirements.txt
```

## 环境配置与依赖安装

```bash
pip install -r requirements.txt
```

### GPU / Apple Silicon（强烈推荐）

`build_faiss.py` 自动选择 CUDA → Apple Silicon（MPS）→ CPU，GPU 上用 fp16（与 fp32 向量余弦相似度 ≥ 0.9998）。编码 48,107 个 chunk：Apple M5（MPS，fp16）实测 20.3 分钟，NVIDIA GPU 更快，CPU 需十小时以上。Apple Silicon 上自动改用 eager attention（PyTorch 的 SDPA 在 MPS 上会持续泄漏显存导致 OOM）。检查 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available())"   # 需输出 True
```

若为 `False`（装成了 CPU 版 torch），需**强制重装** CUDA 版（RTX 50 系 / Blackwell 架构需 CUDA 12.8+）：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

## 问题一（Question 1）：知识库构建

**任务目标**：基于 Amazon 2023 三个商品分类的元数据构建商品知识语料库，并切成适合检索与生成的 passage。

### 构建流程

1. **数据来源**：以下三个分类的 `.jsonl.gz`（[原始数据直链](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/)）
   - `meta_Clothing_Shoes_and_Jewelry.jsonl.gz`（7,218,481 个商品）
   - `meta_Home_and_Kitchen.jsonl.gz`（3,735,584 个商品）
   - `meta_Toys_and_Games.jsonl.gz`（890,874 个商品）
2. **流式加载**：逐行流式读取完整的压缩文件，不落盘，每个分类都扫描全部商品。
3. **过滤**：`title`、`features`、`description`、`categories`、`details`、`price`、`average_rating`、`rating_number`、`store` 全部非空才保留（清洗后再校验一次）。合格商品：服装 835,390（11.6%）、家居 749,624（20.1%）、玩具 268,771（30.2%）。
4. **随机采样**：对每个分类的**全部**合格商品做蓄水池抽样（reservoir sampling，种子 `42`），每个合格商品被抽中的概率相同。原来“读到前 1 万个合格商品就停止”只覆盖文件开头一段：服装类文件开头 3 万条的合格率是 25%，而全量只有 11.6%，说明文件开头的商品分布与整体不同。
5. **清洗**：
   - 去 HTML 标签（174 个字段）与 HTML 实体（15 个字段）、emoji 和装饰符号（7,186 个字段）；
   - 规格字段去掉对检索无意义的噪声：Best Sellers Rank（8,710 个商品）、Date First Available（8,972）、Item model number（8,451）、Is Discontinued（5,226）等，并且不再把 Python dict/list 原样写进文本；
   - 卖点、描述去重；Manufacturer 与 Brand 相同时只保留一个。
6. **去重**：`parent_asin` 去重，标准化标题去重（跨分类；去掉 10 个同名商品）。
7. **文档格式**：正文 `text` 只包含语义字段（标题、类目、卖点、描述、规格）；价格、评分、评论数、店铺、类目存入结构化 `meta`，避免稀释向量，并直接用于生成证据与网页展示。
8. **字段级切块**：每个商品切成若干 chunk，每个 chunk 都以商品标题开头，保证脱离上下文后仍可理解：
   - `overview`：标题 + 类目 + 卖点（卖点超过 200 词则分组）
   - `description`：标题 + 描述（200 词窗口、40 词重叠）
   - `specs`：标题 + 规格

   检索在 chunk 上进行，再按商品取最高分合并（父文档检索），生成时使用完整商品正文。

### 语料统计（`data/corpus_stats.json`）

| 指标 | 数值 |
|---|---|
| 文档（商品）数 | 15,000（三个分类各 5,000） |
| chunk 数 | 48,107（overview 16,483 / description 16,609 / specs 15,015） |
| 词数 | 3,589,422（按空格切分的商品正文） |
| types（不重复词） | 60,141（小写、只保留字母数字）；按空格原样切分为 212,432 |
| 每个商品词数 | 平均 239，中位数 203，p95 508，最长 2,040 |
| 每个 chunk 词数 | 平均 88，最长 421 |

### 运行 Q1

```bash
python dataset.py                     # 完整构建，约 15 分钟（下载约 7.7 GB 压缩数据）
python dataset.py --max-scan 50000    # 快速测试：每个分类只扫描前 N 条
```

**运行后预期输出**：

```text
✅ Corpus built. Documents: 15000, chunks: 48107, total words: 3589422, types: 60141
```

## 问题二（Question 2）：混合检索系统与 Web 演示界面

**检索流水线**：查询 → BM25（chunk）与 bge-m3 / FAISS（chunk）分别检索 → 各自按商品取最高分合并 → RRF 融合 → 商品级 Top-k（附命中的字段与两路名次）。

### 核心组件说明

| 模块 | 实现 | 作用 |
| :--- | :--- | :--- |
| **稀疏检索（BM25）** | 自实现 Okapi BM25（k1=1.5, b=0.75），权重预先存入稀疏矩阵，查询时只对查询词所在列求和；分词为小写、只保留字母数字、去停用词 | 精确关键词匹配。相比原来的 `rank_bm25` + `split()`：`Waterproof`/`waterproof`、`hiking.`/`Hiking` 等不再匹配失败；每次查询约 1 ms（原来约 100 ms） |
| **稠密检索（FAISS）** | `BAAI/bge-m3`（1024 维）+ `IndexFlatIP`（向量归一化，等价余弦相似度），chunk 级索引 | 语义匹配，处理同义表达 |
| **父文档合并** | 两路检索都在 chunk 上打分，按商品取最高分（MaxP） | 短 chunk 匹配更精准，返回仍以商品为单位，生成时使用完整商品信息 |
| **倒数秩融合（RRF）** | k=60，每路取前 20 个商品 | 综合词汇与语义匹配 |
| **Gradio 界面** | `app.py` | 输入查询，展示生成回答、推荐商品（标题、价格、评分）、证据、命中字段与各阶段耗时 |

### 运行 Q2

```bash
python build_faiss.py                 # 构建 chunk 级向量索引（需 GPU / Apple Silicon）
python hybrid_retriever.py            # 终端测试
python benchmark_queries.py           # 五条查询的结果与查询速度 → eval/q2_benchmark/
python app.py                         # 启动网页
```

### 五条查询的速度（`eval/q2_benchmark/`）

Apple M5（MPS，fp16），检索 48,107 个 chunk，预热后每条查询重复 10 次取中位数：

| 阶段 | 耗时 |
|---|---|
| BM25（稀疏矩阵） | 0.3–0.4 ms |
| query 编码（bge-m3） | 20.9–23.1 ms |
| FAISS 搜索（IndexFlatIP，暴力精确搜索） | 1.8–2.2 ms |
| RRF 融合 | < 0.1 ms |
| **总计** | **23–25 ms** |

商品级索引（`RETRIEVAL_UNIT=product`）查询总耗时约 53 ms（单次测量）。

五条查询的 Top-5 结果（标题、类目、价格、评分、命中字段、两路名次）见 `eval/q2_benchmark/summary.md` 与 `results.json`。首次查询还需加载模型（约 10 秒）与 GPU 预热，`app.py` 启动时已自动预热。

## 问题三（Question 3）：检索与排序创新

`hybrid_retriever.search()` 统一支持以下选项，`eval_retrieval.py` 在评测场景上逐项比较：

| 方向 | 实现 | 开关 |
|---|---|---|
| 1. 混合融合 | BM25-only / dense-only / RRF / 加权 RRF | `fusion="bm25"/"dense"/"rrf"/"weighted"`, `weights=(w_bm25, w_dense)` |
| 2. 重排序 | `BAAI/bge-reranker-v2-m3` 交叉编码器（多语言）对融合后前 30 个商品重排 | `rerank=True` |
| 3. 时间线与元数据检索 | 价格、评分、类目、上架日期、人群/年龄/材质作为结构化硬过滤；过滤后结果不足时由 agent 逐级放宽 | `filters={"price_max", "rating_min", "category", "date_after", "date_before", "predicate"}` |
| 4. 多语言检索 | bge-m3 与重排序模型都支持中文查询英文商品；对比“中文直接检索”与“LLM 翻译成英文后检索” | `zh:*` / `zh→en:*` 配置 |
| 5. 切块与索引 | 字段级 chunk + 父文档合并 vs 商品级索引；BM25 分词；BGE 指令前缀 | `RETRIEVAL_UNIT=chunk/product`、`BGE_PROMPT=0/1` |

```bash
python eval_retrieval.py                                   # chunk 级索引：全部配置 + 时间线实验
RETRIEVAL_UNIT=product python eval_retrieval.py --tag product --configs bm25 dense rrf rrf+rerank --no-timeline
```

评测：查询为每个场景的完整需求（英文/中文）；指标为 P@5、R@10、MRR@10、nDCG@10 与延迟。相关性有两种：程序化（知识库中满足隐藏需求全部硬约束的商品，覆盖完整，可算召回）与人工标注的 pooled 候选（`gold_relevance.csv`）。结果在 `eval/q3_retrieval/summary.md`。

## 问题四（Question 4）：多轮澄清式推荐与评测

**任务**：用户与助手多轮对话；用户一开始的描述往往不完整（“想买双鞋”），助手判断信息是否足够，不够就反问，够了再基于检索证据推荐，并给出引用。

**Agent 每一轮**（`agent.py`）：状态追踪 + 查询改写 → 混合检索（过滤 + 重排）→ 决策（反问 / 推荐 / 无匹配）→ 证据充分性检查 → 有据生成 → 论断级自检。

### 配置 API

```bash
export DEEPSEEK_API_KEY=sk-...
export DEEPSEEK_MODEL=deepseek-flash      # 以 https://api-docs.deepseek.com 上的模型名为准
export JUDGE_MODEL=deepseek-v4-pro        # 可选：评判模型，建议与生成模型不同
```

deepseek-flash 默认开启思考模式（`temperature` 失效、输出不可复现、每次多出数百推理 token），本项目默认关闭（`llm.py`）；设置 `DEEPSEEK_THINKING=enabled` 或使用 `*+thinking` 配置可开启。

### 评测集（按用户场景）

```bash
python build_scenarios.py --per-domain 60     # 180 个场景（三个领域各 60）→ eval/scenarios/scenarios.jsonl
python build_annotation_set.py --annotators A B C   # 标注记录与标注表 → eval/annotation/（需 GPU 检索）
python compute_iaa.py                         # 一致率、Cohen's κ、Fleiss' κ，金标准
```

- **场景**：隐藏完整需求（品类、人群/孩子年龄/材质、预算）+ 开场白（模糊 / 部分 / 完整三档，英文或中文），以知识库中真实商品为锚点生成，保证至少 3 个商品满足需求。
- **标注记录**：360 条决策记录（每个场景 2 个对话快照：该问还是该推荐、该问什么）+ 约 1,080 条相关性记录（多种检索配置的结果 pooling 后，判断是否满足需求），共约 1,440 条。标注指南见 `eval/annotation/GUIDELINES.md`。

### 运行评测

```bash
python evaluate_decisions.py                  # 澄清决策 P/R/F1、追问属性准确率 → eval/q4_decisions/
python run_dialogues.py --configs full B0 --workers 6   # 用户模拟器多轮对话 → eval/q5_runs/
python rag_judge.py --configs full B0         # faithfulness / answer relevance / context precision
python app.py                                 # 多轮对话网页
```

- **用户模拟器**（`simulator.py`）：只知道隐藏需求，不知道目标商品（避免泄题）；只回答被问到的属性；推荐失败时指出一个被违反的约束；最多 5 轮。成功 = 前 3 个推荐中至少一个满足全部硬约束（程序化判断）。
- **指标**：成功率、首次推荐成功率、平均轮数、提问数、有效提问率、完整需求下的过度追问率、约束违反率、自检支持率、每轮延迟、每段对话的 LLM 调用数与成本；RAGAS 式 faithfulness、answer relevance、context precision。

## 问题五（Question 5）：下游任务创新与消融

| 创新点 | 内容 | 对应开关（`agent.AgentConfig`） |
|---|---|---|
| I1 澄清决策 + 信息增益选问 | 何时问：规则 / LLM 判断 / 候选分布熵；问什么：熵最大的属性（选项来自候选中最常见的取值） | `clarify`, `select` |
| I2 查询改写 + 约束→元数据过滤 | 对话改写成独立英文查询（同时处理中文）；预算、人群、年龄、材质转成硬过滤，结果不足时逐级放宽 | `rewrite`, `filters` |
| I3 证据充分性检查 + 论断级自检 | CRAG 式：程序化校验硬约束 + LLM 判断候选是否匹配，不足则补检索或如实告知；生成后逐条核对论断，价格用 meta 核对 | `check` |
| I4（可选）方面级推荐理由 | 每个推荐按用户关心的方面给出证据或“未说明” | `aspects` |

```bash
python run_dialogues.py --configs B0 B1 B2 B3 B4 full full-I1 full-I2 full-I3 --workers 6
python run_dialogues.py --configs clarify-rule clarify-llm select-llm select-random full+thinking
```

配置定义见 `agent.CONFIGS`：B0（单轮 RAG，拼接历史）→ B1（+改写）→ B2（+LLM 澄清）→ B3（+熵选问）→ B4（+过滤）→ full（+证据检查与自检）；`full-I*` 为逐项去掉。所有配置共用同一检索（RRF + 重排序），差异只来自 Q5 创新点。结果在 `eval/q5_runs/summary.md`。

## 团队协作说明（Notes for Teammates）

- **Windows 兼容性**：由于 Pyserini 依赖 Java 环境且在 Windows 下容易引发编译报错，本项目跳过 Pyserini，用 scikit-learn + SciPy 稀疏矩阵自实现 BM25（`kb_pyserini.jsonl` 仍按 Pyserini 格式输出，便于对照）。
- **直接流式读取**：不依赖 Hugging Face `datasets`，直接通过直链流式读取原始 JSONL 文件。
- **数据集中管理**：所有生成的语料与向量索引统一存放在 `data/` 目录下。
- **文档唯一标识**：使用 Amazon 元数据中的 `parent_asin` 作为每件商品的唯一检索文档 ID（`docid`）。
- **索引一致性**：`hybrid_retriever.py` 会检查索引说明文件（`data/faiss_<unit>[_prompt].json`）中的模型、检索单元、是否加前缀与当前配置一致，不一致直接报错，避免建索引与查询编码方式不同。
- **断点续编**：`build_faiss.py` 按片编码，每 4 片保存一次断点（`*.partial.npz`），中断后重跑会接着编码；Apple Silicon 上每片后释放 MPS 缓存，避免显存持续增长导致 OOM。
- **评测集**：Q4 的 210 条查询 × Top-5 = 1,050 条人工标注记录，同时可用于 Q3 的 P@5、nDCG@5 等检索指标。
- **HF Hub 警告说明**：终端若出现关于未认证 Hugging Face Hub Token 的警告，仅为请求速率限制提醒，不影响演示系统的正常运行。
