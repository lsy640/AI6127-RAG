# AI6127 课程作业（Amazon 2023 商品数据集）

**NTU AI6127 | 混合检索增强生成系统（Hybrid RAG System）**

- **数据集**：McAuley-Lab Amazon Reviews 2023（`meta_Clothing_Shoes_and_Jewelry` / `meta_Home_and_Kitchen` / `meta_Toys_and_Games`）
- **核心实现**：字段级切块 + 稀疏检索（BM25）+ 稠密检索（bge-m3 / FAISS）+ 父文档合并 + 倒数秩融合（RRF）+ LLM 生成 + Gradio Web 界面

## 项目目录结构

```text
AI6127/
├── dataset.py              # Q1：全量流式读取 + 蓄水池随机采样、过滤、清洗、去重、字段级切块、语料统计
├── corpus.py               # 知识库统一读取（每个进程只加载一次）与 BM25 分词器
├── sparse_bm25.py          # BM25 稀疏检索（稀疏矩阵实现；小写/去标点/去停用词；chunk 合并到商品）
├── build_faiss.py          # bge-m3 稠密向量索引（IndexFlatIP）；--unit chunk|product，--prompt
├── hybrid_retriever.py     # 混合检索：BM25 + FAISS → 父文档合并 → RRF → 商品级 Top-k（含命中字段、分阶段计时）
├── benchmark_queries.py    # Q2：五条查询的结果与查询速度（预热、重复测量、分阶段耗时）
├── generator.py            # Q4：证据整理 + DeepSeek 生成（JSON 输出、引用校验）
├── app.py                  # Q2/Q4：Gradio 网页界面（检索 + 生成 + 价格/评分 + 命中字段 + 性能指标）
├── build_annotation_set.py # Q4：冻结 Top-5 检索结果，生成待标注记录与标注表
├── compute_iaa.py          # Q4：标注一致性（一致率、Cohen's κ、Fleiss' κ）与金标准
├── run_q4_evaluation.py    # Q4：批量生成（支持并发、消融开关）
├── rag_judge.py            # Q4：RAGAS 式指标（faithfulness / answer relevance / context relevance）
├── evaluate_q4.py          # Q4：P/R/F1、RAG 指标、幻觉率、延迟/吞吐/成本汇总
├── eval/
│   ├── q4_queries.jsonl    # Q4 评测查询（210 条）
│   ├── annotation/         # 标注池、标注表、标注指南、金标准
│   └── q2_benchmark/       # Q2 五条查询的结果与耗时（benchmark_queries.py 生成）
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

### 检索配置与消融（Q3 可用）

`build_faiss.py` 与 `hybrid_retriever.py` 用同一组配置，保证索引与查询编码一致：

| 配置 | 建索引 | 检索 |
|---|---|---|
| chunk 级、不加前缀（默认） | `python build_faiss.py` | `python hybrid_retriever.py` |
| 商品级（不切块） | `python build_faiss.py --unit product` | `RETRIEVAL_UNIT=product python hybrid_retriever.py` |
| 加 BGE 指令前缀 | `python build_faiss.py --prompt` | `BGE_PROMPT=1 python hybrid_retriever.py` |

指令前缀 `"Represent this sentence for searching relevant passages: "` 是 BGE v1.5 系列给 query 用的指令，bge-m3 官方说明 query 和文档都不需要指令，因此默认不加。

**注意**：改变检索配置后 Q4 的检索结果会变，必须在开始标注前运行 `python build_annotation_set.py` 重新生成标注池（当前配置记录在 `eval/annotation/pool_meta.json`）。

## 问题四（Question 4）：生成与评测

**流水线**：混合检索 Top-5 → 证据整理（价格、评分、评论数从 `meta` 读取并放在标题后；去 HTML、去重复卖点、正文截断至 220 词）→ DeepSeek 生成 JSON（回答 + 推荐 ASIN + 证据编号）→ 校验 ASIN 与证据编号，记录被过滤的推荐（用于统计幻觉率）。

### 配置 API

```bash
export DEEPSEEK_API_KEY=sk-...
export DEEPSEEK_MODEL=deepseek-flash      # 以 https://api-docs.deepseek.com 上的模型名为准
# 可选：评判模型（建议与生成模型不同，减少自评偏差）
export JUDGE_MODEL=deepseek-v4-pro
```

### 运行步骤

```bash
# 0. 网页演示（检索 + 生成）
python app.py

# 1. 生成标注集（已生成好，仅在修改查询或检索器后重跑）
python build_annotation_set.py --annotators A B C

# 2. 每位标注者独立填写 eval/annotation/sheet_<名字>.xlsx（见 GUIDELINES.md），然后：
python compute_iaa.py                 # 一致率 / κ / 金标准 gold_labels.csv

# 3. 批量生成（默认使用冻结的标注检索结果）
python run_q4_evaluation.py --workers 8 --tag main

# 4. RAGAS 式指标
python rag_judge.py --tag main --workers 8

# 5. 汇总全部 Q4 指标（价格单位：美元 / 百万 token）
python evaluate_q4.py --tag main --price-in 0.30 --price-out 1.20
```

消融示例（Q5 可用）：`python run_q4_evaluation.py --no-clean --tag noclean`，再对 `noclean` 运行第 4、5 步并与 `main` 对比。

### 评测设计

- **标注单位**：(查询, 检索到的商品) 对，210 条查询 × Top-5 = **1,050 条**，二分类（相关 / 不相关），3 人独立标注。
- **任务指标（P / R / F1）**：生成器要从 5 个候选中选出推荐商品，把“是否推荐”与人工“是否相关”比较，计算 micro / macro P、R、F1；另统计 **abstention accuracy**（没有相关商品时是否正确地不推荐）。
- **RAG 指标**：faithfulness（回答中的论断是否都有证据支撑）、answer relevance（RAGAS 反向生成问题再算相似度）、context precision（人工标签版与 LLM 评判版），以及评判模型与人工标签的一致性。
- **可靠性**：基于模型原始输出统计幻觉 ASIN 比例、无效引用比例。
- **性能**：检索 / 生成延迟（均值、p50、p95）、吞吐（条/秒，可调并发）、每次查询 token 数与成本。

## 团队协作说明（Notes for Teammates）

- **Windows 兼容性**：由于 Pyserini 依赖 Java 环境且在 Windows 下容易引发编译报错，本项目跳过 Pyserini，用 scikit-learn + SciPy 稀疏矩阵自实现 BM25（`kb_pyserini.jsonl` 仍按 Pyserini 格式输出，便于对照）。
- **直接流式读取**：不依赖 Hugging Face `datasets`，直接通过直链流式读取原始 JSONL 文件。
- **数据集中管理**：所有生成的语料与向量索引统一存放在 `data/` 目录下。
- **文档唯一标识**：使用 Amazon 元数据中的 `parent_asin` 作为每件商品的唯一检索文档 ID（`docid`）。
- **索引一致性**：`hybrid_retriever.py` 会检查索引说明文件（`data/faiss_<unit>[_prompt].json`）中的模型、检索单元、是否加前缀与当前配置一致，不一致直接报错，避免建索引与查询编码方式不同。
- **断点续编**：`build_faiss.py` 按片编码，每 4 片保存一次断点（`*.partial.npz`），中断后重跑会接着编码；Apple Silicon 上每片后释放 MPS 缓存，避免显存持续增长导致 OOM。
- **评测集**：Q4 的 210 条查询 × Top-5 = 1,050 条人工标注记录，同时可用于 Q3 的 P@5、nDCG@5 等检索指标。
- **HF Hub 警告说明**：终端若出现关于未认证 Hugging Face Hub Token 的警告，仅为请求速率限制提醒，不影响演示系统的正常运行。

## 报告可扩展方向（Possible Extensions）

- **引入 LLM 生成阶段**：在检索结果基础上接入大语言模型（LLM），根据检索到的商品段落自动生成自然语言回答（完整 RAG 闭环）。
- **批量定量评测**：构建测试查询集，计算 `Recall@k`、`MRR` 等标准信息检索评价指标。
- **消融实验（Ablation Study）**：对比分析仅使用 BM25、仅使用 FAISS 与使用混合 RRF 融合检索在准确率与响应耗时上的表现差异。
