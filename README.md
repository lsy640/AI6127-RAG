# AI6127 Group Assignment: Conversational RAG Product Recommender (Amazon Reviews 2023)

**English** | [中文](README_zh.md)

**NTU AI6127 | Hybrid Retrieval-Augmented Generation System**

- **Dataset:** McAuley Lab Amazon Reviews 2023 (`meta_Clothing_Shoes_and_Jewelry` / `meta_Home_and_Kitchen` / `meta_Toys_and_Games`)
- **What it does:** field-level chunking + sparse retrieval (BM25) + dense retrieval (bge-m3 / FAISS) + parent-document merging + reciprocal rank fusion (RRF) + cross-encoder re-ranking + metadata/date filters + a multi-turn agent that asks clarifying questions before recommending (DeepSeek) + a Gradio web UI
- **Report:** `report.md` (full write-up of Q0–Q5 with all results) and `report_latex/` (two-column paper version)

### Status

| Part | Status |
|---|---|
| Q1 knowledge base, Q2 retrieval and UI | Done |
| Q3 retrieval innovations | Done (programmatic and human relevance judgments) |
| Q4 evaluation set | Done: 169 unique scenarios, 1,313 unique records (299 decision + 1,014 relevance), labelled independently by three annotators; agreement 95.8 % (relevance 94.9 %) |
| Q4 / Q5 dialogue evaluation and ablation | Done (15 configurations × 169 dialogues); clarification-decision metrics against human gold; LLM judge vs. human agreement 92.7 % (κ 0.853) |
| Report placeholders (TBD) | Members' names and matriculation numbers, the two download links |
| Report figures | `eval/figures/`: architecture, UI screenshot, retrieval results, ablation results |

## Repository structure

```text
AI6127/
├── dataset.py              # Q1: streaming download + reservoir sampling, filtering, cleaning, de-duplication, field-level chunking, corpus statistics
├── corpus.py               # shared loading of the knowledge base (once per process) and the BM25 tokenizer
├── attributes.py           # structured product attributes (type, audience, age band, price band, material) and constraint checks
├── sparse_bm25.py          # BM25 sparse retrieval (sparse-matrix implementation; chunks merged to products)
├── build_faiss.py          # bge-m3 dense index (IndexFlatIP); --unit chunk|product, --prompt
├── hybrid_retriever.py     # search(): fusion mode / metadata and date filters / re-ranking; parent-document merge; per-stage timing
├── reranker.py             # Q3: bge-reranker-v2-m3 cross-encoder re-ranking
├── benchmark_queries.py    # Q2: results and speed of five queries
├── eval_retrieval.py       # Q3: retrieval evaluation (P@5, R@10, MRR, nDCG, timeline experiment)
├── llm.py                  # DeepSeek wrapper (JSON mode, retries, usage and cache-hit accounting, thinking-mode switch)
├── generator.py            # evidence formatting (price/rating from metadata), JSON parsing and citation validation
├── agent.py                # Q4/Q5: multi-turn clarifying recommender (state tracking, decision, retrieval, evidence check, generation, self-check)
├── simulator.py            # Q4/Q5: user simulator (knows the hidden need, never the target product)
├── run_dialogues.py        # Q4/Q5: multi-turn dialogue evaluation and ablations
├── evaluate_decisions.py   # Q4: clarification decision P/R/F1
├── rag_judge.py            # Q4: faithfulness / answer relevance / context precision; judge calibration
├── build_scenarios.py      # Q4: generates evaluation scenarios
├── build_annotation_set.py # Q4: annotation records and sheets (decisions + pooled relevance)
├── compute_iaa.py          # Q4: inter-annotator agreement and gold labels
├── dedup.py                # Q4: de-duplication of the evaluation set (all metrics use the unique set)
├── app.py                  # Gradio multi-turn chat interface
├── report.md               # assignment report (English)
├── report_latex/           # LaTeX report (two-column paper; main.tex + figures/; compile with XeLaTeX, ready for Overleaf)
├── eval/
│   ├── scenarios/          # evaluation scenarios (hidden need, opening message, dialogue snapshots)
│   ├── annotation/         # annotation records, sheets, guidelines, gold labels, agreement report
│   ├── q2_benchmark/       # Q2 results and timings of the five queries
│   ├── q3_retrieval/       # Q3 retrieval results
│   ├── figures/            # report figures (architecture, UI, retrieval and ablation charts)
│   ├── q4_decisions/       # Q4 clarification-decision results
│   └── q5_runs/            # Q4/Q5 dialogue logs and ablation summaries
├── data/
│   ├── kb.json             # product-level knowledge base: id, text (full product text), meta (price, rating, review count, store, categories, release date, brand, material, colour, recommended age, department)
│   ├── chunks.json         # chunk-level retrieval units: chunk_id, parent_id, field (overview/description/specs), text
│   ├── kb_pyserini.jsonl   # chunk-level corpus in Pyserini format
│   ├── corpus_stats.json   # corpus statistics (for Q1)
│   ├── faiss_chunk.bin     # chunk-level vector index (built by build_faiss.py)
│   ├── faiss_chunk.json    # index sidecar: model, unit, prefix setting, precision, chunk-id order
│   └── faiss_product.*     # product-level index (--unit product; Q3 ablation without chunking)
│                           # note: *.bin files exceed GitHub's 100 MB limit and are git-ignored; rebuild locally or download from the shared drive
├── README.md               # this file (English)
├── README_zh.md            # Chinese version
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### GPU / Apple Silicon (strongly recommended)

`build_faiss.py` picks CUDA → Apple Silicon (MPS) → CPU automatically and encodes in fp16 on GPUs (cosine similarity to fp32 vectors ≥ 0.9998). Encoding the 48,107 chunks took 20.3 minutes on an Apple M5 (MPS, fp16); an NVIDIA GPU is faster, a CPU takes more than ten hours. On Apple Silicon the script switches to eager attention, because PyTorch's SDPA keeps leaking memory on MPS and eventually runs out of memory. Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

If it prints `False` (a CPU build of torch is installed), **force-reinstall** the CUDA build (RTX 50 series / Blackwell needs CUDA 12.8+):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

## Question 1: Knowledge base

**Goal:** build a product knowledge corpus from the Amazon 2023 metadata of three categories and split it into passages suitable for retrieval and generation.

### Pipeline

1. **Source:** the `.jsonl.gz` files of three categories ([raw data](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/))
   - `meta_Clothing_Shoes_and_Jewelry.jsonl.gz` (7,218,481 products)
   - `meta_Home_and_Kitchen.jsonl.gz` (3,735,584 products)
   - `meta_Toys_and_Games.jsonl.gz` (890,874 products)
2. **Streaming:** the compressed files are read line by line without saving them; every product of every category is scanned.
3. **Filtering:** a product is kept only if `title`, `features`, `description`, `categories`, `details`, `price`, `average_rating`, `rating_number` and `store` are all non-empty (checked again after cleaning). Valid products: clothing 835,390 (11.6 %), home 749,624 (20.1 %), toys 268,771 (30.2 %).
4. **Random sampling:** reservoir sampling (seed `42`) over **all** valid products of each category, so every valid product has the same probability of being chosen. The earlier approach (stop after the first 10,000 valid products) only covered the beginning of each file, and the beginning is not representative: the first 30,000 clothing products are 25 % valid versus 11.6 % overall.
5. **Cleaning:**
   - HTML tags (174 fields) and entities (15 fields), emoji and decorative symbols (7,186 fields) removed;
   - noisy specification keys that do not help retrieval dropped: Best Sellers Rank (8,710 products), Date First Available (8,972), Item model number (8,451), Is Discontinued (5,226), etc.; Python dicts/lists are no longer written into the text verbatim;
   - duplicate bullet points and description lines removed; Manufacturer dropped when equal to Brand.
6. **De-duplication:** by `parent_asin` and by normalised title across categories (10 duplicates removed).
7. **Document format:** `text` holds only semantic fields (title, categories, features, description, specifications). Price, rating, review count, store and categories go to structured `meta`, so they do not dilute embeddings and can be used directly for evidence headers and display.
8. **Field-level chunking:** each product is split into chunks that all start with the product title, so they stay understandable out of context:
   - `overview`: title + categories + features (grouped if features exceed 200 words)
   - `description`: title + description (200-word windows with 40-word overlap)
   - `specs`: title + specifications

   Retrieval scores chunks and merges them to products by the best chunk (parent-document retrieval); generation uses the full product text.

### Corpus statistics (`data/corpus_stats.json`)

| Metric | Value |
|---|---|
| Documents (products) | 15,000 (5,000 per category) |
| Chunks | 48,107 (overview 16,483 / description 16,609 / specs 15,015) |
| Words | 3,589,422 (whitespace tokens of the product text) |
| Types (unique words) | 60,141 (lower-cased, alphanumeric only); 212,432 with raw whitespace tokens |
| Words per product | mean 239, median 203, p95 508, max 2,040 |
| Words per chunk | mean 88, max 421 |

### Run Q1

```bash
python dataset.py                     # full build, about 15 minutes (downloads about 7.7 GB of compressed data)
python dataset.py --max-scan 50000    # quick test: scan only the first N products per category
```

**Expected output:**

```text
✅ Corpus built. Documents: 15000, chunks: 48107, total words: 3589422, types: 60141
```

## Question 2: Hybrid retrieval and web interface

**Retrieval pipeline:** query → BM25 (chunks) and bge-m3 / FAISS (chunks) → each merged to products by their best chunk → RRF fusion → top-k products (with the matched field and both ranks).

### Components

| Module | Implementation | Purpose |
| :--- | :--- | :--- |
| **Sparse retrieval (BM25)** | Own Okapi BM25 (k1 = 1.5, b = 0.75) with pre-computed weights in a sparse matrix; a query only sums the columns of its terms. Tokenizer: lower-casing, alphanumerics only, stop-word removal | Exact keyword matching. Compared with the original `rank_bm25` + `split()`, `Waterproof`/`waterproof` and `hiking.`/`Hiking` now match; about 0.3 ms per query (previously about 100 ms) |
| **Dense retrieval (FAISS)** | `BAAI/bge-m3` (1024-d) + `IndexFlatIP` (normalised vectors, i.e. cosine similarity), chunk-level index | Semantic matching of paraphrases |
| **Parent-document merge** | Both retrievers score chunks; a product gets the score of its best chunk (MaxP) | Short chunks match precisely while results stay product-level; generation uses the full product |
| **Reciprocal rank fusion (RRF)** | k = 60, top 20 products from each retriever | Combines lexical and semantic matching |
| **Gradio interface** | `app.py` | Chat with the assistant; shows answers, recommended products (title, price, rating), evidence, matched fields and per-stage timing |

### Run Q2

```bash
python build_faiss.py                 # build the chunk-level vector index (GPU / Apple Silicon)
python hybrid_retriever.py            # terminal test
python benchmark_queries.py           # results and speed of five queries → eval/q2_benchmark/
python app.py                         # start the web interface
```

### Speed of the five queries (`eval/q2_benchmark/`)

Apple M5 (MPS, fp16), 48,107 chunks, warm, median of 10 runs per query:

| Stage | Time |
|---|---|
| BM25 (sparse matrix) | 0.2–0.3 ms |
| Query encoding (bge-m3) | 16.4–16.7 ms |
| FAISS search (IndexFlatIP, exact) | 1.6 ms |
| RRF fusion | < 0.1 ms |
| **Total** | **18.2–18.6 ms** |
| With cross-encoder re-ranking (top 20 products) | about 0.69 s (mean over the 169 evaluation queries) |

With the product-level index (`RETRIEVAL_UNIT=product`) a query takes about 53 ms in total (single measurement).

Top-5 results of the five queries (title, category, price, rating, matched field, both ranks) are in `eval/q2_benchmark/summary.md` and `results.json`. The first query also loads the models (about 10 s) and warms up the GPU; `app.py` warms up at start-up.

## Question 3: Retrieval and ranking innovations

`hybrid_retriever.search()` supports all of the following options; `eval_retrieval.py` compares them on the evaluation scenarios:

| Direction | Implementation | Switch |
|---|---|---|
| 1. Hybrid fusion | BM25 only / dense only / RRF / weighted RRF | `fusion="bm25"/"dense"/"rrf"/"weighted"`, `weights=(w_bm25, w_dense)` |
| 2. Re-ranking | multilingual cross-encoder `BAAI/bge-reranker-v2-m3` re-ranks the top 20 fused products (fp16; inputs padded to 64-token buckets to avoid repeated MPS recompilation) | `rerank=True` |
| 3. Timeline and metadata search | price, rating, category, release date and audience/age/material as structured hard filters; the agent relaxes them step by step if too few results remain | `filters={"price_max", "rating_min", "category", "date_after", "date_before", "predicate"}` |
| 4. Multilingual search | bge-m3 and the re-ranker accept Chinese queries against English products; we compare direct Chinese retrieval with LLM translation to English first | `zh:*` / `zh→en:*` configurations |
| 5. Chunking and indexing | field-level chunks + parent-document merge vs. product-level index; BM25 tokenisation; BGE instruction prefix | `RETRIEVAL_UNIT=chunk/product`, `BGE_PROMPT=0/1` |

```bash
python eval_retrieval.py                                   # chunk-level index: all configurations + timeline experiment
RETRIEVAL_UNIT=product python eval_retrieval.py --tag product --configs bm25 dense rrf rrf+rerank --no-timeline
```

Evaluation: the query is the complete need of each scenario (English/Chinese); metrics are P@5, R@10, MRR@10, nDCG@10 and latency. There are two kinds of relevance: programmatic (every product in the catalogue that satisfies all hard constraints of the hidden need; complete, so recall is defined) and human-labelled pooled candidates (`gold_relevance.csv`). Results are in `eval/q3_retrieval/summary.md`.

### Main Q3 results (169 English queries; P@5 / nDCG@10 use programmatic relevance, human P@5 counts judged products only)

| Configuration | P@5 | nDCG@10 | Human P@5 | Latency |
|---|---|---|---|---|
| BM25 | 0.369 | 0.381 | 0.618 | 0.3 ms |
| Dense (bge-m3) | 0.420 | 0.421 | 0.624 | 22 ms |
| RRF | 0.440 | 0.438 | 0.605 | 20 ms |
| Weighted RRF (BM25 weight 0.5) | 0.451 | 0.449 | 0.635 | 20 ms |
| RRF + re-ranking | 0.499 | 0.494 | 0.676 | 688 ms |
| RRF + metadata filters | 0.656 | 0.660 | 0.862 | 22 ms |
| RRF + re-ranking + filters | **0.698** | **0.705** | **0.895** | 787 ms |
| Product-level index: RRF + re-ranking + filters | **0.715** | **0.718** | **0.901** | 842 ms |
| Chinese query, direct (dense / RRF) | 0.151 / 0.075 | 0.154 / 0.078 | 0.360 / 0.335 | — |
| Chinese query → LLM translation → RRF + re-ranking | 0.496 | 0.484 | 0.676 | 780 ms |

- Programmatic relevance agrees with the human labels on 79.9 % of pairs (κ 0.60) and is stricter; both rank the configurations in almost the same order.
- Timeline: writing "released after 2021" into the query text gives only 21.4 % results that meet the date; a date filter gives 100 %.
- Direct Chinese retrieval is weak; translating to English first reaches the quality of English queries, so the agent retrieves in English and answers in the user's language.
- The product-level index is slightly better than field-level chunks (need-style queries are answered by the title and category path); chunks mainly help by giving the generator citable field-level evidence.

## Question 4: Multi-turn clarifying recommendation and evaluation

**Task:** the user and the assistant talk over several turns. The first message is often incomplete ("I want some shoes"); the assistant decides whether it knows enough, asks a question if not, and otherwise recommends products based on retrieved evidence, with citations.

**Every agent turn** (`agent.py`): state tracking + query rewriting → hybrid retrieval (filters + re-ranking) → decision (ask / recommend / nothing matches) → evidence sufficiency check → grounded generation → claim-level self-check.

### API configuration

```bash
export DEEPSEEK_API_KEY=sk-...
export DEEPSEEK_MODEL=deepseek-flash      # use the model name listed at https://api-docs.deepseek.com
export JUDGE_MODEL=deepseek-v4-pro        # optional: judge model, ideally different from the generator
```

deepseek-flash enables thinking (reasoning) mode by default, which ignores `temperature`, makes outputs non-reproducible and adds hundreds of reasoning tokens per call. This project disables it by default (`llm.py`); set `DEEPSEEK_THINKING=enabled` or use a `*+thinking` configuration to turn it on.

### Evaluation set (scenario-based)

```bash
python build_scenarios.py --per-domain 60     # 180 scenarios (60 per domain) → eval/scenarios/scenarios.jsonl
python build_annotation_set.py --annotators A B C   # annotation records and sheets → eval/annotation/ (needs GPU retrieval)
python compute_iaa.py                         # agreement, Cohen's κ, Fleiss' κ, gold labels
```

- **Scenario:** a hidden complete need (product type, audience / child age / material, budget) + an opening message at one of three levels (vague / partial / complete, in English or Chinese), anchored on a real catalogue product and satisfiable by at least 3 products.
- **Annotation records:** 2 dialogue snapshots per scenario (should the agent ask or recommend, and what to ask about) + 6 pooled candidate products (pooled from several retrieval configurations; does the product satisfy the need), 1,440 records generated. Guidelines: `eval/annotation/GUIDELINES.md`.
- **De-duplication** (`python dedup.py`): 11 scenarios repeat an earlier hidden need and 39 further snapshots repeat an earlier dialogue verbatim (e.g. "I need a toy."). After removing them, 169 scenarios, 299 decision records and 1,014 relevance records remain, **1,313 records** in total (listed in `eval/annotation/eval_set_unique.json`). All evaluation scripts compute metrics on the unique set; existing results can be re-summarised without API calls via `--rescore` / `--resummarize`.

### Run the evaluation

```bash
python evaluate_decisions.py                  # clarification decision P/R/F1, attribute accuracy → eval/q4_decisions/
python run_dialogues.py --configs full B0 --workers 6   # user-simulator dialogues → eval/q5_runs/
python rag_judge.py --configs full B0         # faithfulness / answer relevance / context precision
python app.py                                 # multi-turn web interface
```

- **User simulator** (`simulator.py`): knows the hidden need but never the target product (no answer leakage); answers only what is asked; after a failed recommendation it names one violated constraint; at most 5 turns. Success = at least one of the top-3 recommendations satisfies every hard constraint (checked programmatically).
- **Inter-annotator agreement** (`eval/annotation/iaa_report.json`, unique set): ask vs. recommend 98.7 % (Fleiss' κ 0.969); which attribute to ask about 86.6 % (0.812); relevance 94.9 % (0.897); all 1,313 records 95.8 % (0.913). After one round of guideline clarification (one annotator re-checked the records labelled "wrong product type"), relevance agreement rose from 79.9 % in the first round to 94.9 %; the gold labels (majority vote of three) were unchanged.
- **Clarification decision** (299 snapshots, human gold): entropy criterion P/R/F1 = 0.803 / 1.000 / 0.891; LLM judgement 0.973 / 0.711 / 0.822; LLM + thinking mode 0.972 / 0.838 / 0.900. The rule-based decision matches the human labels exactly (the guidelines encode that rule).
- **LLM judge calibration** (`python rag_judge.py --calibrate`): on the 1,014 human-labelled pairs the judge's relevance verdicts agree with the gold labels 92.7 % of the time (κ 0.853).
- **Full agent:** success 0.876, first-recommendation success 0.852, constraint violation rate 0.372, faithfulness 0.998, per-turn latency p50 3.85 s, about $1.34 per 1,000 dialogues (deepseek-flash off-peak prices with the ~40 % context-cache hit rate seen on the bill; all experiments of the project cost $4.61 in total).
- **Metrics:** success rate, first-recommendation success, average turns, questions asked, useful-question rate, over-asking on complete openings, constraint violation rate, self-check supported-claim rate, per-turn latency, LLM calls and cost per dialogue; RAGAS-style faithfulness, answer relevance and context precision.

## Question 5: Downstream innovations and ablation

| Innovation | Description | Switch (`agent.AgentConfig`) |
|---|---|---|
| I1 Clarification decision + information-gain question selection | When to ask: rule / LLM judgement / entropy of the candidate distribution; what to ask: the attribute with the highest entropy (options are its most frequent values among the candidates) | `clarify`, `select` |
| I2 Query rewriting + constraint-to-filter mapping | the dialogue is rewritten into a standalone English query (also handles Chinese); budget, audience, age and material become hard filters, relaxed step by step if too few results remain | `rewrite`, `filters` |
| I3 Evidence sufficiency check + claim-level self-check | CRAG-style: hard constraints checked programmatically + LLM check that candidates match, retrieve again or say honestly that nothing fits; after generation every claim is verified, prices against metadata | `check` |
| I4 (optional) Aspect-level reasons | each recommendation explains the aspects the user cares about, citing evidence or saying "not stated" | `aspects` |

```bash
python run_dialogues.py --configs B0 B1 B2 B3 B4 full full-I1 full-I2 full-I3 --workers 6
python run_dialogues.py --configs clarify-rule clarify-llm select-llm select-random full+thinking
```

Configurations are defined in `agent.CONFIGS`: B0 (single-turn RAG, history concatenated) → B1 (+ rewriting) → B2 (+ LLM clarification) → B3 (+ entropy question selection) → B4 (+ filters) → full (+ evidence check and self-check); `full-I*` removes one innovation at a time. All configurations share the same retrieval (RRF + re-ranking), so differences come only from the Q5 innovations. Results are in `eval/q5_runs/summary.md` (LLM judge results in `judge_summary.json`).

### Q5 ablation results (169 scenarios, user simulator, at most 5 turns)

| Configuration | Success | First-rec. success | Questions | Violation rate | Faithfulness | Cost / 1k dialogues |
|---|---|---|---|---|---|---|
| B0 single-turn RAG | 0.793 | 0.438 | 0 | 0.727 | 0.993 | $0.73 |
| B1 + rewriting | 0.852 | 0.485 | 0 | 0.697 | 0.995 | $0.92 |
| B2 + LLM clarification | 0.858 | 0.763 | 1.22 | 0.558 | — | $0.80 |
| B3 + entropy decision and selection | 0.828 | 0.805 | 2.11 | 0.424 | — | $0.77 |
| B4 + metadata filters | 0.882 | 0.864 | 2.11 | 0.372 | 0.993 | $0.74 |
| full + evidence check and self-check | 0.876 | 0.852 | 2.11 | 0.372 | **0.998** | $1.34 |
| full − I1 (no clarification) | 0.586 | 0.485 | 0 | 0.580 | — | $1.13 |
| full − I2 | 0.852 | 0.834 | 2.11 | 0.369 | — | $1.23 |
| full − I3 | 0.888 | 0.864 | 2.11 | 0.383 | 0.995 | $0.74 |
| full + thinking mode | 0.893 | 0.864 | 2.13 | **0.332** | — | $1.77 |

Main findings:
- **Clarification (I1) is the core:** without it success drops from 0.876 to 0.586.
- **Constraint-to-filter mapping (I2)** mainly reduces violations (B3 → B4: 0.424 → 0.372).
- **Limits of the entropy criterion:** it over-asks — when the opening already states the full need, it still asks in half of the cases; the rule-based decision has a useful-question rate of 0.965 with similar success. Choosing the attribute by entropy is no better than choosing a random missing attribute.
- **Evidence check and self-check (I3)** do not raise success but give the highest faithfulness and let the agent say honestly when nothing matches.
- **Aspect-level reasons (I4)** did not help and remain optional.

## Notes for teammates

- **Windows compatibility:** Pyserini needs Java and often fails to build on Windows, so the project skips it and implements BM25 with scikit-learn + SciPy sparse matrices (`kb_pyserini.jsonl` is still written in Pyserini format for comparison).
- **Direct streaming:** no dependency on Hugging Face `datasets`; the raw JSONL files are streamed from their direct links.
- **Central data folder:** all generated corpora and vector indexes live in `data/`.
- **Document ID:** the Amazon `parent_asin` is the unique retrieval document ID (`docid`) of each product.
- **Index consistency:** `hybrid_retriever.py` checks that the index sidecar (`data/faiss_<unit>[_prompt].json`) matches the current model, retrieval unit and prefix setting and stops with an error otherwise, so the index and query encodings cannot diverge.
- **Resumable encoding:** `build_faiss.py` encodes in slices and saves a checkpoint every 4 slices (`*.partial.npz`); a rerun continues where it stopped. On Apple Silicon the MPS cache is cleared after every slice to avoid running out of memory.
- **Evaluation set:** 1,440 records generated, 1,313 after de-duplication (`dedup.py`); the labels are in `eval/annotation/sheet_A/B/C.xlsx`. After changing labels, run `compute_iaa.py`, `evaluate_decisions.py --rescore`, `eval_retrieval.py --rescore` and `rag_judge.py --calibrate --rescore`; the human-label metrics are written to the respective summaries.
- **Report figures:** `eval/figures/`; result charts are produced by `eval/figures/charts.py` and the architecture diagram `fig_architecture.svg` by `eval/figures/arch_svg.py`.
- **API key:** keep it in a local environment variable (e.g. `~/.zshrc`); never commit it.
- **HF Hub warning:** a warning about an unauthenticated Hugging Face Hub token only concerns rate limits and does not affect the demo.
