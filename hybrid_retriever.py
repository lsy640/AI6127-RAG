"""
混合检索：BM25（稀疏）+ bge-m3 / FAISS（稠密）→ RRF 融合 → 商品级 Top-k

检索单元默认是字段级 chunk（标题+卖点 / 描述 / 规格）。两路检索都先在 chunk 上打分，
再按商品取最高分合并（父文档检索），最后用 RRF 融合两路的商品排名；返回结果中包含
完整商品正文（供生成）以及命中的 chunk（说明是商品的哪一部分与查询匹配）。

配置（环境变量，必须与 build_faiss.py 建索引时一致）：
    RETRIEVAL_UNIT=chunk|product   默认 chunk
    BGE_PROMPT=0|1                 默认 0（不加指令前缀，bge-m3 官方用法）
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import time

import faiss

import corpus
from build_faiss import BGE_SEARCH_PROMPT, EMBED_MODEL, index_paths, load_embedding_model, pick_device
from sparse_bm25 import bm25_retrieve

RETRIEVAL_UNIT = corpus.RETRIEVAL_UNIT
USE_BGE_PROMPT = os.getenv("BGE_PROMPT", "0") == "1"
FAISS_INDEX_PATH, INDEX_META_PATH = index_paths(RETRIEVAL_UNIT, USE_BGE_PROMPT)

CANDIDATES_PER_RETRIEVER = 20   # 每路检索参与融合的商品数
RRF_K = 60                      # RRF 平滑常数

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(
        f"{FAISS_INDEX_PATH} 不存在，请先运行："
        f"python build_faiss.py --unit {RETRIEVAL_UNIT}" + (" --prompt" if USE_BGE_PROMPT else "")
    )

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
    index_meta = json.load(f)
if (index_meta["model"] != EMBED_MODEL or index_meta["unit"] != RETRIEVAL_UNIT
        or index_meta["prompt"] != USE_BGE_PROMPT
        or index_meta["count"] != faiss_index.ntotal):
    raise ValueError(f"索引说明与当前配置不一致：{INDEX_META_PATH}")

dense_unit_ids = index_meta["ids"]
_parent_of = dict(zip(*corpus.load_units()[:2]))
products = corpus.products_by_id()

query_prefix = BGE_SEARCH_PROMPT if USE_BGE_PROMPT else ""
emb_model = load_embedding_model(pick_device())

# 最近一次 hybrid_search 的分阶段耗时（毫秒），供测速脚本读取
LAST_TIMING = {}


def dense_retrieve(query: str, top_k=20):
    """返回 list[(parent_id, rank_position, cosine, best_unit_id)]，按商品去重。"""
    t0 = time.perf_counter()
    q_emb = emb_model.encode([query_prefix + query], convert_to_numpy=True,
                             normalize_embeddings=True)
    t1 = time.perf_counter()
    n_units = top_k * (8 if RETRIEVAL_UNIT == "chunk" else 1)
    scores, idx = faiss_index.search(q_emb, min(n_units, faiss_index.ntotal))
    t2 = time.perf_counter()
    LAST_TIMING["dense_encode_ms"] = (t1 - t0) * 1000
    LAST_TIMING["dense_search_ms"] = (t2 - t1) * 1000

    out, seen = [], set()
    for score, i in zip(scores[0], idx[0]):
        if i == -1:
            continue
        unit_id = dense_unit_ids[i]
        parent = _parent_of[unit_id]
        if parent in seen:
            continue
        seen.add(parent)
        out.append((parent, len(out), float(score), unit_id))
        if len(out) >= top_k:
            break
    return out


def reciprocal_rank_fusion(rank_lists, k=RRF_K):
    """rank_lists: 若干个 [(doc_id, rank_position, ...)]；返回 [(doc_id, rrf_score)]，分数降序。"""
    scores = {}
    for rlist in rank_lists:
        for item in rlist:
            docid, pos = item[0], item[1]
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + pos + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, top_k=10):
    t0 = time.perf_counter()
    sparse = bm25_retrieve(query, top_k=CANDIDATES_PER_RETRIEVER)
    t1 = time.perf_counter()
    dense = dense_retrieve(query, top_k=CANDIDATES_PER_RETRIEVER)
    t2 = time.perf_counter()
    fused = reciprocal_rank_fusion([sparse, dense])
    t3 = time.perf_counter()

    sparse_by_id = {d[0]: d for d in sparse}
    dense_by_id = {d[0]: d for d in dense}

    out = []
    for asin, rrf in fused[:top_k]:
        product = products[asin]
        meta = product.get("meta", {})
        hit = dense_by_id.get(asin) or sparse_by_id.get(asin)
        field, chunk_text = corpus.unit_info(hit[3])
        out.append({
            "parent_asin": asin,
            "passage": product["text"],
            "price": meta.get("price"),
            "average_rating": meta.get("average_rating"),
            "rating_number": meta.get("rating_number"),
            "store": meta.get("store"),
            "main_category": meta.get("main_category"),
            "matched_field": field,
            "matched_text": chunk_text,
            "rrf_score": round(rrf, 5),
            "bm25_rank": sparse_by_id[asin][1] + 1 if asin in sparse_by_id else None,
            "dense_rank": dense_by_id[asin][1] + 1 if asin in dense_by_id else None,
        })

    total_ms = (time.perf_counter() - t0) * 1000
    LAST_TIMING.update({
        "bm25_ms": (t1 - t0) * 1000,
        "dense_ms": (t2 - t1) * 1000,
        "fusion_ms": (t3 - t2) * 1000,
        "total_ms": total_ms,
    })
    return out, round(total_ms, 2)


if __name__ == "__main__":
    print(f"Unit: {RETRIEVAL_UNIT} | prompt: {USE_BGE_PROMPT} | index: {FAISS_INDEX_PATH}")
    hybrid_search("warm up", top_k=1)
    res, ms = hybrid_search("waterproof men's boots", top_k=5)
    print(f"latency {ms} ms  " + "  ".join(f"{k}={v:.1f}" for k, v in LAST_TIMING.items()))
    for item in res:
        print(item["parent_asin"], item["matched_field"], item["passage"][15:90])
