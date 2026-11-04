"""
混合检索：BM25（稀疏）+ bge-m3 / FAISS（稠密）→ 融合 → 元数据过滤 → 交叉编码器重排序

检索单元默认是字段级 chunk（标题+卖点 / 描述 / 规格）。两路检索都先在 chunk 上打分，
再按商品取最高分合并（父文档检索），然后融合两路的商品排名；返回结果包含完整商品正文
（供生成）、命中的 chunk（说明是商品的哪一部分与查询匹配）以及结构化 meta。

主要接口：
    search(query, top_k, fusion="rrf", weights=(1, 1), filters=None, rerank=False)
        fusion:  "rrf"（等权）| "weighted"（加权 RRF）| "bm25"（只用稀疏）| "dense"（只用稠密）
        filters: {"price_min", "price_max", "rating_min", "category", "date_after",
                  "date_before", "predicate"(可调用对象)} —— 结构化元数据硬过滤
        rerank:  是否用 bge-reranker-v2-m3 对融合后的候选重排序
    hybrid_search(query, top_k)  —— 默认配置（RRF，无过滤、无重排），Q2/Q4 沿用

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

CANDIDATES_PER_RETRIEVER = 20   # 无过滤时每路检索参与融合的商品数
FILTER_CANDIDATES = 200         # 有过滤时扩大候选，保证过滤后仍有足够结果
RERANK_CANDIDATES = 20          # 送入重排序的融合候选数
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

# 最近一次检索的分阶段耗时（毫秒）与过滤信息，供测速和评测脚本读取
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


def reciprocal_rank_fusion(rank_lists, k=RRF_K, weights=None):
    """rank_lists: 若干个 [(doc_id, rank_position, ...)]；返回 [(doc_id, score)]，分数降序。"""
    weights = weights or [1.0] * len(rank_lists)
    scores = {}
    for w, rlist in zip(weights, rank_lists):
        for item in rlist:
            docid, pos = item[0], item[1]
            scores[docid] = scores.get(docid, 0.0) + w / (k + pos + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def passes_filters(product, filters):
    """结构化元数据硬过滤。缺失的 meta 值视为不满足对应的数值/日期约束。"""
    if not filters:
        return True
    meta = product.get("meta", {})
    price = meta.get("price")
    if filters.get("price_min") is not None and (price is None or price < filters["price_min"]):
        return False
    if filters.get("price_max") is not None and (price is None or price > filters["price_max"]):
        return False
    rating = meta.get("average_rating")
    if filters.get("rating_min") is not None and (rating is None or rating < filters["rating_min"]):
        return False
    if filters.get("category") and meta.get("main_category") != filters["category"]:
        return False
    date = meta.get("date_first_available")
    if filters.get("date_after") and (not date or date < filters["date_after"]):
        return False
    if filters.get("date_before") and (not date or date > filters["date_before"]):
        return False
    predicate = filters.get("predicate")
    if predicate is not None and not predicate(product):
        return False
    return True


def _result(asin, score, sparse_by_id, dense_by_id):
    product = products[asin]
    meta = product.get("meta", {})
    hit = dense_by_id.get(asin) or sparse_by_id.get(asin)
    field, chunk_text = corpus.unit_info(hit[3])
    return {
        "parent_asin": asin,
        "passage": product["text"],
        "price": meta.get("price"),
        "average_rating": meta.get("average_rating"),
        "rating_number": meta.get("rating_number"),
        "store": meta.get("store"),
        "main_category": meta.get("main_category"),
        "date_first_available": meta.get("date_first_available"),
        "matched_field": field,
        "matched_text": chunk_text,
        "fusion_score": round(score, 5),
        "bm25_rank": sparse_by_id[asin][1] + 1 if asin in sparse_by_id else None,
        "dense_rank": dense_by_id[asin][1] + 1 if asin in dense_by_id else None,
    }


def search(query: str, top_k=10, fusion="rrf", weights=(1.0, 1.0), filters=None,
           rerank=False, candidates=None):
    """主检索接口，见模块说明。返回 (结果列表, 总耗时毫秒)。"""
    t0 = time.perf_counter()
    depth = candidates or (FILTER_CANDIDATES if filters else CANDIDATES_PER_RETRIEVER)
    if rerank:
        depth = max(depth, RERANK_CANDIDATES)

    sparse = bm25_retrieve(query, top_k=depth) if fusion != "dense" else []
    t1 = time.perf_counter()
    dense = dense_retrieve(query, top_k=depth) if fusion != "bm25" else []
    t2 = time.perf_counter()

    if fusion == "bm25":
        fused = [(d[0], d[2]) for d in sparse]
    elif fusion == "dense":
        fused = [(d[0], d[2]) for d in dense]
    elif fusion == "weighted":
        fused = reciprocal_rank_fusion([sparse, dense], weights=list(weights))
    else:
        fused = reciprocal_rank_fusion([sparse, dense])
    t3 = time.perf_counter()

    n_before = len(fused)
    if filters:
        fused = [(a, s) for a, s in fused if passes_filters(products[a], filters)]
    t4 = time.perf_counter()

    sparse_by_id = {d[0]: d for d in sparse}
    dense_by_id = {d[0]: d for d in dense}
    keep = RERANK_CANDIDATES if rerank else top_k
    out = [_result(a, s, sparse_by_id, dense_by_id) for a, s in fused[:keep]]

    rerank_ms = 0.0
    if rerank and out:
        import reranker
        out, rerank_ms = reranker.rerank(query, out)
    out = out[:top_k]

    total_ms = (time.perf_counter() - t0) * 1000
    LAST_TIMING.update({
        "bm25_ms": (t1 - t0) * 1000,
        "dense_ms": (t2 - t1) * 1000,
        "fusion_ms": (t3 - t2) * 1000,
        "filter_ms": (t4 - t3) * 1000,
        "rerank_ms": rerank_ms,
        "total_ms": total_ms,
        "candidates_before_filter": n_before,
        "candidates_after_filter": len(fused),
    })
    if fusion == "bm25":
        LAST_TIMING["dense_encode_ms"] = LAST_TIMING["dense_search_ms"] = 0.0
    return out, round(total_ms, 2)


def hybrid_search(query: str, top_k=10):
    """默认配置：RRF 融合，无过滤、无重排（Q2 测速与 Q4 生成沿用）。"""
    return search(query, top_k=top_k)


if __name__ == "__main__":
    print(f"Unit: {RETRIEVAL_UNIT} | prompt: {USE_BGE_PROMPT} | index: {FAISS_INDEX_PATH}")
    hybrid_search("warm up", top_k=1)
    res, ms = hybrid_search("waterproof men's boots", top_k=5)
    print(f"latency {ms} ms  " + "  ".join(f"{k}={v:.1f}" for k, v in LAST_TIMING.items()))
    for item in res:
        print(item["parent_asin"], item["matched_field"], item["passage"][15:90])
