"""
Q2：五条查询的检索结果与查询速度

- 先预热（加载模型、首次编码不计入），每条查询重复 N 次，报告各阶段耗时的中位数；
- 分阶段计时：BM25、query 编码、FAISS 搜索、RRF 融合、总耗时；
- 结果保存到 eval/q2_benchmark/：
    results.json   每条查询的 Top-k 结果（标题、价格、评分、命中字段、两路名次）与耗时
    timing.csv     每次运行的分阶段耗时
    summary.md     可直接放进报告的表格

用法：
    python benchmark_queries.py --repeats 10 --top-k 5
"""

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path

import hybrid_retriever as hr
from generator import product_title

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "eval" / "q2_benchmark"

QUERIES = [
    "waterproof men's hiking boots",
    "non-stick frying pan for induction stove",
    "board game for family night with kids",
    "sterling silver earrings for sensitive ears",
    "soft queen size bed sheets",
]

STAGES = ["bm25_ms", "dense_encode_ms", "dense_search_ms", "fusion_ms", "total_ms"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 预热：模型加载与首次推理不计入
    for q in ["warm up", *QUERIES]:
        hr.hybrid_search(q, top_k=args.top_k)

    runs, results = [], []
    for q in QUERIES:
        timings = []
        for r in range(args.repeats):
            hits, _ = hr.hybrid_search(q, top_k=args.top_k)
            timings.append(dict(hr.LAST_TIMING))
            runs.append({"query": q, "run": r + 1,
                         **{s: round(hr.LAST_TIMING[s], 3) for s in STAGES}})
        median = {s: round(statistics.median(t[s] for t in timings), 2) for s in STAGES}
        results.append({
            "query": q,
            "median_ms": median,
            "results": [{
                "rank": i + 1,
                "parent_asin": h["parent_asin"],
                "title": product_title(h["passage"]),
                "category": h["main_category"],
                "price": h["price"],
                "average_rating": h["average_rating"],
                "rating_number": h["rating_number"],
                "matched_field": h["matched_field"],
                "bm25_rank": h["bm25_rank"],
                "dense_rank": h["dense_rank"],
                "rrf_score": h["rrf_score"],
            } for i, h in enumerate(hits)],
        })

    env = {
        "retrieval_unit": hr.RETRIEVAL_UNIT,
        "bge_prompt": hr.USE_BGE_PROMPT,
        "faiss_index": os.path.basename(hr.FAISS_INDEX_PATH),
        "indexed_units": hr.faiss_index.ntotal,
        "embedding_device": str(hr.emb_model.device),
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}".strip(),
        "repeats": args.repeats,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({"environment": env, "queries": results}, f, indent=2, ensure_ascii=False)
    with open(OUT_DIR / "timing.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "run", *STAGES])
        writer.writeheader()
        writer.writerows(runs)

    lines = [
        f"环境：{env['machine']}，编码设备 {env['embedding_device']}，"
        f"检索单元 {env['retrieval_unit']}（{env['indexed_units']} 个），"
        f"每条查询重复 {args.repeats} 次取中位数（已预热）。",
        "",
        "| 查询 | BM25 | query 编码 | FAISS 搜索 | RRF | 总计 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["median_ms"]
        lines.append(f"| {r['query']} | {m['bm25_ms']:.1f} | {m['dense_encode_ms']:.1f} | "
                     f"{m['dense_search_ms']:.1f} | {m['fusion_ms']:.2f} | {m['total_ms']:.1f} |")
    for r in results:
        lines += ["", f"**{r['query']}**", "",
                  "| # | 商品 | 类目 | 价格 | 评分 | 命中字段 | BM25 名次 | 稠密名次 |",
                  "|---|---|---|---|---|---|---|---|"]
        for h in r["results"]:
            lines.append(
                f"| {h['rank']} | {h['title'][:70].replace('|', '/')} | {h['category']} | ${h['price']:.2f} | "
                f"{h['average_rating']} ({h['rating_number']}) | {h['matched_field']} | "
                f"{h['bm25_rank'] or '-'} | {h['dense_rank'] or '-'} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:4 + len(results)]))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
