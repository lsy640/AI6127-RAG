"""
Q4: Run the RAG generator over the evaluation queries.

By default the generator is fed the FROZEN retrieval pool
(eval/annotation/pool.jsonl), i.e. exactly the passages that were
manually labeled, so generation metrics are not affected by later changes
to the retriever. Use --live to retrieve again (end-to-end latency).

Outputs one JSON line per query, including the raw (pre-validation) LLM
recommendations, plus a run-level metadata file with wall-clock
throughput.

Usage:
    python run_q4_evaluation.py                       # all queries, pool
    python run_q4_evaluation.py --limit 10            # quick smoke test
    python run_q4_evaluation.py --workers 8           # concurrent requests
    python run_q4_evaluation.py --no-clean --tag raw  # ablation: no preprocessing
    python run_q4_evaluation.py --live                # retrieve on the fly
"""

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from generator import generate_answer


BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "eval"
QUERIES_FILE = EVAL_DIR / "q4_queries.jsonl"
POOL_FILE = EVAL_DIR / "annotation" / "pool.jsonl"

TOP_K = 5


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pool():
    pool = {}
    for row in load_jsonl(POOL_FILE):
        pool.setdefault(row["case_id"], []).append(row)
    for rows in pool.values():
        rows.sort(key=lambda r: r["rank"])
    return pool


def run_one_case(case, pool, live, clean):
    """Run retrieval (or load frozen pool) and generation for one case."""

    query = case["query"]

    result = {
        "case_id": case["case_id"],
        "query": query,
        "domain": case.get("domain"),
        "query_type": case.get("query_type"),
        "retrieved_asins": [],
        "recommendations": [],
        "raw_recommendations": [],
        "dropped_recommendations": [],
        "insufficient_evidence": None,
        "answer": "",
        "evidence": [],
        "retrieval_latency_ms": None,
        "generation_latency_ms": None,
        "total_latency_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "model": None,
        "error": None,
    }

    start_time = time.perf_counter()

    try:
        # Step 1: Retrieval
        if live:
            from hybrid_retriever import hybrid_search
            retrieved, retrieval_latency = hybrid_search(query, top_k=TOP_K)
            result["retrieval_latency_ms"] = retrieval_latency
        else:
            retrieved = [
                {"parent_asin": r["parent_asin"], "passage": r["passage"]}
                for r in pool.get(case["case_id"], [])[:TOP_K]
            ]

        result["retrieved_asins"] = [
            item["parent_asin"]
            for item in retrieved
            if item.get("parent_asin")
        ]

        # Step 2: Generation
        generated = generate_answer(query, retrieved, clean_evidence=clean)

        result["generation_latency_ms"] = generated.get("latency_ms")
        for key in ("answer", "recommendations", "raw_recommendations",
                    "dropped_recommendations", "insufficient_evidence",
                    "evidence", "prompt_tokens", "completion_tokens",
                    "model"):
            result[key] = generated.get(key, result[key])

    except Exception as e:
        result["error"] = str(e)
        traceback.print_exc()

    result["total_latency_ms"] = round(
        (time.perf_counter() - start_time) * 1000, 2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(QUERIES_FILE))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--live", action="store_true",
                        help="retrieve again instead of using the frozen pool")
    parser.add_argument("--no-clean", action="store_true",
                        help="disable evidence preprocessing (ablation)")
    parser.add_argument("--tag", default="main",
                        help="name of this run, used in output file names")
    args = parser.parse_args()

    cases = load_jsonl(args.queries)
    if args.limit:
        cases = cases[:args.limit]

    pool = {} if args.live else load_pool()
    clean = not args.no_clean

    out_file = EVAL_DIR / f"q4_results_{args.tag}.jsonl"
    meta_file = EVAL_DIR / f"q4_run_{args.tag}.json"

    print(f"Loaded {len(cases)} cases | workers={args.workers} | "
          f"retrieval={'live' if args.live else 'frozen pool'} | "
          f"clean={clean}")
    print("-" * 50)

    if args.live:
        # Load BM25/FAISS/embedding model before timing starts.
        from hybrid_retriever import hybrid_search
        hybrid_search("warm up", top_k=1)

    wall_start = time.perf_counter()

    def task(case):
        return run_one_case(case, pool, args.live, clean)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        all_results = []
        for index, result in enumerate(executor.map(task, cases), start=1):
            all_results.append(result)
            status = "FAILED" if result["error"] else "OK"
            print(f"[{index}/{len(cases)}] {result['case_id']} {status} | "
                  f"recs={len(result['recommendations'])} "
                  f"dropped={len(result['dropped_recommendations'])} | "
                  f"{result['total_latency_ms']} ms")

    wall_seconds = time.perf_counter() - wall_start

    with open(out_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    success = sum(1 for r in all_results if not r["error"])
    meta = {
        "tag": args.tag,
        "cases": len(all_results),
        "successful": success,
        "failed": len(all_results) - success,
        "workers": args.workers,
        "retrieval": "live" if args.live else "frozen_pool",
        "clean_evidence": clean,
        "wall_clock_seconds": round(wall_seconds, 2),
        "queries_per_second": round(len(all_results) / wall_seconds, 3)
        if wall_seconds else None,
        "model": next((r["model"] for r in all_results if r["model"]), None),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("-" * 50)
    print(json.dumps(meta, indent=2))
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    main()
