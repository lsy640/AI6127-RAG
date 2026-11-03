"""
Q4: Build the manual-annotation set.

For every query in eval/q4_queries.jsonl, run the hybrid retriever once and
freeze the Top-K passages into eval/annotation/pool.jsonl. Each
(query, retrieved product) pair is one annotation record:
210 queries x Top-5 = 1,050 records
(clothing / home & kitchen / toys / out-of-scope).

The same pool is later fed to the generator (run_q4_evaluation.py --pool),
so the human labels cover exactly the evidence the LLM sees, even if the
retriever is changed afterwards.

It also writes one Excel sheet per annotator. Candidate order is shuffled
inside each query (same order for every annotator) so that annotators are
not biased by the retriever's ranking.

Usage:
    python build_annotation_set.py --annotators A B C --top-k 5
    python build_annotation_set.py --sheets-only   # re-export sheets only
"""

import argparse
import json
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from generator import build_evidence_passage


BASE_DIR = Path(__file__).resolve().parent
QUERIES_FILE = BASE_DIR / "eval" / "q4_queries.jsonl"
OUT_DIR = BASE_DIR / "eval" / "annotation"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_pool(queries, top_k):
    from hybrid_retriever import hybrid_search

    pool = []

    for case in queries:
        results, _ = hybrid_search(case["query"], top_k=top_k)

        for rank, item in enumerate(results, start=1):
            pool.append({
                "pair_id": f"{case['case_id']}-{rank}",
                "case_id": case["case_id"],
                "query": case["query"],
                "domain": case.get("domain", ""),
                "query_type": case.get("query_type", ""),
                "rank": rank,
                "parent_asin": item["parent_asin"],
                "matched_field": item.get("matched_field"),
                "passage": item["passage"],
            })

    return pool


def write_sheet(path, rows, annotator):
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"

    headers = [
        "pair_id", "case_id", "query", "parent_asin",
        "product_passage", "relevant (1/0)", "comment",
    ]
    ws.append(headers)

    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")

    for row in rows:
        ws.append([
            row["pair_id"], row["case_id"], row["query"],
            # Same text the LLM sees (metadata next to the title).
            row["parent_asin"],
            build_evidence_passage(row["parent_asin"], row["passage"]),
            None, None,
        ])

    widths = [10, 8, 34, 13, 110, 14, 24]
    for col, width in zip("ABCDEFG", widths):
        ws.column_dimensions[col].width = width

    for r in ws.iter_rows(min_row=2):
        for cell in r:
            cell.alignment = Alignment(wrap_text=True, vertical="top")

    dv = DataValidation(
        type="list", formula1='"1,0"', allow_blank=True,
        showErrorMessage=True, errorTitle="Invalid label",
        error="Please enter 1 (relevant) or 0 (not relevant).",
    )
    ws.add_data_validation(dv)
    dv.add(f"F2:F{len(rows) + 1}")

    ws.freeze_panes = "F2"

    info = wb.create_sheet("info")
    info.append(["annotator", annotator])
    info.append(["records", len(rows)])
    info.append(["guidelines", "see eval/annotation/GUIDELINES.md"])

    wb.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sheets-only", action="store_true",
                        help="rebuild the Excel sheets from the existing "
                             "pool.jsonl without running retrieval again")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    queries = load_jsonl(QUERIES_FILE)

    if args.sheets_only:
        pool = load_jsonl(OUT_DIR / "pool.jsonl")
    else:
        pool = build_pool(queries, args.top_k)
        with open(OUT_DIR / "pool.jsonl", "w", encoding="utf-8") as f:
            for row in pool:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Record which retriever produced the frozen pool, so a later change
        # of retriever (and the need to re-label) is easy to detect.
        import hybrid_retriever as retriever
        with open(OUT_DIR / "pool_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "retriever": "hybrid_retriever (BM25 + bge-m3 FAISS + RRF, parent-document)",
                "kb": "data/kb.json",
                "retrieval_unit": retriever.RETRIEVAL_UNIT,
                "faiss_index": Path(retriever.FAISS_INDEX_PATH).name,
                "faiss_built_at": retriever.index_meta.get("built_at"),
                "bge_prompt": retriever.USE_BGE_PROMPT,
                "top_k": args.top_k,
                "records": len(pool),
            }, f, indent=2)

    # Shuffle candidates within each query; keep queries in order.
    rng = random.Random(args.seed)
    shuffled = []
    by_case = {}
    for row in pool:
        by_case.setdefault(row["case_id"], []).append(row)
    for case_id in by_case:
        rows = by_case[case_id][:]
        rng.shuffle(rows)
        shuffled.extend(rows)

    for name in args.annotators:
        write_sheet(OUT_DIR / f"sheet_{name}.xlsx", shuffled, name)

    print(f"Queries: {len(queries)}")
    print(f"Annotation records (query-product pairs): {len(pool)}")
    print(f"Unique products in pool: {len({r['parent_asin'] for r in pool})}")
    print(f"Wrote {OUT_DIR / 'pool.jsonl'} and "
          f"{len(args.annotators)} annotator sheets.")


if __name__ == "__main__":
    main()
