"""
Q4: Compute all Q4 metrics for one run.

Inputs (for --tag main):
    eval/q4_results_main.jsonl          from run_q4_evaluation.py
    eval/q4_run_main.json               run metadata (throughput)
    eval/q4_judge_main.jsonl            from rag_judge.py      (optional)
    eval/annotation/gold_labels.csv     from compute_iaa.py    (optional)

Metric groups:
  A. Task quality vs. human gold labels
     The generator decides which of the 5 retrieved products to recommend,
     so each (query, product) pair is a binary decision:
     recommended (predicted 1) vs. gold relevant (1) / not relevant (0).
       -> precision, recall, F1 (micro over pairs, macro over queries)
       -> abstention accuracy: no relevant product retrieved
          <=> the system should recommend nothing
  B. Context quality vs. gold (RAGAS non-LLM context precision@5, hit@5)
  C. Answer-level RAG metrics from the LLM judge
       faithfulness, answer relevance, LLM context precision,
       judge-vs-human agreement
  D. Grounding / citation checks on the RAW LLM output
       hallucinated-ASIN rate, invalid-citation rate
  E. Performance: latency, throughput, tokens, cost

Usage:
    python evaluate_q4.py --tag main --price-in 0.30 --price-out 1.20
Prices are USD per 1M tokens; check https://api-docs.deepseek.com/quick_start/pricing
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "eval"
GOLD_FILE = EVAL_DIR / "annotation" / "gold_labels.csv"


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gold(path):
    """Return {case_id: {asin: label}}."""
    gold = defaultdict(dict)
    if not path.exists():
        return gold
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold[row["case_id"]][row["parent_asin"]] = int(row["label"])
    return gold


def avg(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return mean(values) if values else None


def percentile(values, q):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    if not values:
        return None
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return p, r, f


def context_precision(relevance):
    """RAGAS context precision@K for a ranked 0/1 list."""
    if not relevance or sum(relevance) == 0:
        return 0.0 if relevance else None
    score, hits = 0.0, 0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            hits += 1
            score += hits / k
    return score / sum(relevance)


def cohen_kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def case_metrics(result, gold_case, judge):
    retrieved = result.get("retrieved_asins", [])
    recommended = {r["parent_asin"] for r in result.get("recommendations", [])}
    raw = result.get("raw_recommendations", []) or []
    dropped = result.get("dropped_recommendations", []) or []

    m = {
        "case_id": result["case_id"],
        "domain": result.get("domain"),
        "query_type": result.get("query_type"),
        "query": result["query"],
        "success": int(not result.get("error")),
        "n_retrieved": len(retrieved),
        "n_raw_recommendations": len(raw),
        "n_recommended": len(recommended),
        "n_hallucinated_asin": sum(d["reason"] == "asin_not_in_evidence" for d in dropped),
        "n_invalid_citation": sum(d["reason"] == "no_valid_evidence_id" for d in dropped),
        "insufficient_evidence_flag": result.get("insufficient_evidence"),
        "generation_latency_ms": result.get("generation_latency_ms"),
        "retrieval_latency_ms": result.get("retrieval_latency_ms"),
        "total_latency_ms": result.get("total_latency_ms"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "error": result.get("error"),
    }

    # A + B: against gold labels (only for retrieved products that are labeled)
    labeled = [a for a in retrieved if a in gold_case]
    if labeled and labeled == retrieved:
        gold_rel = [gold_case[a] for a in retrieved]
        tp = sum(1 for a in retrieved if gold_case[a] == 1 and a in recommended)
        fp = sum(1 for a in retrieved if gold_case[a] == 0 and a in recommended)
        fn = sum(1 for a in retrieved if gold_case[a] == 1 and a not in recommended)
        tn = sum(1 for a in retrieved if gold_case[a] == 0 and a not in recommended)
        p, r, f = prf(tp, fp, fn)
        has_relevant = int(sum(gold_rel) > 0)
        m.update({
            "labeled": 1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f,
            "gold_has_relevant": has_relevant,
            "abstained": int(len(recommended) == 0),
            "abstention_correct": int((len(recommended) == 0) == (has_relevant == 0)),
            "context_precision_gold": context_precision(gold_rel),
            "precision_at_5_gold": sum(gold_rel) / len(gold_rel),
            "hit_at_5_gold": has_relevant,
        })
    else:
        m["labeled"] = 0

    # C: LLM judge
    if judge and not judge.get("error"):
        m["faithfulness"] = judge.get("faithfulness")
        m["n_claims"] = judge.get("n_claims")
        m["answer_relevance"] = judge.get("answer_relevance")
        m["noncommittal"] = judge.get("noncommittal")
        llm_rel = judge.get("llm_context_relevance", {})
        if retrieved and all(a in llm_rel for a in retrieved):
            m["context_precision_llm"] = context_precision(
                [llm_rel[a] for a in retrieved])
        m["judge_tokens"] = judge.get("judge_tokens")

    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="main")
    parser.add_argument("--gold", default=str(GOLD_FILE))
    parser.add_argument("--price-in", type=float, default=0.30,
                        help="USD per 1M input tokens (cache miss)")
    parser.add_argument("--price-out", type=float, default=1.20,
                        help="USD per 1M output tokens")
    args = parser.parse_args()

    results = load_jsonl(EVAL_DIR / f"q4_results_{args.tag}.jsonl")
    if not results:
        raise SystemExit(f"No results for tag '{args.tag}'. "
                         "Run run_q4_evaluation.py first.")
    judges = {j["case_id"]: j for j in load_jsonl(EVAL_DIR / f"q4_judge_{args.tag}.jsonl")}
    gold = load_gold(Path(args.gold))
    meta_path = EVAL_DIR / f"q4_run_{args.tag}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    rows = [case_metrics(r, gold.get(r["case_id"], {}), judges.get(r["case_id"]))
            for r in results]
    ok = [m for m in rows if m["success"]]
    lab = [m for m in ok if m["labeled"]]

    summary = {"tag": args.tag, "model": meta.get("model"),
               "cases": len(rows), "successful": len(ok),
               "failed": len(rows) - len(ok), "labeled_cases": len(lab)}

    # A. Task quality
    if lab:
        TP, FP, FN, TN = (sum(m[k] for m in lab) for k in ("tp", "fp", "fn", "tn"))
        p, r, f = prf(TP, FP, FN)
        summary.update({
            "pairs_evaluated": TP + FP + FN + TN,
            "confusion": {"tp": TP, "fp": FP, "fn": FN, "tn": TN},
            "micro_precision": p, "micro_recall": r, "micro_f1": f,
            "pair_accuracy": (TP + TN) / (TP + FP + FN + TN),
            "macro_precision": avg([m["precision"] for m in lab]),
            "macro_recall": avg([m["recall"] for m in lab]),
            "macro_f1": avg([m["f1"] for m in lab]),
            "abstention_accuracy": avg([m["abstention_correct"] for m in lab]),
            "cases_without_relevant_product": sum(1 - m["gold_has_relevant"] for m in lab),
        })
        neg = [m for m in lab if not m["gold_has_relevant"]]
        summary["correct_abstention_rate_when_nothing_relevant"] = (
            avg([m["abstained"] for m in neg]))

        for field in ("domain", "query_type"):
            groups = {}
            for value in sorted({str(m.get(field)) for m in lab}):
                sub = [m for m in lab if str(m.get(field)) == value]
                tp, fp, fn = (sum(m[k] for m in sub) for k in ("tp", "fp", "fn"))
                bp, br, bf = prf(tp, fp, fn)
                groups[value] = {"cases": len(sub), "precision": bp, "recall": br,
                                 "f1": bf, "abstention_accuracy":
                                 avg([m["abstention_correct"] for m in sub])}
            summary[f"by_{field}"] = groups

        # B. Context quality
        summary["context_precision_gold"] = avg([m["context_precision_gold"] for m in lab])
        summary["precision_at_5_gold"] = avg([m["precision_at_5_gold"] for m in lab])
        summary["hit_at_5_gold"] = avg([m["hit_at_5_gold"] for m in lab])

    # C. RAG judge metrics
    if judges:
        summary["faithfulness"] = avg([m.get("faithfulness") for m in ok])
        summary["answer_relevance"] = avg([m.get("answer_relevance") for m in ok])
        summary["context_precision_llm"] = avg([m.get("context_precision_llm") for m in ok])
        summary["avg_claims_per_answer"] = avg([m.get("n_claims") for m in ok])

        human, llm = [], []
        for res in results:
            j = judges.get(res["case_id"], {})
            verdicts = j.get("llm_context_relevance", {}) or {}
            for asin in res.get("retrieved_asins", []):
                if asin in verdicts and asin in gold.get(res["case_id"], {}):
                    human.append(gold[res["case_id"]][asin])
                    llm.append(verdicts[asin])
        if human:
            summary["judge_vs_human_agreement"] = (
                sum(a == b for a, b in zip(human, llm)) / len(human))
            summary["judge_vs_human_kappa"] = cohen_kappa(human, llm)

    # D. Grounding on raw output
    raw_total = sum(m["n_raw_recommendations"] for m in ok)
    summary["raw_recommendations"] = raw_total
    summary["hallucinated_asin_rate"] = (
        sum(m["n_hallucinated_asin"] for m in ok) / raw_total if raw_total else None)
    summary["invalid_citation_rate"] = (
        sum(m["n_invalid_citation"] for m in ok) / raw_total if raw_total else None)
    summary["avg_recommendations_per_query"] = avg([m["n_recommended"] for m in ok])

    # E. Performance and cost
    gen = [m["generation_latency_ms"] for m in ok]
    tot = [m["total_latency_ms"] for m in ok]
    ret = [m["retrieval_latency_ms"] for m in ok]
    pt = avg([m["prompt_tokens"] for m in ok])
    ct = avg([m["completion_tokens"] for m in ok])
    cost = (pt * args.price_in + ct * args.price_out) / 1e6 if pt is not None and ct is not None else None
    summary.update({
        "retrieval_latency_ms_mean": avg(ret),
        "generation_latency_ms_mean": avg(gen),
        "generation_latency_ms_p50": median(gen) if gen else None,
        "generation_latency_ms_p95": percentile(gen, 0.95),
        "total_latency_ms_mean": avg(tot),
        "workers": meta.get("workers"),
        "queries_per_second": meta.get("queries_per_second"),
        "avg_prompt_tokens": pt,
        "avg_completion_tokens": ct,
        "price_usd_per_1m_in_out": [args.price_in, args.price_out],
        "cost_usd_per_query": cost,
        "cost_usd_per_1000_queries": cost * 1000 if cost is not None else None,
    })
    jt = avg([m.get("judge_tokens") for m in ok])
    if jt:
        summary["avg_judge_tokens_per_query"] = jt

    # Write outputs
    case_file = EVAL_DIR / f"q4_case_metrics_{args.tag}.csv"
    fields = sorted({k for m in rows for k in m},
                    key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
    with open(case_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_file = EVAL_DIR / f"q4_metrics_summary_{args.tag}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Q4 Evaluation Summary ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nFiles: {case_file}\n       {summary_file}")
    if not gold:
        print("\n(No gold labels yet: precision/recall/F1 and gold context "
              "precision will appear after compute_iaa.py.)")


if __name__ == "__main__":
    main()
