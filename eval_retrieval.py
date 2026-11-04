"""
Q3：检索与排序创新的评测（P@5、R@10、MRR@10、nDCG@10、延迟）

查询：每个评测场景的完整需求（英文 full_query_en / 中文 full_query_zh）。
相关性（qrels）两种：
  programmatic  知识库中满足场景隐藏需求全部硬约束的商品（品类叶子、人群/年龄、价格、材质）——
                覆盖完整，可计算召回率
  human         人工标注的 pooled 候选（gold_relevance.csv）；未标注的商品记为不相关，
                同时报告 judged@5（前 5 个结果中有人工标签的比例）

配置覆盖 Q3 的五个方向：
  融合      bm25 / dense / rrf / 加权 rrf
  重排序    rrf+rerank（bge-reranker-v2-m3）
  元数据    rrf(+rerank)+filters（隐藏需求中的价格、人群/年龄、材质作为硬过滤）
  时间线    timeline：查询附加“released after 2021”文本 vs 日期过滤 date_after
  多语言    中文查询直接检索（dense / rrf / rerank） vs LLM 翻译成英文后检索
  切块      用 RETRIEVAL_UNIT=product 再运行一次（商品级索引），结果写入同一汇总

用法：
    python eval_retrieval.py                       # chunk 级索引，全部配置
    RETRIEVAL_UNIT=product python eval_retrieval.py --tag product --configs bm25 dense rrf rrf+rerank
"""

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import attributes as A
import hybrid_retriever as hr

BASE_DIR = Path(__file__).resolve().parent
SCENARIOS = BASE_DIR / "eval" / "scenarios" / "scenarios.jsonl"
ANN_DIR = BASE_DIR / "eval" / "annotation"
OUT_DIR = BASE_DIR / "eval" / "q3_retrieval"
TRANSLATIONS = BASE_DIR / "eval" / "scenarios" / "translations_zh_en.json"

CONFIGS = {
    "bm25": ("en", dict(fusion="bm25"), False),
    "dense": ("en", dict(fusion="dense"), False),
    "rrf": ("en", dict(fusion="rrf"), False),
    "rrf-w(bm25=0.5)": ("en", dict(fusion="weighted", weights=(0.5, 1.0)), False),
    "rrf-w(dense=0.5)": ("en", dict(fusion="weighted", weights=(1.0, 0.5)), False),
    "rrf+rerank": ("en", dict(fusion="rrf", rerank=True), False),
    "rrf+filters": ("en", dict(fusion="rrf"), True),
    "rrf+rerank+filters": ("en", dict(fusion="rrf", rerank=True), True),
    "zh:dense": ("zh", dict(fusion="dense"), False),
    "zh:rrf": ("zh", dict(fusion="rrf"), False),
    "zh:rrf+rerank": ("zh", dict(fusion="rrf", rerank=True), False),
    "zh→en:rrf+rerank": ("zh2en", dict(fusion="rrf", rerank=True), False),
}
TIMELINE_CUTOFF = "2021-01-01"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def programmatic_qrels(scenarios):
    by_type = {}
    for asin, a in A.all_attributes().items():
        for c in a["category_path"]:
            by_type.setdefault(c, []).append(asin)
    qrels = {}
    for sc in scenarios:
        cand = by_type.get(sc["hidden_need"]["product_type"].lower(), [])
        qrels[sc["scenario_id"]] = {a for a in cand if A.satisfies(a, sc["hidden_need"])[0]}
    return qrels


def human_qrels():
    pool_path, gold_path = ANN_DIR / "pool.jsonl", ANN_DIR / "gold_relevance.csv"
    if not (pool_path.exists() and gold_path.exists()):
        return None
    rec = {r["record_id"]: r for r in load_jsonl(pool_path)}
    judged, relevant = {}, {}
    with open(gold_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            r = rec.get(row["record_id"])
            if not r:
                continue
            judged.setdefault(r["scenario_id"], set()).add(r["parent_asin"])
            if str(row["relevant"]).strip() in ("1", "1.0"):
                relevant.setdefault(r["scenario_id"], set()).add(r["parent_asin"])
    return judged, relevant


def metrics(ranked, rel, k_p=5, k=10):
    hits = [1 if a in rel else 0 for a in ranked[:k]]
    p5 = sum(hits[:k_p]) / k_p
    r10 = sum(hits) / len(rel) if rel else 0.0
    mrr = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return p5, r10, mrr, (dcg / idcg if idcg else 0.0)


def translate_queries(scenarios):
    cache = json.loads(TRANSLATIONS.read_text()) if TRANSLATIONS.exists() else {}
    todo = [s for s in scenarios if s["scenario_id"] not in cache]
    if todo:
        import llm
        for s in todo:
            out = llm.chat([{"role": "user", "content":
                             "Translate this shopping request into a concise English product search query. "
                             f'Return JSON {{"query": "..."}}.\nRequest: {s["full_query_zh"]}'}], tag="translate")
            cache[s["scenario_id"]] = out.get("query", "")
        TRANSLATIONS.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache


def run_config(name, scenarios, qrels_prog, human, translations, k=10):
    lang, kwargs, use_filters = CONFIGS[name]
    rows, lat = [], []
    for sc in scenarios:
        q = (sc["full_query_en"] if lang == "en" else sc["full_query_zh"] if lang == "zh"
             else translations.get(sc["scenario_id"], sc["full_query_zh"]))
        filters = A.filters_for(sc["hidden_need"]) if use_filters else None
        res, ms = hr.search(q, top_k=k, filters=filters, **kwargs)
        lat.append(ms)
        ranked = [r["parent_asin"] for r in res]
        row = {"scenario_id": sc["scenario_id"], "ranked": ranked,
               "prog": metrics(ranked, qrels_prog[sc["scenario_id"]])}
        if human:
            judged, relevant = human
            rel = relevant.get(sc["scenario_id"], set())
            row["human"] = metrics(ranked, rel)
            row["judged@5"] = sum(a in judged.get(sc["scenario_id"], set()) for a in ranked[:5]) / 5
        rows.append(row)
    summary = {"queries": len(rows), "latency_ms_mean": statistics.mean(lat),
               "latency_ms_p50": statistics.median(lat)}
    for key in ("prog", "human"):
        if rows and key in rows[0]:
            vals = list(zip(*[r[key] for r in rows]))
            summary[key] = {m: round(statistics.mean(v), 4) for m, v in zip(("P@5", "R@10", "MRR@10", "nDCG@10"), vals)}
    if human:
        summary["judged@5"] = round(statistics.mean(r["judged@5"] for r in rows), 3)
    return summary, rows


def timeline_experiment(scenarios, k=10):
    """时间线检索：把“2021 年以后上架”写进查询文本 vs 作为日期硬过滤。"""
    out = {}
    for name, use_filter in (("text", False), ("date_filter", True)):
        ok_date, ok_both, n = 0, 0, 0
        for sc in scenarios:
            q = sc["full_query_en"].rstrip(".") + (" released after 2021" if not use_filter else "")
            filters = {"date_after": TIMELINE_CUTOFF} if use_filter else None
            res, _ = hr.search(q, top_k=k, filters=filters, rerank=True)
            for r in res:
                n += 1
                d_ok = bool(r["date_first_available"]) and r["date_first_available"] >= TIMELINE_CUTOFF
                ok_date += d_ok
                ok_both += d_ok and A.satisfies(r["parent_asin"], sc["hidden_need"])[0]
        out[name] = {"date_precision@10": round(ok_date / max(1, n), 4),
                     "date_and_need_precision@10": round(ok_both / max(1, n), 4)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="chunk")
    parser.add_argument("--no-timeline", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = load_jsonl(SCENARIOS)[:args.limit] if args.limit else load_jsonl(SCENARIOS)
    qrels_prog = programmatic_qrels(scenarios)
    human = human_qrels()
    translations = translate_queries(scenarios) if any(CONFIGS[c][0] == "zh2en" for c in args.configs) else {}
    hr.search("warm up", top_k=3, rerank=True)
    import reranker
    reranker.warm_up()

    path = OUT_DIR / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for name in args.configs:
        t0 = time.time()
        s, rows = run_config(name, scenarios, qrels_prog, human, translations)
        key = f"{args.tag}:{name}"
        summary[key] = s
        with open(OUT_DIR / f"runs_{args.tag}_{name.replace('/', '_')}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[{key}] prog {s.get('prog')} human {s.get('human')} "
              f"lat {s['latency_ms_mean']:.0f}ms ({time.time() - t0:.0f}s)", flush=True)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.no_timeline and args.tag == "chunk":
        summary["timeline"] = timeline_experiment(scenarios)
        print("[timeline]", summary["timeline"])
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = ["| config | P@5 | R@10 | MRR@10 | nDCG@10 | human P@5 | human nDCG@10 | judged@5 | latency (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, s in summary.items():
        if key == "timeline":
            continue
        p, h = s.get("prog", {}), s.get("human", {})
        lines.append(f"| {key} | {p.get('P@5', '-')} | {p.get('R@10', '-')} | {p.get('MRR@10', '-')} | "
                     f"{p.get('nDCG@10', '-')} | {h.get('P@5', '-')} | {h.get('nDCG@10', '-')} | "
                     f"{s.get('judged@5', '-')} | {s['latency_ms_mean']:.0f} |")
    if "timeline" in summary:
        lines += ["", "| timeline | date precision@10 | date & need precision@10 |", "|---|---|---|"]
        for k, v in summary["timeline"].items():
            lines.append(f"| {k} | {v['date_precision@10']} | {v['date_and_need_precision@10']} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
