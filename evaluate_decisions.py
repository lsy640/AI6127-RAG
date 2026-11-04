"""
Q4：澄清决策的分类指标（“该问还是该推荐”的 P/R/F1，以及“问什么”的准确率）

在人工标注的对话快照（eval/annotation/decisions.jsonl）上重放 Agent 的决策：
  clarify-rule          必填槽位缺失就问
  clarify-llm           LLM 判断（状态追踪里的 should_ask）
  clarify-llm+thinking  同上，开启 DeepSeek 思考模式
  full                  候选分布熵（I1）
金标准：gold_decisions.csv（人工）；尚无人工标签时退回规则推导的 expected_action，
并在结果中标注 provisional。正类 = “该问”。

用法：python evaluate_decisions.py --configs clarify-rule clarify-llm clarify-llm+thinking full
"""

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm
from agent import CONFIGS, Agent

BASE_DIR = Path(__file__).resolve().parent
ANN_DIR = BASE_DIR / "eval" / "annotation"
OUT_DIR = BASE_DIR / "eval" / "q4_decisions"


def load_gold():
    path = ANN_DIR / "gold_decisions.csv"
    if not path.exists():
        return None
    gold = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gold[row["record_id"]] = (int(float(row["should_ask"])), row.get("ask_about") or None)
    return gold


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def decide_one(cfg_name, rec):
    usage = llm.Usage()
    agent = Agent(CONFIGS[cfg_name], usage=usage)
    history = [dict(t) for t in rec["turns"]]
    t0 = time.perf_counter()
    state = agent.track_state(history)
    candidates, _, _ = agent.retrieve(state, max(agent.cfg.candidates, 10))
    action, slot, _, _ = agent.decide(state, history, candidates)
    return {"record_id": rec["record_id"], "pred_ask": int(action == "ask"), "pred_slot": slot,
            "latency_ms": (time.perf_counter() - t0) * 1000, "calls": usage.calls,
            "tokens": usage.prompt_tokens + usage.completion_tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["clarify-rule", "clarify-llm", "clarify-llm+thinking", "full"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(ANN_DIR / "decisions.jsonl", encoding="utf-8")]
    records = records[:args.limit] if args.limit else records
    gold = load_gold()
    provisional = gold is None
    if provisional:
        gold = {r["record_id"]: (int(r["expected_action"] == "ask"),
                                 r["missing_required"][0] if r["missing_required"] else None) for r in records}
    import hybrid_retriever as hr
    hr.search("warm up", top_k=3, rerank=True)
    import reranker
    reranker.warm_up()

    summary = {}
    for name in args.configs:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            preds = list(ex.map(lambda r: decide_one(name, r), records))
        tp = fp = fn = tn = slot_ok = slot_n = 0
        for p in preds:
            g_ask, g_slot = gold.get(p["record_id"], (None, None))
            if g_ask is None:
                continue
            tp += p["pred_ask"] and g_ask
            fp += p["pred_ask"] and not g_ask
            fn += (not p["pred_ask"]) and g_ask
            tn += (not p["pred_ask"]) and not g_ask
            if p["pred_ask"] and g_ask and g_slot:
                slot_n += 1
                slot_ok += p["pred_slot"] == g_slot
        prec, rec, f1 = prf(tp, fp, fn)
        n = tp + fp + fn + tn
        summary[name] = {
            "gold": "rule-derived (provisional)" if provisional else "human",
            "records": n, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / n if n else None,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "ask_slot_accuracy": slot_ok / slot_n if slot_n else None,
            "latency_ms_mean": sum(p["latency_ms"] for p in preds) / len(preds),
            "llm_calls_per_decision": sum(p["calls"] for p in preds) / len(preds),
            "tokens_per_decision": sum(p["tokens"] for p in preds) / len(preds),
        }
        with open(OUT_DIR / f"preds_{name}.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        s = summary[name]
        print(f"[{name}] P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} "
              f"acc {s['accuracy']:.3f} slot-acc {s['ask_slot_accuracy']} ({s['gold']})", flush=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
