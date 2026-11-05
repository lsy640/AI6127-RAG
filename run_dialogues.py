"""
Q4/Q5：用用户模拟器在评测场景上跑多轮对话，比较各 Agent 配置（消融实验）

用法：
    python run_dialogues.py --configs B0 B1 B2 B3 B4 full full-I1 full-I2 full-I3 --workers 6
    python run_dialogues.py --configs full --limit 10            # 冒烟测试
    python run_dialogues.py --resummarize                         # 不重跑，按现有记录重新计算汇总（如修改了价格）
输出（eval/q5_runs/）：
    <config>.jsonl     每段对话的完整记录（轮次、提问、推荐、证据、耗时、token）
    summary.json/.md   各配置的指标汇总
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import llm
from agent import CONFIGS, Agent
from simulator import run_dialogue

BASE_DIR = Path(__file__).resolve().parent
SCENARIOS = BASE_DIR / "eval" / "scenarios" / "scenarios.jsonl"
OUT_DIR = BASE_DIR / "eval" / "q5_runs"

# 价格：美元 / 百万 token（deepseek-flash，https://api-docs.deepseek.com/quick_start/pricing，2026-11）
# 高峰时段（UTC 周一至周五 01:00–04:00、06:00–10:00）为下表的 2 倍；本项目实验均在非高峰时段运行。
PRICES = {
    "offpeak": {"in_hit": 0.003, "in_miss": 0.15, "out": 0.60},
    "peak": {"in_hit": 0.006, "in_miss": 0.30, "out": 1.20},
}
PRICE_TIER = os.getenv("DEEPSEEK_PRICE_TIER", "offpeak")
# 早期记录没有保存缓存命中 token 数，按 DeepSeek 账单上整体约 40% 的缓存命中率估算
ASSUMED_CACHE_HIT_RATE = float(os.getenv("ASSUMED_CACHE_HIT_RATE", "0.4"))


def dialogue_cost(u, tier=PRICE_TIER):
    p = PRICES[tier]
    pin, pout = u["prompt_tokens"], u["completion_tokens"]
    hit = u["cache_hit_tokens"] if "cache_hit_tokens" in u else pin * ASSUMED_CACHE_HIT_RATE
    return (hit * p["in_hit"] + (pin - hit) * p["in_miss"] + pout * p["out"]) / 1e6


def load_scenarios(path, limit=None, ids=None):
    with open(path, encoding="utf-8") as f:
        scs = [json.loads(l) for l in f if l.strip()]
    import dedup                                          # 隐藏需求重复的场景只保留一个
    keep = set(dedup.unique_scenario_ids())
    scs = [s for s in scs if s["scenario_id"] in keep]
    if ids:
        scs = [s for s in scs if s["scenario_id"] in ids]
    return scs[:limit] if limit else scs


def one(config_name, sc, max_turns, k):
    usage = llm.Usage()
    agent = Agent(CONFIGS[config_name], usage=usage)
    t0 = time.perf_counter()
    rec = run_dialogue(agent, sc, max_turns=max_turns, k=k)
    rec["wall_seconds"] = round(time.perf_counter() - t0, 2)
    rec["usage"] = usage.snapshot()
    return rec


def summarize(records):
    ok = [r for r in records if not r["error"]]
    n = len(ok) or 1
    lat = [t["debug"].get("latency_ms", 0) for r in ok for t in r["turns"]]
    calls = [r["usage"]["calls"] for r in ok]
    pin = [r["usage"]["prompt_tokens"] for r in ok]
    pout = [r["usage"]["completion_tokens"] for r in ok]
    complete = [r for r in ok if r["level"] == "complete"]
    unclear = [r for r in ok if r["level"] != "complete"]
    questions = [q for r in ok for q in r["questions"]]
    sc = [t["debug"].get("self_check") for r in ok for t in r["turns"] if t["debug"].get("self_check")]
    cost = [dialogue_cost(r["usage"]) for r in ok]
    cost_peak_nocache = [dialogue_cost({**r["usage"], "cache_hit_tokens": 0}, "peak") for r in ok]
    measured_hits = [r["usage"] for r in ok if "cache_hit_tokens" in r["usage"]]
    return {
        "dialogues": len(records),
        "errors": len(records) - len(ok),
        "success_rate": sum(r["success"] for r in ok) / n,
        "first_recommendation_success": sum(bool(r["first_rec_success"]) for r in ok) / n,
        "success_rate_unclear_openings": (sum(r["success"] for r in unclear) / len(unclear)) if unclear else None,
        "success_rate_complete_openings": (sum(r["success"] for r in complete) / len(complete)) if complete else None,
        "avg_turns": statistics.mean(r["n_turns"] for r in ok) if ok else None,
        "avg_turns_to_success": statistics.mean(r["success_turn"] for r in ok if r["success"]) if any(r["success"] for r in ok) else None,
        "avg_questions": statistics.mean(r["n_questions"] for r in ok) if ok else None,
        "useful_question_rate": (sum(q["useful"] for q in questions) / len(questions)) if questions else None,
        "over_asking_rate_complete": (sum(1 for r in complete if r["n_questions"] > 0) / len(complete)) if complete else None,
        "constraint_violation_rate": (sum(r["n_violating"] for r in ok) / max(1, sum(r["n_recommendations"] for r in ok))),
        "no_match_responses": sum(r["none_responses"] for r in ok),
        "self_check_supported_claim_rate": (sum(x["supported"] for x in sc) / max(1, sum(x["claims"] for x in sc))) if sc else None,
        "turn_latency_ms_p50": statistics.median(lat) if lat else None,
        "turn_latency_ms_p95": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
        "llm_calls_per_dialogue": statistics.mean(calls) if calls else None,
        "tokens_per_dialogue": statistics.mean(a + b for a, b in zip(pin, pout)) if pin else None,
        "cost_usd_per_1000_dialogues": statistics.mean(cost) * 1000 if cost else None,
        "cost_usd_per_1000_dialogues_list_price_no_cache": statistics.mean(cost_peak_nocache) * 1000 if cost else None,
        "cost_basis": f"{PRICE_TIER} price; cache hits " + (
            "measured" if len(measured_hits) == len(ok) else f"assumed {ASSUMED_CACHE_HIT_RATE:.0%}"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["B0", "B1", "B2", "B3", "B4", "full"])
    parser.add_argument("--scenarios", default=str(SCENARIOS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resummarize", action="store_true", help="只根据已有的 <config>.jsonl 重新计算汇总")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"summary{args.tag}.json"
    if args.resummarize:
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        for name in list(summary):
            path = OUT_DIR / f"{name}{args.tag}.jsonl"
            if path.exists():
                import dedup
                keep = set(dedup.unique_scenario_ids())
                records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
                records = [r for r in records if r["scenario_id"] in keep]
                wall = summary[name].get("wall_minutes")
                summary[name] = summarize(records)
                summary[name]["wall_minutes"] = wall
        summary_path.write_text(json.dumps(summary, indent=2))
        write_markdown(summary, args.tag)
        return

    scenarios = load_scenarios(args.scenarios, args.limit, args.ids)
    # 预热检索与重排序模型
    import hybrid_retriever as hr
    hr.search("warm up", top_k=3, rerank=True)
    import reranker
    reranker.warm_up()

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in args.configs:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            records = list(ex.map(lambda sc: one(name, sc, args.max_turns, args.k), scenarios))
        with open(OUT_DIR / f"{name}{args.tag}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        summary[name] = summarize(records)
        summary[name]["wall_minutes"] = round((time.time() - t0) / 60, 1)
        print(f"[{name}] success {summary[name]['success_rate']:.3f} | questions {summary[name]['avg_questions']:.2f} "
              f"| violations {summary[name]['constraint_violation_rate']:.3f} | errors {summary[name]['errors']} "
              f"| {summary[name]['wall_minutes']} min", flush=True)
        summary_path.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, args.tag)


def write_markdown(summary, tag=""):
    cols = ["success_rate", "first_recommendation_success", "avg_turns", "avg_questions",
            "useful_question_rate", "over_asking_rate_complete", "constraint_violation_rate",
            "self_check_supported_claim_rate", "llm_calls_per_dialogue", "turn_latency_ms_p50",
            "cost_usd_per_1000_dialogues"]
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| config | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, s in summary.items():
        lines.append(f"| {name} | " + " | ".join(fmt(s.get(c)) for c in cols) + " |")
    (OUT_DIR / f"summary{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
