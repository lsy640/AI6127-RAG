"""
评测集去重（作业要求评测数据集不含重复）

规则（按文件中的先后顺序保留第一次出现）：
  场景      隐藏需求（品类、人群/年龄/材质、预算）完全相同的场景只保留一个
  决策记录  属于保留场景，且对话文本完全相同的只保留一个（如多个场景都以 "I need a toy." 开场）
  相关性记录 属于保留场景，且（需求, 商品）相同的只保留一个
各评测脚本（compute_iaa / evaluate_decisions / eval_retrieval / run_dialogues / rag_judge）
都只在去重后的集合上计算指标。

用法：python dedup.py      → 打印统计并写出 eval/annotation/eval_set_unique.json
"""

import json
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCENARIOS = BASE_DIR / "eval" / "scenarios" / "scenarios.jsonl"
ANN_DIR = BASE_DIR / "eval" / "annotation"


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@lru_cache(maxsize=None)
def unique_scenario_ids():
    seen, keep = set(), []
    for s in _jsonl(SCENARIOS):
        key = json.dumps(s["hidden_need"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            keep.append(s["scenario_id"])
    return tuple(keep)


@lru_cache(maxsize=None)
def unique_decision_ids():
    scen, seen, keep = set(unique_scenario_ids()), set(), []
    for r in _jsonl(ANN_DIR / "decisions.jsonl"):
        if r["scenario_id"] in scen and r["dialogue"] not in seen:
            seen.add(r["dialogue"])
            keep.append(r["record_id"])
    return tuple(keep)


@lru_cache(maxsize=None)
def unique_relevance_ids():
    scen, seen, keep = set(unique_scenario_ids()), set(), []
    for r in _jsonl(ANN_DIR / "pool.jsonl"):
        key = (r["need_en"], r["parent_asin"])
        if r["scenario_id"] in scen and key not in seen:
            seen.add(key)
            keep.append(r["record_id"])
    return tuple(keep)


def main():
    out = {"scenarios": list(unique_scenario_ids()),
           "decision_records": list(unique_decision_ids()),
           "relevance_records": list(unique_relevance_ids())}
    counts = {k: len(v) for k, v in out.items()}
    counts["total_records"] = counts["decision_records"] + counts["relevance_records"]
    out["counts"] = counts
    (ANN_DIR / "eval_set_unique.json").write_text(json.dumps(out, indent=1))
    print(counts)


if __name__ == "__main__":
    main()
