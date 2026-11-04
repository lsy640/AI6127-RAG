"""
Q4：按“用户场景”生成人工标注集（≥ 1,000 条记录）

输入：eval/scenarios/scenarios.jsonl（build_scenarios.py 生成，180 个场景）
两类标注记录：
  1. decisions  每个场景 2 个对话快照 × 2 个标签
                  should_ask：此时助手应该反问（1）还是直接推荐（0）
                  ask_about ：如果该问，最该问哪个属性（品类 / 适用人群 / 孩子年龄 / 预算 / 材质）
  2. relevance  每个场景的候选商品是否满足用户的完整需求（1/0）
                候选来自多种检索配置的 Top 结果按名次轮流合并（pooling），
                这样 Q3 的不同检索配置都能在同一批人工标签上公平比较。

输出（eval/annotation/）：
  decisions.jsonl / pool.jsonl / pool_meta.json
  sheet_<annotator>.xlsx（两个工作表：decisions、relevance；不显示任何参考答案）

用法：python build_annotation_set.py --annotators A B C --pool-size 6
"""

import argparse
import json
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

import attributes as A
from generator import build_evidence_passage

BASE_DIR = Path(__file__).resolve().parent
SCENARIOS = BASE_DIR / "eval" / "scenarios" / "scenarios.jsonl"
OUT_DIR = BASE_DIR / "eval" / "annotation"

SLOT_LABELS = {"product_type": "product type", "audience": "who it is for",
               "age": "child age", "price_max": "budget", "material": "material"}

# 参与 pooling 的检索配置：(名称, 查询语言, search() 参数, 是否使用隐藏需求的约束作过滤)
POOL_CONFIGS = [
    ("rrf+rerank+filters", "en", dict(fusion="rrf", rerank=True), True),
    ("rrf+rerank", "en", dict(fusion="rrf", rerank=True), False),
    ("rrf", "en", dict(fusion="rrf"), False),
    ("bm25", "en", dict(fusion="bm25"), False),
    ("dense", "en", dict(fusion="dense"), False),
    ("dense-zh", "zh", dict(fusion="dense"), False),
    ("rrf+rerank-zh", "zh", dict(fusion="rrf", rerank=True), False),
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def dialogue_text(turns):
    return "\n".join(f"{'USER' if t['role'] == 'user' else 'ASSISTANT'}: {t['text']}" for t in turns)


def build_pool(scenarios, pool_size, depth=10):
    import hybrid_retriever as hr
    import reranker
    reranker.warm_up()
    pool, runs = [], {}
    for n, sc in enumerate(scenarios, start=1):
        if n % 20 == 0:
            print(f"  pooled {n}/{len(scenarios)} scenarios", flush=True)
        ranked = []
        for name, lang, kwargs, use_filters in POOL_CONFIGS:
            query = sc["full_query_en"] if lang == "en" else sc["full_query_zh"]
            filters = A.filters_for(sc["hidden_need"]) if use_filters else None
            res, _ = hr.search(query, top_k=depth, filters=filters, **kwargs)
            ids = [r["parent_asin"] for r in res]
            runs.setdefault(name, {})[sc["scenario_id"]] = ids
            ranked.append((name, ids, {r["parent_asin"]: r for r in res}))
        chosen, found_by = [], {}
        for rank in range(depth):                       # 按名次轮流合并
            for name, ids, _ in ranked:
                if rank < len(ids):
                    found_by.setdefault(ids[rank], []).append(f"{name}@{rank + 1}")
                    if ids[rank] not in chosen and len(chosen) < pool_size:
                        chosen.append(ids[rank])
        passages = {}
        for _, _, by_id in ranked:
            for asin, r in by_id.items():
                passages.setdefault(asin, r["passage"])
        for i, asin in enumerate(chosen, start=1):
            pool.append({
                "record_id": f"{sc['scenario_id']}-R{i}",
                "scenario_id": sc["scenario_id"],
                "parent_asin": asin,
                "need_en": sc["full_query_en"],
                "need_zh": sc["full_query_zh"],
                "found_by": found_by.get(asin, []),
                "programmatic_relevant": A.satisfies(asin, sc["hidden_need"])[0],
                "passage": passages[asin],
            })
    return pool, runs


def decision_records(scenarios):
    out = []
    for sc in scenarios:
        spec = A.SLOT_SPECS[sc["domain"]]
        for i, snap in enumerate(sc["snapshots"], start=1):
            out.append({
                "record_id": f"{sc['scenario_id']}-D{i}",
                "scenario_id": sc["scenario_id"],
                "domain": sc["domain"],
                "dialogue": dialogue_text(snap["turns"]),
                "turns": snap["turns"],
                "allowed_slots": spec["askable"],
                "expected_action": snap["expected_action"],         # 规则推导，仅供参考，不给标注者
                "missing_required": snap["missing_required"],
            })
    return out


def _sheet(ws, headers, widths):
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
    for col, w in zip("ABCDEFGH", widths):
        ws.column_dimensions[col].width = w


def write_sheet(path, decisions, relevance, annotator):
    wb = Workbook()
    ws = wb.active
    ws.title = "decisions"
    _sheet(ws, ["record_id", "domain", "dialogue", "should_ask (1=ask, 0=recommend)",
                "ask_about (if 1)", "comment"], [12, 13, 70, 16, 18, 24])
    for d in decisions:
        ws.append([d["record_id"], d["domain"], d["dialogue"], None, None, None])
    dv = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"D2:D{len(decisions) + 1}")
    dv2 = DataValidation(type="list", formula1='"product type,who it is for,child age,budget,material"',
                         allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(f"E2:E{len(decisions) + 1}")

    ws2 = wb.create_sheet("relevance")
    _sheet(ws2, ["record_id", "user need", "product (with price and rating)", "relevant (1/0)", "comment"],
           [12, 38, 100, 14, 24])
    for r in relevance:
        ws2.append([r["record_id"], f"{r['need_en']}\n{r['need_zh']}",
                    build_evidence_passage(r["parent_asin"], r["passage"]), None, None])
    dv3 = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    ws2.add_data_validation(dv3)
    dv3.add(f"D2:D{len(relevance) + 1}")

    for sheet in (ws, ws2):
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.freeze_panes = "D2"
    info = wb.create_sheet("info")
    info.append(["annotator", annotator])
    info.append(["decision records", len(decisions)])
    info.append(["relevance records", len(relevance)])
    info.append(["guidelines", "eval/annotation/GUIDELINES.md"])
    wb.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotators", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sheets-only", action="store_true",
                        help="只根据已有的 decisions.jsonl / pool.jsonl 重新导出标注表")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_jsonl(SCENARIOS)

    if args.sheets_only:
        decisions = load_jsonl(OUT_DIR / "decisions.jsonl")
        pool = load_jsonl(OUT_DIR / "pool.jsonl")
    else:
        decisions = decision_records(scenarios)
        pool, runs = build_pool(scenarios, args.pool_size)
        for name, rows in (("decisions.jsonl", decisions), ("pool.jsonl", pool)):
            with open(OUT_DIR / name, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(OUT_DIR / "pool_runs.json", "w", encoding="utf-8") as f:
            json.dump(runs, f)
        import hybrid_retriever as hr
        with open(OUT_DIR / "pool_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "scenarios": len(scenarios),
                "decision_records": len(decisions),
                "relevance_records": len(pool),
                "pool_size_per_scenario": args.pool_size,
                "pool_configs": [c[0] for c in POOL_CONFIGS],
                "retrieval_unit": hr.RETRIEVAL_UNIT,
                "faiss_index": Path(hr.FAISS_INDEX_PATH).name,
                "faiss_built_at": hr.index_meta.get("built_at"),
                "bge_prompt": hr.USE_BGE_PROMPT,
            }, f, indent=2)

    # 标注表中记录顺序在场景内打乱（所有标注者相同），避免检索名次带来的偏差
    rng = random.Random(args.seed)
    by_sc = {}
    for r in pool:
        by_sc.setdefault(r["scenario_id"], []).append(r)
    relevance = []
    for sid in by_sc:
        rows = by_sc[sid][:]
        rng.shuffle(rows)
        relevance.extend(rows)
    for name in args.annotators:
        write_sheet(OUT_DIR / f"sheet_{name}.xlsx", decisions, relevance, name)

    print(f"Scenarios: {len(scenarios)} | decision records: {len(decisions)} | "
          f"relevance records: {len(pool)} | total: {len(decisions) + len(pool)}")
    print(f"Programmatically relevant share in pool: "
          f"{sum(r['programmatic_relevant'] for r in pool) / max(1, len(pool)):.2f}")


if __name__ == "__main__":
    main()
