"""
Q4：标注一致性（IAA）与金标准

读取 eval/annotation/sheet_*.xlsx（每位标注者一份，两个工作表）：
  decisions：should_ask（1/0）与 ask_about（品类 / 适用人群 / 孩子年龄 / 预算 / 材质）
  relevance：relevant（1/0）
对三种标签分别报告：两两一致率、Cohen's κ、Fleiss' κ（≥3 人）。
金标准：≥3 人多数票；2 人时取一致的标签，分歧写入 adjudication_*.csv，
填好 final_label 后再次运行即可合并。
复查：若存在 review_<annotator>.xlsx（标注者按指南复查自己的部分 relevance 标签，
列 record_id 与 revised relevant (1/0)），其标签覆盖该标注者的原标签；
不加 --no-review 时报告的是复查后的一致率。

输出：iaa_report.json、gold_decisions.csv、gold_relevance.csv、adjudication_decisions.csv、
      adjudication_relevance.csv
用法：python compute_iaa.py            （或 --sheets sheet_A.xlsx sheet_B.xlsx）
"""

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

BASE_DIR = Path(__file__).resolve().parent
ANN_DIR = BASE_DIR / "eval" / "annotation"
THRESHOLD = 0.80
SLOT_CODES = {"product type": "product_type", "who it is for": "audience", "child age": "age",
              "budget": "price_max", "material": "material"}


def parse_binary(v):
    t = "" if v is None else str(v).strip().lower()
    return 1 if t in {"1", "1.0", "true", "yes", "y"} else 0 if t in {"0", "0.0", "false", "no", "n"} else None


def load_review(path):
    """复查表：record_id → 修正后的 relevant（按表头定位列）。"""
    ws = load_workbook(path, read_only=True)["review"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h or "").strip().lower() for h in next(rows)]
    col = next(i for i, h in enumerate(header) if h.startswith("revised"))
    return {str(r[0]): parse_binary(r[col]) for r in rows if r and r[0] and parse_binary(r[col]) is not None}


def load_sheet(path):
    wb = load_workbook(path, read_only=True)
    dec, rel = {}, {}
    rows = wb["decisions"].iter_rows(values_only=True)
    next(rows)
    for r in rows:
        if r and r[0]:
            ask = parse_binary(r[3])
            slot = SLOT_CODES.get(str(r[4]).strip().lower()) if r[4] else None
            dec[str(r[0])] = (ask, slot if ask == 1 else None)
    rows = wb["relevance"].iter_rows(values_only=True)
    next(rows)
    for r in rows:
        if r and r[0]:
            rel[str(r[0])] = parse_binary(r[3])
    return dec, rel


def cohen_kappa(a, b):
    """多类别 Cohen's κ。"""
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(items):
    """items: 每个条目的标签列表（标注者数相同）。"""
    if not items:
        return None
    cats = sorted({x for it in items for x in it})
    n, N = len(items[0]), len(items)
    if n < 2:
        return None
    counts = [[it.count(c) for c in cats] for it in items]
    p_j = [sum(row[j] for row in counts) / (N * n) for j in range(len(cats))]
    P_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    return 1.0 if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def agreement(labels_by_ann, ids):
    """labels_by_ann: {annotator: {id: label}}；只统计所有人都标了的条目。"""
    names = list(labels_by_ann)
    report = {"pairwise": {}}
    for x, y in itertools.combinations(names, 2):
        common = [i for i in ids if labels_by_ann[x].get(i) is not None and labels_by_ann[y].get(i) is not None]
        a = [labels_by_ann[x][i] for i in common]
        b = [labels_by_ann[y][i] for i in common]
        report["pairwise"][f"{x}-{y}"] = {
            "items": len(common),
            "percent_agreement": (sum(p == q for p, q in zip(a, b)) / len(common)) if common else None,
            "cohen_kappa": cohen_kappa(a, b),
        }
    pa = [v["percent_agreement"] for v in report["pairwise"].values() if v["percent_agreement"] is not None]
    ks = [v["cohen_kappa"] for v in report["pairwise"].values() if v["cohen_kappa"] is not None]
    report["mean_percent_agreement"] = sum(pa) / len(pa) if pa else None
    report["mean_cohen_kappa"] = sum(ks) / len(ks) if ks else None
    complete = [i for i in ids if all(labels_by_ann[n].get(i) is not None for n in names)]
    report["items_labelled_by_all"] = len(complete)
    if len(names) >= 3:
        report["fleiss_kappa"] = fleiss_kappa([[labels_by_ann[n][i] for n in names] for i in complete])
    report["meets_80_percent"] = report["mean_percent_agreement"] is not None and report["mean_percent_agreement"] >= THRESHOLD
    return report


def gold_labels(labels_by_ann, ids, adjudication_file):
    decided = {}
    if adjudication_file.exists():
        with open(adjudication_file, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("final_label", "").strip() != "":
                    decided[row["record_id"]] = row["final_label"].strip()
    gold, disputes = {}, []
    names = list(labels_by_ann)
    for i in ids:
        votes = [labels_by_ann[n].get(i) for n in names]
        valid = [v for v in votes if v is not None]
        c = Counter(valid)
        if i in decided:
            gold[i] = (decided[i], "adjudicated")
        elif len(valid) >= 2 and len(c) == 1:
            gold[i] = (valid[0], "unanimous")
        elif len(valid) >= 3 and c.most_common(1)[0][1] > len(valid) / 2:
            gold[i] = (c.most_common(1)[0][0], "majority")
        else:
            disputes.append((i, votes))
    with open(adjudication_file, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["record_id", *[f"label_{n}" for n in names], "final_label"])
        for i, votes in disputes:
            w.writerow([i, *["" if v is None else v for v in votes], ""])
    return gold, disputes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", nargs="*")
    parser.add_argument("--no-review", action="store_true", help="忽略 review_<annotator>.xlsx（第一轮原始标签）")
    args = parser.parse_args()
    paths = ([ANN_DIR / p for p in args.sheets] if args.sheets else sorted(ANN_DIR.glob("sheet_*.xlsx")))
    if len(paths) < 2:
        raise SystemExit("Need at least two annotator sheets.")
    names = [p.stem.replace("sheet_", "") for p in paths]
    sheets = {n: load_sheet(p) for n, p in zip(names, paths)}
    reviewed = {}
    for n in names:
        review_path = ANN_DIR / f"review_{n}.xlsx"
        if review_path.exists() and not args.no_review:
            rev = load_review(review_path)
            changed = sum(1 for i, v in rev.items() if sheets[n][1].get(i) != v)
            sheets[n][1].update(rev)
            reviewed[n] = {"reviewed": len(rev), "changed": changed}

    dec_ids = [json.loads(l)["record_id"] for l in open(ANN_DIR / "decisions.jsonl", encoding="utf-8")]
    rel_ids = [json.loads(l)["record_id"] for l in open(ANN_DIR / "pool.jsonl", encoding="utf-8")]

    ask = {n: {i: v[0] for i, v in sheets[n][0].items()} for n in names}
    slot = {n: {i: v[1] for i, v in sheets[n][0].items() if v[0] == 1 and v[1]} for n in names}
    rel = {n: sheets[n][1] for n in names}

    report = {
        "annotators": names,
        "review": reviewed,
        "records": {"decisions": len(dec_ids), "relevance": len(rel_ids),
                    "total": len(dec_ids) + len(rel_ids)},
        "labelled_per_annotator": {n: {"decisions": sum(v is not None for v in ask[n].values()),
                                       "relevance": sum(v is not None for v in rel[n].values())} for n in names},
        "should_ask": agreement(ask, dec_ids),
        "ask_about": agreement(slot, dec_ids),
        "relevance": agreement(rel, rel_ids),
    }
    all_ids = dec_ids + rel_ids
    combined = {n: {**{i: ask[n].get(i) for i in dec_ids}, **{i: rel[n].get(i) for i in rel_ids}} for n in names}
    report["all_binary_records"] = agreement(combined, all_ids)

    gold_ask, disp_a = gold_labels(ask, dec_ids, ANN_DIR / "adjudication_decisions.csv")
    ask_ids = [i for i in dec_ids if str(gold_ask.get(i, ("", ""))[0]) == "1"]
    gold_slot, disp_s = gold_labels(slot, ask_ids, ANN_DIR / "adjudication_slots.csv")
    gold_rel, disp_r = gold_labels(rel, rel_ids, ANN_DIR / "adjudication_relevance.csv")
    with open(ANN_DIR / "gold_decisions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "should_ask", "ask_about", "source"])
        for i, (v, src) in gold_ask.items():
            w.writerow([i, v, gold_slot.get(i, ("", ""))[0], src])
    with open(ANN_DIR / "gold_relevance.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["record_id", "relevant", "source"])
        for i, (v, src) in gold_rel.items():
            w.writerow([i, v, src])
    report["gold"] = {"decisions": len(gold_ask), "relevance": len(gold_rel),
                      "unresolved_decisions": len(disp_a), "unresolved_relevance": len(disp_r),
                      "slots": len(gold_slot), "unresolved_slots": len(disp_s)}

    (ANN_DIR / "iaa_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("records", "gold")}, indent=2))
    for key in ("should_ask", "ask_about", "relevance", "all_binary_records"):
        r = report[key]
        print(f"{key:20s} agreement {r['mean_percent_agreement']}  cohen {r['mean_cohen_kappa']}  "
              f"fleiss {r.get('fleiss_kappa')}  meets80 {r['meets_80_percent']}")


if __name__ == "__main__":
    main()
