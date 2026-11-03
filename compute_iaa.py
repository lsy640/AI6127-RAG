"""
Q4: Inter-annotator agreement and gold labels.

Reads eval/annotation/sheet_*.xlsx (one per annotator, filled with 1/0),
then reports:
    - per-annotator label counts and missing labels
    - pairwise observed agreement and Cohen's kappa
    - Fleiss' kappa (3+ annotators)
and builds the gold standard:
    - 3+ annotators: majority vote (ties -> adjudication)
    - 2 annotators: agreed labels; disagreements -> adjudication.csv

If eval/annotation/adjudication.csv has a filled `final_label` column,
those labels are used for the disputed records.

Outputs:
    eval/annotation/iaa_report.json
    eval/annotation/adjudication.csv
    eval/annotation/gold_labels.csv

Usage:
    python compute_iaa.py                 # all sheet_*.xlsx
    python compute_iaa.py --sheets sheet_A.xlsx sheet_B.xlsx
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
POOL_FILE = ANN_DIR / "pool.jsonl"
ADJ_FILE = ANN_DIR / "adjudication.csv"
GOLD_FILE = ANN_DIR / "gold_labels.csv"
REPORT_FILE = ANN_DIR / "iaa_report.json"

THRESHOLD = 0.80


def parse_label(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"1", "1.0", "True", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "False", "false", "no", "n"}:
        return 0
    return None


def load_sheet(path):
    """Return {pair_id: label or None}."""
    ws = load_workbook(path, read_only=True)["labels"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(rows)]
    pid_col = header.index("pair_id")
    lab_col = [i for i, h in enumerate(header) if h.startswith("relevant")][0]
    return {
        str(row[pid_col]): parse_label(row[lab_col])
        for row in rows
        if row and row[pid_col]
    }


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(matrix):
    """matrix: list of [n_zero, n_one] per item, same rater count."""
    N = len(matrix)
    if N == 0:
        return None
    n = sum(matrix[0])
    p_j = [sum(row[j] for row in matrix) / (N * n) for j in range(2)]
    P_i = [
        (sum(c * c for c in row) - n) / (n * (n - 1))
        for row in matrix
    ]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)
    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def load_adjudication():
    decided = {}
    if not ADJ_FILE.exists():
        return decided
    with open(ADJ_FILE, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            label = parse_label(row.get("final_label"))
            if label is not None:
                decided[row["pair_id"]] = label
    return decided


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", nargs="*")
    args = parser.parse_args()

    if args.sheets:
        paths = [Path(p) if Path(p).is_absolute() else ANN_DIR / p
                 for p in args.sheets]
    else:
        paths = sorted(ANN_DIR.glob("sheet_*.xlsx"))

    if len(paths) < 2:
        raise SystemExit("Need at least two annotator sheets.")

    names = [p.stem.replace("sheet_", "") for p in paths]
    sheets = {name: load_sheet(p) for name, p in zip(names, paths)}

    pool = {}
    with open(POOL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pool[row["pair_id"]] = row

    report = {"annotators": names, "records": len(pool), "per_annotator": {}}

    for name, labels in sheets.items():
        values = [labels.get(pid) for pid in pool]
        report["per_annotator"][name] = {
            "labeled": sum(v is not None for v in values),
            "missing": sum(v is None for v in values),
            "relevant": sum(v == 1 for v in values),
            "not_relevant": sum(v == 0 for v in values),
        }

    # Pairwise agreement and Cohen's kappa.
    pairwise = {}
    for x, y in itertools.combinations(names, 2):
        ids = [pid for pid in pool
               if sheets[x].get(pid) is not None
               and sheets[y].get(pid) is not None]
        a = [sheets[x][pid] for pid in ids]
        b = [sheets[y][pid] for pid in ids]
        agree = sum(i == j for i, j in zip(a, b))
        pairwise[f"{x}-{y}"] = {
            "items": len(ids),
            "percent_agreement": agree / len(ids) if ids else None,
            "cohen_kappa": cohen_kappa(a, b),
        }
    report["pairwise"] = pairwise

    agreements = [v["percent_agreement"] for v in pairwise.values()
                  if v["percent_agreement"] is not None]
    kappas = [v["cohen_kappa"] for v in pairwise.values()
              if v["cohen_kappa"] is not None]
    report["mean_pairwise_agreement"] = (
        sum(agreements) / len(agreements) if agreements else None)
    report["mean_cohen_kappa"] = sum(kappas) / len(kappas) if kappas else None

    complete = [pid for pid in pool
                if all(sheets[n].get(pid) is not None for n in names)]
    report["items_labeled_by_all"] = len(complete)
    report["all_annotators_agree"] = (
        sum(len({sheets[n][pid] for n in names}) == 1 for pid in complete)
        / len(complete) if complete else None)

    if len(names) >= 3 and complete:
        matrix = []
        for pid in complete:
            votes = [sheets[n][pid] for n in names]
            matrix.append([votes.count(0), votes.count(1)])
        report["fleiss_kappa"] = fleiss_kappa(matrix)

    # Gold labels.
    adjudicated = load_adjudication()
    gold, disputes = {}, []

    for pid, row in pool.items():
        votes = [sheets[n].get(pid) for n in names]
        valid = [v for v in votes if v is not None]
        counts = Counter(valid)

        if pid in adjudicated:
            gold[pid] = (adjudicated[pid], "adjudicated")
        elif valid and len(counts) == 1 and len(valid) >= 2:
            gold[pid] = (valid[0], "unanimous")
        elif len(valid) >= 3 and counts.most_common(1)[0][1] > len(valid) / 2:
            gold[pid] = (counts.most_common(1)[0][0], "majority")
        else:
            disputes.append((pid, row, votes))

    with open(ADJ_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "query", "parent_asin", "passage_head",
                         *[f"label_{n}" for n in names], "final_label"])
        for pid, row, votes in disputes:
            writer.writerow([
                pid, row["query"], row["parent_asin"],
                row["passage"][:300],
                *["" if v is None else v for v in votes],
                "",
            ])

    with open(GOLD_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "case_id", "parent_asin", "rank",
                         "label", "source"])
        for pid, (label, source) in gold.items():
            row = pool[pid]
            writer.writerow([pid, row["case_id"], row["parent_asin"],
                             row["rank"], label, source])

    report["gold"] = {
        "labeled": len(gold),
        "unresolved_disputes": len(disputes),
        "relevant": sum(v[0] == 1 for v in gold.values()),
        "not_relevant": sum(v[0] == 0 for v in gold.values()),
        "by_source": dict(Counter(v[1] for v in gold.values())),
    }
    report["meets_80_percent_agreement"] = (
        report["mean_pairwise_agreement"] is not None
        and report["mean_pairwise_agreement"] >= THRESHOLD)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nGold labels: {GOLD_FILE}")
    if disputes:
        print(f"{len(disputes)} records need adjudication: {ADJ_FILE}")


if __name__ == "__main__":
    main()
