"""
Q1：知识库构建

数据来源：McAuley Lab Amazon Reviews 2023 商品元数据（三个分类）。

流程
1. 流式读取：逐行读取完整的 .jsonl.gz（不落盘），每个分类扫描全部商品。
2. 过滤：title / features / description / categories / details / price /
   average_rating / rating_number / store 全部非空才保留（清洗后再校验一次）。
3. 随机采样：对每个分类的全部合格商品做蓄水池抽样（reservoir sampling），
   保证每个合格商品被抽中的概率相同，而不是只取文件开头的一段。
4. 清洗：去 HTML 标签与实体、去 emoji 和装饰符号、统一空白；
   规格字段去掉对检索无意义的噪声（Best Sellers Rank、上架日期、型号、
   是否停产等），不再把 Python dict/list 原样写进文本。
5. 去重：parent_asin 去重，标准化标题去重（跨分类）。
6. 切块（字段级）：每个商品切成若干 chunk，每个 chunk 都带商品标题：
     overview    = 标题 + 类目 + 卖点（features）
     description = 标题 + 描述（长描述按 200 词窗口、40 词重叠切分）
     specs       = 标题 + 规格
   检索在 chunk 上进行，再按商品（parent）合并（父文档检索）。
7. 输出（data/）：
     kb.json            商品级知识库：id、text（完整正文，供生成与展示）、meta
     chunks.json        chunk 级检索单元：chunk_id、parent_id、field、text
     kb_pyserini.jsonl  chunk 级语料（Pyserini 格式）
     corpus_stats.json  语料统计（Q1 报告用：文档数、chunk 数、词数、types 等）

用法：
    python dataset.py                 # 完整构建（扫描全部数据，约 15 分钟）
    python dataset.py --max-scan 50000  # 快速测试：每个分类只扫描前 N 条
"""

import argparse
import gzip
import html
import json
import os
import random
import re
import time
from collections import Counter

import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Amazon 2023 元数据根地址（McAuley Lab）
BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"

# 三个分类数据集（文件名 -> 展示名）
CATEGORIES = {
    "Clothing_Shoes_and_Jewelry": "Clothing, Shoes & Jewelry",
    "Home_and_Kitchen": "Home & Kitchen",
    "Toys_and_Games": "Toys & Games",
}

PER_CATEGORY = 5000      # 每类最终保留条数
RESERVOIR_SIZE = 5600    # 蓄水池略大于目标，留出去重损耗
RANDOM_SEED = 42         # 固定随机种子，保证可复现

CHUNK_WORDS = 200        # 每个 chunk 正文的最大词数（不含标题）
CHUNK_OVERLAP = 40       # 描述切分时相邻窗口的重叠词数

# 规格字段中对检索没有语义价值的键（排名、日期、型号、库存状态等）
DROP_DETAIL_KEYS = {
    "Best Sellers Rank",
    "Date First Available",
    "Item model number",
    "Is Discontinued By Manufacturer",
    "ASIN",
    "UPC",
    "Customer Reviews",
    "Batteries",
    "Batteries Required?",
    "Batteries Included?",
    "Domestic Shipping",
    "International Shipping",
}


# ---------------- 字段校验工具 ----------------

def _is_nonempty_str(v):
    return isinstance(v, str) and v.strip() != ""


def _nonempty_strings(lst):
    """从 list 中过滤出非空字符串条目（同时去重，保持顺序）。"""
    if not isinstance(lst, list):
        return []
    seen = set()
    out = []
    for x in lst:
        if isinstance(x, str) and x.strip():
            s = x.strip()
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def _to_price(v):
    """价格：必须是正数（支持带 $ 或逗号的字符串），否则返回 None。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    if isinstance(v, str):
        s = v.strip().replace("$", "").replace(",", "").strip()
        try:
            f = float(s)
        except ValueError:
            return None
        return f if f > 0 else None
    return None


def _is_positive_number(v):
    """数值字段（评分/评论数）：必须为大于 0 的数字。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        try:
            return float(v.strip()) > 0
        except ValueError:
            return False
    return False


def _details_nonempty(details):
    """details 字段：必须是非空 dict，且至少有一个非空值。"""
    if not isinstance(details, dict) or not details:
        return False
    for k, v in details.items():
        if k is None or (isinstance(k, str) and not k.strip()):
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        return True
    return False


def product_valid(product):
    """校验商品：所有字段必须有数据、不能为空（用于过滤）。"""
    if not isinstance(product, dict):
        return False
    return (
        _is_nonempty_str(product.get("parent_asin"))
        and _is_nonempty_str(product.get("title"))
        and bool(_nonempty_strings(product.get("features")))
        and bool(_nonempty_strings(product.get("description")))
        and bool(_nonempty_strings(product.get("categories")))
        and _details_nonempty(product.get("details"))
        and _to_price(product.get("price")) is not None
        and _is_positive_number(product.get("average_rating"))
        and _is_positive_number(product.get("rating_number"))
        and _is_nonempty_str(product.get("store"))
    )


# ---------------- 文本清洗 ----------------

_TAG = re.compile(r"<[^>]+>")
_SYMBOLS = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji、图形符号
    "☀-➿"           # 杂项符号、dingbats（★ ✔ ❤ 等）
    "⬀-⯿"           # 箭头、几何符号
    "■-◿"           # 几何形状（● ■ ◆ ▶ 等）
    "️‍"            # emoji 变体选择符、零宽连接符
    "【】"            # 【 】
    "]+"
)
_SPACES = re.compile(r"\s+")


class CleanStats:
    def __init__(self):
        self.counter = Counter()


CLEAN = CleanStats()


def clean_text(s, field=""):
    """去 HTML、去 emoji/装饰符号、统一空白。"""
    if not isinstance(s, str):
        s = str(s)
    original = s
    s = html.unescape(s)
    if s != original:
        CLEAN.counter[f"html_entities:{field}"] += 1
    if _TAG.search(s):
        CLEAN.counter[f"html_tags:{field}"] += 1
        s = _TAG.sub(" ", s)
    if _SYMBOLS.search(s):
        CLEAN.counter[f"symbols:{field}"] += 1
        s = _SYMBOLS.sub(" ", s)
    s = _SPACES.sub(" ", s).strip(" ;,")
    return s


def clean_list(items, field):
    out, seen = [], set()
    for x in _nonempty_strings(items):
        c = clean_text(x, field)
        key = c.lower()
        if c and key not in seen:
            seen.add(key)
            out.append(c)
    return out


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


def parse_date(value):
    """'October 9, 2015' -> '2015-10-09'；无法解析返回 None。"""
    m = re.match(r"\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", str(value or ""))
    if not m or m.group(1).lower() not in _MONTHS:
        return None
    return f"{int(m.group(3)):04d}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"


def _detail(p, key):
    v = (p.get("details") or {}).get(key)
    if v is None or isinstance(v, (dict, list)):
        return None
    v = clean_text(str(v), "meta")
    return v or None


def clean_details(details):
    """规格：去噪声键、跳过 dict/list 值、Manufacturer 与 Brand 相同时去重。"""
    specs = []
    brand = str(details.get("Brand", "")).strip().lower()
    for k, v in details.items():
        if not isinstance(k, str) or not k.strip():
            continue
        k = k.strip()
        if k in DROP_DETAIL_KEYS:
            CLEAN.counter[f"spec_dropped:{k}"] += 1
            continue
        if isinstance(v, (dict, list)):
            CLEAN.counter["spec_dropped:<dict/list value>"] += 1
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if k == "Manufacturer" and str(v).strip().lower() == brand:
            continue
        v = clean_text(v, "specs")
        if v:
            specs.append(f"{k}: {v}")
    return specs


# ---------------- 商品记录、正文与切块 ----------------

def product_record(p, category):
    """把原始商品转成清洗后的结构化记录；清洗后关键字段为空则返回 None。"""
    rec = {
        "id": p["parent_asin"].strip(),
        "title": clean_text(p["title"], "title"),
        "categories": clean_list(p.get("categories"), "categories"),
        "features": clean_list(p.get("features"), "features"),
        "description": clean_list(p.get("description"), "description"),
        "specs": clean_details(p.get("details") or {}),
        "meta": {
            "parent_asin": p["parent_asin"].strip(),
            "price": _to_price(p.get("price")),
            "average_rating": float(p.get("average_rating")),
            "rating_number": int(float(p.get("rating_number"))),
            "store": clean_text(p.get("store"), "store"),
            "main_category": CATEGORIES[category],
            # 上架日期不进入检索正文（对语义无用），但保留在 meta 中用于时间线过滤
            "date_first_available": parse_date((p.get("details") or {}).get("Date First Available")),
            # 结构化属性：用于元数据过滤、澄清提问（按属性分布选问题）和约束校验
            "categories": clean_list(p.get("categories"), "categories"),
            "brand": _detail(p, "Brand"),
            "material": _detail(p, "Material") or _detail(p, "Material Type"),
            "color": _detail(p, "Color"),
            "recommended_age": _detail(p, "Manufacturer recommended age"),
            "department": _detail(p, "Department"),
        },
    }
    if not (rec["title"] and rec["categories"] and rec["features"]
            and rec["description"] and rec["specs"]):
        return None
    return rec


def product_text(rec):
    """商品级完整正文（供生成证据与网页展示）。价格、评分等在 meta 中。"""
    return (
        f"Product Title: {rec['title']}; "
        f"Categories: {', '.join(rec['categories'])}; "
        f"Features: {', '.join(rec['features'])}; "
        f"Description: {' '.join(rec['description'])}; "
        f"Specs: {', '.join(rec['specs'])}"
    )


def _group_items(items, max_words):
    """把条目（卖点/规格项）按顺序装箱，每组不超过 max_words 词（单条超长则单独成组）。"""
    groups, cur, n = [], [], 0
    for it in items:
        w = len(it.split())
        if cur and n + w > max_words:
            groups.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += w
    if cur:
        groups.append(cur)
    return groups


def _windows(words, size, overlap):
    if len(words) <= size:
        return [words]
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(words[start:start + size])
        if start + size >= len(words):
            break
    return out


def chunk_product(rec):
    """字段级切块：overview / description / specs，每块都以商品标题开头。"""
    head = f"Product Title: {rec['title']}"
    chunks = []

    def add(field, body):
        n = sum(1 for c in chunks if c["field"] == field) + 1
        chunks.append({
            "chunk_id": f"{rec['id']}#{field}-{n}",
            "parent_id": rec["id"],
            "field": field,
            "text": f"{head}; {body}",
        })

    cats = ", ".join(rec["categories"])
    for group in _group_items(rec["features"], CHUNK_WORDS):
        add("overview", f"Categories: {cats}; Features: {', '.join(group)}")

    desc_words = " ".join(rec["description"]).split()
    for window in _windows(desc_words, CHUNK_WORDS, CHUNK_OVERLAP):
        add("description", "Description: " + " ".join(window))

    for group in _group_items(rec["specs"], CHUNK_WORDS):
        add("specs", "Specs: " + ", ".join(group))

    return chunks


# ---------------- 流式读取 + 蓄水池抽样 ----------------

def load_category(category, max_scan=None):
    url = BASE_URL + f"meta_{category}.jsonl.gz"
    rng = random.Random(f"{RANDOM_SEED}-{category}")
    reservoir = []
    scanned = valid = 0
    t0 = time.time()
    print(f"[{category}] streaming {url} ...")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as f:
            for raw in f:
                if not raw.strip():
                    continue
                scanned += 1
                if max_scan and scanned > max_scan:
                    break
                try:
                    prod = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not product_valid(prod):
                    continue
                valid += 1
                # Algorithm R：第 valid 个合格商品以 RESERVOIR_SIZE/valid 的概率入池
                if len(reservoir) < RESERVOIR_SIZE:
                    reservoir.append(prod)
                else:
                    j = rng.randrange(valid)
                    if j < RESERVOIR_SIZE:
                        reservoir[j] = prod
                if scanned % 500000 == 0:
                    print(f"  scanned {scanned:,}  valid {valid:,}  ({time.time() - t0:.0f}s)")
    rng.shuffle(reservoir)
    print(f"[{category}] scanned {scanned:,} -> valid {valid:,} -> reservoir {len(reservoir)}")
    return reservoir, {"scanned": scanned, "valid": valid, "reservoir": len(reservoir)}


def _title_key(title):
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


# ---------------- 统计 ----------------

def _word_tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def corpus_stats(products, chunks, sampling):
    texts = [p["text"] for p in products]
    words = [len(t.split()) for t in texts]
    types_norm, types_raw = set(), set()
    for t in texts:
        types_norm.update(_word_tokens(t))
        types_raw.update(t.split())
    chunk_words = [len(c["text"].split()) for c in chunks]
    srt = sorted(words)
    return {
        "documents": len(products),
        "chunks": len(chunks),
        "chunks_by_field": dict(Counter(c["field"] for c in chunks)),
        "words": sum(words),
        "words_definition": "whitespace-separated tokens in product text",
        "types": len(types_norm),
        "types_definition": "unique lowercased alphanumeric tokens ([a-z0-9]+)",
        "types_raw_whitespace": len(types_raw),
        "words_per_document": {
            "mean": round(sum(words) / len(words), 1),
            "median": srt[len(srt) // 2],
            "p95": srt[int(0.95 * len(srt))],
            "max": srt[-1],
        },
        "words_per_chunk": {
            "mean": round(sum(chunk_words) / len(chunk_words), 1),
            "max": max(chunk_words),
        },
        "documents_by_category": dict(Counter(p["meta"]["main_category"] for p in products)),
        "sampling": sampling,
        "cleaning": dict(sorted(CLEAN.counter.items())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scan", type=int, default=None,
                        help="每个分类最多扫描的商品数（快速测试用）")
    args = parser.parse_args()

    seen_asins, seen_titles = set(), set()
    products, chunks, sampling = [], [], {}

    for cat in CATEGORIES:
        reservoir, info = load_category(cat, args.max_scan)
        kept = dup_asin = dup_title = invalid_after_clean = 0
        for prod in reservoir:
            if kept >= PER_CATEGORY:
                break
            asin = prod["parent_asin"].strip()
            if asin in seen_asins:
                dup_asin += 1
                continue
            rec = product_record(prod, cat)
            if rec is None:
                invalid_after_clean += 1
                continue
            tkey = _title_key(rec["title"])
            if tkey in seen_titles:
                dup_title += 1
                continue
            seen_asins.add(asin)
            seen_titles.add(tkey)
            products.append({"id": rec["id"], "text": product_text(rec), "meta": rec["meta"]})
            chunks.extend(chunk_product(rec))
            kept += 1
        info.update({"duplicate_asin": dup_asin, "duplicate_title": dup_title,
                     "empty_after_cleaning": invalid_after_clean, "kept": kept})
        sampling[cat] = info
        print(f"[{cat}] kept {kept} (dup asin {dup_asin}, dup title {dup_title}, "
              f"empty after cleaning {invalid_after_clean})")

    with open(os.path.join(DATA_DIR, "kb.json"), "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "kb_pyserini.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c["chunk_id"], "contents": c["text"]},
                               ensure_ascii=False) + "\n")

    stats = corpus_stats(products, chunks, sampling)
    with open(os.path.join(DATA_DIR, "corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"✅ Corpus built. Documents: {stats['documents']}, chunks: {stats['chunks']}, "
          f"total words: {stats['words']}, types: {stats['types']}")


if __name__ == "__main__":
    main()
