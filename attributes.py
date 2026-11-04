"""
商品结构化属性（槽位）与约束校验 —— 澄清提问、元数据过滤、评测共用

每个商品从 meta（类目路径、部门、推荐年龄、材质、价格）中离线推导出槽位取值：
    domain        clothing | home_kitchen | toys
    coarse_type   粗粒度品类（如 Shoes / Kitchen & Dining / Puzzles）
    product_type  细粒度品类 = 类目路径最后一级（如 Hiking Boots / Jigsaw Puzzles）
    audience      women | men | girls | boys | baby | unisex（服装）
    age_band      0-2 | 3-5 | 6-8 | 9-12 | 13+（玩具，按推荐年龄下限）
    price_band    under $15 | $15-30 | $30-60 | $60-120 | over $120
    material      规范化后的主要材质（家居）

“需求”（need）用同样的槽位表示，例如
    {"domain": "toys", "product_type": "Jigsaw Puzzles", "age": 6, "price_max": 25}
satisfies(product, need) 逐个检查硬约束，返回是否满足以及违反了哪些约束。
"""

import math
import re
from collections import Counter
from functools import lru_cache

import corpus

DOMAINS = {
    "Clothing, Shoes & Jewelry": "clothing",
    "Home & Kitchen": "home_kitchen",
    "Toys & Games": "toys",
}

# 每个领域：推荐前必须明确的槽位（规则式澄清）与可以提问的槽位（熵选问）
SLOT_SPECS = {
    "clothing": {"required": ["product_type", "audience", "price_max"],
                 "askable": ["product_type", "audience", "price_max"]},
    "home_kitchen": {"required": ["product_type", "price_max"],
                     "askable": ["product_type", "material", "price_max"]},
    "toys": {"required": ["product_type", "age", "price_max"],
             "askable": ["product_type", "age", "price_max"]},
}

# 槽位 -> 用于计算候选分布的商品属性
SLOT_ATTRIBUTE = {
    "product_type": "product_type",
    "audience": "audience",
    "age": "age_band",
    "price_max": "price_band",
    "material": "material",
}

PRICE_BANDS = [(15, "under $15"), (30, "$15-30"), (60, "$30-60"), (120, "$60-120"),
               (float("inf"), "over $120")]
AGE_BANDS = [(2, "0-2"), (5, "3-5"), (8, "6-8"), (12, "9-12"), (float("inf"), "13+")]

_AUDIENCE = {"women": "women", "men": "men", "girls": "girls", "boys": "boys",
             "baby": "baby", "baby girls": "baby", "baby boys": "baby",
             "womens": "women", "mens": "men", "unisex-adult": "unisex",
             "unisex-child": "unisex", "unisex": "unisex", "girls'": "girls", "boys'": "boys"}

_MATERIALS = ["stainless steel", "cast iron", "aluminum", "ceramic", "glass", "porcelain",
              "wood", "bamboo", "cotton", "polyester", "microfiber", "linen", "silk", "wool",
              "plastic", "silicone", "metal", "iron", "steel", "stone", "marble", "resin",
              "fabric", "leather", "paper", "velvet", "acrylic", "copper", "rubber", "foam"]


def price_band(price):
    if price is None:
        return None
    for upper, label in PRICE_BANDS:
        if price < upper:
            return label
    return PRICE_BANDS[-1][1]


def parse_age_years(text):
    """'3 years and up' -> 3；'36 months and up' -> 3；'8 - 12 years' -> 8。"""
    if not text:
        return None
    t = str(text).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-\s*\d+(?:\.\d+)?\s*)?(year|yr|month|mo)", t)
    if not m:
        return None
    value = float(m.group(1))
    return round(value / 12, 1) if m.group(2).startswith("mo") else value


def age_band(years):
    if years is None:
        return None
    for upper, label in AGE_BANDS:
        if years <= upper:
            return label
    return AGE_BANDS[-1][1]


def normalize_material(text):
    if not text:
        return None
    t = str(text).lower()
    for mat in _MATERIALS:
        if mat in t:
            return mat
    return "other"


def _audience(categories, department):
    for c in categories[1:3]:
        a = _AUDIENCE.get(c.lower())
        if a:
            return a
    if department:
        return _AUDIENCE.get(str(department).lower().strip())
    return None


def derive(product):
    meta = product.get("meta", {})
    cats = meta.get("categories") or []
    domain = DOMAINS.get(meta.get("main_category"))
    if domain == "clothing":
        coarse = cats[2] if len(cats) > 2 else (cats[1] if len(cats) > 1 else None)
    else:
        coarse = cats[1] if len(cats) > 1 else None
    age_years = parse_age_years(meta.get("recommended_age"))
    return {
        "domain": domain,
        "coarse_type": coarse,
        "product_type": cats[-1] if len(cats) > 1 else None,
        "category_path": [c.lower() for c in cats],
        "audience": _audience(cats, meta.get("department")) if domain == "clothing" else None,
        "age_years": age_years,
        "age_band": age_band(age_years) if domain == "toys" else None,
        "price": meta.get("price"),
        "price_band": price_band(meta.get("price")),
        "material": normalize_material(meta.get("material")) if domain == "home_kitchen" else None,
        "rating": meta.get("average_rating"),
        "date": meta.get("date_first_available"),
    }


@lru_cache(maxsize=None)
def all_attributes():
    return {p["id"]: derive(p) for p in corpus.load_products()}


def get(asin):
    return all_attributes()[asin]


# ---------------- 约束校验 ----------------

def _norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def type_matches(attrs, wanted):
    """细粒度品类匹配：需求品类出现在商品类目路径中（大小写、标点无关）。"""
    w = _norm(wanted)
    return any(_norm(c) == w for c in attrs["category_path"])


def satisfies(asin, need):
    """返回 (是否满足全部硬约束, 违反的约束列表)。值为 None/'any' 的槽位不检查。"""
    a = get(asin)
    violations = []

    def active(key):
        v = need.get(key)
        return v not in (None, "", "any")

    if active("domain") and a["domain"] != need["domain"]:
        violations.append("domain")
    if active("product_type") and not type_matches(a, need["product_type"]):
        violations.append("product_type")
    if active("audience"):
        aud = a["audience"]
        if aud not in (need["audience"], "unisex") and not (aud is None and need["audience"] == "unisex"):
            violations.append("audience")
    if active("age"):
        if a["age_years"] is None or a["age_years"] > float(need["age"]):
            violations.append("age")
    if active("price_max") and (a["price"] is None or a["price"] > float(need["price_max"])):
        violations.append("price_max")
    if active("material") and a["material"] != need["material"]:
        violations.append("material")
    return not violations, violations


def filters_for(need, include_type=False):
    """
    把需求中的硬约束转成 hybrid_retriever.search 的 filters。
    对话中用户说的品类是自由文本（如 "hiking boots"），默认不作为过滤条件，
    而是留在检索查询里交给语义检索；评测时的隐藏需求（类目叶子）可用 include_type=True。
    """
    filters = {}
    if need.get("price_max") not in (None, "", "any"):
        filters["price_max"] = float(need["price_max"])
    if need.get("rating_min"):
        filters["rating_min"] = float(need["rating_min"])
    domain_to_cat = {v: k for k, v in DOMAINS.items()}
    if need.get("domain") in domain_to_cat:
        filters["category"] = domain_to_cat[need["domain"]]
    keys = ("product_type", "audience", "age", "material") if include_type else ("audience", "age", "material")
    structural = {k: need.get(k) for k in keys if need.get(k) not in (None, "", "any")}
    if structural:
        sub_need = dict(structural)

        def predicate(product, _need=sub_need):
            return satisfies(product["id"], _need)[0]
        filters["predicate"] = predicate
    return filters


# ---------------- 候选分布与熵（澄清选问） ----------------

def distribution(asins, slot):
    attr = SLOT_ATTRIBUTE[slot]
    values = [get(a)[attr] for a in asins]
    return Counter(v for v in values if v)


def normalized_entropy(counter):
    """归一化熵 H/log(K)，K 为取值个数；只有一个取值时为 0。"""
    total = sum(counter.values())
    if total == 0 or len(counter) < 2:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counter.values())
    return h / math.log(len(counter))


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())
