"""
知识库的统一读取入口（每个进程只加载一次），以及 BM25 使用的分词器。

检索单元（RETRIEVAL_UNIT 环境变量）：
    chunk   （默认）字段级 chunk（data/chunks.json），检索后按商品合并
    product          整个商品作为一个检索单元（data/kb.json），用于消融对比
"""

import json
import os
import re
from functools import lru_cache

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
KB_PATH = os.path.join(DATA_DIR, "kb.json")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")

RETRIEVAL_UNIT = os.getenv("RETRIEVAL_UNIT", "chunk")
if RETRIEVAL_UNIT not in ("chunk", "product"):
    raise ValueError("RETRIEVAL_UNIT must be 'chunk' or 'product'")


@lru_cache(maxsize=None)
def load_products():
    """商品级知识库：[{id, text, meta}, ...]"""
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def products_by_id():
    return {p["id"]: p for p in load_products()}


@lru_cache(maxsize=None)
def load_chunks():
    """chunk 级检索单元：[{chunk_id, parent_id, field, text}, ...]"""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def chunks_by_id():
    return {c["chunk_id"]: c for c in load_chunks()}


def load_units(unit=RETRIEVAL_UNIT):
    """返回检索单元 (unit_ids, parent_ids, texts)。"""
    if unit == "chunk":
        chunks = load_chunks()
        return ([c["chunk_id"] for c in chunks],
                [c["parent_id"] for c in chunks],
                [c["text"] for c in chunks])
    products = load_products()
    ids = [p["id"] for p in products]
    return ids, ids, [p["text"] for p in products]


def unit_info(unit_id, unit=RETRIEVAL_UNIT):
    """检索单元的字段名与文本（用于展示“命中了商品的哪一部分”）。"""
    if unit == "chunk":
        c = chunks_by_id()[unit_id]
        return c["field"], c["text"]
    return "product", products_by_id()[unit_id]["text"]


# ---------------- BM25 分词 ----------------

_TOKEN = re.compile(r"[a-z0-9]+")

# 功能词与查询里常见的口语填充词；保留 no/not 等可能改变语义的词
STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as into about over than then so
is are was were be been being am do does did have has had
i me my we our us you your he she it its they them their this that these those
what which who whom whose where when why how can could would should will shall may might must
need needs want wants looking look find recommend recommended suggest please some any also just
very there here
""".split())


def tokenize(text):
    """小写 + 只保留字母数字 + 去停用词；单字符只保留数字（去掉 men's 中的 s）。"""
    return [t for t in _TOKEN.findall(text.lower())
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]
