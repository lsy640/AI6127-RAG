"""
交叉编码器重排序（Q3 创新点：re-ranking）

模型：BAAI/bge-reranker-v2-m3（多语言，支持中文 query 对英文商品）。
对融合后的前 N 个商品，用 (query, 商品正文前若干词) 逐对打分并重排。

性能处理（Apple Silicon / MPS）：
- eager attention：PyTorch 的 SDPA 在 MPS 上会持续泄漏显存；
- fp16：比 fp32 快约 3 倍；
- 输入按 64 token 分档补齐：MPS 对每种新输入形状都会重新编译计算图，长度各不相同时
  编译开销远大于计算本身；分档后只有少数几种形状。
在 Apple M5 上，20 个候选 × 约 190 token 一次重排约 0.4 s。
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
PASSAGE_WORDS = 128          # 送入重排序的商品文本长度（标题+类目+卖点在前）
MAX_TOKENS = 320
BUCKET = 64

_tok = _model = _device = None


def _load():
    global _tok, _model, _device
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from build_faiss import pick_device
        _device = pick_device()
        kwargs = {"attn_implementation": "eager"} if _device == "mps" else {}
        _tok = AutoTokenizer.from_pretrained(RERANK_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, **kwargs).to(_device).eval()
        if _device in ("cuda", "mps"):
            _model = _model.half()
    return _tok, _model, _device


def score_pairs(query, passages, batch_size=32):
    import torch
    tok, model, device = _load()
    scores = []
    for i in range(0, len(passages), batch_size):
        chunk = passages[i:i + batch_size]
        lengths = tok([query] * len(chunk), chunk, truncation="only_second",
                      max_length=MAX_TOKENS)["input_ids"]
        longest = max(len(x) for x in lengths)
        padded = min(MAX_TOKENS, ((longest + BUCKET - 1) // BUCKET) * BUCKET)
        enc = tok([query] * len(chunk), chunk, padding="max_length", truncation="only_second",
                  max_length=padded, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**enc).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores


def rerank(query, items, text_key="passage"):
    """
    items: hybrid_retriever 的结果列表（含 passage）。返回按重排分数降序的新列表，
    每项增加 rerank_score；同时返回耗时（毫秒）。
    """
    if not items:
        return [], 0.0
    t0 = time.perf_counter()
    passages = [" ".join(str(it[text_key]).split()[:PASSAGE_WORDS]) for it in items]
    scores = score_pairs(query, passages)
    out = []
    for it, s in zip(items, scores):
        it = dict(it)
        it["rerank_score"] = round(float(s), 4)
        out.append(it)
    out.sort(key=lambda x: x["rerank_score"], reverse=True)
    return out, (time.perf_counter() - t0) * 1000


def warm_up():
    for n in (1, 20):
        rerank("warm up", [{"passage": "warm up " * 100}] * n)
