"""
BM25 稀疏检索（Okapi BM25，k1=1.5，b=0.75）

- 分词：小写、只保留字母数字、去停用词（corpus.tokenize），
  解决原来 split() 下 "Waterproof" / "waterproof"、"hiking." / "Hiking" 匹配不上的问题。
- 实现：预先把每个 (文档, 词) 的 BM25 权重算进稀疏矩阵，查询时只取查询词对应的列求和，
  比 rank_bm25 逐词遍历全部文档快一个数量级以上。
- idf 采用 Lucene 形式 log(1 + (N - df + 0.5) / (df + 0.5))，恒为正。
- 检索单元为 chunk 时，按商品取最高分合并（MaxP 父文档检索）。
"""

import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

import corpus

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, texts):
        self.vectorizer = CountVectorizer(
            tokenizer=corpus.tokenize, lowercase=False, token_pattern=None
        )
        tf = self.vectorizer.fit_transform(texts).tocoo().astype(np.float32)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean()
        df = np.bincount(tf.col, minlength=tf.shape[1])
        self.idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        norm = K1 * (1.0 - B + B * doc_len[tf.row] / avg_len)
        weights = self.idf[tf.col] * tf.data * (K1 + 1.0) / (tf.data + norm)
        self.weights = sp.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        self.vocab = self.vectorizer.vocabulary_
        self.n_docs = n_docs

    def scores(self, query):
        """每个检索单元的 BM25 分数（重复的查询词按出现次数计入）。"""
        cols = [self.vocab[t] for t in corpus.tokenize(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.n_docs, dtype=np.float32)
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()


_t0 = time.perf_counter()
unit_ids, parent_ids, _texts = corpus.load_units()
bm25 = BM25Index(_texts)
del _texts
BUILD_SECONDS = time.perf_counter() - _t0


def bm25_retrieve(query: str, top_k=20):
    """
    返回 list[(parent_id, rank_position, score, best_unit_id)]，按商品去重后的前 top_k 个。
    """
    scores = bm25.scores(query)
    n_candidates = min(len(scores), top_k * 20)
    candidates = np.argpartition(-scores, n_candidates - 1)[:n_candidates]
    candidates = candidates[np.argsort(-scores[candidates])]

    out, seen = [], set()
    for idx in candidates:
        if scores[idx] <= 0:
            break
        parent = parent_ids[idx]
        if parent in seen:
            continue
        seen.add(parent)
        out.append((parent, len(out), float(scores[idx]), unit_ids[idx]))
        if len(out) >= top_k:
            break
    return out


# quick test
if __name__ == "__main__":
    print(f"BM25 index: {bm25.n_docs} {corpus.RETRIEVAL_UNIT}s, "
          f"{len(bm25.vocab)} terms, built in {BUILD_SECONDS:.1f}s")
    q = "waterproof men's boots"
    t = time.perf_counter()
    res = bm25_retrieve(q, top_k=5)
    print(f"==== BM25 test result ({(time.perf_counter() - t) * 1000:.1f} ms) ====")
    for docid, pos, score, unit in res:
        print(f"docid:{docid}, pos:{pos}, score:{score:.4f}, matched:{unit}")
