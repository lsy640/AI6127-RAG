"""
构建稠密向量索引（BAAI/bge-m3，1024 维，归一化后用 IndexFlatIP = 余弦相似度）

用法：
    python build_faiss.py                       # 默认：chunk 级、不加指令前缀
    python build_faiss.py --unit product        # 商品级（消融：不切块）
    python build_faiss.py --prompt              # 文档加 BGE 指令前缀（消融）

输出（data/）：
    faiss_<unit>[_prompt].bin   向量索引
    faiss_<unit>[_prompt].json  索引说明：模型、检索单元、是否加前缀、向量顺序对应的 id 列表等
                                （替代原来把全文又存一遍的 id_map.json）

关于指令前缀："Represent this sentence for searching relevant passages: " 是 BGE v1.5
系列给 *query* 用的指令；bge-m3 官方说明 query 和文档都不需要加指令，所以默认不加。
hybrid_retriever.py 按 RETRIEVAL_UNIT / BGE_PROMPT 环境变量选择对应的索引，并保证
query 的编码方式与索引一致。
"""

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import argparse
import gc
import json
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EMBED_MODEL = "BAAI/bge-m3"
BGE_SEARCH_PROMPT = "Represent this sentence for searching relevant passages: "


def index_paths(unit, prompt):
    name = f"faiss_{unit}" + ("_prompt" if prompt else "")
    return os.path.join(DATA_DIR, name + ".bin"), os.path.join(DATA_DIR, name + ".json")


SLICE_SIZE = int(os.getenv("SLICE_SIZE", "512"))  # 每次送入 encode 的条数；每片之后释放显存缓存并保存断点


def empty_cache(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def load_embedding_model(device):
    """
    加载 bge-m3（建索引与查询编码共用，保证一致）。
    - Apple Silicon（MPS）上使用 eager attention：PyTorch 的 SDPA 在 MPS 上每个 batch 都会
      泄漏显存（实测每条约 25 MB，几百条后 OOM），eager 实现显存恒定。
    - GPU（CUDA / MPS）上用 fp16：Apple M5 上快约 3.4 倍，与 fp32 向量的余弦相似度 ≥ 0.9998。
    """
    kwargs = {"attn_implementation": "eager"} if device == "mps" else {}
    model = SentenceTransformer(EMBED_MODEL, device=device, model_kwargs=kwargs)
    if device in ("cuda", "mps"):
        model = model.half()
    return model


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", choices=["chunk", "product"], default="chunk",
                        help="检索单元：chunk（字段级切块，默认）或 product（整个商品）")
    parser.add_argument("--prompt", action="store_true",
                        help="文档前加 BGE 指令前缀（默认不加，bge-m3 官方用法）")
    parser.add_argument("--batch-size", type=int, default=8, help="显存小可改为 4")
    parser.add_argument("--limit", type=int, default=None, help="只编码前 N 条（冒烟测试用）")
    args = parser.parse_args()

    os.environ["RETRIEVAL_UNIT"] = args.unit
    import corpus  # 读取 RETRIEVAL_UNIT 之后再导入

    unit_ids, parent_ids, texts = corpus.load_units(args.unit)
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("检测到重复 id，请检查 dataset.py 的输出")
    if args.limit:
        unit_ids, texts = unit_ids[:args.limit], texts[:args.limit]

    device = pick_device()
    print(f"⚙️  Device: {device} {'(CPU —— bge-m3 会很慢，建议用 GPU / Apple Silicon)' if device == 'cpu' else '(GPU 加速)'}")
    print(f"Unit: {args.unit} | prompt on documents: {args.prompt} | units: {len(texts)}")

    prefix = BGE_SEARCH_PROMPT if args.prompt else ""
    model = load_embedding_model(device)
    model.max_seq_length = 8192  # bge-m3 上限；本语料最长约 2k token，不会截断

    # 按长度从长到短分片编码：显存问题会在最开始暴露；每片之后释放缓存
    # （MPS 不释放缓存时会持续增长直至 OOM），并保存断点，中断后重跑会接着编码。
    index_path, meta_path = index_paths(args.unit, args.prompt)
    ckpt_path = index_path + ".partial.npz"
    order = np.argsort([-len(t) for t in texts], kind="stable")
    n = len(texts)
    embeddings, done = None, 0
    if os.path.exists(ckpt_path):
        ckpt = np.load(ckpt_path)
        if int(ckpt["n"]) == n and bool(ckpt["prompt"]) == args.prompt:
            embeddings, done = ckpt["embeddings"], int(ckpt["done"])
            print(f"Resuming from checkpoint: {done}/{n} encoded")

    first_done = done
    t0 = time.time()
    for start in range(done, n, SLICE_SIZE):
        idx = order[start:start + SLICE_SIZE]
        part = model.encode(
            [prefix + texts[i] for i in idx],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=args.batch_size,
        ).astype(np.float32)
        if embeddings is None:
            embeddings = np.zeros((n, part.shape[1]), dtype=np.float32)
        embeddings[idx] = part
        done = min(start + SLICE_SIZE, n)
        empty_cache(device)
        if (done // SLICE_SIZE) % 4 == 0 and done < n:
            np.savez(ckpt_path, embeddings=embeddings, done=done, n=n, prompt=args.prompt)
        elapsed = time.time() - t0
        rate = (done - first_done) / elapsed
        print(f"  {done}/{n} encoded | {rate:.1f} units/s | "
              f"{elapsed / 60:.1f} min elapsed, ~{(n - done) / rate / 60:.0f} min left", flush=True)
    seconds = time.time() - t0

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": EMBED_MODEL,
            "unit": args.unit,
            "prompt": args.prompt,
            "dimension": int(embeddings.shape[1]),
            "count": int(index.ntotal),
            "limit": args.limit,
            "device": device,
            "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "encode_seconds": round(seconds, 1),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ids": unit_ids,
        }, f, ensure_ascii=False)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"✅ FAISS index saved: {index_path}")
    print(f"✅ Index info saved: {meta_path}")
    print(f"✅ Dimension: {embeddings.shape[1]}, vectors: {index.ntotal}, "
          f"encoding took {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
