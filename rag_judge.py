"""
Q4: Answer-level RAG metrics, following the RAGAS definitions
(Es et al., 2024, "RAGAS: Automated Evaluation of Retrieval Augmented
Generation"), with an LLM as the judge.

For every generated answer in eval/q4_results_<tag>.jsonl:

1. Faithfulness
   The judge splits the answer (answer text + recommendation reasons) into
   atomic factual claims and decides, for each claim, whether it can be
   inferred from the evidence passages.
       faithfulness = supported claims / all claims

2. Answer relevance
   The judge writes N questions that the answer would answer; the score is
   the mean cosine similarity between those questions and the original
   query (all-MiniLM-L6-v2 embeddings). A non-committal answer
   ("I cannot recommend...") scores 0, as in RAGAS.

3. Context relevance (LLM-judged)
   The judge marks every retrieved passage as useful / not useful for the
   query. Gives an LLM version of context precision that needs no human
   labels, and lets us measure judge-vs-human agreement on the gold set.

Judge model: $JUDGE_MODEL if set, otherwise $DEEPSEEK_MODEL.
Input: dialogue records from run_dialogues.py (eval/q5_runs/<config>.jsonl). Using a
different (or stronger) model than the generator reduces self-preference
bias.

Usage:
    python rag_judge.py --configs B0 full --workers 6
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from generator import create_client, parse_json_response


BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "eval"

N_QUESTIONS = 3

FAITHFULNESS_PROMPT = """You are evaluating whether an answer is grounded in evidence.

Evidence passages:
{evidence}

Answer to check:
{answer}

Step 1: Break the answer into short, self-contained factual claims about
products (features, materials, suitability, prices, ratings, etc.).
Ignore pure opinions without factual content and statements that only
say the evidence is insufficient.
Step 2: For each claim, decide whether it can be directly inferred from the
evidence passages (1 = supported, 0 = not supported).

Return JSON only:
{{"claims": [{{"claim": "...", "supported": 1, "evidence_id": "E1"}}]}}
"""

ANSWER_RELEVANCE_PROMPT = """Read the following answer from a shopping assistant.

Answer:
{answer}

1. Write {n} different questions that this answer would be a good answer to.
2. Decide whether the answer is non-committal (evasive, vague, or says it
   cannot help / the evidence is insufficient): 1 = non-committal, 0 = not.

Return JSON only:
{{"questions": ["...", "..."], "noncommittal": 0}}
"""

CONTEXT_RELEVANCE_PROMPT = """A user asked a shopping assistant:
"{query}"

For each retrieved product passage, decide whether the product satisfies
the user's need (correct product type, target group and any explicit
constraints stated in the query). 1 = relevant, 0 = not relevant.

Passages:
{evidence}

Return JSON only:
{{"verdicts": [{{"evidence_id": "E1", "relevant": 1}}]}}
"""


def judge_model():
    model = os.getenv("JUDGE_MODEL") or os.getenv("DEEPSEEK_MODEL")
    if not model:
        raise ValueError("Set JUDGE_MODEL or DEEPSEEK_MODEL.")
    return model


def ask(prompt):
    client = create_client()
    response = client.chat.completions.create(
        model=judge_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1500,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    usage = response.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    return parse_json_response(response.choices[0].message.content or ""), tokens


def answer_text(result):
    parts = [result.get("answer", "")]
    for rec in result.get("recommendations", []):
        parts.append(f"[{rec['parent_asin']}] {rec.get('reason', '')}")
    return "\n".join(p for p in parts if p)


def evidence_text(result):
    return "\n\n".join(
        f"{e['evidence_id']} (ASIN {e['parent_asin']}): {e['passage']}"
        for e in result.get("evidence", [])
    )


_EMB = None


def embed(texts):
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def judge_one(result):
    out = {"case_id": result["case_id"], "judge_model": judge_model(),
           "judge_tokens": 0, "error": None}

    if result.get("error") or not result.get("evidence"):
        out["error"] = result.get("error") or "no evidence"
        return out

    try:
        text = answer_text(result)
        evidence = evidence_text(result)

        # 1. Faithfulness
        data, tokens = ask(FAITHFULNESS_PROMPT.format(
            evidence=evidence, answer=text))
        out["judge_tokens"] += tokens
        claims = data.get("claims", []) if isinstance(data, dict) else []
        claims = [c for c in claims if isinstance(c, dict)]
        out["claims"] = claims
        out["n_claims"] = len(claims)
        out["faithfulness"] = (
            sum(int(c.get("supported", 0) in (1, "1", True)) for c in claims)
            / len(claims) if claims else None)

        # 2. Answer relevance
        data, tokens = ask(ANSWER_RELEVANCE_PROMPT.format(
            answer=text, n=N_QUESTIONS))
        out["judge_tokens"] += tokens
        questions = [q for q in data.get("questions", []) if isinstance(q, str)]
        noncommittal = int(data.get("noncommittal", 0) in (1, "1", True))
        out["generated_questions"] = questions
        out["noncommittal"] = noncommittal
        if questions:
            vectors = embed([result["query"], *questions])
            sims = vectors[1:] @ vectors[0]
            out["answer_relevance"] = (
                0.0 if noncommittal else float(np.mean(sims)))
        else:
            out["answer_relevance"] = None

        # 3. Context relevance per passage
        data, tokens = ask(CONTEXT_RELEVANCE_PROMPT.format(
            query=result["query"], evidence=evidence))
        out["judge_tokens"] += tokens
        id_to_asin = {e["evidence_id"]: e["parent_asin"]
                      for e in result["evidence"]}
        verdicts = {}
        for v in data.get("verdicts", []):
            if isinstance(v, dict) and v.get("evidence_id") in id_to_asin:
                verdicts[id_to_asin[v["evidence_id"]]] = int(
                    v.get("relevant", 0) in (1, "1", True))
        out["llm_context_relevance"] = verdicts

    except Exception as error:
        out["error"] = str(error)

    return out


def context_precision(relevance):
    """RAGAS context precision：按证据顺序的平均 precision@k（只在相关位置累计）。"""
    if not relevance or not any(relevance):
        return 0.0 if relevance else None
    hits, score = 0, 0.0
    for k, r in enumerate(relevance, start=1):
        if r:
            hits += 1
            score += hits / k
    return score / sum(relevance)


def dialogue_results(path, scenarios):
    """把 run_dialogues.py 的对话记录转成评判输入：每段对话最后一次推荐。"""
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        final = r.get("final")
        if not final or not final.get("evidence"):
            continue
        sc = scenarios[r["scenario_id"]]
        out.append({"case_id": r["scenario_id"],
                    "query": sc["full_query_zh"] if sc["language"] == "zh" else sc["full_query_en"],
                    "answer": final["answer"], "recommendations": final["recommendations"],
                    "evidence": final["evidence"], "error": None})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["full"])
    parser.add_argument("--runs-dir", default=str(EVAL_DIR / "q5_runs"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    runs = Path(args.runs_dir)
    scenarios = {json.loads(l)["scenario_id"]: json.loads(l)
                 for l in open(EVAL_DIR / "scenarios" / "scenarios.jsonl", encoding="utf-8")}
    embed(["warm up"])
    summary_path = runs / "judge_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in args.configs:
        results = dialogue_results(runs / f"{name}.jsonl", scenarios)
        results = results[:args.limit] if args.limit else results
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            judged = list(executor.map(judge_one, results))
        for res, j in zip(results, judged):
            rel = j.get("llm_context_relevance") or {}
            order = [e["parent_asin"] for e in res["evidence"]]
            j["context_precision"] = context_precision([rel.get(a, 0) for a in order]) if rel else None
        with open(runs / f"judge_{name}.jsonl", "w", encoding="utf-8") as f:
            for j in judged:
                f.write(json.dumps(j, ensure_ascii=False) + "\n")
        ok = [j for j in judged if not j.get("error")]
        mean = lambda key: (sum(j[key] for j in ok if j.get(key) is not None)
                            / max(1, sum(1 for j in ok if j.get(key) is not None)))
        summary[name] = {"judged": len(ok), "faithfulness": mean("faithfulness"),
                         "answer_relevance": mean("answer_relevance"),
                         "context_precision": mean("context_precision"),
                         "judge_model": judge_model()}
        print(f"[{name}] {summary[name]}", flush=True)
        summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
