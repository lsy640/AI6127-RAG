"""
用户模拟器（多轮评测用）

- 模拟器只知道场景的隐藏需求（槽位取值），不知道任何目标商品，避免“泄题”
  （Zhu et al., WWW 2024 Companion 指出 LLM 模拟器会直接说出目标商品名，使结果虚高）。
- 只回答被问到的那个槽位；问到隐藏需求中没有的内容时回答“没有偏好”。
- 回答由模板生成（中/英），可复现、零成本，不引入 LLM 评判偏差。
- 推荐后用程序化规则判断是否成功（前 k 个推荐中至少一个满足全部硬约束）；
  失败且还有轮次时，用户指出一个被违反的约束（例如“超预算了，我最多花 30 美元”），对话继续。
"""

from collections import Counter

import attributes as A
from build_scenarios import AUDIENCE_TEXT, MATERIAL_ZH, a as article, answer_text

NO_PREFERENCE = ("No particular preference.", "都可以，没有特别要求。")


def _names(sc):
    return (sc["type_names"]["en"], sc["type_names"]["zh"], "", "")


def answer(sc, slot):
    need, lang = sc["hidden_need"], sc["language"]
    if slot in need and slot != "domain":
        return answer_text(slot, need, _names(sc), lang)
    return NO_PREFERENCE[0 if lang == "en" else 1]


def feedback(sc, slot):
    need, lang = sc["hidden_need"], sc["language"]
    en_type, zh_type = sc["type_names"]["en"], sc["type_names"]["zh"]
    texts = {
        "price_max": (f"These are over my budget — I want to spend at most ${need.get('price_max')}.",
                      f"超预算了，我最多花{need.get('price_max')}美元。"),
        "audience": (f"I need it {AUDIENCE_TEXT.get(need.get('audience'), ('', ''))[0]}.",
                     f"我要的是{AUDIENCE_TEXT.get(need.get('audience'), ('', ''))[1]}的。"),
        "age": (f"It's for a {need.get('age')}-year-old.", f"是给{need.get('age')}岁孩子的。"),
        "product_type": (f"Actually I'm looking for {article(en_type)}.", f"我其实想要{zh_type}。"),
        "domain": (f"Actually I'm looking for {article(en_type)}.", f"我其实想要{zh_type}。"),
        "material": (f"I'd like it in {need.get('material')}.",
                     f"想要{MATERIAL_ZH.get(need.get('material'), '')}材质的。"),
    }
    return texts.get(slot, NO_PREFERENCE)[0 if lang == "en" else 1]


def judge_recommendations(sc, recs, k=3):
    """返回 (是否成功, 每个推荐违反的约束列表)。"""
    need = sc["hidden_need"]
    verdicts = [A.satisfies(r["parent_asin"], need)[1] for r in recs[:k]]
    return any(not v for v in verdicts) and bool(verdicts), verdicts


def run_dialogue(agent, sc, max_turns=5, k=3):
    history = [{"role": "user", "text": sc["opening"]}]
    given = set(sc["given_slots"])
    record = {"scenario_id": sc["scenario_id"], "level": sc["level"], "language": sc["language"],
              "domain": sc["domain"], "config": agent.cfg.name, "turns": [], "questions": [],
              "success": False, "success_turn": None, "first_rec_success": None,
              "n_recommendations": 0, "n_violating": 0, "none_responses": 0, "error": None}
    try:
        for turn in range(1, max_turns + 1):
            out = agent.respond(history)
            history.append({"role": "assistant", "text": out.text, "action": out.action,
                            "asked_slot": out.asked_slot})
            record["turns"].append({"turn": turn, "action": out.action, "asked_slot": out.asked_slot,
                                    "text": out.text, "state": out.state, "debug": out.debug,
                                    "recommendations": out.recommendations,
                                    "evidence_ids": [e["evidence_id"] for e in out.evidence]})
            if out.action == "ask":
                useful = out.asked_slot in sc["hidden_need"] and out.asked_slot not in given
                record["questions"].append({"turn": turn, "slot": out.asked_slot, "useful": useful})
                given.add(out.asked_slot)
                history.append({"role": "user", "text": answer(sc, out.asked_slot)})
                continue
            if out.action == "none":
                # 场景都是可解的（知识库中至少有 3 个满足需求的商品），“找不到”即失败
                record["none_responses"] += 1
                break
            ok, verdicts = judge_recommendations(sc, out.recommendations, k)
            record["n_recommendations"] += len(verdicts)
            record["n_violating"] += sum(1 for v in verdicts if v)
            record["turns"][-1]["violations"] = verdicts
            if record["first_rec_success"] is None:
                record["first_rec_success"] = ok
            if ok:
                record["success"], record["success_turn"] = True, turn
                record["final"] = {"answer": out.text, "recommendations": out.recommendations,
                                   "evidence": out.evidence}
                break
            violated = Counter(s for v in verdicts for s in v)
            slot = violated.most_common(1)[0][0] if violated else "product_type"
            given.add(slot)
            history.append({"role": "user", "text": feedback(sc, slot)})
            record["final"] = {"answer": out.text, "recommendations": out.recommendations,
                               "evidence": out.evidence}
    except Exception as error:   # 记录错误，继续评测其它对话
        record["error"] = f"{type(error).__name__}: {error}"
    record["n_turns"] = len(record["turns"])
    record["n_questions"] = len(record["questions"])
    record["history"] = history
    return record
