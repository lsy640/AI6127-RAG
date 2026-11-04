"""
Q4/Q5：生成“用户场景”评测集（多轮澄清式推荐）

每个场景 = 一个用户的隐藏完整需求 + 一句（可能很模糊的）开场白：
    hidden_need     {"domain", "product_type", "audience" | "age" | "material", "price_max"}
    opening         用户第一句话（英文或中文），按模糊程度分三档：
                      vague    只说大类（“想买双鞋”）       → 应该反问
                      partial  说了品类但缺人群/年龄/预算    → 应该反问
                      complete 必填信息都给了               → 应该直接推荐（测是否过度追问）
    full_query_en / full_query_zh   完整需求的一句话描述（Q3 检索评测用的查询）
    snapshots       两个对话时点（用于人工标注“该问还是该推荐、该问什么”）

隐藏需求以知识库中的一个真实商品为锚点生成（价格上限取锚点价格的 1.1–1.6 倍并取整），
并要求知识库中至少有 MIN_FEASIBLE 个商品满足全部硬约束，保证场景可解。

用法：python build_scenarios.py --per-domain 60
输出：eval/scenarios/scenarios.jsonl
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import attributes as A
import corpus

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "eval" / "scenarios"
MIN_FEASIBLE = 3

# product_type（类目叶子）-> 英文名、中文名、模糊说法（英/中）
TYPES = {
    "clothing": {
        "T-Shirts": ("t-shirt", "T恤", "some clothes", "几件衣服"),
        "Hoodies": ("hoodie", "连帽卫衣", "some clothes", "几件衣服"),
        "Jeans": ("pair of jeans", "牛仔裤", "some clothes", "几件衣服"),
        "Dresses": ("dress", "连衣裙", "some clothes", "几件衣服"),
        "Socks": ("pack of socks", "袜子", "some clothes", "几件衣服"),
        "Leggings": ("pair of leggings", "打底裤", "some clothes", "几件衣服"),
        "Polos": ("polo shirt", "Polo 衫", "some clothes", "几件衣服"),
        "Cardigans": ("cardigan", "开衫", "some clothes", "几件衣服"),
        "Fashion Sneakers": ("pair of sneakers", "运动休闲鞋", "some shoes", "一双鞋"),
        "Flats": ("pair of flats", "平底鞋", "some shoes", "一双鞋"),
        "Sandals": ("pair of sandals", "凉鞋", "some shoes", "一双鞋"),
        "Slippers": ("pair of slippers", "拖鞋", "some shoes", "一双鞋"),
        "Loafers & Slip-Ons": ("pair of loafers", "乐福鞋", "some shoes", "一双鞋"),
        "Pumps": ("pair of pumps", "高跟鞋", "some shoes", "一双鞋"),
        "Wrist Watches": ("wristwatch", "手表", "an accessory", "一个配饰"),
        "Sunglasses": ("pair of sunglasses", "太阳镜", "an accessory", "一个配饰"),
        "Baseball Caps": ("baseball cap", "棒球帽", "an accessory", "一个配饰"),
        "Belts": ("belt", "皮带", "an accessory", "一个配饰"),
        "Wallets": ("wallet", "钱包", "an accessory", "一个配饰"),
        "Pendant Necklaces": ("pendant necklace", "吊坠项链", "some jewelry", "一件首饰"),
        "Rings": ("ring", "戒指", "some jewelry", "一件首饰"),
        "Drop & Dangle": ("pair of drop earrings", "耳坠", "some jewelry", "一件首饰"),
        "Hoop": ("pair of hoop earrings", "圈形耳环", "some jewelry", "一件首饰"),
        "Stud": ("pair of stud earrings", "耳钉", "some jewelry", "一件首饰"),
        "Bracelets": ("bracelet", "手链", "some jewelry", "一件首饰"),
        "Crossbody Bags": ("crossbody bag", "斜挎包", "a bag", "一个包"),
        "Totes": ("tote bag", "托特包", "a bag", "一个包"),
    },
    "home_kitchen": {
        "Coffee Mugs": ("coffee mug", "咖啡杯", "something for the kitchen", "厨房用品"),
        "Wine Glasses": ("set of wine glasses", "红酒杯", "something for the kitchen", "厨房用品"),
        "Tumblers & Water Glasses": ("tumbler", "水杯", "something for the kitchen", "厨房用品"),
        "Cutting Boards": ("cutting board", "砧板", "something for the kitchen", "厨房用品"),
        "Container Sets": ("set of food storage containers", "保鲜盒", "something for the kitchen", "厨房用品"),
        "Cookie Cutters": ("set of cookie cutters", "饼干模具", "something for the kitchen", "厨房用品"),
        "Teapots": ("teapot", "茶壶", "something for the kitchen", "厨房用品"),
        "Aprons": ("apron", "围裙", "something for the kitchen", "厨房用品"),
        "Tablecloths": ("tablecloth", "桌布", "something for the dining table", "餐桌用品"),
        "Table Runners": ("table runner", "桌旗", "something for the dining table", "餐桌用品"),
        "Place Mats": ("set of placemats", "餐垫", "something for the dining table", "餐桌用品"),
        "Coasters": ("set of coasters", "杯垫", "something for the dining table", "餐桌用品"),
        "Throw Pillow Covers": ("set of throw pillow covers", "抱枕套", "something cozy for my living room", "客厅软装"),
        "Throws": ("throw blanket", "盖毯", "something cozy for my living room", "客厅软装"),
        "Area Rugs": ("area rug", "地毯", "something cozy for my living room", "客厅软装"),
        "Sheet & Pillowcase Sets": ("bed sheet set", "床单枕套套装", "some bedding", "床上用品"),
        "Comforter Sets": ("comforter set", "被子套装", "some bedding", "床上用品"),
        "Duvet Cover Sets": ("duvet cover set", "被套套装", "some bedding", "床上用品"),
        "Wall Clocks": ("wall clock", "挂钟", "some home decor", "家居装饰"),
        "Vases": ("vase", "花瓶", "some home decor", "家居装饰"),
        "Artificial Flowers": ("bunch of artificial flowers", "仿真花", "some home decor", "家居装饰"),
        "Posters & Prints": ("poster", "海报", "some home decor", "家居装饰"),
        "Jar Candles": ("jar candle", "香薰蜡烛", "some home decor", "家居装饰"),
        "Panels": ("pair of curtain panels", "窗帘", "some home decor", "家居装饰"),
        "Shower Curtains": ("shower curtain", "浴帘", "something for the bathroom", "浴室用品"),
        "Hand Towels": ("set of hand towels", "擦手巾", "something for the bathroom", "浴室用品"),
        "Bath Rugs": ("bath mat", "浴室地垫", "something for the bathroom", "浴室用品"),
    },
    "toys": {
        "Board Games": ("board game", "桌游", "a game", "一个游戏"),
        "Card Games": ("card game", "卡牌游戏", "a game", "一个游戏"),
        "Jigsaw Puzzles": ("jigsaw puzzle", "拼图", "a toy", "一个玩具"),
        "Building Sets": ("building set", "积木", "a toy", "一个玩具"),
        "Action Figures": ("action figure", "可动人偶", "a toy", "一个玩具"),
        "Dolls": ("doll", "娃娃", "a toy", "一个玩具"),
        "Plush Figures": ("plush toy", "毛绒玩具", "a toy", "一个玩具"),
        "Stuffed Animals & Teddy Bears": ("stuffed animal", "毛绒公仔", "a toy", "一个玩具"),
        "Pre-Kindergarten Toys": ("preschool toy", "学前益智玩具", "a toy", "一个玩具"),
        "Cars, Trucks, Boats & Planes": ("toy vehicle", "玩具车", "a toy", "一个玩具"),
        "Role Play & Dress Up": ("pretend play set", "角色扮演玩具", "a toy", "一个玩具"),
        "Art & Creativity": ("arts and crafts kit", "手工美术套装", "a toy", "一个玩具"),
        "Science Kits & Toys": ("science kit", "科学实验玩具", "a toy", "一个玩具"),
        "Sorting & Stacking Toys": ("stacking toy", "叠叠乐玩具", "a toy", "一个玩具"),
        "Remote & App Controlled Vehicles": ("remote control car", "遥控车", "a toy", "一个玩具"),
        "Magic Kits & Accessories": ("magic kit", "魔术道具", "a toy", "一个玩具"),
        "Balloons": ("pack of balloons", "气球", "some party supplies", "派对用品"),
        "Stickers": ("pack of stickers", "贴纸", "some party supplies", "派对用品"),
        "Multi-Item Party Favor Packs": ("party favor pack", "派对小礼品套装", "some party supplies", "派对用品"),
    },
}

AUDIENCE_TEXT = {
    "women": ("for women", "女款"), "men": ("for men", "男款"),
    "girls": ("for a girl", "女童款"), "boys": ("for a boy", "男童款"),
    "baby": ("for a baby", "婴儿款"), "unisex": ("unisex", "男女通用"),
}
CLEAR_MATERIALS = {"cotton", "ceramic", "glass", "wood", "stainless steel", "polyester",
                   "metal", "plastic", "microfiber", "bamboo", "porcelain", "silk", "wool"}
MATERIAL_ZH = {"cotton": "纯棉", "ceramic": "陶瓷", "glass": "玻璃", "wood": "木质",
               "stainless steel": "不锈钢", "polyester": "涤纶", "metal": "金属",
               "plastic": "塑料", "microfiber": "超细纤维", "bamboo": "竹制",
               "porcelain": "瓷", "silk": "丝绸", "wool": "羊毛"}

LEVEL_SHARE = [("vague", 0.45), ("partial", 0.35), ("complete", 0.20)]
ZH_SHARE = 0.30


def a(word):
    """不定冠词：a / an。"""
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def nice_ceiling(x):
    step = 5 if x < 100 else 10
    return int(max(10, math.ceil(x / step) * step))


def need_text_en(need, names):
    parts = [a(names[0])]
    if need.get("audience"):
        parts.append(AUDIENCE_TEXT[need["audience"]][0])
    if need.get("age") is not None:
        parts.append(f"for a {need['age']}-year-old child")
    if need.get("material"):
        parts[0] = a(f"{need['material']} {names[0]}")
    parts.append(f"under ${need['price_max']}")
    return " ".join(parts)


def need_text_zh(need, names):
    s = ""
    if need.get("audience"):
        s += AUDIENCE_TEXT[need["audience"]][1]
    if need.get("age") is not None:
        s += f"适合{need['age']}岁孩子的"
    if need.get("material"):
        s += MATERIAL_ZH[need["material"]]
    return f"{s}{names[1]}，预算{need['price_max']}美元以内"


def opening(level, lang, need, names, rng):
    if level == "complete":
        return (f"I'm looking for {need_text_en(need, names)}." if lang == "en"
                else f"我想买{need_text_zh(need, names)}。")
    if level == "vague":
        en = [f"I'm looking for {names[2]}.", f"Can you recommend {names[2]}?",
              f"I need {names[2]}.", f"Any suggestions for {names[2]}?"]
        zh = [f"我想买{names[3]}。", f"能推荐{names[3]}吗？", f"帮我找{names[3]}。"]
        return rng.choice(en if lang == "en" else zh)
    # partial：给出具体品类，最多再给一个非预算信息，预算留空
    extra_en, extra_zh = "", ""
    if need.get("material") and rng.random() < 0.5:
        return (f"I want {a(need['material'] + ' ' + names[0])}." if lang == "en"
                else f"想要一个{MATERIAL_ZH[need['material']]}{names[1]}。")
    if need.get("audience") and rng.random() < 0.4:
        extra_en, extra_zh = " " + AUDIENCE_TEXT[need["audience"]][0], AUDIENCE_TEXT[need["audience"]][1]
    return (f"I want {a(names[0])}{extra_en}." if lang == "en" else f"想买{extra_zh}{names[1]}。")


def mentioned_slots(level, text, need):
    """开场白中已经给出的槽位（用于生成快照和期望追问）。"""
    if level == "complete":
        return set(k for k in need if k != "domain")
    given = set()
    if level == "partial":
        given.add("product_type")
        if need.get("audience") and (AUDIENCE_TEXT[need["audience"]][0] in text
                                     or AUDIENCE_TEXT[need["audience"]][1] in text):
            given.add("audience")
        if need.get("material") and (need["material"] in text or MATERIAL_ZH[need["material"]] in text):
            given.add("material")
    return given


ANSWER_TEMPLATES = {
    "product_type": ("{type_en_a}.", "{type_zh}。"),
    "audience": ("It's {aud_en}.", "要{aud_zh}的。"),
    "age": ("My kid is {age} years old.", "孩子{age}岁。"),
    "price_max": ("My budget is up to ${price}.", "预算{price}美元以内。"),
    "material": ("I'd prefer {material}.", "最好是{material_zh}的。"),
}
QUESTION_TEMPLATES = {
    "product_type": ("What kind of item are you looking for exactly?", "您具体想要哪一类商品？"),
    "audience": ("Who is it for — men, women or kids?", "是给谁买的？男款、女款还是儿童款？"),
    "age": ("How old is the child?", "孩子多大了？"),
    "price_max": ("What's your budget?", "您的预算是多少？"),
    "material": ("Do you have a preferred material?", "对材质有偏好吗？"),
}


def answer_text(slot, need, names, lang):
    en, zh = ANSWER_TEMPLATES[slot]
    fmt = dict(type_en=names[0], type_en_a=a(names[0]).capitalize(), type_zh=names[1], age=need.get("age"),
               price=need.get("price_max"), material=need.get("material"),
               material_zh=MATERIAL_ZH.get(need.get("material"), ""),
               aud_en=AUDIENCE_TEXT.get(need.get("audience"), ("", ""))[0],
               aud_zh=AUDIENCE_TEXT.get(need.get("audience"), ("", ""))[1])
    return (en if lang == "en" else zh).format(**fmt)


def make_snapshots(sc, names):
    """两个标注时点：开场白之后；以及（若需要）回答了一个问题之后。"""
    need, lang = sc["hidden_need"], sc["language"]
    spec = A.SLOT_SPECS[need["domain"]]
    given = set(sc["given_slots"])
    order = [s for s in spec["required"] if s in need] + [s for s in need if s not in spec["required"] and s != "domain"]
    missing = [s for s in order if s not in given]
    snaps = [{"turns": [{"role": "user", "text": sc["opening"]}],
              "missing_required": [s for s in spec["required"] if s not in given]}]
    if missing:
        slot = missing[0]
        q = QUESTION_TEMPLATES[slot][0 if lang == "en" else 1]
        a = answer_text(slot, need, names, lang)
        given2 = given | {slot}
        snaps.append({"turns": [{"role": "user", "text": sc["opening"]},
                                {"role": "assistant", "text": q, "action": "ask", "asked_slot": slot},
                                {"role": "user", "text": a}],
                      "missing_required": [s for s in spec["required"] if s not in given2]})
    else:
        extra = ("Ideally something with good reviews." if lang == "en" else "最好评价好一点的。")
        snaps.append({"turns": [{"role": "user", "text": sc["opening"]},
                                {"role": "user", "text": extra}],
                      "missing_required": []})
    for s in snaps:
        s["expected_action"] = "ask" if s["missing_required"] else "recommend"
    return snaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    att = A.all_attributes()
    by_type = {}
    for asin, a in att.items():
        by_type.setdefault((a["domain"], a["product_type"]), []).append(asin)

    scenarios = []
    for domain, types in TYPES.items():
        n = args.per_domain
        levels = [lvl for lvl, share in LEVEL_SHARE for _ in range(round(n * share))][:n]
        while len(levels) < n:
            levels.append("vague")
        rng.shuffle(levels)
        type_cycle = list(types)
        rng.shuffle(type_cycle)
        made, attempts = 0, 0
        while made < n and attempts < n * 50:
            attempts += 1
            ptype = type_cycle[(made + attempts) % len(type_cycle)]
            pool = [x for x in by_type.get((domain, ptype), []) if att[x]["price"]]
            if not pool:
                continue
            anchor = att[rng.choice(pool)]
            need = {"domain": domain, "product_type": ptype}
            if domain == "clothing":
                if anchor["audience"] not in AUDIENCE_TEXT:
                    continue
                need["audience"] = anchor["audience"]
            if domain == "toys":
                if anchor["age_years"] is None:
                    continue
                need["age"] = int(min(14, math.ceil(anchor["age_years"]) + rng.randint(0, 3)))
            if domain == "home_kitchen" and anchor["material"] in CLEAR_MATERIALS and rng.random() < 0.5:
                need["material"] = anchor["material"]
            need["price_max"] = nice_ceiling(anchor["price"] * rng.uniform(1.1, 1.6))

            feasible = [x for x in pool if A.satisfies(x, need)[0]]
            if len(feasible) < MIN_FEASIBLE:
                continue
            names = types[ptype]
            level = levels[made]
            lang = "zh" if rng.random() < ZH_SHARE else "en"
            text = opening(level, lang, need, names, rng)
            sc = {
                "scenario_id": f"S{len(scenarios) + 1:03d}",
                "domain": domain,
                "level": level,
                "language": lang,
                "hidden_need": need,
                "opening": text,
                "given_slots": sorted(mentioned_slots(level, text, need)),
                "full_query_en": f"I'm looking for {need_text_en(need, names)}.",
                "full_query_zh": f"我想买{need_text_zh(need, names)}。",
                "type_names": {"en": names[0], "zh": names[1]},
                "n_feasible": len(feasible),
            }
            sc["snapshots"] = make_snapshots(sc, names)
            scenarios.append(sc)
            made += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "scenarios.jsonl", "w", encoding="utf-8") as f:
        for sc in scenarios:
            f.write(json.dumps(sc, ensure_ascii=False) + "\n")

    print(f"Scenarios: {len(scenarios)}")
    print("by domain:", dict(Counter(s["domain"] for s in scenarios)))
    print("by level:", dict(Counter(s["level"] for s in scenarios)))
    print("by language:", dict(Counter(s["language"] for s in scenarios)))
    print("snapshot expected actions:", dict(Counter(sn["expected_action"] for s in scenarios for sn in s["snapshots"])))
    print(f"feasible products per scenario: min {min(s['n_feasible'] for s in scenarios)}, "
          f"median {sorted(s['n_feasible'] for s in scenarios)[len(scenarios) // 2]}")


if __name__ == "__main__":
    main()
