# Q4 标注指南（场景式对话推荐）

每位标注者**独立**完成自己的 `sheet_<名字>.xlsx`，标注期间不要互相讨论，否则一致率（IAA）无效。表格有两个工作表。

背景：系统是一个购物助手，用户可能只说一句很模糊的话（例如“想买双鞋”）。助手要么先反问补全需求，要么直接推荐商品。

---

## 工作表 1：decisions（360 条）

每条是一段对话的当前状态。请回答：

**should_ask**：此时助手应该 **反问（1）** 还是 **直接推荐（0）**？
**ask_about**（仅当填 1 时）：最应该问的**一个**属性，从下拉框中选：`product type` / `who it is for` / `child age` / `budget` / `material`

### 判定规则

推荐前需要知道的关键信息：

| 领域 | 必须知道 | 可选 |
|---|---|---|
| 服装、鞋、首饰、包（clothing） | 具体品类（如“凉鞋”，不是“鞋”）、给谁（男/女/儿童）、预算 | — |
| 家居厨房（home_kitchen） | 具体品类（如“砧板”，不是“厨房用品”）、预算 | 材质 |
| 玩具（toys） | 具体品类（如“拼图”，不是“玩具”）、孩子年龄、预算 | — |

1. 任何一项“必须知道”的信息缺失 → `should_ask = 1`；都已知道 → `0`。
2. 用户说“都可以 / 没有偏好” → 视为已知道。
3. `ask_about` 选**最影响推荐结果**的缺失项。品类不明确时优先问品类，其次是人群或年龄，最后是预算。
4. 只根据对话内容判断，不要猜测用户没说的信息。

### 示例

| 对话 | should_ask | ask_about |
|---|---|---|
| USER: I'm looking for some shoes. | 1 | product type |
| USER: 想买饼干模具。 | 1 | budget |
| USER: I want a hoodie for men. / ASSISTANT: What's your budget? / USER: My budget is up to $40. | 0 | （空） |
| USER: I'm looking for a jigsaw puzzle for a 6-year-old child under $25. | 0 | （空） |

---

## 工作表 2：relevance（约 1,080 条）

每条是一个用户的**完整需求**（英文 + 中文）和一个候选商品（标题后附价格、评分）。请判断：

**relevant**：这个商品**是否满足**该需求？ 满足 = `1`，不满足 = `0`

### 判定规则

1. **品类必须对**：要“耳钉”，给的是“项链” → 0；要“遥控车”，给的是“遥控车配件” → 0。近义品类算对（“毛绒玩具”与“毛绒公仔”都算毛绒玩具）。
2. **人群 / 年龄必须对**：要女款，商品明确是男款 → 0；男女通用或未注明 → 可以算 1。要给 3 岁孩子，商品写“8 岁以上” → 0；未注明年龄 → 按常识判断。
3. **预算是硬约束**：价格超过预算 → 0（看标题后面的 `Price`）。
4. **材质**：需求写了材质，商品明确是其他材质 → 0；没写材质 → 按常识判断。
5. **只看给出的商品信息**，不要上网查。
6. 拿不准时，按“如果我是这个用户，会不会愿意买它”判断，并在 comment 写原因。

### 示例

| 需求 | 商品 | relevant | 原因 |
|---|---|---|---|
| a pair of sandals for women under $80 | Women's Flat Sandals …; Price: $29.99 | 1 | 品类、人群、价格都满足 |
| a pair of sandals for women under $80 | Men's Leather Slide Sandals …; Price: $45.00 | 0 | 男款 |
| a jigsaw puzzle for a 6-year-old child under $25 | 1000 Piece Puzzle, Ages 12+ …; Price: $18.99 | 0 | 年龄不符 |
| a cutting board under $30 | Bamboo Cutting Board Set …; Price: $36.99 | 0 | 超预算 |

---

## 流程

1. 在各自的表中填写，不要修改其他列，不要排序或删除行。
2. 收齐后运行 `python compute_iaa.py`：分别计算 should_ask、ask_about、relevance 的一致率、Cohen's κ 和 Fleiss' κ，生成金标准 `gold_decisions.csv`、`gold_relevance.csv`。
3. 3 人标注时按多数票定标；2 人标注时，分歧写入 `adjudication_*.csv`，由第三人在 `final_label` 列裁决后重新运行。
