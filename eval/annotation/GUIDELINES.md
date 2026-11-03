# Q4 标注指南（Annotation Guidelines）

## 任务

每条记录是一对 **(用户查询, 检索到的商品段落)**。判断：**这个商品是否满足该查询的需求？**

- `1` = 相关（relevant）：推荐给提出这个查询的用户是合理的
- `0` = 不相关（not relevant）

共 1,050 条（210 个查询 × Top-5），覆盖服装、家居厨房、玩具三个类目，外加 20 个超出语料范围的查询。每位标注者**独立**完成，标注期间不要讨论，否则一致率（IAA）无效。

## 判定规则

1. **只看段落内容判断**，不要上网查商品，不要凭品牌印象。
2. **品类必须对**：查询要“手表”，商品是“表带” → `0`；查询要“靴子”，商品是“袜子” → `0`；查询要“遥控车”，商品是“遥控车配件” → `0`（除非查询本身要配件）。
3. **人群 / 年龄必须对**：查询写明 men's / women's / kids / baby，商品明确是另一人群 → `0`；商品为 unisex 或未注明 → 可以是 `1`。玩具查询写了年龄（如 2-year-old、ages 8+），商品标注的适用年龄明显不符 → `0`；未标注年龄 → 按常识判断。
   **尺寸 / 规格**：查询写明尺寸（queen bed、18x18 inches、1000-piece），段落写了不同尺寸 → `0`；段落列出多个可选尺寸且包含所需尺寸 → `1`。
4. **明确写出的约束必须满足**：查询提到的硬性属性（waterproof、100% cotton、sterling silver、steel toe、with pockets 等）：
   - 段落中**明确写了**该属性 → 满足
   - 段落中**明确相反** → `0`
   - 段落中**没提到** → `0`（因为系统无法依据证据确认）
   - 近义表达算满足：water resistant ≈ waterproof（**例外**：查询明确要求 "waterproof for swimming" 时不算）
   - **价格**：每条商品段落在标题后都列出了 `Price`、`Average rating`、评论数。查询写了明确价格上限（under $50）→ 按价格判断，超出即 `0`。
5. **主观形容词放宽**：comfortable、soft、affordable、budget、good for everyday 这类词，只要品类和人群对、段落没有明显矛盾 → 可标 `1`（affordable / budget 不按具体价格卡死，除非价格明显离谱）。
6. **问句型查询**（Which…/What…）：商品能作为该问题的一个正确答案 → `1`。
7. **超出语料范围的查询**（笔记本电脑、手机、汽车机油等）：几乎都应为 `0`，除非检索到的商品确实满足需求。
8. **拿不准时**：按“如果我是这个用户，看到这个推荐会觉得有用吗？”判断，并在 `comment` 栏写原因，方便后续仲裁。

## 示例

| 查询 | 商品 | 标签 | 原因 |
|---|---|---|---|
| I need waterproof men's boots for hiking. | Foxelli Men's Hiking Boots – Waterproof Suede Leather… | 1 | 品类、人群、waterproof 都满足 |
| I need waterproof men's boots for hiking. | C.PARAVANO Pointed Chelsea Boots Women… | 0 | 女款，且非徒步靴 |
| A men's watch with a leather strap. | DITOU Watch Band 20mm… Leather Watch Strap | 0 | 表带不是手表 |
| I want a pair of yoga pants. | PERSIT Yoga Pants for Women with Pockets… | 1 | 未限定性别，品类正确 |
| Which earrings are made of 925 sterling silver? | Moonstone Stud Earrings 925 Sterling Silver… | 1 | 明确写了 925 sterling silver |
| Find a non-stick frying pan. | … Nonstick Frying Pan, 10 inch … | 1 | 品类正确，明确 non-stick |
| A 1000-piece jigsaw puzzle with a landscape picture. | … 500 Piece Jigsaw Puzzle, Mountain Lake … | 0 | 片数不符 |
| Suggest a toy for a 2-year-old toddler. | … Building Blocks, Ages 8+ … | 0 | 适用年龄不符 |
| What is the best smartphone under $500? | Thread Tank Women's T-Shirt | 0 | 超出语料范围 |

## 流程

1. 打开自己的 `sheet_<名字>.xlsx`，在 `relevant (1/0)` 列填 1 或 0，不要改动其他列，不要排序或删行。
2. 交回文件后运行 `python compute_iaa.py`，计算一致率和 Cohen's / Fleiss' κ。
3. 存在分歧的记录会导出到 `adjudication.csv`：
   - 3 人标注时按多数票自动定标；
   - 2 人标注时由第三人或讨论后在 `final_label` 列填写最终标签，再运行一次 `compute_iaa.py` 生成金标准 `gold_labels.csv`。
