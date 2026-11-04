环境：Darwin arm64 arm，编码设备 mps:0，检索单元 chunk（48107 个），每条查询重复 10 次取中位数（已预热）。

| 查询 | BM25 | query 编码 | FAISS 搜索 | RRF | 总计 (ms) |
|---|---|---|---|---|---|
| waterproof men's hiking boots | 0.3 | 16.7 | 1.6 | 0.01 | 18.6 |
| non-stick frying pan for induction stove | 0.2 | 16.7 | 1.6 | 0.01 | 18.5 |
| board game for family night with kids | 0.2 | 16.5 | 1.6 | 0.01 | 18.4 |
| sterling silver earrings for sensitive ears | 0.2 | 16.5 | 1.6 | 0.01 | 18.4 |
| soft queen size bed sheets | 0.2 | 16.4 | 1.6 | 0.01 | 18.2 |

**waterproof men's hiking boots**

| # | 商品 | 类目 | 价格 | 评分 | 命中字段 | BM25 名次 | 稠密名次 |
|---|---|---|---|---|---|---|---|
| 1 | riemot Women's Men's Winter Snow Boots Waterproof Outdoor Boots Anti-s | Clothing, Shoes & Jewelry | $29.99 | 4.0 (133) | description | 1 | 3 |
| 2 | Nord Trail Mt. Washington Hiking Shoes for Men - Lightweight High Trac | Clothing, Shoes & Jewelry | $64.99 | 4.5 (22) | overview | 5 | 2 |
| 3 | BEARPAW Men's Traverse Hiking Boot | Clothing, Shoes & Jewelry | $119.99 | 3.8 (22) | description | 10 | 1 |
| 4 | Sorel Men's Kezar Storm WP Boot — Waterproof Leather Winter Boots | Clothing, Shoes & Jewelry | $249.99 | 4.3 (113) | specs | 4 | 8 |
| 5 | Rocky Men's Red Mountain Hiking Boot | Clothing, Shoes & Jewelry | $54.56 | 4.2 (24) | description | 8 | 4 |

**non-stick frying pan for induction stove**

| # | 商品 | 类目 | 价格 | 评分 | 命中字段 | BM25 名次 | 稠密名次 |
|---|---|---|---|---|---|---|---|
| 1 | Trudeau Pure Ceramic Frying Pan, 12-Inch, Grey | Home & Kitchen | $46.98 | 4.3 (31) | overview | 3 | 2 |
| 2 | Phantom Chef 12" Deep Frying Pan with Granite finish & Lid / PFOA Free | Home & Kitchen | $49.97 | 4.5 (20) | overview | 6 | 1 |
| 3 | Frying Pan Double Side Pan Folding Frying Pan Fried Egg for Household  | Home & Kitchen | $21.19 | 1.0 (1) | overview | 1 | 8 |
| 4 | OXO Good Grips Hard Anodized PFOA-Free Nonstick 12" Frying Pan Skillet | Home & Kitchen | $94.94 | 4.7 (67) | description | 2 | 12 |
| 5 | Nuwave 3-quart Grill Pan with Tempered Glass Lid, Forged Lightweight,  | Home & Kitchen | $57.04 | 4.9 (25) | specs | 9 | 5 |

**board game for family night with kids**

| # | 商品 | 类目 | 价格 | 评分 | 命中字段 | BM25 名次 | 稠密名次 |
|---|---|---|---|---|---|---|---|
| 1 | Pandasaurus Games The Mind - Family-Friendly Board Games - Game Night  | Toys & Games | $14.95 | 4.6 (5740) | description | 1 | 1 |
| 2 | eeBoo: Gathering a Garden Board Game, Educational Games and Activities | Toys & Games | $14.99 | 4.7 (415) | overview | 3 | 5 |
| 3 | Hasbro Gaming Sorry! Board Game: Disney Villains Edition Kids, Family  | Toys & Games | $23.05 | 4.8 (2380) | description | 4 | 6 |
| 4 | 2 Pack 29” Board Games for Kids, Large Dart Board for Kids with 20 Sti | Toys & Games | $19.99 | 4.6 (123) | description | 7 | 8 |
| 5 | A Little Wordy by Exploding Kittens - A Clever Scramble Word Game of T | Toys & Games | $14.88 | 4.7 (1296) | overview | 8 | 10 |

**sterling silver earrings for sensitive ears**

| # | 商品 | 类目 | 价格 | 评分 | 命中字段 | BM25 名次 | 稠密名次 |
|---|---|---|---|---|---|---|---|
| 1 | Ai.Moichien Women 925 Sterling Silver Stud Earrings Crystal Starfish F | Clothing, Shoes & Jewelry | $0.99 | 5.0 (1) | overview | 1 | 1 |
| 2 | Crawler Earrings for Women 14K Gold Plated Sterling Silver Post Ear Cu | Clothing, Shoes & Jewelry | $11.99 | 4.1 (35) | description | 7 | 4 |
| 3 | 925 Sterling Silver Heart Earrings Dangle Earrings Light Drop Earrings | Clothing, Shoes & Jewelry | $10.99 | 4.3 (66) | overview | 11 | 3 |
| 4 | Hoops & Loops 925 Sterling Silver 2mm Oval Click-Top Hoop Earrings for | Clothing, Shoes & Jewelry | $26.99 | 4.4 (128) | overview | 3 | 15 |
| 5 | SLUYNZ 925 Sterling Silver Ear Cuff Earrings Chain for Women Teen Girl | Clothing, Shoes & Jewelry | $13.99 | 4.3 (351) | overview | 4 | 18 |

**soft queen size bed sheets**

| # | 商品 | 类目 | 价格 | 评分 | 命中字段 | BM25 名次 | 稠密名次 |
|---|---|---|---|---|---|---|---|
| 1 | Diamond Bedding 800 Thread Count 100% Organic Bamboo Sheets - 6 pcs Ba | Home & Kitchen | $93.99 | 3.8 (3) | overview | 1 | 8 |
| 2 | Top Selling on Amazon Queen Size Sheets Luxury Soft 550-TC Egyptian Co | Home & Kitchen | $89.99 | 2.0 (1) | overview | 9 | 1 |
| 3 | MattRest Luxury Silky Soft - Wrinkle Resistant 1500 Thread Count Egypt | Home & Kitchen | $25.99 | 4.4 (5) | specs | 10 | 6 |
| 4 | RV Sheets, RV Short Queen (60x75) Inch, Upto 8" Deep Pocket, 400 Threa | Home & Kitchen | $41.99 | 4.5 (25) | overview | 11 | 12 |
| 5 | Scmkd Fitted Sheet Only-Bottom Sheet - Deep Pocket -Ultra-Soft Mricefi | Home & Kitchen | $18.47 | 1.0 (1) | overview | 15 | 10 |
