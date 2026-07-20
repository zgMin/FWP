# 扩展版答案可验证任务报告

四种方法共享严格 `<answer>...</answer>` 输出协议；知识上下文只出现在 Full prompt，或由闭式解转换为可一次融合的固定权重。全部生成均为 greedy。

## 设置

- 模型：`/root/zgm/models/Qwen/Qwen3.5-0.8B`
- 数据：774 条；中文 312，英文 462。
- QTraj：最后 8 层，prefix=4，rank=8，ridge=0.001；lm_head tokens=32。
- Teacher-token：lm_head tokens=32，rank=128，ridge=0.1；margin=4.0；Top-k=8。
- Base 与两种 query-only 路径都保留共享格式协议，但不接收知识段落。

## 总体

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| 全部 | 774 | Base (query only) | 88.89% | 88.89% | 4.91% | 15.20% | 7.11% |
| 全部 | 774 | Full prompt | 97.29% | 97.29% | 56.33% | 71.34% | 67.05% |
| 全部 | 774 | QTraj + Teacher-token | 96.90% | 96.90% | 55.94% | 70.98% | 66.80% |
| 全部 | 774 | QTraj + Top-k | 97.29% | 97.29% | 54.91% | 70.04% | 65.63% |
| 全部 | 774 | QTraj + Auto-Margin | 97.29% | 97.29% | 56.33% | 71.34% | 67.05% |

## 按语言

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| en | 462 | Base (query only) | 92.64% | 92.64% | 6.71% | 11.62% | 9.31% |
| en | 462 | Full prompt | 98.70% | 98.70% | 56.28% | 66.49% | 67.10% |
| en | 462 | QTraj + Teacher-token | 98.27% | 98.27% | 55.84% | 66.04% | 66.88% |
| en | 462 | QTraj + Top-k | 98.92% | 98.92% | 54.33% | 64.86% | 64.94% |
| en | 462 | QTraj + Auto-Margin | 98.70% | 98.70% | 56.28% | 66.49% | 67.10% |
| zh | 312 | Base (query only) | 83.33% | 83.33% | 2.24% | 20.50% | 3.85% |
| zh | 312 | Full prompt | 95.19% | 95.19% | 56.41% | 78.52% | 66.99% |
| zh | 312 | QTraj + Teacher-token | 94.87% | 94.87% | 56.09% | 78.29% | 66.67% |
| zh | 312 | QTraj + Top-k | 94.87% | 94.87% | 55.77% | 77.71% | 66.67% |
| zh | 312 | QTraj + Auto-Margin | 95.19% | 95.19% | 56.41% | 78.52% | 66.99% |

## 按来源

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| cmrc2018 | 155 | Base (query only) | 73.55% | 73.55% | 0.65% | 17.99% | 1.94% |
| cmrc2018 | 155 | Full prompt | 93.55% | 93.55% | 50.97% | 78.04% | 62.58% |
| cmrc2018 | 155 | QTraj + Teacher-token | 93.55% | 93.55% | 50.97% | 78.01% | 61.94% |
| cmrc2018 | 155 | QTraj + Top-k | 92.90% | 92.90% | 50.32% | 76.81% | 62.58% |
| cmrc2018 | 155 | QTraj + Auto-Margin | 93.55% | 93.55% | 50.97% | 78.04% | 62.58% |
| drcd | 155 | Base (query only) | 92.90% | 92.90% | 3.87% | 23.28% | 5.81% |
| drcd | 155 | Full prompt | 96.77% | 96.77% | 61.29% | 78.72% | 70.97% |
| drcd | 155 | QTraj + Teacher-token | 96.13% | 96.13% | 60.65% | 78.30% | 70.97% |
| drcd | 155 | QTraj + Top-k | 96.77% | 96.77% | 60.65% | 78.32% | 70.32% |
| drcd | 155 | QTraj + Auto-Margin | 96.77% | 96.77% | 61.29% | 78.72% | 70.97% |
| natural_questions | 150 | Base (query only) | 87.33% | 87.33% | 4.00% | 8.98% | 6.00% |
| natural_questions | 150 | Full prompt | 97.33% | 97.33% | 44.00% | 56.52% | 49.33% |
| natural_questions | 150 | QTraj + Teacher-token | 96.67% | 96.67% | 43.33% | 56.00% | 49.33% |
| natural_questions | 150 | QTraj + Top-k | 98.67% | 98.67% | 41.33% | 54.50% | 46.67% |
| natural_questions | 150 | QTraj + Auto-Margin | 97.33% | 97.33% | 44.00% | 56.52% | 49.33% |
| relation_extraction | 150 | Base (query only) | 96.00% | 96.00% | 8.67% | 13.55% | 11.33% |
| relation_extraction | 150 | Full prompt | 100.00% | 100.00% | 70.67% | 82.40% | 86.67% |
| relation_extraction | 150 | QTraj + Teacher-token | 99.33% | 99.33% | 70.00% | 81.54% | 86.00% |
| relation_extraction | 150 | QTraj + Top-k | 99.33% | 99.33% | 68.67% | 81.23% | 84.67% |
| relation_extraction | 150 | QTraj + Auto-Margin | 100.00% | 100.00% | 70.67% | 82.40% | 86.67% |
| squad | 10 | Base (query only) | 60.00% | 60.00% | 0.00% | 6.22% | 0.00% |
| squad | 10 | Full prompt | 100.00% | 100.00% | 60.00% | 63.33% | 70.00% |
| squad | 10 | QTraj + Teacher-token | 100.00% | 100.00% | 60.00% | 63.33% | 70.00% |
| squad | 10 | QTraj + Top-k | 90.00% | 90.00% | 40.00% | 44.81% | 60.00% |
| squad | 10 | QTraj + Auto-Margin | 100.00% | 100.00% | 60.00% | 63.33% | 70.00% |
| synthetic | 4 | Base (query only) | 75.00% | 75.00% | 0.00% | 0.00% | 0.00% |
| synthetic | 4 | Full prompt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| synthetic | 4 | QTraj + Teacher-token | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| synthetic | 4 | QTraj + Top-k | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| synthetic | 4 | QTraj + Auto-Margin | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| triviaqa | 150 | Base (query only) | 97.33% | 97.33% | 8.00% | 12.85% | 11.33% |
| triviaqa | 150 | Full prompt | 98.67% | 98.67% | 53.33% | 60.32% | 64.67% |
| triviaqa | 150 | QTraj + Teacher-token | 98.67% | 98.67% | 53.33% | 60.32% | 64.67% |
| triviaqa | 150 | QTraj + Top-k | 99.33% | 99.33% | 53.33% | 59.72% | 63.33% |
| triviaqa | 150 | QTraj + Auto-Margin | 98.67% | 98.67% | 53.33% | 60.32% | 64.67% |

## 按数据分区

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| base_short | 24 | Base (query only) | 75.00% | 75.00% | 0.00% | 9.46% | 0.00% |
| base_short | 24 | Full prompt | 100.00% | 100.00% | 79.17% | 83.89% | 87.50% |
| base_short | 24 | QTraj + Teacher-token | 100.00% | 100.00% | 79.17% | 83.89% | 87.50% |
| base_short | 24 | QTraj + Top-k | 95.83% | 95.83% | 70.83% | 76.17% | 83.33% |
| base_short | 24 | QTraj + Auto-Margin | 100.00% | 100.00% | 79.17% | 83.89% | 87.50% |
| expanded_public | 750 | Base (query only) | 89.33% | 89.33% | 5.07% | 15.39% | 7.33% |
| expanded_public | 750 | Full prompt | 97.20% | 97.20% | 55.60% | 70.94% | 66.40% |
| expanded_public | 750 | QTraj + Teacher-token | 96.80% | 96.80% | 55.20% | 70.57% | 66.13% |
| expanded_public | 750 | QTraj + Top-k | 97.33% | 97.33% | 54.40% | 69.85% | 65.07% |
| expanded_public | 750 | QTraj + Auto-Margin | 97.20% | 97.20% | 55.60% | 70.94% | 66.40% |

## 输入输出示例

### pair--zh-drcd-dc7204b7760e (drcd)

- Prompt：资料：1913年7月畢業，8月16日入天津南開學校，因表現優異而為學校創辦人嚴范孫、張伯苓器之為「宰相之才」，特免其學雜費，這是南開當時唯一一個免費生。青年周恩來相貌英俊瀟灑，在南開曾反串表演，相識了妻子和革命伴侶鄧穎超。1917年至1919年，周恩來赴日本求學。修讀日語一年後，曾應考東京高等師範學校以及第一高等學校，但皆未錄取。1919年4月，得悉南開學校即將創...
请仅依据资料回答查询。
- Query：周恩來在哪一年時進入天津南開學校就讀?
- Gold：`['1913年']`
- Base (query only)：`<answer>1921</answer>` (EM=False, F1=0.667)
- Full prompt：`<answer>1913 年</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>1913 年 8 月 13 日 13 日 13 日 13` (EM=False, F1=0.345)
- QTraj + Top-k：`<answer>1913 年</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>1913 年</answer>` (EM=True, F1=1.000)

### pair--zh-cmrc2018-c985eb8fbb73 (cmrc2018)

- Prompt：资料：...4年内，又进行了《内蒙古植物志》第二版的编辑工作。第二版共5卷，于1998年全部出版完毕，总共收录野生维管植物134科、681属、2270种及栽培植物70属、172种。到目前为止，是介绍内蒙古植物的最新、最全面的志书。和中国大陆大部分地方植物志一样，《内蒙古植物志》的蕨类植物采用1978年秦仁昌系统，裸子植物采用1978年郑万钧系统，被子植物采用1936年恩...
请仅依据资料回答查询。
- Query：《内蒙古植物志》到目前为止有着怎样的地位？
- Gold：`['是介绍内蒙古植物的最新、最全面的志书', '介绍内蒙古植物的最新、最全面的志书']`
- Base (query only)：`<answer>《内蒙古植物志》是内蒙古地区植物分类学研究的重要基础文献，其地位在于系统整理了内蒙古地区的植物种类、分布及生态特征` (EM=False, F1=0.237)
- Full prompt：`<answer>介绍内蒙古植物的最新、最全面的志书</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>介绍内蒙古植物的最新、最全面的志书</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>《内蒙古植物志》是介绍内蒙古植物的最新、最全面的志书，是内蒙古植物志的志书</answer>` (EM=False, F1=0.680)
- QTraj + Auto-Margin：`<answer>介绍内蒙古植物的最新、最全面的志书</answer>` (EM=True, F1=1.000)

### pair--zh-knowledge-011 (synthetic)

- Prompt：资料：在虚构的澄星档案中，霁川站由青岑计划建造。青岑计划的总设计师是林砚秋，计划于2037年4月12日正式启动。
请仅依据资料回答查询。
- Query：霁川站所属计划的总设计师是谁？
- Gold：`['林砚秋']`
- Base (query only)：`<answer>陈明</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>林砚秋</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>林砚秋</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>林砚秋</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>林砚秋</answer>` (EM=True, F1=1.000)

### pair--en-natural-questions-99f9e9c6b825 (natural_questions)

- Prompt：Reference: The confluence of the Mississippi and Ohio rivers is at Cairo , Illinois .
Answer the query using only the reference.
- Query：where does the ohio river and the mississippi river meet
- Gold：`['Cairo , Illinois']`
- Base (query only)：`<answer>Ohio River and Mississippi River meet at the Ohio River Delta, located in the state of Indiana, USA.</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>Cairo, Illinois</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>Cairo, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois, Illinois` (EM=False, F1=0.222)
- QTraj + Top-k：`<answer>Cairo, Illinois</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>Cairo, Illinois</answer>` (EM=True, F1=1.000)

### pair--en-relation-extraction-5560e49828db (relation_extraction)

- Prompt：Reference: The Honourable Gerald David Lascelles (21 August 1924 -- 27 February 1998) was the younger son of Henry Lascelles, 6th Earl of Harewood and Mary, Princess Royal, the only daughter of King George V of the United Kingdom and Mary of Teck.
Answer the query using only the reference.
- Query：Who is Gerald David Lascelles's dad?
- Gold：`['Henry Lascelles, 6th Earl of Harewood']`
- Base (query only)：`<answer>John Lascelles</answer>` (EM=False, F1=0.250)
- Full prompt：`<answer>Henry Lascelles, 6th Earl of Harewood</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>Henry Lascelles, 6th Earl of Harewood, 6th Earl of Harewood, 6th Earl of H` (EM=False, F1=0.571)
- QTraj + Top-k：`<answer>Henry Lascelles, 6th Earl of Harewood</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>Henry Lascelles, 6th Earl of Harewood</answer>` (EM=True, F1=1.000)

### pair--en-knowledge-001 (squad)

- Prompt：Reference: Several barriers protect organisms from infection, including mechanical, chemical, and biological barriers. The waxy cuticle of many leaves, the exoskeleton of insects, the shells and membranes of externally deposited eggs, and skin are examples of mechanical barr...
Answer the query from this reference only.
- Query：What is an example of a mechanical barrier on leaves?
- Gold：`['The waxy cuticle', 'waxy cuticle']`
- Base (query only)：`<answer>leaves</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>waxy cuticle</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>waxy cuticle</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>leaves</answer>` (EM=False, F1=0.000)
- QTraj + Auto-Margin：`<answer>waxy cuticle</answer>` (EM=True, F1=1.000)

### pair--en-triviaqa-28a74dc15ee4 (triviaqa)

- Prompt：Reference: [DOC] [TLE] Busy Lizzie - definition of busy Lizzie by The Free DictionaryBusy Lizzie - definition of busy Lizzie by The Free Dictionary [PAR] Busy Lizzie - definition of busy Lizzie by The Free Dictionary [PAR] http://www.thefreedictionary.com/busy+Lizzie [PAR] (ˈlɪzɪ) [PAR] n [PAR] (Plants) a balsaminaceous plant, Impatiens balsamina, that has pink, red, or white flowers and is often grown as a pot plant [PAR] Want to thank TFD for its existence? Tell a friend about us , add a link to this page, or visit the webmaster's page for free fun content . [PAR] Link to this page: [PAR] impatiens [PAR] References in periodicals archive ? [PAR] Racing off the same mark in the Busy Lizzie Handicap Premier Project should now be near his peak and is taken to get ...
Answer the query using only the reference.
- Query：What plant do we often call the 'Busy Lizzie'?
- Gold：`['impatiens']`
- Base (query only)：`<answer>Chrysanthemum</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>impatiens</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>impatiens</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>impala</answer>` (EM=False, F1=0.000)
- QTraj + Auto-Margin：`<answer>impatiens</answer>` (EM=True, F1=1.000)
