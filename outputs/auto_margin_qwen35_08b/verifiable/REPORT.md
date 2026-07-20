# 扩展版答案可验证任务报告

四种方法共享严格 `<answer>...</answer>` 输出协议；知识上下文只出现在 Full prompt，或由闭式解转换为可一次融合的固定权重。全部生成均为 greedy。

## 设置

- 模型：`/root/zgm/models/Qwen/Qwen3.5-0.8B`
- 数据：30 条；中文 10，英文 20。
- QTraj：最后 8 层，prefix=4，rank=8，ridge=0.001；lm_head tokens=32。
- Teacher-token：lm_head tokens=32，rank=128，ridge=0.1；margin=4.0；Top-k=8。
- Base 与两种 query-only 路径都保留共享格式协议，但不接收知识段落。

## 总体

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| 全部 | 30 | Base (query only) | 86.67% | 86.67% | 3.33% | 10.28% | 3.33% |
| 全部 | 30 | Full prompt | 96.67% | 96.67% | 50.00% | 64.58% | 60.00% |
| 全部 | 30 | QTraj + Teacher-token | 96.67% | 96.67% | 50.00% | 64.58% | 60.00% |
| 全部 | 30 | QTraj + Top-k | 96.67% | 96.67% | 46.67% | 64.05% | 56.67% |
| 全部 | 30 | QTraj + Auto-Margin | 96.67% | 96.67% | 50.00% | 64.58% | 60.00% |

## 按语言

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| en | 20 | Base (query only) | 90.00% | 90.00% | 5.00% | 10.96% | 5.00% |
| en | 20 | Full prompt | 95.00% | 95.00% | 55.00% | 64.26% | 65.00% |
| en | 20 | QTraj + Teacher-token | 95.00% | 95.00% | 55.00% | 64.26% | 65.00% |
| en | 20 | QTraj + Top-k | 95.00% | 95.00% | 50.00% | 61.26% | 60.00% |
| en | 20 | QTraj + Auto-Margin | 95.00% | 95.00% | 55.00% | 64.26% | 65.00% |
| zh | 10 | Base (query only) | 80.00% | 80.00% | 0.00% | 8.93% | 0.00% |
| zh | 10 | Full prompt | 100.00% | 100.00% | 40.00% | 65.23% | 50.00% |
| zh | 10 | QTraj + Teacher-token | 100.00% | 100.00% | 40.00% | 65.23% | 50.00% |
| zh | 10 | QTraj + Top-k | 100.00% | 100.00% | 40.00% | 69.64% | 50.00% |
| zh | 10 | QTraj + Auto-Margin | 100.00% | 100.00% | 40.00% | 65.23% | 50.00% |

## 按来源

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| cmrc2018 | 4 | Base (query only) | 50.00% | 50.00% | 0.00% | 12.16% | 0.00% |
| cmrc2018 | 4 | Full prompt | 100.00% | 100.00% | 25.00% | 62.18% | 50.00% |
| cmrc2018 | 4 | QTraj + Teacher-token | 100.00% | 100.00% | 25.00% | 62.18% | 50.00% |
| cmrc2018 | 4 | QTraj + Top-k | 100.00% | 100.00% | 25.00% | 62.18% | 50.00% |
| cmrc2018 | 4 | QTraj + Auto-Margin | 100.00% | 100.00% | 25.00% | 62.18% | 50.00% |
| drcd | 4 | Base (query only) | 100.00% | 100.00% | 0.00% | 10.15% | 0.00% |
| drcd | 4 | Full prompt | 100.00% | 100.00% | 25.00% | 50.89% | 25.00% |
| drcd | 4 | QTraj + Teacher-token | 100.00% | 100.00% | 25.00% | 50.89% | 25.00% |
| drcd | 4 | QTraj + Top-k | 100.00% | 100.00% | 25.00% | 61.93% | 25.00% |
| drcd | 4 | QTraj + Auto-Margin | 100.00% | 100.00% | 25.00% | 50.89% | 25.00% |
| natural_questions | 5 | Base (query only) | 100.00% | 100.00% | 0.00% | 2.50% | 0.00% |
| natural_questions | 5 | Full prompt | 80.00% | 80.00% | 20.00% | 41.03% | 40.00% |
| natural_questions | 5 | QTraj + Teacher-token | 80.00% | 80.00% | 20.00% | 41.03% | 40.00% |
| natural_questions | 5 | QTraj + Top-k | 80.00% | 80.00% | 20.00% | 49.03% | 40.00% |
| natural_questions | 5 | QTraj + Auto-Margin | 80.00% | 80.00% | 20.00% | 41.03% | 40.00% |
| relation_extraction | 5 | Base (query only) | 100.00% | 100.00% | 20.00% | 20.00% | 20.00% |
| relation_extraction | 5 | Full prompt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| relation_extraction | 5 | QTraj + Teacher-token | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| relation_extraction | 5 | QTraj + Top-k | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| relation_extraction | 5 | QTraj + Auto-Margin | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| squad | 4 | Base (query only) | 75.00% | 75.00% | 0.00% | 10.00% | 0.00% |
| squad | 4 | Full prompt | 100.00% | 100.00% | 50.00% | 50.00% | 50.00% |
| squad | 4 | QTraj + Teacher-token | 100.00% | 100.00% | 50.00% | 50.00% | 50.00% |
| squad | 4 | QTraj + Top-k | 100.00% | 100.00% | 25.00% | 25.00% | 25.00% |
| squad | 4 | QTraj + Auto-Margin | 100.00% | 100.00% | 50.00% | 50.00% | 50.00% |
| synthetic | 4 | Base (query only) | 75.00% | 75.00% | 0.00% | 0.00% | 0.00% |
| synthetic | 4 | Full prompt | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| synthetic | 4 | QTraj + Teacher-token | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| synthetic | 4 | QTraj + Top-k | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| synthetic | 4 | QTraj + Auto-Margin | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| triviaqa | 4 | Base (query only) | 100.00% | 100.00% | 0.00% | 16.67% | 0.00% |
| triviaqa | 4 | Full prompt | 100.00% | 100.00% | 25.00% | 45.00% | 50.00% |
| triviaqa | 4 | QTraj + Teacher-token | 100.00% | 100.00% | 25.00% | 45.00% | 50.00% |
| triviaqa | 4 | QTraj + Top-k | 100.00% | 100.00% | 25.00% | 45.00% | 50.00% |
| triviaqa | 4 | QTraj + Auto-Margin | 100.00% | 100.00% | 25.00% | 45.00% | 50.00% |

## 按数据分区

| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |
|---|---:|---|---:|---:|---:|---:|---:|
| base_short | 8 | Base (query only) | 75.00% | 75.00% | 0.00% | 5.00% | 0.00% |
| base_short | 8 | Full prompt | 100.00% | 100.00% | 75.00% | 75.00% | 75.00% |
| base_short | 8 | QTraj + Teacher-token | 100.00% | 100.00% | 75.00% | 75.00% | 75.00% |
| base_short | 8 | QTraj + Top-k | 100.00% | 100.00% | 62.50% | 62.50% | 62.50% |
| base_short | 8 | QTraj + Auto-Margin | 100.00% | 100.00% | 75.00% | 75.00% | 75.00% |
| expanded_public | 22 | Base (query only) | 90.91% | 90.91% | 4.55% | 12.20% | 4.55% |
| expanded_public | 22 | Full prompt | 95.45% | 95.45% | 40.91% | 60.79% | 54.55% |
| expanded_public | 22 | QTraj + Teacher-token | 95.45% | 95.45% | 40.91% | 60.79% | 54.55% |
| expanded_public | 22 | QTraj + Top-k | 95.45% | 95.45% | 40.91% | 64.62% | 54.55% |
| expanded_public | 22 | QTraj + Auto-Margin | 95.45% | 95.45% | 40.91% | 60.79% | 54.55% |

## 输入输出示例

### pair--zh-cmrc2018-0519a39ece2b (cmrc2018)

- Prompt：资料：黄天竺鲷，又名环天竺鲷、环尾天竺鲷，俗名大目侧仔，为辐鳍鱼纲鲈形目鲈亚目天竺鲷科的其中一个种。本鱼分布于印度洋－西太平洋区，包括红海、东非、南非、留尼旺、葛摩、模里西斯、塞席尔群岛、马达加斯加、印度、斯里兰卡、泰国、马来西亚、台湾、巴布亚纽几内亚、日本、圣诞岛、澳洲、新喀里多尼亚、东加等海域。水深10－40公尺。本鱼体长圆而侧扁。头大、吻长、眼大。体呈金黄色...
请仅依据资料回答查询。
- Query：黄天竺鲷的俗名是什么？
- Gold：`['大目侧仔']`
- Base (query only)：`<answer>黄天竺鲷</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>大目侧仔</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>大目侧仔</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>大目侧仔</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>大目侧仔</answer>` (EM=True, F1=1.000)

### pair--zh-drcd-008168d8c75f (drcd)

- Prompt：资料：...正哲由維也納到訪日本，根據信用卡的紀錄，金正恩曾經逗留東京迪士尼樂園。而金正恩在瑞士留學時的照片於2010年6月曝光。1996年至2001年，金正恩和妹妹金與正及金正怡分別化名為「朴銀」、「鄭順」一同假扮為北韓駐瑞士大使館職員的孩子於瑞士伯爾尼的一間國際學校就讀，學習英語、德語及法語。另外，金正男，於2001年涉嫌用假護照攜妻子兒女至日本東京迪士尼樂園旅遊並...
请仅依据资料回答查询。
- Query：金與正於1996年至2001年時以什麼化名至瑞士伯爾尼的一間國際學校就讀?
- Gold：`['朴銀']`
- Base (query only)：`<answer>金與正以「金與正」的化名至瑞士伯爾尼的一間國際學校就讀</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>朴銀</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>朴銀</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>朴銀</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>朴銀</answer>` (EM=True, F1=1.000)

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

### pair--en-knowledge-011 (synthetic)

- Prompt：Reference: In the fictional Northglass archive, Lumen Station was built under Project Alder. The chief designer of Project Alder was Mira Solen, and the project formally began on April 12, 2037.
Answer the query from this reference only.
- Query：Who was the chief designer of the project that built Lumen Station?
- Gold：`['Mira Solen']`
- Base (query only)：`<answer>John H. H. H. H. H. H. H. H. H. H. H. H. H. H.` (EM=False, F1=0.000)
- Full prompt：`<answer>Mira Solen</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>Mira Solen</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>Mira Solen</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>Mira Solen</answer>` (EM=True, F1=1.000)

### pair--en-natural-questions-0675afb8fe06 (natural_questions)

- Prompt：Reference: `` Lotta Love '' is a song written and recorded by Neil Young and released on his 1978 Comes a Time album . `` Lotta Love '' was also covered by Nicolette Larson in 1978 . Larson 's version reached No. 8 on the Billboard Hot 100 chart and No. 8 on the Cash Box Top 100 in February 1979 . It also hit No. 1 on the Easy Listening chart and was a hit in Australia ( No. 11 ) and New Zealand ( No. 22 ) .
Answer the query using only the reference.
- Query：who wrote it 's gon na take a lot of love
- Gold：`['Neil Young']`
- Base (query only)：`<answer>John Steinbeck</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>Neil Young</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>Neil Young</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>Neil Young</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>Neil Young</answer>` (EM=True, F1=1.000)

### pair--en-relation-extraction-01076e3a5587 (relation_extraction)

- Prompt：Reference: Puyi's last surviving younger half-brother Puren (b. 1918) has adopted the Chinese name Jin Youzhi and lived in China until his death in 2015.
Answer the query using only the reference.
- Query：What is Puyi's brothers name?
- Gold：`['Jin Youzhi']`
- Base (query only)：`<answer>Yi</answer>` (EM=False, F1=0.000)
- Full prompt：`<answer>Jin Youzhi</answer>` (EM=True, F1=1.000)
- QTraj + Teacher-token：`<answer>Jin Youzhi</answer>` (EM=True, F1=1.000)
- QTraj + Top-k：`<answer>Jin Youzhi</answer>` (EM=True, F1=1.000)
- QTraj + Auto-Margin：`<answer>Jin Youzhi</answer>` (EM=True, F1=1.000)
