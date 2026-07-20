# 非可验证任务 Full-prompt 一致性报告

276 个中英文非可验证 prompt-query 配对，覆盖描述、风格、格式和输出控制任务，以及 long_redundant, medium_redundant, short 三档 prompt。BLEU、ROUGE、语义、格式和长度均以 Full prompt 的 greedy 输出为 reference。KL 在 Full-prompt greedy teacher trajectory 上逐 token 对齐。
Prompt 长度扩展方式：redundant, semantic_elaboration。

## 实验设置

- 模型：`/root/zgm/models/Qwen/Qwen3.5-0.8B`
- 生成：greedy，`max_new_tokens=256`。
- QTraj：最后 8 层，prefix=4，rank=8，ridge=0.001；lm_head tokens=32。
- Teacher-token：lm_head tokens=32，rank=128，margin=4.0，Top-k=8。
- 语义模型：`/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181`，BGE-M3 CLS embedding cosine。
- 所有 patch 都是一次求解、整段生成固定不变的可合并权重；KL 评估不进行逐 token 重拟合。

## 总体

| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 276 | Base (query only) | 0.1549 | 0.4159 | 0.2242 | 0.2762 | 0.8213 | 50.72% | 5.00% | 0.6832 | 59.96 | 2.641 | 41.30% | 0.337550 | 0.149603 |
| 全部 | 276 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 68.33% | 1.0000 | 0.00 | 1.000 | 50.36% | 0.000000 | 0.000000 |
| 全部 | 276 | QTraj + Teacher-token | 0.4270 | 0.6222 | 0.4829 | 0.5260 | 0.9307 | 80.43% | 53.33% | 0.8721 | 32.29 | 1.258 | 42.39% | 0.363717 | 0.312127 |
| 全部 | 276 | QTraj + Top-k | 0.4113 | 0.6322 | 0.4706 | 0.5153 | 0.9229 | 77.90% | 58.33% | 0.8743 | 23.71 | 1.273 | 50.00% | 0.123606 | 0.145939 |
| 全部 | 276 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 68.33% | 1.0000 | 0.00 | 1.000 | 50.36% | 0.067789 | 0.067129 |

## 按语言

| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 138 | Base (query only) | 0.2099 | 0.4740 | 0.2844 | 0.3528 | 0.8298 | 56.52% | 0.00% | 0.7052 | 65.60 | 2.722 | 54.35% | 0.343231 | 0.147605 |
| en | 138 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 80.00% | 1.0000 | 0.00 | 1.000 | 60.87% | 0.000000 | 0.000000 |
| en | 138 | QTraj + Teacher-token | 0.4885 | 0.6888 | 0.5482 | 0.6040 | 0.9315 | 83.33% | 63.33% | 0.8773 | 30.96 | 1.150 | 54.35% | 0.363025 | 0.363724 |
| en | 138 | QTraj + Top-k | 0.4813 | 0.6975 | 0.5468 | 0.6028 | 0.9334 | 83.33% | 63.33% | 0.8890 | 25.78 | 1.047 | 63.04% | 0.146360 | 0.203334 |
| en | 138 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 80.00% | 1.0000 | 0.00 | 1.000 | 60.87% | 0.059612 | 0.062154 |
| zh | 138 | Base (query only) | 0.1000 | 0.3578 | 0.1639 | 0.1996 | 0.8128 | 44.93% | 10.00% | 0.6611 | 54.33 | 2.560 | 28.26% | 0.331870 | 0.151318 |
| zh | 138 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 56.67% | 1.0000 | 0.00 | 1.000 | 39.86% | 0.000000 | 0.000000 |
| zh | 138 | QTraj + Teacher-token | 0.3655 | 0.5557 | 0.4175 | 0.4479 | 0.9298 | 77.54% | 43.33% | 0.8668 | 33.62 | 1.366 | 30.43% | 0.364409 | 0.267866 |
| zh | 138 | QTraj + Top-k | 0.3413 | 0.5670 | 0.3943 | 0.4278 | 0.9125 | 72.46% | 53.33% | 0.8595 | 21.64 | 1.498 | 36.96% | 0.100851 | 0.096704 |
| zh | 138 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 56.67% | 1.0000 | 0.00 | 1.000 | 39.86% | 0.075967 | 0.071397 |

## 按任务族

| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| descriptive | 180 | Base (query only) | 0.1784 | 0.4724 | 0.2550 | 0.3089 | 0.8464 | 57.22% | - | 0.7596 | 40.09 | 1.159 | 43.33% | 0.112887 | 0.109222 |
| descriptive | 180 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 40.00% | 0.000000 | 0.000000 |
| descriptive | 180 | QTraj + Teacher-token | 0.3718 | 0.5981 | 0.4276 | 0.4722 | 0.9252 | 78.33% | - | 0.8536 | 23.17 | 1.131 | 33.89% | 0.264272 | 0.237717 |
| descriptive | 180 | QTraj + Top-k | 0.3599 | 0.6084 | 0.4191 | 0.4652 | 0.9168 | 77.22% | - | 0.8672 | 15.68 | 1.015 | 38.89% | 0.070968 | 0.080439 |
| descriptive | 180 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 40.00% | 0.054400 | 0.059945 |
| format | 36 | Base (query only) | 0.0347 | 0.1942 | 0.0742 | 0.1270 | 0.7070 | 16.67% | 8.33% | 0.4314 | 114.69 | 6.759 | 41.67% | 0.898659 | 0.272184 |
| format | 36 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 77.78% | 1.0000 | 0.00 | 1.000 | 66.67% | 0.000000 | 0.000000 |
| format | 36 | QTraj + Teacher-token | 0.6056 | 0.7263 | 0.6648 | 0.7051 | 0.9565 | 91.67% | 69.44% | 0.9699 | 37.33 | 1.319 | 72.22% | 0.700735 | 0.891540 |
| format | 36 | QTraj + Top-k | 0.5341 | 0.6788 | 0.5995 | 0.6431 | 0.9404 | 86.11% | 80.56% | 0.9537 | 51.86 | 1.399 | 77.78% | 0.373830 | 0.726776 |
| format | 36 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 77.78% | 1.0000 | 0.00 | 1.000 | 66.67% | 0.083084 | 0.076497 |
| output_control | 24 | Base (query only) | 0.0932 | 0.2653 | 0.1599 | 0.2145 | 0.7576 | 29.17% | 0.00% | 0.4633 | 105.38 | 7.944 | 37.50% | 1.155663 | 0.323949 |
| output_control | 24 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 54.17% | 1.0000 | 0.00 | 1.000 | 75.00% | 0.000000 | 0.000000 |
| output_control | 24 | QTraj + Teacher-token | 0.6159 | 0.7363 | 0.6802 | 0.7186 | 0.9449 | 87.50% | 29.17% | 0.8738 | 57.42 | 1.253 | 70.83% | 0.490341 | 0.345906 |
| output_control | 24 | QTraj + Top-k | 0.5864 | 0.7221 | 0.6371 | 0.6817 | 0.9387 | 87.50% | 25.00% | 0.8871 | 34.96 | 3.187 | 79.17% | 0.157502 | 0.120737 |
| output_control | 24 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 54.17% | 1.0000 | 0.00 | 1.000 | 75.00% | 0.097415 | 0.060056 |
| style | 36 | Base (query only) | 0.1990 | 0.4556 | 0.2625 | 0.3028 | 0.8525 | 66.67% | - | 0.6991 | 74.33 | 2.399 | 33.33% | 0.354349 | 0.255707 |
| style | 36 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 69.44% | 0.000000 | 0.000000 |
| style | 36 | QTraj + Teacher-token | 0.3981 | 0.5626 | 0.4458 | 0.4872 | 0.9232 | 75.00% | - | 0.8654 | 56.14 | 1.837 | 36.11% | 0.439503 | 0.344084 |
| style | 36 | QTraj + Top-k | 0.4289 | 0.6449 | 0.4880 | 0.5270 | 0.9257 | 66.67% | - | 0.8215 | 28.22 | 1.160 | 58.33% | 0.113971 | 0.140074 |
| style | 36 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 69.44% | 0.099692 | 0.114327 |

## 按 Prompt 长度

| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_redundant | 92 | Base (query only) | 0.1427 | 0.4026 | 0.2126 | 0.2644 | 0.8218 | 48.91% | 5.00% | 0.6740 | 65.15 | 2.589 | 41.30% | 0.322233 | 0.152770 |
| long_redundant | 92 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 70.00% | 1.0000 | 0.00 | 1.000 | 53.26% | 0.000000 | 0.000000 |
| long_redundant | 92 | QTraj + Teacher-token | 0.4203 | 0.6111 | 0.4814 | 0.5217 | 0.9303 | 82.61% | 55.00% | 0.8662 | 39.17 | 1.327 | 45.65% | 0.379843 | 0.340219 |
| long_redundant | 92 | QTraj + Top-k | 0.4334 | 0.6449 | 0.4923 | 0.5321 | 0.9268 | 77.17% | 60.00% | 0.8638 | 25.91 | 1.104 | 53.26% | 0.136712 | 0.175119 |
| long_redundant | 92 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 70.00% | 1.0000 | 0.00 | 1.000 | 53.26% | 0.065540 | 0.066292 |
| medium_redundant | 92 | Base (query only) | 0.1562 | 0.4169 | 0.2243 | 0.2764 | 0.8229 | 47.83% | 5.00% | 0.6790 | 58.14 | 2.472 | 41.30% | 0.320015 | 0.163222 |
| medium_redundant | 92 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 65.00% | 1.0000 | 0.00 | 1.000 | 48.91% | 0.000000 | 0.000000 |
| medium_redundant | 92 | QTraj + Teacher-token | 0.4206 | 0.6244 | 0.4788 | 0.5235 | 0.9297 | 78.26% | 50.00% | 0.8583 | 29.25 | 1.191 | 40.22% | 0.381133 | 0.350276 |
| medium_redundant | 92 | QTraj + Top-k | 0.3832 | 0.6176 | 0.4495 | 0.4923 | 0.9223 | 79.35% | 60.00% | 0.8747 | 26.88 | 1.082 | 51.09% | 0.143309 | 0.183146 |
| medium_redundant | 92 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 65.00% | 1.0000 | 0.00 | 1.000 | 48.91% | 0.071770 | 0.076024 |
| short | 92 | Base (query only) | 0.1659 | 0.4283 | 0.2355 | 0.2878 | 0.8192 | 55.43% | 5.00% | 0.6965 | 56.60 | 2.863 | 41.30% | 0.370403 | 0.132610 |
| short | 92 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 70.00% | 1.0000 | 0.00 | 1.000 | 48.91% | 0.000000 | 0.000000 |
| short | 92 | QTraj + Teacher-token | 0.4401 | 0.6311 | 0.4885 | 0.5328 | 0.9321 | 80.43% | 55.00% | 0.8917 | 28.46 | 1.257 | 41.30% | 0.330174 | 0.245469 |
| short | 92 | QTraj + Top-k | 0.4172 | 0.6342 | 0.4699 | 0.5216 | 0.9197 | 77.17% | 55.00% | 0.8844 | 18.34 | 1.633 | 45.65% | 0.090796 | 0.079159 |
| short | 92 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 70.00% | 1.0000 | 0.00 | 1.000 | 48.91% | 0.066058 | 0.058911 |

## 逐 Token KL 文件

- `token_kl.jsonl` / `token_kl.csv`：每个 pair、方法和 teacher token 的 KL、token log-prob、EOS 标记。
- `token_kl_by_position.csv`：按语言、方法和 token 位置聚合的均值、中位数、最小值、最大值，可直接绘制折线图。
- `token_kl_by_length_position.csv`：按 prompt 长度、方法和 token 位置聚合，用于分析长度敏感性。

## 输入输出示例

### pair--zh-descriptive-003--short--zh-descriptive-003--q003

- Prompt：回答查询时覆盖关键方面，并保持内容前后一致。
- Query：创建生日策划清单。
- Base (query only)：为庆祝生日，一份精心策划的清单不仅能让你感到惊喜，还能让家人朋友感受到你的用心。以下是一份**通用且实用的生日策划清单**，你可以根据你的喜好、预算和特殊需求进行调整：

### 🎂 核心主题与氛围
1. **确定主题**
   - 选择适合你的主题（如：科技未来、自然治愈、复古怀旧、运动活力、艺术灵感等）。
   - 提前收集相关素材（如：定制卡片、照片、音乐、装饰道具）。
2. **确定时间**
   - 设定一个具体的日期（例如：3 月 15 日）。
   - 预留 1-2 天用于布置和准备。

### 🎁 惊喜环节（惊喜清单）
3. **定制惊喜**
   - 制作专属的生日蛋糕（可定制口味、图案或数字）。
   - 准备定制礼物（如：定制钥匙扣、定制徽章、定制玩偶）。
4. **互动游戏**
   - 准备简单的互动游戏（如：猜谜、抽奖、拼图、盲盒）。
   - 提前准备好奖品或奖励。
5. **特殊仪式** (BLEU=0.135, R-L=0.233, semantic=0.839, KL=0.05485)
- Full prompt：# 🎂 2024 年生日策划清单

## 一、核心主题：庆祝与成长
*   **主题选择**：选择与个人兴趣、职业或价值观最契合的生日主题（如：科技、艺术、运动、自然等）。
*   **视觉设计**：准备一张精美的生日贺卡或照片，包含个人照片、祝福语和个性化装饰（如定制徽章、贴纸）。
*   **氛围营造**：布置温馨的生日派对，确保灯光、音乐和氛围感符合预期。

## 二、饮食与活动安排
*   **菜单规划**：
    *   **主菜**：选择一顿丰盛、营养均衡的晚餐（如：海鲜、烤肉、意面等）。
    *   **甜点**：准备一份精致的生日蛋糕或甜点，搭配水果。
    *   **饮品**：准备一杯热咖啡或茶，作为开场或收尾的饮品。
*   **互动环节**：
    *   **游戏**：设计一个简短的生日游戏（如：猜谜语、猜生肖、猜谜语等），增加趣味性。
    *   **分享**：邀请朋友或家人分享当天的趣事或感悟。
* (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：# 🎂 2024 年生日策划清单

## 一、核心主题：庆祝与成长
*   **主题选择**：庆祝生日、成长里程碑、家庭团聚、旅行探索、健康养生、艺术创作等。
*   **主题口号**：
    *   “庆祝 [你的姓名] 的生日，见证成长！”
    *   “在这个特别的日子里，拥抱 [你的爱好/职业] 与家庭。”
    *   “庆祝 [你的姓名] 的生日，开启新的旅程！”

## 二、时间规划：从筹备到庆祝
*   **准备期（第 1 周）**
    *   **日期**：2024 年 X 月 X 日
    *   **任务**：
        1.  **确定主题**：根据上述口号选择具体主题。
        2.  **确定主题**：确定主题。
        3.  **确定主题**：确定主题。
        4.  **确定主题**：确定主题。
        5.  **确定主题**：确定主题。
        6.  **确定主题**：确定主题。 (BLEU=0.119, R-L=0.192, semantic=0.929, KL=0.15085)
- QTraj + Top-k：# 🎂 2024 年生日策划清单

## 一、核心主题：致敬成长与感恩
**主题建议**：《时光里的星光》、《感恩相遇》、《成长的里程碑》
**核心精神**：记录过去一年的高光时刻，感谢家人朋友，展望未来。

---

## 二、时间规划（建议 3 天）

### 第一天：回顾与感恩（Day 1）
**目标**：回顾过去一年的高光时刻，感谢重要的人。

| 时间 | 活动 | 备注 |
| :--- | :--- | :--- |
| **08:00** | **生日派对** | 享用精心准备的生日蛋糕、饮品，开启庆祝氛围。 |
| **09:00** | **回顾回顾** | 播放过去一年最精彩的一集视频（如电影、纪录片、播客），回顾成就。 |
| **10:00** | **感谢仪式** | 感谢父母、伴侣、朋友、同事等，每人分享一个感谢瞬间。 |
| **11:00** | **感恩分享** | 邀请亲友分享过去一年最难忘的一件事，或共同 (BLEU=0.120, R-L=0.218, semantic=0.861, KL=0.04229)
- QTraj + Auto-Margin：# 🎂 2024 年生日策划清单

## 一、核心主题：庆祝与成长
*   **主题选择**：选择与个人兴趣、职业或价值观最契合的生日主题（如：科技、艺术、运动、自然等）。
*   **视觉设计**：准备一张精美的生日贺卡或照片，包含个人照片、祝福语和个性化装饰（如定制徽章、贴纸）。
*   **氛围营造**：布置温馨的生日派对，确保灯光、音乐和氛围感符合预期。

## 二、饮食与活动安排
*   **菜单规划**：
    *   **主菜**：选择一顿丰盛、营养均衡的晚餐（如：海鲜、烤肉、意面等）。
    *   **甜点**：准备一份精致的生日蛋糕或甜点，搭配水果。
    *   **饮品**：准备一杯热咖啡或茶，作为开场或收尾的饮品。
*   **互动环节**：
    *   **游戏**：设计一个简短的生日游戏（如：猜谜语、猜生肖、猜谜语等），增加趣味性。
    *   **分享**：邀请朋友或家人分享当天的趣事或感悟。
* (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.04017)

### pair--zh-format-002--medium_redundant--zh-format-002--q001

- Prompt：请将全部回答放在<answer>和</answer>标签之间，标签外不得输出内容。
为避免歧义，上述内容只规定输出格式。所要求的格式不变，再次说明: 请将全部回答放在<answer>和</answer>标签之间，标签外不得输出内容。
- Query：给一个快速的跑步热身的建议。简要解释每个动作。
- Base (query only)：为了快速提升心率并激活肌肉，建议采用"**原地高抬腿**"（High Knees）作为核心动作。

**动作解释**：
双脚与肩同宽站立，身体重心前倾，膝盖弯曲，同时**膝盖向上抬起**，直到膝盖触及胸口或胸部。保持这个姿势约 10-15 秒，然后迅速回到起始位置。

**快速节奏建议**：
*   **10 秒**：原地高抬腿
*   **10 秒**：快速回正（保持膝盖弯曲，不要完全伸直）
*   **10 秒**：原地高抬腿
*   **10 秒**：快速回正

**注意**：此动作能显著增加心脏负荷，但需注意膝盖不要过度弯曲导致受伤，且不要长时间保持膝盖伸直。 (BLEU=0.095, R-L=0.199, semantic=0.844, KL=0.11085)
- Full prompt：<answer>
1. **原地踏步**：保持身体直立，交替抬起和放下膝盖，每次 10-15 秒，帮助激活核心和腿部肌肉。
2. **高抬腿**：双脚同时抬起，保持膝盖弯曲，重复 15-20 次，增强心肺功能和腿部爆发力。
3. **开合跳**：双脚同时打开，保持膝盖弯曲，重复 15-20 次，提高心肺耐力。
4. **高抬腿 + 开合跳**：结合前两个动作，增加心肺负荷和协调性。
5. **深蹲**：双脚分开与肩同宽，保持膝盖弯曲，重复 15-20 次，锻炼下肢力量。
6. **弓步蹲**：一只脚后跟落地，另一只脚前脚掌着地，重复 10-15 次，增强全身协调性。
7. **平板支撑**：保持身体呈直线，保持 30-60 秒，锻炼核心稳定性。
8. **臀桥**：仰卧，将臀部抬起，保持 15-20 秒，强化臀部肌肉。 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：<answer>
1. **原地踏步**：保持身体直立，交替抬起和放下膝盖，每次 10-15 秒，帮助激活核心和腿部肌肉，避免膝盖过度弯曲。 (BLEU=0.015, R-L=0.303, semantic=0.870, KL=0.41947)
- QTraj + Top-k：<answer>1. 原地踏步：保持身体直立，用脚后跟和脚趾用力蹬地，感受腿部肌肉发力，保持 10-15 秒，重复 10 次。 (BLEU=0.005, R-L=0.195, semantic=0.868, KL=0.22186)
- QTraj + Auto-Margin：<answer>
1. **原地踏步**：保持身体直立，交替抬起和放下膝盖，每次 10-15 秒，帮助激活核心和腿部肌肉。
2. **高抬腿**：双脚同时抬起，保持膝盖弯曲，重复 15-20 次，增强心肺功能和腿部爆发力。
3. **开合跳**：双脚同时打开，保持膝盖弯曲，重复 15-20 次，提高心肺耐力。
4. **高抬腿 + 开合跳**：结合前两个动作，增加心肺负荷和协调性。
5. **深蹲**：双脚分开与肩同宽，保持膝盖弯曲，重复 15-20 次，锻炼下肢力量。
6. **弓步蹲**：一只脚后跟落地，另一只脚前脚掌着地，重复 10-15 次，增强全身协调性。
7. **平板支撑**：保持身体呈直线，保持 30-60 秒，锻炼核心稳定性。
8. **臀桥**：仰卧，将臀部抬起，保持 15-20 秒，强化臀部肌肉。 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.05880)

### pair--zh-output-control-003--long_redundant--zh-output-control-003--q001

- Prompt：使用列表回答，并在每个列表项的末尾添加固定字符串“I2W”。
为避免歧义，上述内容只规定一项输出控制规则。该规则保持不变，再次说明: 使用列表回答，并在每个列表项的末尾添加固定字符串“I2W”。
不增加任何新条件，上述内容只规定一项输出控制规则。该规则保持不变，再次说明: 使用列表回答，并在每个列表项的末尾添加固定字符串“I2W”。
为避免歧义，上述内容只规定一项输出控制规则。该规则保持不变，再次说明: 使用列表回答，并在每个列表项的末尾添加固定字符串“I2W”。
- Query：列出一个健康饮食的购物清单。
- Base (query only)：制定一份健康饮食的购物清单，关键在于**减少高糖、高油、高盐的摄入**，并增加**优质蛋白、膳食纤维和新鲜蔬果**的摄入。

以下是一份为您整理的“减脂/增肌友好型”健康饮食购物清单，涵盖了从日常采购到特殊场景的必备物品：

### 🛒 核心必买清单（基础款）

这部分是日常购物的核心，建议优先购买，因为它们是身体代谢的基础。

| 类别 | 推荐商品 | 选购理由 |
| :--- | :--- | :--- |
| **优质蛋白** | **鸡胸肉、瘦牛肉、鱼虾、鸡蛋、豆腐、低脂牛奶** | 提供必需氨基酸，满足肌肉生长需求，减少脂肪堆积。 |
| **膳食纤维** | **全谷物（燕麦、糙米、荞麦）、绿叶蔬菜、豆类** | 增加饱腹感，帮助控制血糖和体重。 |
| **优质脂肪** | **坚果（杏仁、核桃）、牛油果、橄榄油** | 提供健康脂肪，维持激素平衡，增加饱腹感。 |
| **新鲜蔬果** | **苹果、蓝莓、草莓、西兰花、胡萝卜、 (BLEU=0.069, R-L=0.134, semantic=0.793, KL=0.10456)
- Full prompt：- 选择低糖水果：苹果、香蕉、橙子
- 选择优质蛋白：鸡胸肉、鱼、豆腐
- 选择健康脂肪：橄榄油、坚果、牛油果
- 选择充足水分：白开水、绿茶、柠檬水
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：- 选择低糖水果：苹果、香蕉、橙子
- 选择优质蛋白：鸡胸肉、鱼、豆腐
- 选择健康脂肪：橄榄油、坚果
- 选择健康脂肪：橄榄油、坚果 (BLEU=0.007, R-L=0.241, semantic=0.939, KL=0.46751)
- QTraj + Top-k：- 选择低糖水果（如苹果、香蕉、橙子）
- 选择低脂乳制品（如牛奶、酸奶）
- 选择优质蛋白（如鱼、虾、瘦肉）
- 选择大量蔬菜（如绿叶蔬菜、西兰花）
- 选择适量主食（如全麦面包、糙米） (BLEU=0.021, R-L=0.259, semantic=0.939, KL=0.32928)
- QTraj + Auto-Margin：- 选择低糖水果：苹果、香蕉、橙子
- 选择优质蛋白：鸡胸肉、鱼、豆腐
- 选择健康脂肪：橄榄油、坚果、牛油果
- 选择充足水分：白开水、绿茶、柠檬水
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜
- 选择适量主食：糙米、燕麦、红薯
- 选择适量蛋白质：鸡蛋、牛奶、酸奶
- 选择适量蔬菜：西兰花、菠菜、胡萝卜 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.04103)

### pair--en-format-001--short--en-format-001--q001

- Prompt：Return exactly one JSON object containing only the string field answer.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Return exactly one JSON object containing only the string field answer.
- Query：I'm considering buying a bicycle. How do I know which one I should buy?
- Base (query only)：Choosing the right bicycle is a great way to start your journey, but it's not as simple as picking the "best" one. The right bike depends heavily on your **riding style**, **budget**, **maintenance needs**, and **personal preferences**.

Here is a step-by-step guide to help you make an informed decision:

### 1. Define Your Riding Style
Before looking at specs, understand how you will ride. This is the most critical factor.
*   **City Commuter:** Focus on comfort, handling, and speed. You want to ride for hours without getting tired.
*   **Hiking/Trail Rider:** Focus on durability, suspension, and weight. You need to handle rough terrain and carry gear.
*   **Tourist/Adventure:** Focus on aesthetics, comfort, and versatility. You might mix city riding with off-road capability.
*   **Rider with a Child:** Consider the weight distribution and seat height for a child's safety and comfort.

### 2. Analyze Your Budget
Bicycles come in different price points.
*   **Entry Level:** $200 – $500. Good for learning, occasional city riding, and basic maintenance.
*   **Mid (BLEU=0.013, R-L=0.125, semantic=0.832, KL=0.70317)
- Full prompt：{
  "answer": "You should consider the brand, the type of frame (e.g., road, mountain, hybrid), the price, the warranty, and the maintenance costs. A good bike is one that fits your riding style, has good durability, and offers good value for money."
} (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：{
  "answer": "You should consider the brand, the type of frame (e.g., road, mountain, hybrid), the price, the warranty, the warranty period, the warranty type (e.g., manufacturer warranty, warranty by frame manufacturer), the warranty period (e.g., 1 year, 3 years), the warranty type (e.g., manufacturer warranty, warranty by frame manufacturer), the warranty period (e.g., 1 year, 3 years), the warranty type (e.g., manufacturer warranty, warranty by frame manufacturer), the warranty period (e.g., 1 year, 3 years), the warranty type (e.g., manufacturer warranty, warranty by frame manufacturer), the warranty period (e.g., 1 year, 3 years), the warranty type (e.g., manufacturer warranty, warranty by frame manufacturer), the warranty period (e.g., 1 year, 3 years), the warranty type (e.g., manufacturer warranty, warranty by frame manufacturer), the warranty period (e.g., 1 year, 3 years), the warranty type (e.g., manufacturer warranty, warranty by frame manufacturer), the warranty period (e.g., 1 year, 3 years), the warranty type (e.g., manufacturer warranty (BLEU=0.121, R-L=0.227, semantic=0.758, KL=0.78286)
- QTraj + Top-k：{
  "answer": "You should consider the brand, the type of frame (road vs. mountain), the tire size, the price, and your budget. Here is a breakdown of how to evaluate your options:  \n\n1. **Look for brand reputation and reliability**: Brands like Trek, Specialized, and Cannondale are known for durability and good pricing.  \n2. **Check the frame type**:  \n   - **Road**: For daily commuting or light trails, a road frame is usually sufficient.  \n   - **Mountain**: If you plan to climb hills or do long-distance touring, a mountain frame (like Trek's \"Bolt\" or Specialized's \"Bolt-2\") is better.  \n3. **Consider tire size and type**:  \n   - **Road**: 26-inch tires are standard for road bikes.  \n   - **Mountain**: 27.5-inch tires are more suitable for trails.  \n   - **Road**: 2.0-inch tires are for light trails (e.g., gravel).  \n   - **Mountain**: 2.0-inch tires are for trails.  \n4. **Check (BLEU=0.107, R-L=0.228, semantic=0.804, KL=0.21018)
- QTraj + Auto-Margin：{
  "answer": "You should consider the brand, the type of frame (e.g., road, mountain, hybrid), the price, the warranty, and the maintenance costs. A good bike is one that fits your riding style, has good durability, and offers good value for money."
} (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.18201)

### pair--en-format-006--medium_redundant--en-format-006--q001

- Prompt：Output one line only and separate its parts with the vertical bar character |.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
Without adding any new condition, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
- Query：Think up a bunch of different meals I can make that include potatoes.
- Base (query only)：Here is a diverse list of potato-based meals ranging from savory and comforting to spicy and savory, categorized by how you might use them.

### 🥔 **Savory & Creamy**
*Great for dipping, salads, or as a main course.*

1.  **Potato Frittata**
    *   *Why:* A classic, fluffy, and savory breakfast.
    *   *How:* Mix eggs, spinach, and a splash of milk or cream. Bake until golden. Serve with a side of roasted potatoes.
2.  **Potato Soup**
    *   *Why:* A comforting, neutral base for any meal.
    *   *How:* Sauté diced potatoes with onions, carrots, and celery. Add chicken broth, thyme, and a splash of milk or cream. Simmer until thickened.
3.  **Potato Stew**
    *   *Why:* A hearty, slow-cooked meal that is incredibly filling.
    *   *How:* Sauté potatoes with onions, carrots, and celery. Add beef broth, bay leaves, and a splash of milk. Simmer for 2–3 hours until the potatoes are tender.
4. (BLEU=0.003, R-L=0.008, semantic=0.680, KL=0.11360)
- Full prompt：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=0.047, R-L=0.395, semantic=0.960, KL=5.14089)
- QTraj + Top-k：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=0.035, R-L=0.375, semantic=0.956, KL=5.15423)
- QTraj + Auto-Margin：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00472)

### pair--en-format-006--long_redundant--en-format-006--q001

- Prompt：Output one line only and separate its parts with the vertical bar character |.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
Without adding any new condition, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
Without adding any new condition, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Output one line only and separate its parts with the vertical bar character |.
- Query：Think up a bunch of different meals I can make that include potatoes.
- Base (query only)：Here is a diverse list of potato-based meals ranging from savory and comforting to spicy and savory, categorized by how you might use them.

### 🥔 **Savory & Creamy**
*Great for dipping, salads, or as a main course.*

1.  **Potato Frittata**
    *   *Why:* A classic, fluffy, and savory breakfast.
    *   *How:* Mix eggs, spinach, and a splash of milk or cream. Bake until golden. Serve with a side of roasted potatoes.
2.  **Potato Soup**
    *   *Why:* A comforting, neutral base for any meal.
    *   *How:* Sauté diced potatoes with onions, carrots, and celery. Add chicken broth, thyme, and a splash of milk or cream. Simmer until thickened.
3.  **Potato Stew**
    *   *Why:* A hearty, slow-cooked meal that is incredibly filling.
    *   *How:* Sauté potatoes with onions, carrots, and celery. Add beef broth, bay leaves, and a splash of milk. Simmer for 2–3 hours until the potatoes are tender.
4. (BLEU=0.003, R-L=0.008, semantic=0.680, KL=0.11175)
- Full prompt：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=0.044, R-L=0.390, semantic=0.959, KL=5.26087)
- QTraj + Top-k：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=0.041, R-L=0.385, semantic=0.959, KL=5.27764)
- QTraj + Auto-Margin：| Potato Soup | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00486)
