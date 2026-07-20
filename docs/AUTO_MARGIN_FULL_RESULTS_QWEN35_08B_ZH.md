# Qwen3.5-0.8B Auto-Margin 全量实验结果

## 实验设置

- 模型：`/root/zgm/models/Qwen/Qwen3.5-0.8B`
- Conda 环境：`thoughtpatch-qwen25`
- 可验证数据集：774 条，英文 462 条、中文 312 条
- 描述性数据集：276 条，英文 138 条、中文 138 条
- 生成：greedy；QTraj language head 使用 32 个 teacher token
- Auto-Margin：无手工 margin、negative scale、boost cap、anchor ridge、
  anchor rank 或 anchor scale
- 描述性 KL：Full greedy trajectory 上的逐 token teacher-forced
  `KL(P_full || P_method)`，包含 Full 实际生成的 EOS

全部 1050 条样本的 Auto-Margin 求解均收敛，且自由生成逐字等于
Full-prompt greedy 输出。

## 可验证任务

| 方法 | 严格格式 | EM | F1 | Containment |
|---|---:|---:|---:|---:|
| Base | 0.88889 | 0.04910 | 0.15202 | 0.07106 |
| Full Prompt | 0.97287 | 0.56331 | 0.71341 | 0.67054 |
| QTraj + Margin | 0.96899 | 0.55943 | 0.70982 | 0.66796 |
| QTraj + Top-k | 0.97287 | 0.54910 | 0.70041 | 0.65633 |
| QTraj + Auto-Margin | **0.97287** | **0.56331** | **0.71341** | **0.67054** |

按语言：

| 语言 | N | Auto EM | Auto F1 | Auto Containment | Auto 严格格式 |
|---|---:|---:|---:|---:|---:|
| 英文 | 462 | 0.56277 | 0.66491 | 0.67100 | 0.98701 |
| 中文 | 312 | 0.56410 | 0.78522 | 0.66987 | 0.95192 |

## 描述性任务

| 方法 | Exact | BLEU-4 | ROUGE-L | Semantic | 格式匹配 | 绝对长度差 | KL macro | KL micro |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.00725 | 0.15492 | 0.27618 | 0.82131 | 0.50725 | 59.96 | 0.33755 | 0.14960 |
| Full Prompt | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 0.00 | 0.00000 | 0.00000 |
| QTraj + Margin | 0.13043 | 0.42697 | 0.52596 | 0.93069 | 0.80435 | 32.29 | 0.36372 | 0.31213 |
| QTraj + Top-k | 0.10870 | 0.41128 | 0.51531 | 0.92294 | 0.77899 | 23.71 | 0.12361 | 0.14594 |
| QTraj + Auto-Margin | **1.00000** | **1.00000** | **1.00000** | **1.00000** | **1.00000** | **0.00** | **0.06779** | **0.06713** |

Auto-Margin 的 KL 虽不为零，但 greedy 决策在全部 48,385 个 Full
trajectory token 上足以保持同一条输出轨迹。它的 macro/micro KL 均低于
Margin，micro KL 也低于 Top-k。

按语言的 Auto-Margin KL：

| 语言 | N | Exact | KL macro | KL micro |
|---|---:|---:|---:|---:|
| 英文 | 138 | 1.00000 | 0.05961 | 0.06215 |
| 中文 | 138 | 1.00000 | 0.07597 | 0.07140 |

按任务族的 Auto-Margin KL macro：

| 任务族 | N | Exact | KL macro |
|---|---:|---:|---:|
| descriptive | 180 | 1.00000 | 0.05440 |
| format | 36 | 1.00000 | 0.08308 |
| output_control | 24 | 1.00000 | 0.09742 |
| style | 36 | 1.00000 | 0.09969 |

按 Prompt 长度：

| 长度 | N | Exact | KL macro |
|---|---:|---:|---:|
| short | 92 | 1.00000 | 0.06606 |
| medium_redundant | 92 | 1.00000 | 0.07177 |
| long_redundant | 92 | 1.00000 | 0.06554 |

冗余 Prompt 变长没有造成 KL 单调上升或遵循能力下降。

## 自适应参数规模

参数量按实际存储的 dense low-rank factors `A` 和 `B` 统计，不包含各方法
共享的 QTraj 参数。

| 数据集 | Zero anchor | Rank mean | Rank median | Rank p95 | Rank max | 参数 mean | 参数 p95 | 参数 max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 可验证 | 74.29% | 0.43 | 0 | 2.00 | 25 | 0.107M | 0.499M | 6.234M |
| 描述性 | 3.99% | 26.31 | 21 | 80.25 | 117 | 6.560M | 20.010M | 29.173M |

旧 Margin 固定 rank 128 的 anchor 约为 31.916M 参数。Auto-Margin 在描述性
任务上的平均 anchor 参数减少约 79.4%，且最大样本仍低于固定 rank 128。

Auto-Margin 阶段平均耗时为可验证 2.69 秒/条、描述性 42.60 秒/条；QTraj
拟合平均耗时分别为 17.77 秒/条和 15.09 秒/条。

## 示例

知识问答：

```text
Context: The confluence of the Mississippi and Ohio rivers is at Cairo, Illinois.
Query: where does the ohio river and the mississippi river meet
Full:   <answer>Cairo, Illinois</answer>
Margin: <answer>Cairo, Illinois, Illinois, Illinois, ...
Auto:   <answer>Cairo, Illinois</answer>
```

该样本 Auto-Margin 只使用 rank 1。

JSON 格式任务中，旧 Margin 和 Top-k 会在 `answer` 字段中重复片段并耗尽
生成长度；Auto-Margin 使用 rank 12，完整复现 Full 的合法单字段 JSON。

## 文件位置

远端完整报告：

- `outputs/auto_margin_full/reports/verifiable/`
- `outputs/auto_margin_full/reports/descriptive/`
- `outputs/auto_margin_full/reports/descriptive/token_kl.jsonl`
- `outputs/auto_margin_full/reports/descriptive/token_kl_by_position.csv`
- `outputs/auto_margin_full/reports/descriptive/token_kl_by_length_position.csv`
