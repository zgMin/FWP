# 项目交接文档

## 目标与边界

本项目研究将上下文 `C` 对某个查询 `x` 的影响近似编译为可融合权重：

    Full prompt:  C + x -> y_1, ..., y_T
    Query patch:  x     -> y'_1, ..., y'_T

主要对象是查询相关更新 `Delta W(C, x)`，不是与查询无关的 `Delta W(C)`。全部实验
使用 greedy decoding，不对基础模型进行反向传播、SGD 或 Adam 训练。

## 当前主方法

1. `src/eval_query_dependent_patch.py`
   - QTraj 通过 Full 与 Query-only teacher path 的中间表示/输出残差构造岭回归。
   - 求解闭式解，再保存为固定低秩线性层补丁。

2. `src/auto_margin_teacher_anchor.py`
   - 对 QTraj 后仍偏离 teacher token 的位置，构造 teacher-vs-top-1 competitor
     不等式，并求最小 Frobenius 范数解。
   - 对偶目标为 `max_{alpha >= 0} r^T alpha - 0.5 alpha^T G alpha`。它只数值求解
     小规模凸问题，不训练模型参数。
   - 更新写为 `Delta W = B @ A`，可一次融合进 `lm_head.weight`。

3. `src/eval_auto_margin_dataset.py`
   - 当前推荐的统一入口，支持可验证和描述性数据集。
   - `--auto-margin-teacher-tokens 32` 限制 Auto-Margin 只使用前 32 个 Full token；
     QTraj 的 `lm_head` 校准也采用 32 token。

完整数学推导见 [AUTO_MARGIN_TEACHER_ANCHOR_ZH.md](AUTO_MARGIN_TEACHER_ANCHOR_ZH.md)。

## 最近可复现配置

    模型:       /root/zgm/models/Qwen/Qwen3.5-0.8B
    环境:       thoughtpatch-qwen25
    数据:       verifiable_expanded_150 (774), p2w_bench_semantic_lengths (276)
    解码:       greedy
    QTraj:      last 8 layers, rank 8, ridge 1e-3,
                lm_head rank 64, scale 0.6, teacher tokens 32
    Auto-Margin: first 32 Full-trajectory tokens

完整前 32 token 实验结果：

| 任务 | Auto-Margin 主要结果 | 说明 |
|---|---|---|
| 可验证，774 条 | EM 0.5633，F1 0.7134，Containment 0.6705 | 与 Full Prompt 完全一致 |
| 描述性，276 条 | BLEU 0.4811，ROUGE-L 0.5693，语义 0.9420 | Exact 0.1558，KL macro 0.0811 |

描述性任务上，前 32 token Auto-Margin 优于旧 Margin 与 Top-k 的 BLEU、ROUGE-L、
语义、格式、长度差和 KL；但它不保证长回答在第 33 token 后仍完全复现 Full。平均
有效秩为 3.20，P95 为 8，平均 anchor 参数约 0.80M。

## 目录和责任划分

    src/run_thought_patch_qwen.py        Patch 配置、注入、融合和通用模型工具
    src/eval_query_dependent_patch.py    QTraj 拟合
    src/auto_margin_teacher_anchor.py    Auto-Margin 对偶、factor 重建和融合
    src/eval_auto_margin_dataset.py      当前 Auto-Margin 评测入口
    src/eval_*_dataset.py                Margin/Top-k 与其他消融实验入口
    src/summarize_*.py                   指标汇总和 Markdown/CSV 报告
    src/merge_auto_margin_results.py     将 Auto-Margin 补入已有基线结果
    tests/                               Auto-Margin 最小范数数值测试
    docs/                                理论、用法、实验记录
    data/                                本地挂载的数据集，不提交 JSONL
    outputs/                             生成的 patch、分片和报告，不提交

远端完整结果位于：

    /root/zgm/thoughtpatch_qwen25/outputs/auto_margin_qwen35_08b_t32/

## 已知限制

- Full-trajectory Auto-Margin 将固定 `(C, x)` 的 teacher answer 作为硬约束，
  因而是 answer compilation。100% 输出复现不能解释为跨查询 Prompt 泛化。
- 前 32 token 版本显著压缩参数，但只保证输出前缀的 greedy 决策，长回答后续仍可能分叉。
- 部分 Qwen 模型共享 `lm_head` 与 input embedding。融合时必须使用
  `merge_bundle_into_model`，它会先解除共享，避免错误修改输入 embedding。
- 代码对 Qwen 风格层路径支持最充分；新模型应先跑单条 smoke test，并检查
  `model.model.layers`、chat template 与 EOS 处理。
- 描述性语义相似度依赖本地 BGE-M3；该模型不随仓库提交。

## 上传 GitHub 前检查

    git init
    git status --ignored
    PYTHONPATH=src python -m pytest -q
    git add README.md requirements.txt .gitignore data/ docs/ src/ tests/
    git status

确认 `data/*.jsonl`、`outputs/`、模型目录、Hugging Face cache 和 `.pt` patch 权重
均没有进入暂存区。若需要发布结果，建议只提交小型 CSV 或 Markdown 汇总表，并在
release 或外部存储中提供大文件。
