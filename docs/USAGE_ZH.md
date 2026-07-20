# 使用说明

## 环境

已验证环境为 A6000、Python 3.10、conda 环境 `thoughtpatch-qwen25`。关键版本为
PyTorch 2.6.0+cu124、Transformers 5.13.0、SciPy 1.13.1、NumPy 2.2.5 与
NLTK 3.9.4。

新机器上先安装与 CUDA 匹配的 PyTorch，再安装其余依赖：

    conda create -n thoughtpatch-qwen25 python=3.10 -y
    conda activate thoughtpatch-qwen25
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt

所有命令从仓库根目录运行，并使用 `PYTHONPATH=src`。模型必须是已经下载到本地的
Hugging Face 目录，代码默认 `local_files_only=True`。

    export MODEL_PATH=/root/zgm/models/Qwen/Qwen3.5-0.8B
    export PY=/root/anaconda3/envs/thoughtpatch-qwen25/bin/python

## 数据准备

将最终 JSONL 放到 [data/README.md](../data/README.md) 规定的位置：

    data/verifiable_expanded_150/answer_verifiable_expanded.jsonl
    data/p2w_bench_semantic_lengths/answer_nonverifiable.jsonl

前者是可验证答案任务，后者是描述性任务。数据构建和原始语料获取不包含在本仓库中。

## 单样本冒烟测试

先验证模型结构、chat template、数据字段与 Auto-Margin 合成流程：

    PYTHONPATH=src "$PY" src/eval_auto_margin_dataset.py \
      --task verifiable \
      --dataset data/verifiable_expanded_150/answer_verifiable_expanded.jsonl \
      --out outputs/smoke/verifiable.json \
      --model "$MODEL_PATH" --device cuda:0 --limit 1 \
      --prefix-count 4 --qtraj-lm-teacher-tokens 32 \
      --auto-margin-teacher-tokens 32

    PYTHONPATH=src "$PY" src/eval_auto_margin_dataset.py \
      --task descriptive \
      --dataset data/p2w_bench_semantic_lengths/answer_nonverifiable.jsonl \
      --out outputs/smoke/descriptive.json \
      --model "$MODEL_PATH" --device cuda:0 --limit 1 \
      --max-new-tokens 256 --prefix-count 4 \
      --qtraj-lm-teacher-tokens 32 --auto-margin-teacher-tokens 32

    PYTHONPATH=src "$PY" -m pytest -q

`--auto-margin-teacher-tokens 32` 只约束 Full trajectory 的前 32 个位置。它限制的
是可用约束位置，不是严格固定秩；最终有效秩等于活跃约束数，至多为 32。省略该参数
时，Auto-Margin 会约束整条 Full trajectory。

Auto-Margin 入口也支持 `--methods`。例如只保存和评估 Auto-Margin：

    --methods qtraj_teacher_auto_margin

此时 Full Prompt 仍会在内部生成，以提供 teacher trajectory；Base 不会额外生成。

## 完整 Auto-Margin 评测

下面以三张 GPU、三分片为例。每个进程内部只看到一张 GPU，因此
`CUDA_VISIBLE_DEVICES` 后仍传入 `--device cuda:0`。输出文件可以恢复：已完成的
`pair_id` 会被跳过。

    for shard in 0 1 2; do
      gpu=$((shard + 1))
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src "$PY" src/eval_auto_margin_dataset.py \
        --task verifiable \
        --dataset data/verifiable_expanded_150/answer_verifiable_expanded.jsonl \
        --out "outputs/auto_margin_t32/verifiable/shard${shard}.json" \
        --model "$MODEL_PATH" --device cuda:0 \
        --num-shards 3 --shard-index "$shard" \
        --prefix-count 4 --qtraj-lm-teacher-tokens 32 \
        --auto-margin-teacher-tokens 32 &
    done
    wait

将上面命令的 `--task`、`--dataset`、输出目录替换为下面描述性设置，即可运行描述性集：

    --task descriptive
    --dataset data/p2w_bench_semantic_lengths/answer_nonverifiable.jsonl
    --out "outputs/auto_margin_t32/descriptive/shard${shard}.json"
    --max-new-tokens 256

运行整条轨迹版本时，移除 `--auto-margin-teacher-tokens 32`。该版本可以精确复制
Full 的 greedy 输出，但其语义是每个 `(C, x)` 的答案轨迹编译，不能作为通用 Prompt
替代方案来宣称。

## 比较 Margin、Top-k 与 Auto-Margin

先用下列入口生成 Base、Full Prompt、QTraj + Margin 和 QTraj + Top-k：

    PYTHONPATH=src "$PY" src/eval_verifiable_expanded.py --help
    PYTHONPATH=src "$PY" src/eval_descriptive_dataset.py --help

若只需要一个 patch 方案，在任一基线入口后加入 `--methods`。例如只跑 Top-k：

    --methods qtraj_topk_delta

或只跑 Margin：

    --methods qtraj_teacher_margin

Base、Full Prompt 与 QTraj 拟合仍会在内部执行，因为它们是 anchor 合成与对齐评测的
共同依赖；未选中的 patch 不会拟合、生成或计算逐 token KL。

将 Auto-Margin 分片合并到已有基线结果后，统一汇总描述性指标：

    PYTHONPATH=src "$PY" src/merge_auto_margin_results.py \
      --task descriptive \
      --base outputs/baseline/nonverifiable_report/combined_results.json \
      --auto outputs/auto_margin_t32/descriptive/shard0.json \
             outputs/auto_margin_t32/descriptive/shard1.json \
             outputs/auto_margin_t32/descriptive/shard2.json \
      --out outputs/auto_margin_t32/descriptive_merged.json

    PYTHONPATH=src "$PY" src/summarize_descriptive_eval.py \
      --inputs outputs/auto_margin_t32/descriptive_merged.json \
      --out-dir outputs/auto_margin_t32/descriptive_report \
      --expected-rows 276 --semantic-model /path/to/bge-m3 \
      --semantic-device cuda:0 --semantic-batch-size 8

可验证数据使用同样的合并步骤，将 `--task` 改为 `verifiable`，再执行：

    PYTHONPATH=src "$PY" src/summarize_verifiable_expanded.py \
      --inputs outputs/auto_margin_t32/verifiable_merged.json \
      --out-dir outputs/auto_margin_t32/verifiable_report \
      --expected-rows 774

描述性 KL 定义为 Full greedy teacher trajectory 上逐 token 的
`KL(P_full || P_method)`。即使自由生成已经分叉，KL 仍在相同 `y_<t` 前缀上
teacher-force 对齐，因此可以比较不同方法。
