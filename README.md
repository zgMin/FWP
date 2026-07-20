# Prompt-to-Weight Patches

This repository studies no-training, query-dependent prompt-to-weight patches
for decoder-only language models. Given a context `C` and query `x`, the
primary method synthesizes a mergeable update `Delta W(C, x)` so that a
query-only model better follows the behavior of `C + x`.

The current experimental stack is built around QTraj and Auto-Margin:

- QTraj uses closed-form ridge regression to match Full-prompt residual trajectories on `x + y_<t`.
- Auto-Margin builds a minimum-norm `lm_head` update from violated teacher-token decision boundaries.
- All updates are fixed low-rank factors and can be fused with `linear.weight += B @ A`.
- No gradient update is applied to the base model.

`Delta W(C, x)` is query-dependent. Full-trajectory Auto-Margin is therefore
answer-trajectory compilation, not a query-agnostic replacement for a prompt.

## Quick Start

The tested remote setup is Python 3.10 with conda environment
`thoughtpatch-qwen25`. See [docs/USAGE_ZH.md](docs/USAGE_ZH.md) for setup,
dataset placement, smoke tests, sharded evaluation, and report generation.

    PYTHONPATH=src python src/eval_auto_margin_dataset.py \
      --task descriptive \
      --dataset data/p2w_bench_semantic_lengths/answer_nonverifiable.jsonl \
      --out outputs/smoke/descriptive.json \
      --model /path/to/Qwen3.5-0.8B \
      --device cuda:0 --limit 1 --max-new-tokens 256 \
      --qtraj-lm-teacher-tokens 32 \
      --auto-margin-teacher-tokens 32

## Layout

    src/       Method implementation, evaluation, merging, and summarization
    tests/     Numerical tests for the Auto-Margin convex dual
    docs/      Theory, usage, handoff notes, and reported results
    data/      Dataset mount point; JSONL data are deliberately not versioned
    outputs/   Generated reports and patches; deliberately not versioned

## Documentation

- [Usage guide (Chinese)](docs/USAGE_ZH.md)
- [Handoff (Chinese)](docs/HANDOFF_ZH.md)
- [Auto-Margin theory (Chinese)](docs/AUTO_MARGIN_TEACHER_ANCHOR_ZH.md)
- [Full-trajectory Auto-Margin result (Chinese)](docs/AUTO_MARGIN_FULL_RESULTS_QWEN35_08B_ZH.md)
- [Dataset placement](data/README.md)

## Tested Models

The code was tested with Qwen2.5-3B-Instruct, Qwen3.5-0.8B,
Llama-3.2-1B-Instruct, and Gemma-3-1B-IT. Always pass `--model` explicitly
and run a one-row smoke test before launching a new architecture.
