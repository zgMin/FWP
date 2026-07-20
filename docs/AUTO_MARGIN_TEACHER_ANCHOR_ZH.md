# QTraj + Auto-Margin Teacher Anchor

## 动机

原 Teacher-Token Margin 方法需要手工指定 `margin`、`negative_scale`、
`max_boost`、`rank`、`ridge` 和 `anchor_scale`。这些量会影响遵循能力、
参数规模和稳定性，也会随模型与任务变化。Auto-Margin 的目标是直接从
Full-prompt 与 query-only 的真实生成轨迹中确定必要边界，只求满足这些
边界的最小范数固定权重。

该方法不使用反向传播、梯度或模型训练。求解结束后得到一组固定的
`Delta W = B A`，整个生成过程只使用同一份权重，并可一次融合到模型中。

## 约束目标

设 Full-prompt greedy 输出为 `y_1, ..., y_T`，QTraj query-only 缓存轨迹在
位置 `t` 的最终隐藏状态为 `h_t`，未加 Auto-Margin 时的 logits 为 `q_t`。
当该位置的当前最强竞争 token 为 `b_t` 时，要求：

```text
((W + Delta W) h_t)[y_t] >= ((W + Delta W) h_t)[b_t] + epsilon_t
```

等价于：

```text
<Delta W, (e_y_t - e_b_t) h_t^T>
    >= q_t[b_t] - q_t[y_t] + epsilon_t
```

其中 `epsilon_t` 不是任务超参数，而是由实际注入 dtype 在当前 logit
附近的一个 ULP 自动确定，仅用于避免 BF16/FP16 舍入后出现平局。

在所有已发现的竞争边界上求最小 Frobenius 范数解：

```text
min_DeltaW  1/2 ||Delta W||_F^2
s.t.        <Delta W, v_i> >= r_i
```

其对偶问题为：

```text
max_alpha>=0  r^T alpha - 1/2 alpha^T G alpha
G_ij = <v_i, v_j>
```

解出 `alpha` 后：

```text
Delta W = sum_i alpha_i (e_y_i - e_b_i) h_i^T = B A
```

因此实际 rank 不超过活跃约束数，不再指定固定 rank，也不做低秩截断。

## Active-set 轨迹过程

1. 用 Full prompt 进行 greedy generation，保留原始 token ID，并只在 Full
   实际终止时把 EOS 纳入 teacher 轨迹。
2. 安装固定 QTraj 权重，使用 `model.generate(use_cache=True)` 的同一路径，
   通过 logits processor 强制输入 teacher token，同时抓取强制前的 logits
   和 `lm_head` 输入 hidden state。
3. 对所有 `argmax != y_t` 的位置加入 teacher-vs-current-argmax 约束。
4. 解上述凸对偶问题，构造固定 `B @ A`，再用真实 BF16 注入前向检查。
5. 若耦合更新引入新的竞争 token，增加对应约束；若仅由数值舍入导致旧
   边界失败，则按实测缺口自动收紧。没有违规位置时结束。

使用缓存生成轨迹很重要。Qwen3.5 含混合注意力模块，一次性 teacher-forcing
forward 与逐 token cache generation 可能有足以改变 argmax 的数值差异。

## 权重融合

QTraj 层更新和 Auto-Margin 输出头更新都由 `LowRankPatchedLinear` 表示，
最终执行：

```text
module.weight += B @ A
```

Qwen3.5 默认共享 input embedding 和 `lm_head.weight`。为了使融合行为与
“只注入输出头”严格一致，融合前必须复制并 untie `lm_head`，再把输出头
更新写入独立权重；否则会意外同时修改所有输入 token embedding。

## 实现入口

- `src/auto_margin_teacher_anchor.py`：约束构造、凸对偶求解、固定因子构造与融合。
- `src/eval_auto_margin_dataset.py`：可验证/描述性数据集评测。
- `src/merge_auto_margin_results.py`：与旧 Margin/Top-k 对齐合并。
- `tests/test_auto_margin_teacher_anchor.py`：最小范数约束数值测试。

远端环境仍为 `thoughtpatch-qwen25`，模型为
`/root/zgm/models/Qwen/Qwen3.5-0.8B`。

## 本轮平衡抽样结果

本轮用于方法验证，不替代完整数据集主实验：可验证任务 30 条
（英文 20、中文 10），描述性任务 18 条（英文 12、中文 6）。所有 48 条
均收敛并逐字复现 Full-prompt greedy 输出。

- 可验证：Auto-Margin EM `0.5000`、F1 `0.6458`、Containment `0.6000`、
  严格格式 `0.9667`，与 Full 和旧 Margin 相同。
- 描述性：Exact/BLEU-4/ROUGE-L/格式匹配均为 `1.0000`，长度绝对差为 `0`。
- 描述性逐 token KL：macro `0.07090`，micro `0.06611`。
- 可验证活跃 rank：均值 `0.57`，中位数 `0`，最大 `3`。
- 描述性活跃 rank：均值 `17.00`，中位数 `12.5`，最大 `43`。
- Auto anchor 参数量：可验证均值 `0.141M`；描述性均值 `4.239M`。
  旧固定 rank-128 anchor 为约 `31.916M` 参数（均不含两者共享的 QTraj）。

在 1 条额外 merge smoke test 中，模型先 untie `lm_head`，再永久合并
33 个模块；融合后输出与注入态和 Full-prompt 输出完全一致。
