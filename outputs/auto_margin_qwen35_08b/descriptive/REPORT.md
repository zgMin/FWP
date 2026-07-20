# 非可验证任务 Full-prompt 一致性报告

18 个中英文非可验证 prompt-query 配对，覆盖描述、风格、格式和输出控制任务，以及 long_redundant, medium_redundant, short 三档 prompt。BLEU、ROUGE、语义、格式和长度均以 Full prompt 的 greedy 输出为 reference。KL 在 Full-prompt greedy teacher trajectory 上逐 token 对齐。
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
| 全部 | 18 | Base (query only) | 0.2525 | 0.4793 | 0.3280 | 0.3690 | 0.8725 | 50.00% | 0.00% | 0.5998 | 86.72 | 3.001 | 50.00% | 0.737551 | 0.300180 |
| 全部 | 18 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 77.78% | 0.000000 | 0.000000 |
| 全部 | 18 | QTraj + Teacher-token | 0.5688 | 0.7004 | 0.6217 | 0.6393 | 0.9407 | 72.22% | 77.78% | 0.8937 | 42.44 | 1.732 | 72.22% | 0.432735 | 0.306395 |
| 全部 | 18 | QTraj + Top-k | 0.4812 | 0.6453 | 0.5416 | 0.5635 | 0.9240 | 77.78% | 77.78% | 0.9178 | 51.11 | 2.000 | 61.11% | 0.062767 | 0.080179 |
| 全部 | 18 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 77.78% | 0.070898 | 0.066105 |

## 按语言

| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | 12 | Base (query only) | 0.3338 | 0.5205 | 0.4100 | 0.4605 | 0.8803 | 58.33% | 0.00% | 0.6560 | 76.75 | 2.728 | 75.00% | 0.912240 | 0.434123 |
| en | 12 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 91.67% | 0.000000 | 0.000000 |
| en | 12 | QTraj + Teacher-token | 0.6801 | 0.7736 | 0.7181 | 0.7354 | 0.9469 | 66.67% | 66.67% | 0.8910 | 47.33 | 1.672 | 83.33% | 0.400440 | 0.336382 |
| en | 12 | QTraj + Top-k | 0.6084 | 0.7378 | 0.6580 | 0.6820 | 0.9296 | 75.00% | 66.67% | 0.9135 | 48.50 | 1.798 | 66.67% | 0.065886 | 0.113334 |
| en | 12 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 91.67% | 0.074215 | 0.086113 |
| zh | 6 | Base (query only) | 0.0900 | 0.3969 | 0.1639 | 0.1860 | 0.8568 | 33.33% | 0.00% | 0.4875 | 106.67 | 3.547 | 0.00% | 0.388175 | 0.157000 |
| zh | 6 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 50.00% | 0.000000 | 0.000000 |
| zh | 6 | QTraj + Teacher-token | 0.3463 | 0.5540 | 0.4291 | 0.4470 | 0.9283 | 83.33% | 100.00% | 0.8993 | 32.67 | 1.852 | 50.00% | 0.497324 | 0.274340 |
| zh | 6 | QTraj + Top-k | 0.2268 | 0.4605 | 0.3087 | 0.3264 | 0.9128 | 83.33% | 100.00% | 0.9264 | 56.33 | 2.402 | 50.00% | 0.056528 | 0.044737 |
| zh | 6 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 50.00% | 0.064264 | 0.044718 |

## 按任务族

| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| descriptive | 6 | Base (query only) | 0.1643 | 0.5621 | 0.2686 | 0.2948 | 0.9083 | 50.00% | - | 0.7244 | 25.50 | 0.892 | 50.00% | 0.116530 | 0.114466 |
| descriptive | 6 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 33.33% | 0.000000 | 0.000000 |
| descriptive | 6 | QTraj + Teacher-token | 0.3017 | 0.5790 | 0.3687 | 0.4029 | 0.8986 | 50.00% | - | 0.7447 | 26.17 | 0.894 | 50.00% | 0.218356 | 0.216697 |
| descriptive | 6 | QTraj + Top-k | 0.2851 | 0.5954 | 0.3637 | 0.3879 | 0.9061 | 66.67% | - | 0.8248 | 22.00 | 0.945 | 16.67% | 0.082140 | 0.081368 |
| descriptive | 6 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 33.33% | 0.059350 | 0.058647 |
| format | 6 | Base (query only) | 0.0249 | 0.1888 | 0.0610 | 0.1253 | 0.7724 | 0.00% | 0.00% | 0.2179 | 212.50 | 6.163 | 0.00% | 0.820823 | 0.794132 |
| format | 6 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 100.00% | 0.000000 | 0.000000 |
| format | 6 | QTraj + Teacher-token | 0.4047 | 0.5222 | 0.4966 | 0.5149 | 0.9235 | 66.67% | 66.67% | 0.9365 | 101.17 | 3.301 | 66.67% | 0.855305 | 0.836371 |
| format | 6 | QTraj + Top-k | 0.1585 | 0.3406 | 0.2610 | 0.3026 | 0.8658 | 66.67% | 66.67% | 0.9286 | 131.33 | 4.054 | 66.67% | 0.094747 | 0.105928 |
| format | 6 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 100.00% | 0.113688 | 0.119172 |
| output_control | 3 | Base (query only) | 0.2017 | 0.3855 | 0.3457 | 0.3855 | 0.8755 | 100.00% | 0.00% | 0.7143 | 44.00 | 2.913 | 100.00% | 2.239694 | 2.239694 |
| output_control | 3 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 100.00% | 0.000000 | 0.000000 |
| output_control | 3 | QTraj + Teacher-token | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 100.00% | 0.243973 | 0.243973 |
| output_control | 3 | QTraj + Top-k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 100.00% | 0.018054 | 0.018054 |
| output_control | 3 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 100.00% | 0.046486 | 0.046486 |
| style | 3 | Base (query only) | 0.9351 | 0.9885 | 0.9630 | 0.9885 | 0.9978 | 100.00% | - | 1.0000 | 0.33 | 0.981 | 100.00% | 0.310909 | 0.313405 |
| style | 3 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 100.00% | 0.000000 | 0.000000 |
| style | 3 | QTraj + Teacher-token | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 100.00% | 0.205113 | 0.206181 |
| style | 3 | QTraj + Top-k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 100.00% | 0.004773 | 0.004795 |
| style | 3 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | - | 1.0000 | 0.00 | 1.000 | 100.00% | 0.032826 | 0.032973 |

## 按 Prompt 长度

| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_redundant | 6 | Base (query only) | 0.2720 | 0.4859 | 0.3435 | 0.3784 | 0.8724 | 50.00% | 0.00% | 0.6268 | 83.50 | 2.907 | 50.00% | 0.719841 | 0.303695 |
| long_redundant | 6 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 83.33% | 0.000000 | 0.000000 |
| long_redundant | 6 | QTraj + Teacher-token | 0.5584 | 0.6959 | 0.6165 | 0.6415 | 0.9505 | 66.67% | 66.67% | 0.8967 | 40.00 | 1.975 | 66.67% | 0.417739 | 0.314359 |
| long_redundant | 6 | QTraj + Top-k | 0.4754 | 0.6368 | 0.5525 | 0.5567 | 0.9386 | 66.67% | 66.67% | 0.9011 | 62.83 | 2.046 | 66.67% | 0.061434 | 0.073761 |
| long_redundant | 6 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 83.33% | 0.068708 | 0.064168 |
| medium_redundant | 6 | Base (query only) | 0.2566 | 0.4736 | 0.3282 | 0.3706 | 0.8549 | 50.00% | 0.00% | 0.6285 | 88.50 | 3.439 | 50.00% | 0.764690 | 0.282131 |
| medium_redundant | 6 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 83.33% | 0.000000 | 0.000000 |
| medium_redundant | 6 | QTraj + Teacher-token | 0.6472 | 0.7574 | 0.6796 | 0.6895 | 0.9538 | 100.00% | 100.00% | 0.9873 | 30.17 | 1.663 | 83.33% | 0.455851 | 0.279301 |
| medium_redundant | 6 | QTraj + Top-k | 0.5202 | 0.6667 | 0.5750 | 0.6000 | 0.9278 | 100.00% | 100.00% | 0.9586 | 38.67 | 1.956 | 66.67% | 0.036617 | 0.047968 |
| medium_redundant | 6 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 83.33% | 0.059364 | 0.050153 |
| short | 6 | Base (query only) | 0.2290 | 0.4784 | 0.3121 | 0.3581 | 0.8901 | 50.00% | 0.00% | 0.5441 | 88.17 | 2.656 | 50.00% | 0.728123 | 0.313230 |
| short | 6 | Full prompt | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 66.67% | 0.000000 | 0.000000 |
| short | 6 | QTraj + Teacher-token | 0.5008 | 0.6479 | 0.5692 | 0.5868 | 0.9179 | 50.00% | 66.67% | 0.7972 | 57.17 | 1.557 | 66.67% | 0.424614 | 0.323520 |
| short | 6 | QTraj + Top-k | 0.4480 | 0.6325 | 0.4973 | 0.5338 | 0.9055 | 66.67% | 66.67% | 0.8937 | 51.83 | 1.997 | 50.00% | 0.090250 | 0.115101 |
| short | 6 | QTraj + Auto-Margin | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 100.00% | 100.00% | 1.0000 | 0.00 | 1.000 | 66.67% | 0.084623 | 0.082262 |

## 逐 Token KL 文件

- `token_kl.jsonl` / `token_kl.csv`：每个 pair、方法和 teacher token 的 KL、token log-prob、EOS 标记。
- `token_kl_by_position.csv`：按语言、方法和 token 位置聚合的均值、中位数、最小值、最大值，可直接绘制折线图。
- `token_kl_by_length_position.csv`：按 prompt 长度、方法和 token 位置聚合，用于分析长度敏感性。

## 输入输出示例

### pair--zh-descriptive-001--short--zh-descriptive-001--q001

- Prompt：围绕查询的核心问题作答，并完整说明主要观点。
- Query：解释人类行为。

输入：行为：哭泣。
- Base (query only)：人类行为是一个复杂且多维的概念，它不仅仅是生理反应或情绪表达，更是个体与环境互动、认知加工以及社会适应的结果。

以"**哭泣**"为例，我们可以从以下几个层面来理解其本质：

### 1. 生理与心理的即时反应
从生物学角度看，哭泣是身体对疼痛、恐惧或悲伤的一种**保护性反射**。
*   **生理机制**：当情绪激动时，大脑分泌肾上腺素和皮质醇，导致心率加快、血压升高，进而引发眼泪分泌。
*   **心理意义**：哭泣通常被视为一种**情绪宣泄**或**自我安抚**的方式。它允许个体释放积压的压力，缓解焦虑，并可能带来短暂的平静感。

### 2. 社会与文化意义的投射
在人类社会中，哭泣往往超越了单纯的生理需求，成为一种**社会符号**。
*   **情感表达**：哭泣是表达痛苦、哀悼或共情的重要方式。它向他人传递了“我感受到了你的痛苦”这一信息。
*   **社会功能**：在集体生活中，哭泣有助于建立信任、促进和解（如葬礼上的哭泣）或维持群体凝聚力。

### 3. 行为背后的深层动机
理解人类行为 (BLEU=0.142, R-L=0.219, semantic=0.934, KL=0.07627)
- Full prompt：人类行为是一个复杂且多维度的概念，其核心在于个体在特定情境下，通过特定的心理机制和生理反应，对周围环境做出反应并产生相应后果的过程。

首先，从**功能主义**的角度来看，人类行为具有生存和繁衍的适应性意义。例如，哭泣作为一种防御性行为，其核心功能是缓解痛苦、表达悲伤或寻求安慰，从而帮助个体维持心理平衡。这种行为并非单纯的生理反应，而是个体在面临威胁或不适时，通过释放情绪能量来应对环境压力的策略。

其次，从**认知心理学**的角度分析，人类行为深受**情绪调节**和**认知评估**的影响。当个体感知到负面刺激（如疼痛、恐惧）时，大脑会激活杏仁核等区域，引发恐惧反应，进而导致哭泣。这一过程不仅涉及生理上的出汗、流泪等生理变化，还涉及认知层面的“评估”——即个体判断当前情境是否值得哭泣，以及哭泣是否能带来某种心理上的解脱或缓解。

最后，从**社会文化**视角看，人类行为还受到**文化规范**和**社会互动**的深刻塑造。哭泣作为一种社会行为，往往承载着特定的意义，如表达哀悼、分享喜悦或寻求群体支持。 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：人类行为是一个复杂且多维度的概念，其核心在于个体在特定情境下，通过特定的心理机制和生理反应，对周围环境做出反应并产生相应后果的过程。

从**定义层面**来看，人类行为并非单一动作，而是由多种因素共同构成的动态系统，主要包括：
*   **生理基础**：包括神经系统（如大脑皮层）、激素分泌（如肾上腺素、皮质醇）以及身体机能（如呼吸、心跳）的协调反应。
*   **认知与情感**：涉及情绪体验（如恐惧、愤怒）、认知评估（如判断风险）以及自我调节能力。
*   **社会互动**：在群体中，行为往往受到文化规范、语言符号、互惠原则以及社会期望的引导。
*   **环境刺激**：外部压力源（如威胁、资源匮乏）和内部需求（如饥饿、孤独）是驱动行为变化的关键变量。

从**功能层面**分析，人类行为具有显著的适应性价值：
1.  **生存与繁衍**：通过觅食、筑巢、交配等行为，个体确保自身生存并延续基因。
2.  **社会整合**：合作、冲突、仪式等行为维系群体秩序 (BLEU=0.290, R-L=0.334, semantic=0.865, KL=0.21042)
- QTraj + Top-k：人类行为是一个复杂且多维度的概念，其核心在于个体在特定情境下，通过特定的心理机制和生理反应，对周围环境做出反应并产生相应后果的过程。

从**定义层面**来看，人类行为并非单一动作，而是由多种因素共同构成的动态系统。它通常包含以下关键要素：

1.  **情境触发**：行为的发生往往依赖于外部刺激（如疼痛、恐惧、饥饿）或内部情绪（如愤怒、悲伤）。
2.  **认知评估**：个体会分析刺激与自身需求或价值观之间的匹配度，评估行为的合理性与必要性。
3.  **动机驱动**：行为背后通常隐藏着深层的动机，例如生存本能、社会规范、自我实现或逃避痛苦。
4.  **执行与反应**：个体通过神经系统（如大脑皮层、边缘系统）将认知转化为具体的动作，并伴随生理反应（如心跳加速、肌肉收缩）。
5.  **后果与反馈**：行为最终会引发结果（如疼痛、快乐、恐惧），并可能引发新的行为模式或情绪调节。

**举例说明**：
当面对被咬伤时，人类行为表现为：
*   **认知评估**：个体 (BLEU=0.278, R-L=0.309, semantic=0.855, KL=0.04499)
- QTraj + Auto-Margin：人类行为是一个复杂且多维度的概念，其核心在于个体在特定情境下，通过特定的心理机制和生理反应，对周围环境做出反应并产生相应后果的过程。

首先，从**功能主义**的角度来看，人类行为具有生存和繁衍的适应性意义。例如，哭泣作为一种防御性行为，其核心功能是缓解痛苦、表达悲伤或寻求安慰，从而帮助个体维持心理平衡。这种行为并非单纯的生理反应，而是个体在面临威胁或不适时，通过释放情绪能量来应对环境压力的策略。

其次，从**认知心理学**的角度分析，人类行为深受**情绪调节**和**认知评估**的影响。当个体感知到负面刺激（如疼痛、恐惧）时，大脑会激活杏仁核等区域，引发恐惧反应，进而导致哭泣。这一过程不仅涉及生理上的出汗、流泪等生理变化，还涉及认知层面的“评估”——即个体判断当前情境是否值得哭泣，以及哭泣是否能带来某种心理上的解脱或缓解。

最后，从**社会文化**视角看，人类行为还受到**文化规范**和**社会互动**的深刻塑造。哭泣作为一种社会行为，往往承载着特定的意义，如表达哀悼、分享喜悦或寻求群体支持。 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.04197)

### pair--zh-format-001--medium_redundant--zh-format-001--q001

- Prompt：请严格输出一个JSON对象，只包含字符串字段answer。
为避免歧义，上述内容只规定输出格式。所要求的格式不变，再次说明: 请严格输出一个JSON对象，只包含字符串字段answer。
不增加任何新条件，上述内容只规定输出格式。所要求的格式不变，再次说明: 请严格输出一个JSON对象，只包含字符串字段answer。
- Query：给定一种运动的名称，说明如何做这个运动。

输入：搏击操
- Base (query only)：**搏击操**（Barko）是一种结合了传统武术（如中国武术、日本空手道、巴西柔术等）与现代健身元素的综合运动形式。它强调动作的连贯性、力量控制以及身体协调性，通常以“练功”或“练气”为核心目标，而非单纯的竞技对抗。

以下是进行搏击操的主要步骤和核心要点：

### 1. 准备阶段
*   **热身**：在正式训练前，必须进行充分的热身。包括关节活动、拉伸和动态拉伸，以预防运动损伤。
*   **装备**：通常穿着专业的运动服，佩戴护具（如护腕、护膝、护肘），并带上必要的器械（如拳套、护肘、护膝等）。
*   **心态**：保持专注，将注意力集中在动作的发力点和节奏上，而非单纯追求速度。

### 2. 动作分解与组合
搏击操的动作通常分为几个主要部分，每个部分都有特定的训练目的：
*   **基础动作**：如站桩、弓步、猫牛式等，用于建立基础力量和柔韧性。
*   **核心力量**：如单腿支撑、侧踢、 (BLEU=0.042, R-L=0.138, semantic=0.799, KL=0.79281)
- Full prompt：{
  "answer": "搏击操是一种结合了拳击、踢击和格斗技巧的武术运动，通过模拟真实的搏击动作来锻炼身体协调性和力量。"
} (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：{
  "answer": "搏击操是一种结合了拳击、踢击和格斗技巧的武术运动，通过模拟真实的搏击动作来锻炼身体协调性和力量。具体做法包括：\n1. 准备阶段：穿着专业的搏击服，佩戴护具（如头盔、护腕、护膝等），并检查装备的完整性。\n2. 热身：进行动态拉伸和基础动作练习，如左右腿开合、手臂挥击等，提高身体灵活性。\n3. 核心训练：使用模拟的靶子或地面，练习快速转身、侧身闪避和快速反击等核心力量。\n4. 实战模拟：在教练指导下进行模拟的搏击动作，包括步法移动、击打模拟和防守反击，强化反应速度和力量控制。\n5. 放松与恢复：运动结束后进行拉伸和休息，避免过度疲劳，确保身体恢复。"
} (BLEU=0.190, R-L=0.326, semantic=0.936, KL=0.97767)
- QTraj + Top-k：{
  "answer": "搏击操是一种结合了拳击、踢击和格斗技巧的武术运动，通过模拟真实的搏击动作来锻炼身体的协调性、力量和反应能力。具体做法包括：\n1. 准备阶段：穿着专业的搏击服，佩戴护具（如头盔、护腕、护膝等），并检查装备是否完好。\n2. 热身：进行动态拉伸，提高身体灵活性，避免运动损伤。\n3. 基础训练：学习基本的击打动作，如左拳右腿、左腿右腿等，练习正确的发力方向和角度。\n4. 进阶训练：增加复杂动作，如交叉腿、侧踢、上步等，提高身体协调性和平衡能力。\n5. 实战模拟：在教练指导下进行模拟对抗，练习应对对手攻击的战术和策略。\n6. 休息与恢复：运动后充分休息，避免过度疲劳，保证身体恢复。"
} (BLEU=0.178, R-L=0.311, semantic=0.932, KL=0.05789)
- QTraj + Auto-Margin：{
  "answer": "搏击操是一种结合了拳击、踢击和格斗技巧的武术运动，通过模拟真实的搏击动作来锻炼身体协调性和力量。"
} (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.08269)

### pair--zh-descriptive-001--long_redundant--zh-descriptive-001--q001

- Prompt：你是一个有用且可靠的助手。用户会提出一个需要回答、解释或完成的询问。请先把握其中的核心任务，让回复始终围绕这一任务展开。回答时应把直接相关的主要观点说明完整，使最终回复能够充分回应用户真正想了解或完成的内容。
- Query：解释人类行为。

输入：行为：哭泣。
- Base (query only)：人类行为是一个复杂且多维的概念，它不仅仅是生理反应或情绪表达，更是个体与环境互动、认知加工以及社会适应的结果。

以"**哭泣**"为例，我们可以从以下几个层面来理解其本质：

### 1. 生理与心理的即时反应
从生物学角度看，哭泣是身体对疼痛、恐惧或悲伤的一种**保护性反射**。
*   **生理机制**：当情绪激动时，大脑分泌肾上腺素和皮质醇，导致心率加快、血压升高，进而引发眼泪分泌。
*   **心理意义**：哭泣通常被视为一种**情绪宣泄**或**自我安抚**的方式。它允许个体释放积压的压力，缓解焦虑，并可能带来短暂的平静感。

### 2. 社会与文化意义的投射
在人类社会中，哭泣往往超越了单纯的生理需求，成为一种**社会符号**。
*   **情感表达**：哭泣是表达痛苦、哀悼或共情的重要方式。它向他人传递了“我感受到了你的痛苦”这一信息。
*   **社会功能**：在集体生活中，哭泣有助于建立信任、促进和解（如葬礼上的哭泣）或维持群体凝聚力。

### 3. 行为背后的深层动机
理解人类行为 (BLEU=0.123, R-L=0.201, semantic=0.882, KL=0.05527)
- Full prompt：人类行为中，**哭泣**是一种普遍且重要的情绪表达，它反映了个体内心的感受、需求或压力。以下是关于哭泣的核心解释：

### 1. 情绪表达与宣泄
哭泣是许多人在感到痛苦、悲伤或愤怒时释放情绪的自然方式。当一个人感到难过、委屈或愤怒时，通过哭泣可以：
*   **释放压力**：将内心的负面情绪转化为身体上的释放，缓解心理负担。
*   **沟通需求**：向他人表达自己无法用语言完全传达的感受，寻求理解或安慰。

### 2. 生理反应与身体调节
哭泣是一个复杂的生理过程，涉及多个系统的协调：
*   **神经系统**：触发副交感神经，使心跳减慢、呼吸变浅，帮助身体从紧张状态恢复平静。
*   **内分泌系统**：分泌催产素（Oxytocin）和催乳素，促进亲密关系和信任感。
*   **皮肤与体温**：皮肤血管收缩，体温下降，使身体感觉凉爽。

### 3. 社会功能与意义
在人类社会中，哭泣具有深远的社会意义：
*   **建立连接**：哭泣是建立情感纽带、表达 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：人类行为中，**哭泣**是一种普遍且重要的情绪表达，它反映了个体内心的感受、需求或压力。以下是关于哭泣的核心解释：

### 1. 核心功能：表达情绪与沟通
哭泣的主要功能是**释放负面情绪**，帮助个体向他人或自己传达内心的感受。
*   **表达悲伤**：当人们感到难过、失落或失去亲人时，哭泣是自然且正常的反应，它向周围人传递“我很伤心”的信号。
*   **表达恐惧与焦虑**：面对威胁、疼痛或压力时，哭泣是身体释放压力激素（如皮质醇）的机制，同时也能向他人表明“我很害怕”或“我很痛苦”。
*   **表达愤怒**：哭泣是愤怒情绪的出口，它既可能是为了宣泄怒火，也可能是为了寻求理解或向他人道歉。

### 2. 生理机制：身体调节
哭泣是一个复杂的生理过程，涉及多个系统：
*   **情绪中枢**：大脑的**杏仁核**（负责情绪识别）和**前额叶皮层**（负责理性控制）会同时工作。当情绪强烈时，前额叶皮层会抑制杏仁核的过度反应，使哭泣行为变得理性 (BLEU=0.334, R-L=0.384, semantic=0.958, KL=0.18221)
- QTraj + Top-k：人类行为中，**哭泣**是一种普遍且重要的情绪表达，它反映了个体内心的感受、需求或压力。以下是关于哭泣的核心解释：

### 1. 核心功能：表达与沟通
哭泣的主要功能是**向他人传达情绪**，帮助他人理解你的感受。
*   **情感宣泄**：当感到悲伤、愤怒、恐惧或委屈时，哭泣是释放负面情绪最直接的方式。
*   **建立连接**：哭泣通常伴随着微笑或拥抱，这能向他人传递“我感受到了你的情绪”的信号，从而增强人际关系的信任与亲密感。

### 2. 常见触发场景
哭泣并非总是由悲伤引起，它也可能由其他强烈的情绪触发，例如：
*   **极度悲伤**：如失去亲人、重病或重大挫折。
*   **愤怒**：面对不公、侮辱或挑衅时，哭泣是抗议和自我保护的体现。
*   **恐惧**：面对危险、灾难或无法控制的突发状况时，哭泣是本能反应。
*   **委屈与难过**：面对被误解、被拒绝或生活低谷时，这是表达无助和寻求安慰的常见行为。

### 3. 生理与心理机制
*   ** (BLEU=0.273, R-L=0.323, semantic=0.959, KL=0.03689)
- QTraj + Auto-Margin：人类行为中，**哭泣**是一种普遍且重要的情绪表达，它反映了个体内心的感受、需求或压力。以下是关于哭泣的核心解释：

### 1. 情绪表达与宣泄
哭泣是许多人在感到痛苦、悲伤或愤怒时释放情绪的自然方式。当一个人感到难过、委屈或愤怒时，通过哭泣可以：
*   **释放压力**：将内心的负面情绪转化为身体上的释放，缓解心理负担。
*   **沟通需求**：向他人表达自己无法用语言完全传达的感受，寻求理解或安慰。

### 2. 生理反应与身体调节
哭泣是一个复杂的生理过程，涉及多个系统的协调：
*   **神经系统**：触发副交感神经，使心跳减慢、呼吸变浅，帮助身体从紧张状态恢复平静。
*   **内分泌系统**：分泌催产素（Oxytocin）和催乳素，促进亲密关系和信任感。
*   **皮肤与体温**：皮肤血管收缩，体温下降，使身体感觉凉爽。

### 3. 社会功能与意义
在人类社会中，哭泣具有深远的社会意义：
*   **建立连接**：哭泣是建立情感纽带、表达 (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.03491)

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

### pair--en-descriptive-001--medium_redundant--en-descriptive-001--q001

- Prompt：You are a helpful assistant. Understand the central task in the user's query, respond to that task, and explain the main points needed for a complete answer.
- Query：What is the third limb of yoga?

Task context: sanas originated in India. In his Yoga Sutras, Patanjali (c. 2nd to 4th century CE) describes asana practice as the third of the eight limbs (Sanskrit: अष्टाङ्ग, aṣṭāṅga, from अष्ट् aṣṭ, eight, and अङ्ग aṅga, limb) of classical, or raja yoga.
- Base (query only)：Based on the context provided and classical yoga tradition, the third limb of yoga is **Asana** (Sanskrit: *Aṣṭāṅga*).

In the *Yoga Sutras of Patanjali*, the eight limbs of Raja Yoga are ordered as follows:
1.  **Niyama** (Purification)
2.  **Asana** (Posture)
3.  **Pranayama** (Breath Control)
4.  **Kriya** (Meditation)
5.  **Dharana** (Concentration)
6.  **Dhyana** (Deep Meditation)
7.  **Samadhi** (Samadhi)
8.  **Nidra** (Sleep)

Therefore, the third limb is **Asana**. (BLEU=0.135, R-L=0.333, semantic=0.909, KL=0.11960)
- Full prompt：The **third limb of yoga** is **Asana** (Sanskrit: *Asana*).

Here are the key points needed for a complete answer:

*   **Definition**: Asana refers to the physical postures or positions that practitioners adopt to prepare the body for meditation and to cultivate mental clarity.
*   **Context**: It is the third of the eight limbs of *Raja Yoga* (also known as *Yoga Sutras* by Patanjali), which are the foundational principles of classical yoga practice.
*   **Purpose**: Asana serves as the preparatory stage for the subsequent four limbs (Pranayama, Pratyahara, Dharana, and Dhyana). It helps the practitioner build strength, flexibility, and balance before entering the meditative state.
*   **Philosophical Basis**: While the specific physical details of Asana are often described in detail in texts like the *Yoga Sutras*, the core concept is that it is the first step in the practice of yoga, leading to the realization of the Self (*Atman*). (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：The **third limb of yoga** is **Asana** (Sanskrit: *Asana*).

Here are the key points needed for a complete answer:

*   **Definition**: Asana refers to the preparatory phase of yoga, where the practitioner performs physical postures (postures) to build strength, flexibility, and balance.
*   **Context**: In the *Yoga Sutras of Patanjali* (c. 2nd–4th century CE), Asana is explicitly listed as the **third limb** of the eight limbs (*Ashtanga*).
*   **Purpose**: It is considered the most important and foundational limb, as it is the primary method for developing the necessary physical and mental qualities for the subsequent stages of yoga.
*   **Related Concept**: The **fourth limb** is **Bhava** (mental state), which is the state of mind that arises after the physical postures are completed. (BLEU=0.429, R-L=0.485, semantic=0.944, KL=0.20223)
- QTraj + Top-k：The **third limb of yoga** is **Asana** (Sanskrit: *Asana*).

Here are the key points needed for a complete answer:

*   **Definition**: Asana refers to the preparatory phase of yoga, where the practitioner performs physical postures (postures) to build strength, flexibility, and balance.
*   **Context**: In the *Yoga Sutras of Patanjali* (c. 2nd–4th century CE), Asana is explicitly listed as the **third limb** (*aṣṭāṅga*, meaning "eight limbs").
*   **Purpose**: It is considered the most important and foundational stage of yoga, as it is where the body is first prepared for the subsequent stages of Vinyasa and Hatha yoga.
*   **Comparison**:
    *   **First limb**: Prana (Energy)
    *   **Second limb**: Pratyahara (Withdrawal of senses)
    *   **Third limb**: Asana (Physical postures)
    *   **Fourth limb**: Dharana (Concentration)
    *   **Fifth limb**: Dhyana (Deep Meditation)
    * (BLEU=0.382, R-L=0.434, semantic=0.947, KL=0.06714)
- QTraj + Auto-Margin：The **third limb of yoga** is **Asana** (Sanskrit: *Asana*).

Here are the key points needed for a complete answer:

*   **Definition**: Asana refers to the physical postures or positions that practitioners adopt to prepare the body for meditation and to cultivate mental clarity.
*   **Context**: It is the third of the eight limbs of *Raja Yoga* (also known as *Yoga Sutras* by Patanjali), which are the foundational principles of classical yoga practice.
*   **Purpose**: Asana serves as the preparatory stage for the subsequent four limbs (Pranayama, Pratyahara, Dharana, and Dhyana). It helps the practitioner build strength, flexibility, and balance before entering the meditative state.
*   **Philosophical Basis**: While the specific physical details of Asana are often described in detail in texts like the *Yoga Sutras*, the core concept is that it is the first step in the practice of yoga, leading to the realization of the Self (*Atman*). (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.05818)

### pair--en-format-001--long_redundant--en-format-001--q001

- Prompt：Return exactly one JSON object containing only the string field answer.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Return exactly one JSON object containing only the string field answer.
Without adding any new condition, the statement above specifies only the output format. The unchanged format requirement is restated: Return exactly one JSON object containing only the string field answer.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Return exactly one JSON object containing only the string field answer.
Without adding any new condition, the statement above specifies only the output format. The unchanged format requirement is restated: Return exactly one JSON object containing only the string field answer.
For clarity, the statement above specifies only the output format. The unchanged format requirement is restated: Return exactly one JSON object containing only the string field answer.
Without adding any new condition, the statement above specifies only the output format. The unchanged format requirement is restated: Return exactly one JSON object containing only the string field answer.
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
*   **Mid (BLEU=0.010, R-L=0.115, semantic=0.772, KL=0.94838)
- Full prompt：{
  "answer": "You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your riding style and budget."
} (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.00000)
- QTraj + Teacher-token：{
  "answer": "You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your needs. You should consider the brand, (BLEU=0.118, R-L=0.224, semantic=0.879, KL=0.94367)
- QTraj + Top-k：{
  "answer": "You should consider the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) of the brand, price, and features (e.g., battery life, durability, comfort) (BLEU=0.081, R-L=0.174, semantic=0.785, KL=0.10287)
- QTraj + Auto-Margin：{
  "answer": "You should consider the brand, price, and features (e.g., battery, frame material, tire type) that match your riding style and budget."
} (BLEU=1.000, R-L=1.000, semantic=1.000, KL=0.12012)
