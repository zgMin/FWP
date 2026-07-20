#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModel, AutoTokenizer

from p2w_output_validators import validate_output


METHODS = [
    "base_no_prompt",
    "full_prompt",
    "qtraj_teacher_margin",
    "qtraj_topk_delta",
    "qtraj_teacher_auto_margin",
]
LABELS = {
    "base_no_prompt": "Base (query only)",
    "full_prompt": "Full prompt",
    "qtraj_teacher_margin": "QTraj + Teacher-token",
    "qtraj_topk_delta": "QTraj + Top-k",
    "qtraj_teacher_auto_margin": "QTraj + Auto-Margin",
}


def text_tokens(text: str, language: str) -> list[str]:
    text = text.lower()
    if language == "zh":
        return re.findall(r"[\u3400-\u9fff]|[a-z]+|\d+(?:\.\d+)?", text)
    return re.findall(r"[a-z0-9]+(?:['’-][a-z0-9]+)?|[^\w\s]", text)


def bleu4(reference: str, hypothesis: str, language: str) -> float:
    ref = text_tokens(reference, language)
    hyp = text_tokens(hypothesis, language)
    if not ref or not hyp:
        return float(ref == hyp)
    return float(
        sentence_bleu(
            [ref],
            hyp,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method3,
        )
    )


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1)))


def overlap_f1(reference: list[str], hypothesis: list[str], n: int) -> float:
    ref = ngrams(reference, n)
    hyp = ngrams(hypothesis, n)
    if not ref or not hyp:
        return float(ref == hyp)
    overlap = sum((ref & hyp).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: list[str], hypothesis: list[str]) -> float:
    if not reference or not hypothesis:
        return float(reference == hypothesis)
    overlap = lcs_length(reference, hypothesis)
    if not overlap:
        return 0.0
    precision = overlap / len(hypothesis)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def format_signature(text: str) -> dict:
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", stripped) if part.strip()]
    container = "plain"
    try:
        parsed = json.loads(stripped)
        container = "json_object" if isinstance(parsed, dict) else "json_array" if isinstance(parsed, list) else "json_scalar"
    except (json.JSONDecodeError, TypeError):
        if re.fullmatch(r"\s*<([A-Za-z][\w:-]*)\b[^>]*>.*</\1>\s*", stripped, re.DOTALL):
            container = "xml"
        elif stripped.startswith("```") and stripped.endswith("```"):
            container = "code_fence"

    numbered = sum(bool(re.match(r"^\s*\d+[.)、．]\s*", line)) for line in lines)
    bullets = sum(bool(re.match(r"^\s*[-*•]\s+", line)) for line in lines)
    if numbered >= 2:
        list_style = "numbered"
        list_count = numbered
    elif bullets >= 2:
        list_style = "bulleted"
        list_count = bullets
    else:
        list_style = "none"
        list_count = 0
    return {
        "container": container,
        "list_style": list_style,
        "list_count": list_count,
        "multiline": len(lines) > 1,
        "line_count": len(lines),
        "paragraph_count": len(paragraphs),
        "has_markdown_heading": any(re.match(r"^#{1,6}\s+", line) for line in lines),
    }


def count_similarity(left: int, right: int) -> float:
    return 1.0 - abs(left - right) / max(left, right, 1)


def format_metrics(reference: str, hypothesis: str) -> tuple[bool, float, dict, dict]:
    ref = format_signature(reference)
    hyp = format_signature(hypothesis)
    type_match = ref["container"] == hyp["container"] and ref["list_style"] == hyp["list_style"]
    features = [
        float(ref["container"] == hyp["container"]),
        float(ref["list_style"] == hyp["list_style"]),
        float(ref["multiline"] == hyp["multiline"]),
        float(ref["has_markdown_heading"] == hyp["has_markdown_heading"]),
        count_similarity(ref["list_count"], hyp["list_count"]),
        count_similarity(ref["line_count"], hyp["line_count"]),
        count_similarity(ref["paragraph_count"], hyp["paragraph_count"]),
    ]
    return type_match, sum(features) / len(features), ref, hyp


@torch.no_grad()
def semantic_embeddings(texts: list[str], model_path: str, device: str, batch_size: int) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(device)
    model.eval()
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        output = model(**inputs)
        embedding = F.normalize(output.last_hidden_state[:, 0, :].float(), p=2, dim=-1)
        embeddings.append(embedding.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(embeddings, dim=0)


def add_semantic_scores(rows: list[dict], model_path: str, device: str, batch_size: int) -> None:
    texts = []
    locations = []
    for row_index, row in enumerate(rows):
        for method in METHODS:
            texts.append(row["outputs"][method])
            locations.append((row_index, method))
    embeddings = semantic_embeddings(texts, model_path, device, batch_size)
    by_location = {location: embeddings[index] for index, location in enumerate(locations)}
    for row_index, row in enumerate(rows):
        reference = by_location[(row_index, "full_prompt")]
        for method in METHODS:
            row["comparison_metrics"][method]["semantic_cosine_bge_m3"] = float(
                torch.dot(reference, by_location[(row_index, method)]).item()
            )


def evaluate_text(row: dict, method: str) -> dict:
    reference = row["outputs"]["full_prompt"]
    hypothesis = row["outputs"][method]
    language = row["language"]
    ref_tokens = text_tokens(reference, language)
    hyp_tokens = text_tokens(hypothesis, language)
    type_match, structure_score, ref_format, hyp_format = format_metrics(reference, hypothesis)
    reference_model_tokens = row["output_token_counts"]["full_prompt"]
    hypothesis_model_tokens = row["output_token_counts"][method]
    validator_applicable = row.get("validator", {}).get("type") != "full_prompt_similarity"
    validator_pass = validate_output(hypothesis, row["validator"])[0] if validator_applicable else None
    return {
        "exact_output_match": reference == hypothesis,
        "bleu4": bleu4(reference, hypothesis, language),
        "rouge1_f1": overlap_f1(ref_tokens, hyp_tokens, 1),
        "rouge2_f1": overlap_f1(ref_tokens, hyp_tokens, 2),
        "rougeL_f1": rouge_l(ref_tokens, hyp_tokens),
        "format_type_match": type_match,
        "format_structure_score": structure_score,
        "reference_format": ref_format,
        "hypothesis_format": hyp_format,
        "reference_model_tokens": reference_model_tokens,
        "hypothesis_model_tokens": hypothesis_model_tokens,
        "length_diff_tokens": hypothesis_model_tokens - reference_model_tokens,
        "absolute_length_diff_tokens": abs(hypothesis_model_tokens - reference_model_tokens),
        "length_ratio_tokens": hypothesis_model_tokens / max(reference_model_tokens, 1),
        "terminated_with_eos": row["terminated_with_eos"][method],
        "reference_chars": len(reference),
        "hypothesis_chars": len(hypothesis),
        "length_diff_chars": len(hypothesis) - len(reference),
        "trajectory_kl_macro": row["trajectory_kl_mean"][method],
        "validator_applicable": validator_applicable,
        "validator_pass": validator_pass,
    }


def aggregate(rows: list[dict]) -> dict:
    result = {"rows": len(rows)}
    metric_keys = [
        "exact_output_match",
        "bleu4",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "semantic_cosine_bge_m3",
        "format_type_match",
        "format_structure_score",
        "length_diff_tokens",
        "absolute_length_diff_tokens",
        "length_ratio_tokens",
        "terminated_with_eos",
        "length_diff_chars",
        "trajectory_kl_macro",
    ]
    for method in METHODS:
        metrics = [row["comparison_metrics"][method] for row in rows]
        result[method] = {
            key: sum(float(item[key]) for item in metrics) / len(metrics) if metrics else 0.0
            for key in metric_keys
        }
        validator_metrics = [item for item in metrics if item["validator_applicable"]]
        result[method]["validator_adherence"] = (
            sum(float(item["validator_pass"]) for item in validator_metrics) / len(validator_metrics)
            if validator_metrics
            else None
        )
        result[method]["validator_rows"] = len(validator_metrics)
        if method == "full_prompt":
            token_kls = [0.0 for row in rows for _ in range(row["teacher_trajectory_token_count"])]
        else:
            token_kls = [
                token["kl_full_to_method"]
                for row in rows
                for token in row["token_kl"].get(method, [])
            ]
        result[method]["trajectory_kl_micro"] = sum(token_kls) / len(token_kls) if token_kls else 0.0
        result[method]["trajectory_token_count"] = len(token_kls)
    return result


def grouped(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: aggregate(value) for key, value in sorted(groups.items())}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def optional_pct(value: float | None) -> str:
    return "-" if value is None else pct(value)


def metric_table(title: str, groups: dict) -> str:
    lines = [
        f"## {title}",
        "",
        "| 分组 | N | 方法 | BLEU-4 | R-1 | R-2 | R-L | BGE cosine | 格式匹配 | 规则遵循 | 结构分 | |Δtokens| | 长度比 | EOS率 | KL macro | KL micro |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, values in groups.items():
        for method in METHODS:
            score = values[method]
            lines.append(
                f"| {group} | {values['rows']} | {LABELS[method]} | {score['bleu4']:.4f} | "
                f"{score['rouge1_f1']:.4f} | {score['rouge2_f1']:.4f} | {score['rougeL_f1']:.4f} | "
                f"{score['semantic_cosine_bge_m3']:.4f} | {pct(score['format_type_match'])} | {optional_pct(score['validator_adherence'])} | "
                f"{score['format_structure_score']:.4f} | {score['absolute_length_diff_tokens']:.2f} | "
                f"{score['length_ratio_tokens']:.3f} | {pct(score['terminated_with_eos'])} | {score['trajectory_kl_macro']:.6f} | "
                f"{score['trajectory_kl_micro']:.6f} |"
            )
    return "\n".join(lines)


def choose_examples(rows: list[dict]) -> list[dict]:
    selected = []
    for language in ("zh", "en"):
        for length_variant in ("short", "medium_redundant", "long_redundant"):
            candidates = [
                row
                for row in rows
                if row["language"] == language and row["length_variant"] == length_variant
            ]
            candidates.sort(
                key=lambda row: (
                    row["comparison_metrics"]["qtraj_teacher_margin"]["bleu4"],
                    -row["comparison_metrics"]["qtraj_teacher_margin"]["trajectory_kl_macro"],
                )
            )
            selected.extend(candidates[:1])
    return selected


def write_token_trajectories(rows: list[dict], out_dir: Path) -> None:
    flat = []
    for row in rows:
        for method in METHODS:
            tokens = row["token_kl"].get(method, [])
            if method == "full_prompt":
                tokens = [
                    {
                        **token,
                        "method": "full_prompt",
                        "kl_full_to_method": 0.0,
                        "method_teacher_token_logprob": token["full_teacher_token_logprob"],
                    }
                    for token in row["token_kl"]["base_no_prompt"]
                ]
            for token in tokens:
                flat.append(
                    {
                        "pair_id": row["pair_id"],
                        "query_id": row["query_id"],
                        "language": row["language"],
                        "task_family": row["task_family"],
                        "subtype": row["subtype"],
                        "length_variant": row["length_variant"],
                        **token,
                    }
                )
    with (out_dir / "token_kl.jsonl").open("w", encoding="utf-8") as handle:
        for item in flat:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    columns = [
        "pair_id", "query_id", "language", "task_family", "subtype", "length_variant", "method", "step", "teacher_token_id",
        "teacher_token", "is_eos", "kl_full_to_method", "full_teacher_token_logprob",
        "method_teacher_token_logprob",
    ]
    with (out_dir / "token_kl.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)

    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for item in flat:
        groups[(item["language"], item["method"], item["step"])].append(item["kl_full_to_method"])
        groups[("all", item["method"], item["step"])].append(item["kl_full_to_method"])
    with (out_dir / "token_kl_by_position.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "method", "step", "sample_count", "mean_kl", "median_kl", "min_kl", "max_kl"])
        for (language, method, step), values in sorted(groups.items()):
            writer.writerow([language, method, step, len(values), sum(values) / len(values), statistics.median(values), min(values), max(values)])

    length_groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for item in flat:
        length_groups[(item["length_variant"], item["method"], item["step"])].append(
            item["kl_full_to_method"]
        )
    with (out_dir / "token_kl_by_length_position.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["length_variant", "method", "step", "sample_count", "mean_kl", "median_kl", "min_kl", "max_kl"]
        )
        for (length_variant, method, step), values in sorted(length_groups.items()):
            writer.writerow(
                [
                    length_variant,
                    method,
                    step,
                    len(values),
                    sum(values) / len(values),
                    statistics.median(values),
                    min(values),
                    max(values),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=12)
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--semantic-device", default="cuda:0")
    parser.add_argument("--semantic-batch-size", type=int, default=4)
    args = parser.parse_args()

    rows = []
    configs = []
    runtimes = []
    for path in args.inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["results"])
        configs.append(payload.get("config", {}))
        runtimes.append(payload.get("runtime_seconds", 0.0))
    rows.sort(key=lambda row: row["pair_id"])
    if len(rows) != args.expected_rows or len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(f"expected {args.expected_rows} unique rows, got {len(rows)}")

    for row in rows:
        row["comparison_metrics"] = {method: evaluate_text(row, method) for method in METHODS}
    add_semantic_scores(rows, args.semantic_model, args.semantic_device, args.semantic_batch_size)
    expansion_types = sorted({row.get("expansion_type") for row in rows if row.get("expansion_type")})
    summary = {
        "overall": aggregate(rows),
        "by_language": grouped(rows, "language"),
        "by_task_family": grouped(rows, "task_family"),
        "by_length_variant": grouped(rows, "length_variant"),
        "by_subtype": grouped(rows, "subtype"),
        "expansion_types": expansion_types,
        "semantic_model": args.semantic_model,
        "runtime_seconds_max_shard": max(runtimes, default=0.0),
        "metric_definitions": {
            "bleu4": "Sentence BLEU-4 with NLTK method3 smoothing; English word/punctuation tokens, Chinese Han-character/alphanumeric tokens.",
            "rouge": "ROUGE-1/2/L F1 on the same language-aware tokens.",
            "semantic": "Cosine similarity of normalized BGE-M3 CLS embeddings.",
            "format_type_match": "Exact match of coarse container (plain/JSON/XML/code) and list style (none/numbered/bulleted).",
            "format_structure_score": "Mean similarity over container, list style, multiline, headings, list count, line count, and paragraph count.",
            "validator_adherence": "Pass rate of the dataset-specified deterministic validator; not applicable to descriptive/style full-prompt-similarity rows.",
            "kl_macro": "Mean of per-example mean token KL.",
            "kl_micro": "Mean over every aligned teacher-forced output token across examples.",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_token_trajectories(rows, args.out_dir)
    (args.out_dir / "combined_results.json").write_text(json.dumps({"configs": configs, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["breakdown", "group", "rows", "method", "exact_match", "bleu4", "rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_cosine_bge_m3", "format_type_match", "format_structure_score", "validator_adherence", "validator_rows", "length_diff_tokens", "absolute_length_diff_tokens", "length_ratio_tokens", "terminated_with_eos", "length_diff_chars", "trajectory_kl_macro", "trajectory_kl_micro", "trajectory_token_count"])
        for breakdown, groups in {
            "overall": {"all": summary["overall"]},
            "language": summary["by_language"],
            "task_family": summary["by_task_family"],
            "length_variant": summary["by_length_variant"],
            "subtype": summary["by_subtype"],
        }.items():
            for group, values in groups.items():
                for method in METHODS:
                    score = values[method]
                    writer.writerow([breakdown, group, values["rows"], method, *[score[key] for key in ["exact_output_match", "bleu4", "rouge1_f1", "rouge2_f1", "rougeL_f1", "semantic_cosine_bge_m3", "format_type_match", "format_structure_score", "validator_adherence", "validator_rows", "length_diff_tokens", "absolute_length_diff_tokens", "length_ratio_tokens", "terminated_with_eos", "length_diff_chars", "trajectory_kl_macro", "trajectory_kl_micro", "trajectory_token_count"]]])

    examples = choose_examples(rows)
    config = configs[0]
    sections = [
        "# 非可验证任务 Full-prompt 一致性报告",
        "",
        f"{len(rows)} 个中英文非可验证 prompt-query 配对，覆盖描述、风格、格式和输出控制任务，以及 {', '.join(sorted({row['length_variant'] for row in rows}))} 三档 prompt。BLEU、ROUGE、语义、格式和长度均以 Full prompt 的 greedy 输出为 reference。KL 在 Full-prompt greedy teacher trajectory 上逐 token 对齐。",
        *([f"Prompt 长度扩展方式：{', '.join(expansion_types)}。"] if expansion_types else []),
        "",
        "## 实验设置",
        "",
        f"- 模型：`{config.get('model')}`",
        f"- 生成：greedy，`max_new_tokens={config.get('max_new_tokens')}`。",
        f"- QTraj：最后 {config.get('last_layers')} 层，prefix={config.get('prefix_count')}，rank={config.get('qtraj_rank')}，ridge={config.get('qtraj_ridge')}；lm_head tokens={config.get('qtraj_lm_teacher_tokens')}。",
        f"- Teacher-token：lm_head tokens={config.get('teacher_tokens')}，rank={config.get('teacher_rank')}，margin={config.get('margin')}，Top-k={config.get('top_k_delta')}。",
        f"- 语义模型：`{args.semantic_model}`，BGE-M3 CLS embedding cosine。",
        "- 所有 patch 都是一次求解、整段生成固定不变的可合并权重；KL 评估不进行逐 token 重拟合。",
        "",
        metric_table("总体", {"全部": summary["overall"]}),
        "",
        metric_table("按语言", summary["by_language"]),
        "",
        metric_table("按任务族", summary["by_task_family"]),
        "",
        metric_table("按 Prompt 长度", summary["by_length_variant"]),
        "",
        "## 逐 Token KL 文件",
        "",
        "- `token_kl.jsonl` / `token_kl.csv`：每个 pair、方法和 teacher token 的 KL、token log-prob、EOS 标记。",
        "- `token_kl_by_position.csv`：按语言、方法和 token 位置聚合的均值、中位数、最小值、最大值，可直接绘制折线图。",
        "- `token_kl_by_length_position.csv`：按 prompt 长度、方法和 token 位置聚合，用于分析长度敏感性。",
        "",
        "## 输入输出示例",
        "",
    ]
    for row in examples:
        sections.extend([f"### {row['pair_id']}", "", f"- Prompt：{row['prompt']}", f"- Query：{row['query']}"])
        for method in METHODS:
            score = row["comparison_metrics"][method]
            sections.append(
                f"- {LABELS[method]}：{row['outputs'][method]} "
                f"(BLEU={score['bleu4']:.3f}, R-L={score['rougeL_f1']:.3f}, semantic={score['semantic_cosine_bge_m3']:.3f}, KL={score['trajectory_kl_macro']:.5f})"
            )
        sections.append("")
    (args.out_dir / "REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
