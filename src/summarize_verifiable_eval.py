#!/usr/bin/env python3
"""Merge sharded verifiable-task results and build accuracy reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


METHODS = ["base_no_prompt", "full_prompt", "qtraj_teacher_margin", "qtraj_topk_delta"]
METHOD_LABELS = {
    "base_no_prompt": "Base (query only)",
    "full_prompt": "Full prompt",
    "qtraj_teacher_margin": "QTraj + Teacher-token",
    "qtraj_topk_delta": "QTraj + Top-k",
}

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


def compact_normalize(text: str, language: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "zh" and _OPENCC is not None:
        text = _OPENCC.convert(text)
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text)


def english_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if char in string.punctuation else char for char in text)
    return [token for token in text.split() if token not in {"a", "an", "the"}]


def answer_units(text: str, language: str) -> list[str]:
    if language == "en":
        return english_tokens(text)
    return list(compact_normalize(text, language))


def sequence_f1(prediction: str, gold: str, language: str) -> float:
    pred_units = answer_units(prediction, language)
    gold_units = answer_units(gold, language)
    if not pred_units or not gold_units:
        return float(pred_units == gold_units)
    overlap = sum((Counter(pred_units) & Counter(gold_units)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_units)
    recall = overlap / len(gold_units)
    return 2.0 * precision * recall / (precision + recall)


def normalized_exact_match(prediction: str, answers: list[str], language: str) -> bool:
    normalized = compact_normalize(prediction, language)
    return any(normalized == compact_normalize(answer, language) for answer in answers)


def max_answer_f1(prediction: str, answers: list[str], language: str) -> float:
    return max((sequence_f1(prediction, answer, language) for answer in answers), default=0.0)


def first_sentence(text: str, language: str) -> str:
    line = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
    if language == "zh":
        match = re.search(r"[。！？]", line)
        return line[: match.end()].strip() if match else line

    abbreviations = {"mr", "mrs", "ms", "dr", "prof", "lt", "col", "st", "vs", "e.g", "i.e"}
    for match in re.finditer(r"[.!?](?:\s+|$)", line):
        if match.group(0).startswith("."):
            prefix = line[: match.start()].rstrip()
            previous = prefix.rsplit(None, 1)[-1].lower().rstrip(".") if prefix else ""
            if previous in abbreviations or (len(previous) == 1 and previous.isalpha()):
                continue
        return line[: match.start() + 1].strip()
    return line


def extract_answer_text(text: str, language: str) -> str:
    tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("answer"), (str, int, float)):
            return str(payload["answer"]).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    sentence = first_sentence(text, language)
    sentence = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", sentence)
    sentence = re.sub(r"^\s*(?:answer|final answer)\s*[:：]\s*", "", sentence, flags=re.IGNORECASE)
    sentence = re.sub(r"^\s*(?:答案|最终答案)\s*(?:是|为|[:：])\s*", "", sentence)
    return sentence.strip()


def evaluate_prediction(output: str, answers: list[str], language: str, containment: bool) -> dict:
    extracted = extract_answer_text(output, language)
    return {
        "containment": bool(containment),
        "whole_output_em": normalized_exact_match(output, answers, language),
        "whole_output_f1": max_answer_f1(output, answers, language),
        "extracted_text": extracted,
        "extracted_em": normalized_exact_match(extracted, answers, language),
        "extracted_f1": max_answer_f1(extracted, answers, language),
    }


def metric(rows: list[dict]) -> dict:
    result = {"row_count": len(rows), "query_count": len({row["query_id"] for row in rows})}
    for method in METHODS:
        correct = sum(bool(row["correct"][method]) for row in rows)
        evaluations = [row["evaluation_metrics"][method] for row in rows]
        result[method] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else 0.0,
            "whole_output_em": sum(item["whole_output_em"] for item in evaluations) / len(evaluations) if evaluations else 0.0,
            "whole_output_f1": sum(item["whole_output_f1"] for item in evaluations) / len(evaluations) if evaluations else 0.0,
            "extracted_em": sum(item["extracted_em"] for item in evaluations) / len(evaluations) if evaluations else 0.0,
            "extracted_f1": sum(item["extracted_f1"] for item in evaluations) / len(evaluations) if evaluations else 0.0,
        }
    return result


def grouped(rows: list[dict], fields: tuple[str, ...]) -> dict:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[field]) for field in fields)].append(row)
    return {" / ".join(key): metric(group_rows) for key, group_rows in sorted(groups.items())}


def all_lengths_metric(rows: list[dict]) -> dict:
    by_query: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    result = {"query_count": len(by_query)}
    for method in METHODS:
        correct = sum(all(row["correct"][method] for row in query_rows) for query_rows in by_query.values())
        result[method] = {
            "correct": correct,
            "total": len(by_query),
            "accuracy": correct / len(by_query) if by_query else 0.0,
        }
    return result


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def markdown_table(title: str, groups: dict) -> str:
    lines = [f"## {title}", "", "| 分组 | 样本数 | Query 数 | 基础模型 | Full prompt | QTraj + Teacher-token | QTraj + Top-k |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in groups.items():
        scores = [f"{values[method]['correct']}/{values[method]['total']} ({pct(values[method]['accuracy'])})" for method in METHODS]
        lines.append(f"| {name} | {values['row_count']} | {values['query_count']} | " + " | ".join(scores) + " |")
    return "\n".join(lines)


def metric_markdown_table(title: str, groups: dict) -> str:
    lines = [
        f"## {title}",
        "",
        "| 分组 | 方法 | 包含率 | 完整输出 EM | 完整输出 F1 | 抽取后 EM | 抽取后 F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group, values in groups.items():
        for method in METHODS:
            score = values[method]
            lines.append(
                f"| {group} | {METHOD_LABELS[method]} | {pct(score['accuracy'])} | "
                f"{pct(score['whole_output_em'])} | {pct(score['whole_output_f1'])} | "
                f"{pct(score['extracted_em'])} | {pct(score['extracted_f1'])} |"
            )
    return "\n".join(lines)


def choose_examples(rows: list[dict], per_language: int = 3) -> list[dict]:
    selected = []
    for language in ("zh", "en"):
        candidates = [row for row in rows if row["language"] == language and row["length_variant"] == "short"]
        candidates.sort(
            key=lambda row: (
                int(
                    row["correct"]["qtraj_teacher_margin"] != row["correct"]["full_prompt"]
                    or row["correct"]["qtraj_topk_delta"] != row["correct"]["full_prompt"]
                ),
                len(set(row["correct"].values())),
                int(row["correct"]["full_prompt"]),
                -sum(row["correct"].values()),
            ),
            reverse=True,
        )
        seen_subtypes = set()
        language_rows = []
        for row in candidates:
            if row["subtype"] not in seen_subtypes:
                language_rows.append(row)
                seen_subtypes.add(row["subtype"])
            if len(language_rows) == per_language:
                break
        for row in candidates:
            if len(language_rows) == per_language:
                break
            if row not in language_rows:
                language_rows.append(row)
        selected.extend(language_rows)
    return selected


def clipped(text: str, limit: int = 500) -> str:
    text = text.replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=72)
    args = parser.parse_args()

    rows = []
    configs = []
    normalization = None
    for path in args.inputs:
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(report["results"])
        configs.append(report.get("config", {}))
        normalization = report.get("normalization", normalization)
    rows.sort(key=lambda row: row["pair_id"])
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate pair_id found across shards")
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, got {len(rows)}")
    for row in rows:
        row["evaluation_metrics"] = {
            method: evaluate_prediction(
                row["outputs"][method],
                row["gold_answers"],
                row["language"],
                row["correct"][method],
            )
            for method in METHODS
        }

    summary = {
        "methods": METHOD_LABELS,
        "metric_definitions": {
            "answer_containment_accuracy": normalization,
            "whole_output_em": "compact-normalized complete output exactly equals any gold alias",
            "whole_output_f1": "max gold-alias SQuAD-style token F1 for English or normalized character F1 for Chinese",
            "extracted_em_f1": "same EM/F1 after deterministic <answer>/JSON-answer/first-sentence extraction",
            "traditional_to_simplified": _OPENCC is not None,
        },
        "overall": metric(rows),
        "by_language": grouped(rows, ("language",)),
        "by_subtype": grouped(rows, ("subtype",)),
        "by_language_and_subtype": grouped(rows, ("language", "subtype")),
        "by_source": grouped(rows, ("source",)),
        "by_length_variant": grouped(rows, ("length_variant",)),
        "all_three_lengths_correct": {
            "overall": all_lengths_metric(rows),
            **{language: all_lengths_metric([row for row in rows if row["language"] == language]) for language in ("zh", "en")},
        },
    }
    examples = choose_examples(rows)
    run_config = configs[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "combined_results.json").write_text(
        json.dumps({"configs": configs, "normalization": normalization, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "accuracy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.out_dir / "accuracy_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "breakdown", "group", "rows", "queries", "method", "correct", "total", "containment_accuracy",
            "whole_output_em", "whole_output_f1", "extracted_em", "extracted_f1",
        ])
        breakdowns = {
            "overall": {"all": summary["overall"]},
            "language": summary["by_language"],
            "subtype": summary["by_subtype"],
            "language_subtype": summary["by_language_and_subtype"],
            "source": summary["by_source"],
            "length_variant": summary["by_length_variant"],
        }
        for breakdown, groups in breakdowns.items():
            for group, values in groups.items():
                for method in METHODS:
                    score = values[method]
                    writer.writerow([
                        breakdown, group, values["row_count"], values["query_count"], method,
                        score["correct"], score["total"], score["accuracy"], score["whole_output_em"],
                        score["whole_output_f1"], score["extracted_em"], score["extracted_f1"],
                    ])

    sections = [
        "# 答案可验证任务正确率报告",
        "",
        "本报告同时计算答案包含率、完整输出 normalized EM/F1，以及规则抽取答案后的 EM/F1。英文 F1 使用 SQuAD 风格 token F1，中文使用繁简归一化后的字符 F1；多个 gold alias 取最高分。生成策略为 greedy。",
        "",
        "## 实验设置",
        "",
        f"- 模型：`{run_config.get('model')}`",
        f"- 数据：{len(rows)} 个 prompt-query 配对，{len({row['query_id'] for row in rows})} 个唯一 query；中英文各半，每个 query 对应 short、medium_redundant、long_redundant 三档 prompt。",
        f"- 生成：greedy，`max_new_tokens={run_config.get('max_new_tokens')}`；full prompt 使用单个 user message `{{prompt}}\\n{{query}}`。",
        f"- QTraj：最后 {run_config.get('last_layers')} 层，`prefix_count={run_config.get('prefix_count')}`、`rank={run_config.get('qtraj_rank')}`、`ridge={run_config.get('qtraj_ridge')}`；输出头 `rank={run_config.get('qtraj_lm_rank')}`、`scale={run_config.get('qtraj_lm_scale')}`、teacher tokens={run_config.get('qtraj_lm_teacher_tokens')}。",
        f"- Teacher-token：teacher tokens={run_config.get('teacher_tokens')}、`rank={run_config.get('teacher_rank')}`、`ridge={run_config.get('teacher_ridge')}`；margin 版 `margin={run_config.get('margin')}`，Top-k 版 `k={run_config.get('top_k_delta')}`。",
        "- QTraj 与 Teacher-token 参数均由闭式岭回归得到，不使用梯度、优化器或训练样本更新。",
        "- 为隔离逐样本评测，参数以 factor 形式临时注入；每个样本得到的固定 `Delta W = B @ A` 都可一次性执行 `W <- W + Delta W` 融合，生成时无需逐 token 重算。",
        "",
        "## 指标定义",
        "",
        "- 答案包含率：归一化后的完整输出包含任一 gold alias。",
        "- 完整输出 EM：归一化后的完整输出与任一 gold alias 完全相等。",
        "- 完整输出 F1：完整输出与 gold 的 token/字符重叠 F1，对多个 alias 取最大值。",
        "- 抽取后 EM/F1：优先读取 `<answer>` 或 JSON `answer`，否则确定性截取首个非空句子，再计算 EM/F1。抽取过程不读取 gold。",
        "",
        metric_markdown_table("多指标总体结果", {"全部": summary["overall"]}),
        "",
        metric_markdown_table("多指标按语言", summary["by_language"]),
        "",
        metric_markdown_table("多指标按知识子类", summary["by_subtype"]),
        "",
        markdown_table("总体", {"全部": summary["overall"]}),
        "",
        markdown_table("按语言", summary["by_language"]),
        "",
        markdown_table("按知识子类", summary["by_subtype"]),
        "",
        markdown_table("按语言与知识子类", summary["by_language_and_subtype"]),
        "",
        markdown_table("按数据来源", summary["by_source"]),
        "",
        markdown_table("按 Prompt 长度", summary["by_length_variant"]),
        "",
        "## 三档 Prompt 长度全部答对",
        "",
        "| 分组 | Query 数 | 基础模型 | Full prompt | QTraj + Teacher-token | QTraj + Top-k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, values in summary["all_three_lengths_correct"].items():
        scores = [f"{values[method]['correct']}/{values[method]['total']} ({pct(values[method]['accuracy'])})" for method in METHODS]
        sections.append(f"| {group} | {values['query_count']} | " + " | ".join(scores) + " |")
    sections.extend(["", "## 输入输出示例", ""])
    for row in examples:
        sections.extend([
            f"### {row['pair_id']}",
            "",
            f"- 语言/子类/来源：`{row['language']}` / `{row['subtype']}` / `{row['source']}`",
            f"- Gold answers：`{', '.join(row['gold_answers'])}`",
            f"- Prompt: {clipped(row['prompt'])}",
            f"- Query：{clipped(row['query'])}",
        ])
        for method in METHODS:
            evaluation = row["evaluation_metrics"][method]
            sections.append(f"- {METHOD_LABELS[method]}（{'正确' if row['correct'][method] else '错误'}）：{clipped(row['outputs'][method])}")
            sections.append(
                f"  指标：完整 EM={int(evaluation['whole_output_em'])}，完整 F1={evaluation['whole_output_f1']:.4f}，"
                f"抽取 EM={int(evaluation['extracted_em'])}，抽取 F1={evaluation['extracted_f1']:.4f}。"
            )
        sections.append("")
    (args.out_dir / "REPORT.md").write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
