#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


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

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


def normalize_english(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if char in string.punctuation else char for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_chinese(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text)


def normalize(text: str, language: str) -> str:
    return normalize_english(text) if language == "en" else normalize_chinese(text)


def units(text: str, language: str) -> list[str]:
    normalized = normalize(text, language)
    return normalized.split() if language == "en" else list(normalized)


def f1(prediction: str, answer: str, language: str) -> float:
    prediction_units = units(prediction, language)
    answer_units = units(answer, language)
    if not prediction_units or not answer_units:
        return float(prediction_units == answer_units)
    overlap = sum((Counter(prediction_units) & Counter(answer_units)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_units)
    recall = overlap / len(answer_units)
    return 2 * precision * recall / (precision + recall)


def strict_answer_format(output: str) -> bool:
    return re.fullmatch(r"\s*<answer>\s*[^<>\n]+?\s*</answer>\s*", output, re.IGNORECASE) is not None


def extract_answer(output: str, language: str) -> tuple[str, int]:
    matches = [match.strip() for match in re.findall(r"<answer>\s*(.*?)\s*</answer>", output, re.IGNORECASE | re.DOTALL) if match.strip()]
    if matches:
        return ((" and " if language == "en" else "、").join(matches), len(matches))
    fallback = output.strip().splitlines()[0].strip() if output.strip() else ""
    fallback = re.sub(r"^(?:answer|final answer)\s*[:：]\s*", "", fallback, flags=re.IGNORECASE)
    fallback = re.sub(r"^(?:答案|最终答案)\s*(?:是|为|[:：])\s*", "", fallback)
    return fallback, 0


def evaluate(output: str, answers: list[str], language: str) -> dict:
    prediction, tag_count = extract_answer(output, language)
    normalized_prediction = normalize(prediction, language)
    normalized_answers = [normalize(answer, language) for answer in answers]
    return {
        "strict_format": strict_answer_format(output),
        "tag_extracted": tag_count > 0,
        "tag_count": tag_count,
        "extracted_text": prediction,
        "em": any(normalized_prediction == answer for answer in normalized_answers),
        "f1": max((f1(prediction, answer, language) for answer in answers), default=0.0),
        "containment": any(answer and answer in normalized_prediction for answer in normalized_answers),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {"rows": len(rows)}
    for method in METHODS:
        metrics = [row["metrics"][method] for row in rows]
        result[method] = {
            "strict_format": sum(item["strict_format"] for item in metrics) / len(metrics) if metrics else 0.0,
            "tag_extraction": sum(item["tag_extracted"] for item in metrics) / len(metrics) if metrics else 0.0,
            "em": sum(item["em"] for item in metrics) / len(metrics) if metrics else 0.0,
            "f1": sum(item["f1"] for item in metrics) / len(metrics) if metrics else 0.0,
            "containment": sum(item["containment"] for item in metrics) / len(metrics) if metrics else 0.0,
        }
    return result


def grouped(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: aggregate(value) for key, value in sorted(groups.items())}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def table(title: str, groups: dict) -> str:
    lines = [
        f"## {title}",
        "",
        "| 分组 | N | 方法 | 严格格式 | 标签抽取 | EM | F1 | 包含率 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for group, values in groups.items():
        for method in METHODS:
            score = values[method]
            lines.append(
                f"| {group} | {values['rows']} | {LABELS[method]} | {pct(score['strict_format'])} | "
                f"{pct(score['tag_extraction'])} | {pct(score['em'])} | {pct(score['f1'])} | {pct(score['containment'])} |"
            )
    return "\n".join(lines)


def choose_examples(rows: list[dict]) -> list[dict]:
    selected = []
    for language in ("zh", "en"):
        candidates = [row for row in rows if row["language"] == language]
        candidates.sort(
            key=lambda row: (
                int(row["metrics"]["full_prompt"]["em"] and not row["metrics"]["qtraj_teacher_margin"]["em"]),
                int(row["metrics"]["full_prompt"]["em"] and not row["metrics"]["qtraj_topk_delta"]["em"]),
                len({row["metrics"][method]["em"] for method in METHODS}),
            ),
            reverse=True,
        )
        used_sources = set()
        for row in candidates:
            if row["source"] in used_sources:
                continue
            selected.append(row)
            used_sources.add(row["source"])
            if len(used_sources) == min(4, len({item["source"] for item in candidates})):
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int, default=96)
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
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, got {len(rows)}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate pair_id across shards")

    for row in rows:
        row["metrics"] = {
            method: evaluate(row["outputs"][method], row["gold_answers"], row["language"])
            for method in METHODS
        }
    summary = {
        "overall": aggregate(rows),
        "by_language": grouped(rows, "language"),
        "by_source": grouped(rows, "source"),
        "by_partition": grouped(rows, "dataset_partition"),
        "runtime_seconds_max_shard": max(runtimes, default=0.0),
        "metric_definition": {
            "em": "English uses SQuAD lowercase/punctuation/article/whitespace normalization; Chinese uses NFKC, traditional-to-simplified conversion when OpenCC is available, and punctuation/whitespace removal.",
            "f1": "English token F1 and Chinese normalized-character F1; maximum over gold aliases.",
            "containment": "Normalized extracted prediction contains any normalized gold alias.",
            "strict_format": "The complete output is exactly one non-empty <answer>...</answer> element.",
        },
    }
    examples = choose_examples(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "combined_results.json").write_text(json.dumps({"configs": configs, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    with (args.out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["breakdown", "group", "rows", "method", "strict_format", "tag_extraction", "em", "f1", "containment"])
        for breakdown, groups in {
            "overall": {"all": summary["overall"]},
            "language": summary["by_language"],
            "source": summary["by_source"],
            "partition": summary["by_partition"],
        }.items():
            for group, values in groups.items():
                for method in METHODS:
                    score = values[method]
                    writer.writerow([breakdown, group, values["rows"], method, score["strict_format"], score["tag_extraction"], score["em"], score["f1"], score["containment"]])

    config = configs[0]
    sections = [
        "# 扩展版答案可验证任务报告",
        "",
        "四种方法共享严格 `<answer>...</answer>` 输出协议；知识上下文只出现在 Full prompt，或由闭式解转换为可一次融合的固定权重。全部生成均为 greedy。",
        "",
        "## 设置",
        "",
        f"- 模型：`{config.get('model')}`",
        f"- 数据：{len(rows)} 条；中文 {sum(row['language']=='zh' for row in rows)}，英文 {sum(row['language']=='en' for row in rows)}。",
        f"- QTraj：最后 {config.get('last_layers')} 层，prefix={config.get('prefix_count')}，rank={config.get('qtraj_rank')}，ridge={config.get('qtraj_ridge')}；lm_head tokens={config.get('qtraj_lm_teacher_tokens')}。",
        f"- Teacher-token：lm_head tokens={config.get('teacher_tokens')}，rank={config.get('teacher_rank')}，ridge={config.get('teacher_ridge')}；margin={config.get('margin')}；Top-k={config.get('top_k_delta')}。",
        "- Base 与两种 query-only 路径都保留共享格式协议，但不接收知识段落。",
        "",
        table("总体", {"全部": summary["overall"]}),
        "",
        table("按语言", summary["by_language"]),
        "",
        table("按来源", summary["by_source"]),
        "",
        table("按数据分区", summary["by_partition"]),
        "",
        "## 输入输出示例",
        "",
    ]
    for row in examples:
        sections.extend([
            f"### {row['pair_id']} ({row['source']})",
            "",
            f"- Prompt：{row['prompt']}",
            f"- Query：{row['query']}",
            f"- Gold：`{row['gold_answers']}`",
        ])
        for method in METHODS:
            score = row["metrics"][method]
            sections.append(f"- {LABELS[method]}：`{row['outputs'][method]}` (EM={score['em']}, F1={score['f1']:.3f})")
        sections.append("")
    (args.out_dir / "REPORT.md").write_text("\n".join(sections), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
