#!/usr/bin/env python3
"""Evaluate formatted full-prompt answers on one prompt variant per query."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_thought_patch_qwen import PatchConfig, apply_chat, generate_text
from summarize_verifiable_eval import compact_normalize, max_answer_f1, normalized_exact_match


METHOD = "full_prompt"
FORMAT_INSTRUCTIONS = {
    "zh": (
        "你必须严格输出且只输出一个XML元素，格式为：<answer>答案文本</answer>。"
        "字面量开始标签<answer>和结束标签</answer>都必须原样保留；即使答案只有一个词，也不得省略标签。"
        "标签之外禁止输出任何文字。示例：问题是‘法国的首都是哪里？’时，必须输出<answer>巴黎</answer>。"
    ),
    "en": (
        "You must output exactly one XML element in this form: <answer>answer text</answer>. "
        "The literal opening tag <answer> and closing tag </answer> are mandatory, even for a one-word answer. "
        "Do not output any text outside the tags. Example: for 'What is the capital of France?', output <answer>Paris</answer>."
    ),
}


def extract_tag(text: str) -> str | None:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def strict_format(text: str) -> bool:
    return bool(re.fullmatch(r"\s*<answer>\s*[^<>]+?\s*</answer>\s*", text, re.IGNORECASE | re.DOTALL))


def answer_containment(prediction: str, answers: list[str], language: str) -> bool:
    normalized = compact_normalize(prediction, language)
    return any(compact_normalize(answer, language) in normalized for answer in answers)


def aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "count": count,
        "strict_format": sum(row["metrics"]["strict_format"] for row in rows) / count if count else 0.0,
        "tag_extraction": sum(row["metrics"]["tag_extracted"] for row in rows) / count if count else 0.0,
        "extracted_em": sum(row["metrics"]["extracted_em"] for row in rows) / count if count else 0.0,
        "extracted_f1": sum(row["metrics"]["extracted_f1"] for row in rows) / count if count else 0.0,
        "extracted_containment": sum(row["metrics"]["extracted_containment"] for row in rows) / count if count else 0.0,
    }


def grouped(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: aggregate(group_rows) for name, group_rows in sorted(groups.items())}


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def summary_table(title: str, groups: dict) -> str:
    lines = [
        f"## {title}",
        "",
        "| 分组 | 数量 | 严格格式率 | 标签抽取率 | 抽取后 EM | 抽取后 F1 | 抽取后包含率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in groups.items():
        lines.append(
            f"| {name} | {values['count']} | {pct(values['strict_format'])} | {pct(values['tag_extraction'])} | "
            f"{pct(values['extracted_em'])} | {pct(values['extracted_f1'])} | {pct(values['extracted_containment'])} |"
        )
    return "\n".join(lines)


def clipped(text: str, limit: int = 600) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def markdown_inline(text: str | None) -> str:
    return (text or "").replace("`", "\\`")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--length-variant", default="short")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    all_rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in all_rows if row["length_variant"] == args.length_variant]
    if len(rows) != len({row["query_id"] for row in rows}):
        raise ValueError("selected length variant does not contain exactly one row per query")
    if not all(row.get("gold_answers") for row in rows):
        raise ValueError("dataset includes rows without gold answers")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    cfg = PatchConfig(
        max_eval_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    results = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        instruction = FORMAT_INSTRUCTIONS[row["language"]]
        formatted_prompt = row["prompt"] + "\n" + instruction
        model_input = apply_chat(tokenizer, "", formatted_prompt + "\n" + row["query"])
        output = generate_text(model, tokenizer, model_input, cfg)
        extracted = extract_tag(output)
        prediction = extracted if extracted is not None else ""
        metrics = {
            "strict_format": strict_format(output),
            "tag_extracted": extracted is not None,
            "extracted_em": normalized_exact_match(prediction, row["gold_answers"], row["language"]),
            "extracted_f1": max_answer_f1(prediction, row["gold_answers"], row["language"]),
            "extracted_containment": answer_containment(prediction, row["gold_answers"], row["language"]),
        }
        results.append(
            {
                **{key: row[key] for key in [
                    "pair_id", "query_id", "language", "task_family", "subtype", "source", "source_id",
                    "gold_answers", "prompt", "query",
                ]},
                "format_instruction": instruction,
                "formatted_prompt": formatted_prompt,
                "output": output,
                "extracted_answer": extracted,
                "metrics": metrics,
            }
        )
        print(json.dumps({"progress": f"{index}/{len(rows)}", "pair_id": row["pair_id"], "output": output, "metrics": metrics}, ensure_ascii=False), flush=True)

    summary = {
        "overall": aggregate(results),
        "by_language": grouped(results, "language"),
        "by_subtype": grouped(results, "subtype"),
        "by_source": grouped(results, "source"),
    }
    payload = {
        "config": {**vars(args), "dataset": str(args.dataset), "out": str(args.out), "report": str(args.report)},
        "format_instructions": FORMAT_INSTRUCTIONS,
        "runtime_seconds": round(time.time() - started, 3),
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    sections = [
        "# Full Prompt 固定答案格式试跑",
        "",
        f"数据使用 `{args.length_variant}` 版本，每个 query 仅保留一条，共 {len(results)} 条。仅评估 full prompt，greedy 生成。",
        "",
        "格式约束作为 prompt 的一部分放在 query 之前，full input 为 `{prompt + format_instruction}\\n{query}`。",
        "",
        summary_table("总体", {"全部": summary["overall"]}),
        "",
        summary_table("按语言", summary["by_language"]),
        "",
        summary_table("按知识子类", summary["by_subtype"]),
        "",
        summary_table("按来源", summary["by_source"]),
        "",
        "## 全部输入输出",
        "",
    ]
    for row in results:
        metric = row["metrics"]
        sections.extend(
            [
                f"### {row['pair_id']}",
                "",
                f"- 语言/子类/来源：`{row['language']}` / `{row['subtype']}` / `{row['source']}`",
                f"- Gold：`{', '.join(row['gold_answers'])}`",
                f"- Prompt：{clipped(row['formatted_prompt'])}",
                f"- Query：{clipped(row['query'])}",
                f"- Output：`{markdown_inline(row['output'])}`",
                f"- Extracted：`{markdown_inline(row['extracted_answer'])}`",
                f"- 指标：严格格式={int(metric['strict_format'])}，EM={int(metric['extracted_em'])}，F1={metric['extracted_f1']:.4f}，包含={int(metric['extracted_containment'])}",
                "",
            ]
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(sections), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "report": str(args.report), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
