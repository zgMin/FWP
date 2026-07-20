#!/usr/bin/env python3
"""Evaluate the expanded single-hop benchmark with four inference paths.

All paths receive the same answer-format protocol. Only the knowledge prompt is
present in Full prompt or converted into fixed, mergeable QTraj/teacher weights.
No gradients or optimizer steps are used.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_query_dependent_patch import fit_query_patch
from eval_teacher_token_adherence_patch import combine_lm_head, fit_teacher_token_anchor
from run_thought_patch_qwen import (
    PatchConfig,
    apply_chat,
    configure_chat_terminator,
    generate_text,
    install_lm_head_patch,
    install_patch,
    remove_lm_head_patch,
    remove_patch,
)


METHODS = ["base_no_prompt", "full_prompt", "qtraj_teacher_margin", "qtraj_topk_delta"]

FORMAT_PROTOCOLS = {
    "zh": (
        "输出协议：只输出一个 XML 元素，严格写成 <answer>答案文本</answer>。"
        "必须逐字包含开始标签 <answer> 和结束标签 </answer>，标签外不要输出任何内容。"
        "例如，问题“法国首都是什么？”只应输出 <answer>巴黎</answer>。"
    ),
    "en": (
        "Output protocol: return exactly one XML element in the form "
        "<answer>answer text</answer>. Include the literal opening tag <answer> and closing tag "
        "</answer>, with nothing outside them. For example, for 'What is the capital of France?' "
        "return only <answer>Paris</answer>."
    ),
}


def query_with_protocol(row: dict) -> str:
    return row["query"] + "\n\n" + FORMAT_PROTOCOLS[row["language"]]


@torch.no_grad()
def generate_with_bundle(model, tokenizer, cfg, query_prompt: str, bundle: dict, dtype: torch.dtype) -> str:
    remove_patch(model)
    remove_lm_head_patch(model)
    install_patch(model, bundle.get("patches", {}), cfg.device, dtype)
    install_lm_head_patch(
        model,
        bundle.get("lm_head_patch"),
        cfg.device,
        dtype,
        scale=float(bundle.get("lm_scale", 1.0)),
    )
    try:
        return generate_text(model, tokenizer, query_prompt, cfg)
    finally:
        remove_patch(model)
        remove_lm_head_patch(model)


def fit_anchor_bundle(
    model,
    tokenizer,
    cfg,
    prompt,
    query,
    base_bundle,
    teacher_text,
    target_mode,
    dtype,
    args,
):
    anchor, _, constraints = fit_teacher_token_anchor(
        model,
        tokenizer,
        cfg,
        prompt,
        query,
        base_bundle,
        args.teacher_tokens,
        args.teacher_rank,
        args.teacher_ridge,
        args.margin,
        args.negative_scale,
        args.max_boost,
        args.add_eos,
        target_mode,
        args.delta_max_scale,
        args.top_k_delta,
        dtype,
        input_mode="concat_user",
        teacher_text_override=teacher_text,
    )
    combined = combine_lm_head(
        base_bundle.get("lm_head_patch"),
        float(base_bundle.get("lm_scale", 1.0)),
        anchor,
        args.anchor_scale,
    )
    return {
        "patches": base_bundle.get("patches", {}),
        "lm_head_patch": combined,
        "lm_scale": 1.0,
    }, constraints


def load_rows(path: Path, shard_index: int, num_shards: int, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(row.get("gold_answers") for row in rows):
        raise ValueError("input contains rows without gold_answers")
    rows = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
    return rows[:limit] if limit is not None else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-calib-tokens", type=int, default=768)
    parser.add_argument("--max-eval-prompt-tokens", type=int, default=768)
    parser.add_argument("--prefix-count", type=int, default=4)
    parser.add_argument("--qtraj-rank", type=int, default=8)
    parser.add_argument("--qtraj-ridge", type=float, default=1.0e-3)
    parser.add_argument("--qtraj-lm-rank", type=int, default=64)
    parser.add_argument("--qtraj-lm-scale", type=float, default=0.6)
    parser.add_argument("--qtraj-lm-teacher-tokens", type=int, default=32)
    parser.add_argument("--last-layers", type=int, default=8)
    parser.add_argument("--teacher-tokens", type=int, default=32)
    parser.add_argument("--teacher-rank", type=int, default=128)
    parser.add_argument("--teacher-ridge", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=4.0)
    parser.add_argument("--negative-scale", type=float, default=0.5)
    parser.add_argument("--max-boost", type=float, default=12.0)
    parser.add_argument("--anchor-scale", type=float, default=1.0)
    parser.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--delta-max-scale", type=float, default=8.0)
    parser.add_argument("--top-k-delta", type=int, default=8)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["qtraj_teacher_margin", "qtraj_topk_delta"],
        default=["qtraj_teacher_margin", "qtraj_topk_delta"],
        help="Patch methods to evaluate. Base and Full Prompt remain shared references.",
    )
    args = parser.parse_args()
    selected_methods = list(dict.fromkeys(args.methods))

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    rows = load_rows(args.dataset, args.shard_index, args.num_shards, args.limit)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    cfg = PatchConfig(
        max_calib_tokens=args.max_calib_tokens,
        max_eval_prompt_tokens=args.max_eval_prompt_tokens,
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
    configure_chat_terminator(model, tokenizer)
    n_layers = len(model.model.layers)
    selected_layers = list(range(max(0, n_layers - args.last_layers), n_layers))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if args.out.exists():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        previous = payload.get("config", {})
        for key in ("dataset", "shard_index", "num_shards", "prefix_count", "qtraj_lm_teacher_tokens", "teacher_tokens", "methods"):
            if str(previous.get(key)) != str(getattr(args, key)):
                raise ValueError(f"cannot resume: config mismatch for {key}: {previous.get(key)!r} != {getattr(args, key)!r}")
        results = payload.get("results", [])
    completed_ids = {row["pair_id"] for row in results}
    if len(completed_ids) != len(results):
        raise ValueError("cannot resume: duplicate pair_id in existing output")
    total_rows = len(rows)
    rows = [row for row in rows if row["pair_id"] not in completed_ids]
    started = time.time()
    for row in rows:
        item_started = time.time()
        prompt = row["prompt"]
        query = query_with_protocol(row)
        query_prompt = apply_chat(tokenizer, "", query)
        full_prompt = apply_chat(tokenizer, "", prompt + "\n" + query)
        base_text = generate_text(model, tokenizer, query_prompt, cfg)
        full_text = generate_text(model, tokenizer, full_prompt, cfg)

        fit_started = time.time()
        patches, lm_head_patch, _, fit_meta = fit_query_patch(
            model,
            tokenizer,
            cfg,
            prompt,
            query,
            selected_layers,
            args.qtraj_rank,
            args.qtraj_ridge,
            args.qtraj_lm_rank,
            args.qtraj_lm_scale,
            args.qtraj_lm_teacher_tokens,
            args.prefix_count,
            "qtraj",
            dtype,
            input_mode="concat_user",
            answer_text_override=full_text,
        )
        base_bundle = {"patches": patches, "lm_head_patch": lm_head_patch, "lm_scale": args.qtraj_lm_scale}
        qtraj_seconds = time.time() - fit_started

        outputs = {"base_no_prompt": base_text, "full_prompt": full_text}
        timing_seconds = {"qtraj_fit": round(qtraj_seconds, 3)}
        constraint_counts = {}
        margin_bundle = topk_bundle = None
        if "qtraj_teacher_margin" in selected_methods:
            margin_started = time.time()
            margin_bundle, margin_constraints = fit_anchor_bundle(
                model, tokenizer, cfg, prompt, query, base_bundle, fit_meta["answer_text"], "margin", dtype, args
            )
            outputs["qtraj_teacher_margin"] = generate_with_bundle(model, tokenizer, cfg, query_prompt, margin_bundle, dtype)
            timing_seconds["teacher_margin"] = round(time.time() - margin_started, 3)
            constraint_counts["teacher_margin"] = len(margin_constraints)
        if "qtraj_topk_delta" in selected_methods:
            topk_started = time.time()
            topk_bundle, topk_constraints = fit_anchor_bundle(
                model, tokenizer, cfg, prompt, query, base_bundle, fit_meta["answer_text"], "topk_delta", dtype, args
            )
            outputs["qtraj_topk_delta"] = generate_with_bundle(model, tokenizer, cfg, query_prompt, topk_bundle, dtype)
            timing_seconds["topk_delta"] = round(time.time() - topk_started, 3)
            constraint_counts["topk_delta"] = len(topk_constraints)
        timing_seconds["total"] = round(time.time() - item_started, 3)

        result = {
            **row,
            "evaluation_query": query,
            "format_protocol": FORMAT_PROTOCOLS[row["language"]],
            "outputs": outputs,
            "teacher_output": fit_meta["answer_text"],
            "timing_seconds": timing_seconds,
            "constraint_counts": constraint_counts,
        }
        results.append(result)
        args.out.write_text(
            json.dumps({"config": vars(args), "results": results}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps({"progress": f"{len(results)}/{total_rows}", "pair_id": row["pair_id"], "seconds": result["timing_seconds"]["total"]}, ensure_ascii=False), flush=True)

        del patches, lm_head_patch, base_bundle, margin_bundle, topk_bundle
        remove_patch(model)
        remove_lm_head_patch(model)
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "config": vars(args),
        "methods": ["base_no_prompt", "full_prompt", *selected_methods],
        "selected_layers": selected_layers,
        "result_count": len(results),
        "runtime_seconds": round(time.time() - started, 3),
        "results": results,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(results), "runtime_seconds": report["runtime_seconds"]}))


if __name__ == "__main__":
    main()
