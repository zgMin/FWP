#!/usr/bin/env python3
"""Evaluate answer-verifiable P2W-Bench rows with closed-form patches.

No gradients, optimizer, or parameter training are used. QTraj and teacher-token
updates are solved analytically and installed as temporary mergeable factors.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_query_dependent_patch import fit_query_patch
from eval_teacher_token_adherence_patch import combine_lm_head, fit_teacher_token_anchor
from run_thought_patch_qwen import (
    PatchConfig,
    apply_chat,
    generate_text,
    install_lm_head_patch,
    install_patch,
    remove_lm_head_patch,
    remove_patch,
)


try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


METHODS = ["base_no_prompt", "full_prompt", "qtraj_teacher_margin", "qtraj_topk_delta"]


def normalize_answer(text: str, language: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "zh" and _OPENCC is not None:
        text = _OPENCC.convert(text)
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", text)


def answer_is_correct(output: str, answers: list[str], language: str) -> bool:
    normalized_output = normalize_answer(output, language)
    return any(normalize_answer(answer, language) in normalized_output for answer in answers if answer.strip())


def make_full_prompt(tokenizer, prompt: str, query: str) -> str:
    return apply_chat(tokenizer, "", prompt + "\n" + query)


def make_query_prompt(tokenizer, query: str) -> str:
    return apply_chat(tokenizer, "", query)


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
    bundle = {
        "patches": base_bundle.get("patches", {}),
        "lm_head_patch": combined,
        "lm_scale": 1.0,
    }
    return bundle, constraints


def load_rows(path: Path, shard_index: int, num_shards: int, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(row.get("gold_answers") for row in rows):
        raise ValueError("input contains rows without gold_answers")
    query_ids = list(dict.fromkeys(row["query_id"] for row in rows))
    selected_ids = {query_id for index, query_id in enumerate(query_ids) if index % num_shards == shard_index}
    selected = [row for row in rows if row["query_id"] in selected_ids]
    return selected[:limit] if limit is not None else selected


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
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-calib-tokens", type=int, default=512)
    parser.add_argument("--max-eval-prompt-tokens", type=int, default=512)
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
    args = parser.parse_args()

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
    n_layers = len(model.model.layers)
    selected_layers = list(range(max(0, n_layers - args.last_layers), n_layers))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    base_cache: dict[str, str] = {}
    started = time.time()
    for index, row in enumerate(rows, start=1):
        item_started = time.time()
        query = row["query"]
        prompt = row["prompt"]
        query_prompt = make_query_prompt(tokenizer, query)
        full_prompt = make_full_prompt(tokenizer, prompt, query)
        if row["query_id"] not in base_cache:
            base_cache[row["query_id"]] = generate_text(model, tokenizer, query_prompt, cfg)
        base_text = base_cache[row["query_id"]]
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
        )
        base_bundle = {"patches": patches, "lm_head_patch": lm_head_patch, "lm_scale": args.qtraj_lm_scale}
        qtraj_seconds = time.time() - fit_started

        margin_started = time.time()
        margin_bundle, margin_constraints = fit_anchor_bundle(
            model, tokenizer, cfg, prompt, query, base_bundle, fit_meta["answer_text"], "margin", dtype, args
        )
        margin_text = generate_with_bundle(model, tokenizer, cfg, query_prompt, margin_bundle, dtype)
        margin_seconds = time.time() - margin_started

        topk_started = time.time()
        topk_bundle, topk_constraints = fit_anchor_bundle(
            model, tokenizer, cfg, prompt, query, base_bundle, fit_meta["answer_text"], "topk_delta", dtype, args
        )
        topk_text = generate_with_bundle(model, tokenizer, cfg, query_prompt, topk_bundle, dtype)
        topk_seconds = time.time() - topk_started

        outputs = {
            "base_no_prompt": base_text,
            "full_prompt": full_text,
            "qtraj_teacher_margin": margin_text,
            "qtraj_topk_delta": topk_text,
        }
        correctness = {
            method: answer_is_correct(text, row["gold_answers"], row["language"])
            for method, text in outputs.items()
        }
        result = {
            **{key: row[key] for key in [
                "pair_id", "query_id", "prompt_variant_id", "language", "task_family", "subtype",
                "length_variant", "source", "source_id", "prompt", "query", "gold_answers",
            ]},
            "outputs": outputs,
            "correct": correctness,
            "timing_seconds": {
                "qtraj_fit": round(qtraj_seconds, 3),
                "teacher_margin": round(margin_seconds, 3),
                "topk_delta": round(topk_seconds, 3),
                "total": round(time.time() - item_started, 3),
            },
            "constraint_counts": {
                "teacher_margin": len(margin_constraints),
                "topk_delta": len(topk_constraints),
            },
        }
        results.append(result)
        args.out.write_text(
            json.dumps({"config": vars(args), "results": results}, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(rows)}",
                    "pair_id": row["pair_id"],
                    "correct": correctness,
                    "seconds": result["timing_seconds"]["total"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del patches, lm_head_patch, base_bundle, margin_bundle, topk_bundle
        remove_patch(model)
        remove_lm_head_patch(model)
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "config": vars(args),
        "methods": METHODS,
        "normalization": {
            "unicode": "NFKC",
            "casefold": "lower",
            "remove_non_alphanumeric": True,
            "traditional_to_simplified": _OPENCC is not None,
            "criterion": "any normalized gold answer is a substring of normalized output",
        },
        "selected_layers": selected_layers,
        "result_count": len(results),
        "runtime_seconds": round(time.time() - started, 3),
        "results": results,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(results), "runtime_seconds": report["runtime_seconds"]}))


if __name__ == "__main__":
    main()
