#!/usr/bin/env python3
"""Evaluate fixed QTraj + Auto-Margin bundles on either final dataset."""

from __future__ import annotations

import argparse
import collections
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_margin_teacher_anchor import fit_auto_margin_teacher_anchor, merge_bundle_into_model
from eval_descriptive_dataset import aligned_trajectory, generate_text_and_ids, teacher_forced_logits
from eval_query_dependent_patch import fit_query_patch
from eval_verifiable_expanded import FORMAT_PROTOCOLS
from run_thought_patch_qwen import (
    PatchConfig,
    apply_chat,
    configure_chat_terminator,
    install_lm_head_patch,
    install_patch,
    remove_lm_head_patch,
    remove_patch,
)


METHOD = "qtraj_teacher_auto_margin"


def load_rows(
    path: Path,
    task: str,
    shard_index: int,
    num_shards: int,
    limit: int | None,
    length_variants: list[str],
    balanced: bool,
    languages: list[str],
) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("language") in languages]
    if task == "descriptive":
        rows = [row for row in rows if row.get("length_variant") in length_variants]
        rows.sort(key=lambda row: row["pair_id"])
    else:
        if not all(row.get("gold_answers") for row in rows):
            raise ValueError("verifiable dataset contains rows without gold_answers")
    if balanced:
        buckets: dict[tuple, list[dict]] = collections.defaultdict(list)
        for row in rows:
            if task == "descriptive":
                key = (row.get("language"), row.get("task_family"), row.get("length_variant"))
            else:
                key = (row.get("language"), row.get("source"), row.get("partition"))
            buckets[key].append(row)
        rows = []
        while buckets:
            empty = []
            for key in sorted(buckets, key=lambda item: tuple(str(value) for value in item)):
                rows.append(buckets[key].pop(0))
                if not buckets[key]:
                    empty.append(key)
            for key in empty:
                del buckets[key]
    rows = [row for index, row in enumerate(rows) if index % num_shards == shard_index]
    return rows[:limit] if limit is not None else rows


def install_bundle(model, bundle: dict, cfg: PatchConfig, dtype: torch.dtype) -> None:
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


def evaluation_query(row: dict, task: str) -> str:
    if task == "descriptive":
        return row["query"]
    return row["query"] + "\n\n" + FORMAT_PROTOCOLS[row["language"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["verifiable", "descriptive"], required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="/root/zgm/models/Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--languages", nargs="+", choices=["en", "zh"], default=["en", "zh"])
    parser.add_argument(
        "--length-variants",
        nargs="+",
        choices=["short", "medium_redundant", "long_redundant"],
        default=["short", "medium_redundant", "long_redundant"],
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-calib-tokens", type=int)
    parser.add_argument("--max-eval-prompt-tokens", type=int)
    parser.add_argument("--prefix-count", type=int, default=4)
    parser.add_argument("--qtraj-rank", type=int, default=8)
    parser.add_argument("--qtraj-ridge", type=float, default=1.0e-3)
    parser.add_argument("--qtraj-lm-rank", type=int, default=64)
    parser.add_argument("--qtraj-lm-scale", type=float, default=0.6)
    parser.add_argument("--qtraj-lm-teacher-tokens", type=int, default=32)
    parser.add_argument(
        "--auto-margin-teacher-tokens",
        type=int,
        help="Limit Auto-Margin constraints to this many leading Full-trajectory tokens. "
        "The generated Full answer and QTraj calibration remain unchanged.",
    )
    parser.add_argument("--last-layers", type=int, default=8)
    parser.add_argument("--verify-merge", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["base_no_prompt", "full_prompt", METHOD],
        default=["base_no_prompt", "full_prompt", METHOD],
        help="Methods to save and evaluate. Auto-Margin still needs Full Prompt internally as its teacher.",
    )
    args = parser.parse_args()
    selected_methods = list(dict.fromkeys(args.methods))

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.auto_margin_teacher_tokens is not None and args.auto_margin_teacher_tokens <= 0:
        raise ValueError("auto-margin-teacher-tokens must be positive")
    if args.verify_merge and METHOD not in selected_methods:
        raise ValueError("--verify-merge requires qtraj_teacher_auto_margin in --methods")
    if args.max_new_tokens is None:
        args.max_new_tokens = 32 if args.task == "verifiable" else 256
    if args.max_calib_tokens is None:
        args.max_calib_tokens = 768 if args.task == "verifiable" else 1024
    if args.max_eval_prompt_tokens is None:
        args.max_eval_prompt_tokens = 768 if args.task == "verifiable" else 1024

    rows = load_rows(
        args.dataset,
        args.task,
        args.shard_index,
        args.num_shards,
        args.limit,
        args.length_variants,
        args.balanced,
        args.languages,
    )
    if args.verify_merge and len(rows) != 1:
        raise ValueError("--verify-merge is destructive and requires an evaluation set of exactly one row")
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
    selected_layers = list(range(max(0, len(model.model.layers) - args.last_layers), len(model.model.layers)))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    results = []
    if args.out.exists():
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        previous = payload.get("config", {})
        for key in (
            "task",
            "dataset",
            "shard_index",
            "num_shards",
            "prefix_count",
            "qtraj_lm_teacher_tokens",
            "auto_margin_teacher_tokens",
            "methods",
        ):
            if str(previous.get(key)) != str(getattr(args, key)):
                raise ValueError(f"cannot resume: config mismatch for {key}")
        results = payload.get("results", [])
    completed_ids = {row["pair_id"] for row in results}
    pending = [row for row in rows if row["pair_id"] not in completed_ids]
    started = time.time()

    for row in pending:
        item_started = time.time()
        prompt = row["prompt"]
        query = evaluation_query(row, args.task)
        query_prompt = apply_chat(tokenizer, "", query)
        full_prompt = apply_chat(tokenizer, "", prompt + "\n" + query)
        query_prefix_ids = tokenizer(query_prompt, add_special_tokens=False)["input_ids"]
        full_prefix_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]

        needs_auto = METHOD in selected_methods
        needs_base = "base_no_prompt" in selected_methods
        needs_full = "full_prompt" in selected_methods or needs_auto or (args.task == "descriptive" and needs_base)
        remove_patch(model)
        remove_lm_head_patch(model)
        full_text = ""
        full_visible_ids: list[int] = []
        full_eos = False
        if needs_full:
            full_text, full_visible_ids, full_eos = generate_text_and_ids(model, tokenizer, full_prompt, cfg)
        base_text = ""
        base_ids: list[int] = []
        base_eos = False
        if needs_base:
            base_text, base_ids, base_eos = generate_text_and_ids(model, tokenizer, query_prompt, cfg)
        teacher_ids = [*full_visible_ids, *([tokenizer.eos_token_id] if full_eos else [])]
        outputs = {}
        output_token_counts = {}
        terminated_with_eos = {}
        if needs_base:
            outputs["base_no_prompt"] = base_text
            output_token_counts["base_no_prompt"] = len(base_ids)
            terminated_with_eos["base_no_prompt"] = base_eos
        if "full_prompt" in selected_methods:
            outputs["full_prompt"] = full_text
            output_token_counts["full_prompt"] = len(full_visible_ids)
            terminated_with_eos["full_prompt"] = full_eos

        auto_bundle = None
        auto_meta = None
        fit_meta = None
        timing_seconds = {}
        if needs_auto:
            auto_margin_teacher_ids = teacher_ids[: args.auto_margin_teacher_tokens] if args.auto_margin_teacher_tokens is not None else teacher_ids
            qtraj_started = time.time()
            patches, lm_head_patch, _, fit_meta = fit_query_patch(
                model, tokenizer, cfg, prompt, query, selected_layers, args.qtraj_rank, args.qtraj_ridge,
                args.qtraj_lm_rank, args.qtraj_lm_scale, args.qtraj_lm_teacher_tokens, args.prefix_count,
                "qtraj", dtype, input_mode="concat_user", answer_text_override=full_text,
            )
            base_bundle = {"patches": patches, "lm_head_patch": lm_head_patch, "lm_scale": args.qtraj_lm_scale}
            timing_seconds["qtraj_fit"] = round(time.time() - qtraj_started, 3)
            auto_started = time.time()
            auto_bundle, auto_meta = fit_auto_margin_teacher_anchor(
                model, cfg, query_prefix_ids, auto_margin_teacher_ids, base_bundle, dtype
            )
            install_bundle(model, auto_bundle, cfg, dtype)
            auto_text, auto_ids, auto_eos = generate_text_and_ids(model, tokenizer, query_prompt, cfg)
            timing_seconds["auto_margin"] = round(time.time() - auto_started, 3)
            outputs[METHOD] = auto_text
            output_token_counts[METHOD] = len(auto_ids)
            terminated_with_eos[METHOD] = auto_eos

        result = {
            **row,
            "evaluation_query": query,
            "outputs": outputs,
            "output_token_counts": output_token_counts,
            "terminated_with_eos": terminated_with_eos,
            "timing_seconds": timing_seconds,
        }
        if needs_auto:
            result["teacher_output"] = fit_meta["answer_text"]
            result["auto_margin_target"] = {
                "teacher_tokens_total": len(teacher_ids),
                "teacher_tokens_used": len(auto_margin_teacher_ids),
                "teacher_token_limit": args.auto_margin_teacher_tokens,
            }
            result["auto_margin"] = auto_meta

        if args.task == "descriptive" and (needs_base or needs_auto):
            remove_patch(model)
            remove_lm_head_patch(model)
            full_logits = teacher_forced_logits(model, full_prefix_ids, teacher_ids, cfg.device)
            trajectory_kl_mean = {"full_prompt": 0.0} if "full_prompt" in selected_methods else {}
            token_kl = {"full_prompt": []} if "full_prompt" in selected_methods else {}
            if needs_base:
                base_logits = teacher_forced_logits(model, query_prefix_ids, teacher_ids, cfg.device)
                base_trajectory, base_kl = aligned_trajectory(full_logits, base_logits, teacher_ids, tokenizer, "base_no_prompt")
                trajectory_kl_mean["base_no_prompt"] = base_kl
                token_kl["base_no_prompt"] = base_trajectory
                del base_logits
            if needs_auto:
                install_bundle(model, auto_bundle, cfg, dtype)
                auto_logits = teacher_forced_logits(model, query_prefix_ids, teacher_ids, cfg.device)
                auto_trajectory, auto_kl = aligned_trajectory(full_logits, auto_logits, teacher_ids, tokenizer, METHOD)
                trajectory_kl_mean[METHOD] = auto_kl
                token_kl[METHOD] = auto_trajectory
                del auto_logits
            result["teacher_trajectory_token_count"] = len(teacher_ids)
            result["trajectory_kl_mean"] = trajectory_kl_mean
            result["token_kl"] = token_kl
            del full_logits
        timing_seconds["total"] = round(time.time() - item_started, 3)

        if args.verify_merge:
            merge_meta = merge_bundle_into_model(model, auto_bundle, cfg, dtype)
            merged_text, merged_ids, merged_eos = generate_text_and_ids(model, tokenizer, query_prompt, cfg)
            result["merge_verification"] = {
                **merge_meta,
                "output": merged_text,
                "output_token_count": len(merged_ids),
                "terminated_with_eos": merged_eos,
                "matches_injected_output": merged_text == auto_text,
                "matches_full_prompt_output": merged_text == full_text,
            }

        results.append(result)
        report = {
            "config": vars(args),
            "methods": selected_methods,
            "selected_layers": selected_layers,
            "result_count": len(results),
            "runtime_seconds": round(time.time() - started, 3),
            "results": results,
        }
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        progress = {
            "progress": f"{len(results)}/{len(rows)}",
            "pair_id": row["pair_id"],
            "methods": selected_methods,
            "seconds": result["timing_seconds"]["total"],
        }
        if needs_auto:
            progress.update(
                {
                    "exact": auto_text == full_text,
                    "converged": auto_meta["converged"],
                    "constraints": auto_meta["constraint_count"],
                    "active": auto_meta["active_constraints"],
                }
            )
        print(json.dumps(progress, ensure_ascii=False), flush=True)

        if needs_auto:
            del patches, lm_head_patch, base_bundle, auto_bundle
        remove_patch(model)
        remove_lm_head_patch(model)
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps({"out": str(args.out), "rows": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
