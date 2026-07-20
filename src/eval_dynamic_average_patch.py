#!/usr/bin/env python3
"""Average dynamic per-token patches into one mergeable patch.

This script first follows the expensive dynamic-token procedure:
for each generated prefix, fit a fresh Delta(C, x + prefix) and use it to pick
the next token.  It then averages all temporary Delta_t patches into a single
low-rank mergeable adapter and evaluates that adapter with one static install.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_dynamic_token_patch import fit_prefix_patch
from run_thought_patch_qwen import (
    PatchConfig,
    apply_chat,
    build_default_sets,
    char_bleu,
    char_f1,
    compress_factorized_delta,
    format_score,
    generate_text,
    install_lm_head_patch,
    install_patch,
    kl_divergence_from_logits,
    logits_for_text,
    remove_lm_head_patch,
    remove_patch,
    seq_ratio,
    token_bleu,
)


PatchDict = Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]


def average_factorized_delta(factors: List[Tuple[torch.Tensor, torch.Tensor]], rank: int):
    if not factors:
        return None
    if len(factors) == 1:
        b_full, a_full = factors[0][1], factors[0][0]
        return compress_factorized_delta(b_full, a_full, rank)
    scale = 1.0 / len(factors)
    b_cat = torch.cat([b.float() * scale for a, b in factors], dim=1)
    a_cat = torch.cat([a.float() for a, b in factors], dim=0)
    return compress_factorized_delta(b_cat, a_cat, rank)


def average_patch_sequence(patch_steps: List[PatchDict], rank: int) -> PatchDict:
    by_key: Dict[Tuple[int, str], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for patches in patch_steps:
        for layer_idx, layer_patches in patches.items():
            for proj_name, pair in layer_patches.items():
                by_key.setdefault((int(layer_idx), proj_name), []).append(pair)
    averaged: PatchDict = {}
    for (layer_idx, proj_name), factors in by_key.items():
        out = average_factorized_delta(factors, rank)
        if out is None:
            continue
        a_lr, b_lr, _energy, _fro_norm = out
        averaged.setdefault(layer_idx, {})[proj_name] = (a_lr, b_lr)
    return averaged


def average_lm_head_sequence(lm_steps: List[Tuple[torch.Tensor, torch.Tensor] | None], rank: int):
    factors = [pair for pair in lm_steps if pair is not None]
    out = average_factorized_delta(factors, rank)
    if out is None:
        return None
    a_lr, b_lr, _energy, _fro_norm = out
    return a_lr, b_lr


@torch.no_grad()
def collect_dynamic_then_average(
    model,
    tokenizer,
    cfg: PatchConfig,
    system_prompt: str,
    query: str,
    selected_layers: List[int],
    rank: int,
    avg_rank: int,
    ridge: float,
    lm_rank: int,
    avg_lm_rank: int,
    lm_scale: float,
    dtype: torch.dtype,
    max_new_tokens: int,
):
    full_prompt = apply_chat(tokenizer, system_prompt, query)
    query_prompt = apply_chat(tokenizer, "", query)
    generated_ids: List[int] = []
    patch_steps: List[PatchDict] = []
    lm_steps: List[Tuple[torch.Tensor, torch.Tensor] | None] = []
    step_rows = []

    for step in range(max_new_tokens):
        prefix = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = full_prompt + prefix
        query_text = query_prompt + prefix
        patches, lm_head_patch, pre_lm_kl, _meta = fit_prefix_patch(
            model,
            tokenizer,
            cfg,
            full_text,
            query_text,
            selected_layers,
            rank,
            ridge,
            lm_rank,
            lm_scale,
            dtype,
        )
        patch_steps.append(patches)
        lm_steps.append(lm_head_patch)

        install_patch(model, patches, cfg.device, dtype)
        install_lm_head_patch(model, lm_head_patch, cfg.device, dtype, scale=lm_scale)
        full_logits = logits_for_text(model, tokenizer, full_text, cfg.device, cfg.max_eval_prompt_tokens)
        patched_logits = logits_for_text(model, tokenizer, query_text, cfg.device, cfg.max_eval_prompt_tokens)
        post_lm_kl = kl_divergence_from_logits(full_logits, patched_logits)
        next_id = int(torch.argmax(patched_logits[0]).item())
        token_text = tokenizer.decode([next_id], skip_special_tokens=False)
        remove_patch(model)
        remove_lm_head_patch(model)
        step_rows.append(
            {
                "step": step,
                "token_id": next_id,
                "token": token_text,
                "kl_before_lm_head": pre_lm_kl,
                "kl_after_lm_head": post_lm_kl,
            }
        )
        if next_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_id)

    averaged_patch = average_patch_sequence(patch_steps, avg_rank)
    averaged_lm_head = average_lm_head_sequence(lm_steps, avg_lm_rank)
    dynamic_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return averaged_patch, averaged_lm_head, dynamic_text, step_rows


def strict_tag_score(text: str) -> float:
    return 1.0 if re.match(r"^<think>.+?</think>\s+<answer>.+?</answer>\s*$", text, re.S) else 0.0


def evaluate_average_patch(
    model,
    tokenizer,
    cfg: PatchConfig,
    system_prompt: str,
    query: str,
    patches: PatchDict,
    lm_head_patch,
    lm_scale: float,
    dtype: torch.dtype,
    dynamic_text: str,
    steps: List[dict],
    format_mode: str,
):
    remove_patch(model)
    remove_lm_head_patch(model)
    full_prompt = apply_chat(tokenizer, system_prompt, query)
    query_prompt = apply_chat(tokenizer, "", query)
    baseline_logits = logits_for_text(model, tokenizer, full_prompt, cfg.device, cfg.max_eval_prompt_tokens)
    baseline_text = generate_text(model, tokenizer, full_prompt, cfg)
    install_patch(model, patches, cfg.device, dtype)
    install_lm_head_patch(model, lm_head_patch, cfg.device, dtype, scale=lm_scale)
    avg_logits = logits_for_text(model, tokenizer, query_prompt, cfg.device, cfg.max_eval_prompt_tokens)
    avg_text = generate_text(model, tokenizer, query_prompt, cfg)
    remove_patch(model)
    remove_lm_head_patch(model)

    return {
        "query": query,
        "kl_avg_patch_next_token": kl_divergence_from_logits(baseline_logits, avg_logits),
        "kl_dynamic_mean_after_lm_head": float(sum(s["kl_after_lm_head"] for s in steps) / max(len(steps), 1)),
        "baseline_length_tokens": len(tokenizer.encode(baseline_text, add_special_tokens=False)),
        "avg_length_tokens": len(tokenizer.encode(avg_text, add_special_tokens=False)),
        "dynamic_length_tokens": len(tokenizer.encode(dynamic_text, add_special_tokens=False)),
        "length_diff_tokens": len(tokenizer.encode(avg_text, add_special_tokens=False))
        - len(tokenizer.encode(baseline_text, add_special_tokens=False)),
        "bleu": token_bleu(baseline_text, avg_text),
        "char_bleu": char_bleu(baseline_text, avg_text),
        "char_f1": char_f1(baseline_text, avg_text),
        "sequence_similarity": seq_ratio(baseline_text, avg_text),
        "dynamic_char_bleu": char_bleu(baseline_text, dynamic_text),
        "dynamic_char_f1": char_f1(baseline_text, dynamic_text),
        "dynamic_sequence_similarity": seq_ratio(baseline_text, dynamic_text),
        "format_score": format_score(avg_text, format_mode),
        "dynamic_format_score": format_score(dynamic_text, format_mode),
        "baseline_format_score": format_score(baseline_text, format_mode),
        "strict_tag_score": strict_tag_score(avg_text),
        "dynamic_strict_tag_score": strict_tag_score(dynamic_text),
        "baseline_strict_tag_score": strict_tag_score(baseline_text),
        "baseline": baseline_text,
        "dynamic": dynamic_text,
        "averaged": avg_text,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--system-prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-out-dir", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--avg-rank", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1.0e-3)
    ap.add_argument("--lm-rank", type=int, default=64)
    ap.add_argument("--avg-lm-rank", type=int, default=64)
    ap.add_argument("--lm-scale", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-calib-tokens", type=int, default=160)
    ap.add_argument("--max-eval-prompt-tokens", type=int, default=224)
    ap.add_argument("--layer-start", type=int, default=28)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--format-mode", default="tags")
    ap.add_argument("--eval-limit", type=int, default=1)
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    cfg = PatchConfig(
        rank=args.rank,
        ridge=args.ridge,
        max_calib_tokens=args.max_calib_tokens,
        max_eval_prompt_tokens=args.max_eval_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    _, _, eval_queries = build_default_sets()
    eval_queries = eval_queries[: args.eval_limit]

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
    layer_start = max(0, min(args.layer_start, n_layers))
    layer_end = n_layers if args.layer_end is None else max(layer_start, min(args.layer_end, n_layers))
    selected_layers = list(range(layer_start, layer_end))
    if not selected_layers:
        raise ValueError(f"empty layer range: start={args.layer_start}, end={args.layer_end}, n_layers={n_layers}")
    patch_dir = Path(args.patch_out_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = []
    for idx, query in enumerate(eval_queries):
        print(f"Dynamic-average patch {idx+1}/{len(eval_queries)}: {query}", flush=True)
        patches, lm_head_patch, dynamic_text, steps = collect_dynamic_then_average(
            model,
            tokenizer,
            cfg,
            system_prompt,
            query,
            selected_layers,
            args.rank,
            args.avg_rank,
            args.ridge,
            args.lm_rank,
            args.avg_lm_rank,
            args.lm_scale,
            dtype,
            args.max_new_tokens,
        )
        patch_path = patch_dir / f"dynamic_avg_query{idx}.pt"
        torch.save(
            {
                "patches": patches,
                "lm_head_patch": lm_head_patch,
                "rank": args.avg_rank,
                "lm_rank": args.avg_lm_rank,
                "lm_scale": args.lm_scale,
                "ridge": args.ridge,
                "selected_layers": selected_layers,
                "mode": "dynamic_average",
                "query": query,
                "system_prompt": system_prompt,
                "dynamic_steps": len(steps),
                "note": "Average of per-token Delta(C, x+prefix) patches; merge with weight += B @ A.",
            },
            patch_path,
        )
        row = evaluate_average_patch(
            model, tokenizer, cfg, system_prompt, query, patches, lm_head_patch, args.lm_scale, dtype, dynamic_text, steps, args.format_mode
        )
        row["patch_bundle"] = str(patch_path)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "query",
                        "kl_avg_patch_next_token",
                        "char_bleu",
                        "char_f1",
                        "dynamic_char_f1",
                        "format_score",
                        "dynamic_format_score",
                    ]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        results.append(row)

    keys = [
        "kl_avg_patch_next_token",
        "kl_dynamic_mean_after_lm_head",
        "length_diff_tokens",
        "bleu",
        "char_bleu",
        "char_f1",
        "sequence_similarity",
        "dynamic_char_bleu",
        "dynamic_char_f1",
        "dynamic_sequence_similarity",
        "format_score",
        "dynamic_format_score",
        "baseline_format_score",
        "strict_tag_score",
        "dynamic_strict_tag_score",
        "baseline_strict_tag_score",
    ]
    averages = {k: float(sum(r[k] for r in results) / max(len(results), 1)) for k in keys}
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "dynamic_average",
        "system_prompt_file": args.system_prompt_file,
        "config": vars(args),
        "averages": averages,
        "results": results,
        "runtime_seconds": round(time.time() - t0, 3),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out, "averages": averages, "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
