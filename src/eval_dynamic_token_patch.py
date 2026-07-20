#!/usr/bin/env python3
"""Dynamic exact-token thought patch evaluator.

This is intentionally expensive and is meant as an analysis baseline:
before every generated token, fit a fresh Delta(C, x + generated_prefix), use
that temporary low-rank patch for one next-token prediction, then discard it.

Unlike qtraj, this is not one static mergeable adapter for the whole answer.
Each per-step patch is still a mergeable B @ A update, but it is query-prefix
dependent and changes at every decoding step.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_thought_patch_qwen import (
    PatchConfig,
    apply_chat,
    build_default_sets,
    char_bleu,
    char_f1,
    collect_layer_mlp_inputs,
    compress_factorized_delta,
    format_score,
    generate_text,
    install_lm_head_patch,
    install_patch,
    kl_divergence_from_logits,
    logits_and_last_hidden,
    logits_for_text,
    remove_lm_head_patch,
    remove_patch,
    seq_ratio,
    solve_ridge_factors,
    token_bleu,
)


PatchDict = Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]


def solve_one(a_col: torch.Tensor, t_col: torch.Tensor, ridge: float, rank: int):
    b_full, a_full = solve_ridge_factors(a_col[:, None].float(), t_col[:, None].float(), ridge)
    return compress_factorized_delta(b_full, a_full, rank)


def fit_prefix_patch(
    model,
    tokenizer,
    cfg: PatchConfig,
    full_text: str,
    query_text: str,
    selected_layers: List[int],
    rank: int,
    ridge: float,
    lm_rank: int,
    lm_scale: float,
    dtype: torch.dtype,
) -> Tuple[PatchDict, Tuple[torch.Tensor, torch.Tensor] | None, float, dict]:
    remove_patch(model)
    remove_lm_head_patch(model)
    full_acts = collect_layer_mlp_inputs(model, tokenizer, full_text, cfg.device, cfg.max_calib_tokens, selected_layers)
    patches: PatchDict = {}
    stats = []

    for layer_idx in selected_layers:
        remove_patch(model)
        install_patch(model, patches, cfg.device, dtype)
        record = collect_layer_mlp_inputs(model, tokenizer, query_text, cfg.device, cfg.max_calib_tokens, [layer_idx])[
            layer_idx
        ]
        layer = model.model.layers[layer_idx]
        patches[layer_idx] = {}

        a_lr, b_lr, energy, fro_norm = solve_one(
            record["attn_o_in"],
            full_acts[layer_idx]["attn_out"] - record["attn_out"],
            ridge,
            rank,
        )
        patches[layer_idx]["attn_o_proj"] = (a_lr, b_lr)
        stats.append({"layer": layer_idx, "projection": "attn_o_proj", "energy": energy, "fro": fro_norm})

        remove_patch(model)
        install_patch(model, patches, cfg.device, dtype)
        record = collect_layer_mlp_inputs(model, tokenizer, query_text, cfg.device, cfg.max_calib_tokens, [layer_idx])[
            layer_idx
        ]
        delta_in = full_acts[layer_idx]["mlp_in"] - record["mlp_in"]
        for proj_name in ("gate_proj", "up_proj"):
            weight = getattr(layer.mlp, proj_name).weight.detach().float().cpu()
            target = weight @ delta_in.float()
            a_lr, b_lr, energy, fro_norm = solve_one(record["mlp_in"], target, ridge, rank)
            patches[layer_idx][proj_name] = (a_lr, b_lr)
            stats.append({"layer": layer_idx, "projection": proj_name, "energy": energy, "fro": fro_norm})

        remove_patch(model)
        install_patch(model, patches, cfg.device, dtype)
        patched_acts = collect_layer_mlp_inputs(
            model, tokenizer, query_text, cfg.device, cfg.max_calib_tokens, [layer_idx]
        )[layer_idx]
        a_lr, b_lr, energy, fro_norm = solve_one(
            patched_acts["down_in"],
            full_acts[layer_idx]["mlp_out"] - patched_acts["mlp_out"],
            ridge,
            rank,
        )
        patches[layer_idx]["down_proj"] = (a_lr, b_lr)
        stats.append({"layer": layer_idx, "projection": "down_proj", "energy": energy, "fro": fro_norm})

    lm_head_patch = None
    remove_patch(model)
    remove_lm_head_patch(model)
    full_logits = logits_for_text(model, tokenizer, full_text, cfg.device, cfg.max_eval_prompt_tokens)
    install_patch(model, patches, cfg.device, dtype)
    query_logits, query_hidden = logits_and_last_hidden(model, tokenizer, query_text, cfg.device, cfg.max_eval_prompt_tokens)
    step_kl = kl_divergence_from_logits(full_logits, query_logits)
    if lm_rank > 0 and lm_scale != 0:
        target = (full_logits[0] - query_logits[0]).float()
        a_lr, b_lr, energy, fro_norm = solve_one(query_hidden, target, ridge, lm_rank)
        lm_head_patch = (a_lr, b_lr)
        stats.append({"layer": "output", "projection": "lm_head", "scale": lm_scale, "energy": energy, "fro": fro_norm})
    remove_patch(model)
    remove_lm_head_patch(model)
    return patches, lm_head_patch, step_kl, {"patch_stats": stats}


@torch.no_grad()
def generate_dynamic(
    model,
    tokenizer,
    cfg: PatchConfig,
    system_prompt: str,
    query: str,
    selected_layers: List[int],
    rank: int,
    ridge: float,
    lm_rank: int,
    lm_scale: float,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> Tuple[str, List[dict]]:
    full_prompt = apply_chat(tokenizer, system_prompt, query)
    query_prompt = apply_chat(tokenizer, "", query)
    generated_ids: List[int] = []
    steps = []
    for step in range(max_new_tokens):
        prefix = tokenizer.decode(generated_ids, skip_special_tokens=True)
        full_text = full_prompt + prefix
        query_text = query_prompt + prefix
        patches, lm_head_patch, pre_lm_kl, meta = fit_prefix_patch(
            model, tokenizer, cfg, full_text, query_text, selected_layers, rank, ridge, lm_rank, lm_scale, dtype
        )
        install_patch(model, patches, cfg.device, dtype)
        install_lm_head_patch(model, lm_head_patch, cfg.device, dtype, scale=lm_scale)
        full_logits = logits_for_text(model, tokenizer, full_text, cfg.device, cfg.max_eval_prompt_tokens)
        patched_logits = logits_for_text(model, tokenizer, query_text, cfg.device, cfg.max_eval_prompt_tokens)
        post_lm_kl = kl_divergence_from_logits(full_logits, patched_logits)
        next_id = int(torch.argmax(patched_logits[0]).item())
        token_text = tokenizer.decode([next_id], skip_special_tokens=False)
        remove_patch(model)
        remove_lm_head_patch(model)
        steps.append(
            {
                "step": step,
                "token_id": next_id,
                "token": token_text,
                "kl_before_lm_head": pre_lm_kl,
                "kl_after_lm_head": post_lm_kl,
                "prefix_chars": len(prefix),
                "num_layer_patches": sum(len(v) for v in patches.values()),
                "meta": meta,
            }
        )
        if next_id == tokenizer.eos_token_id:
            break
        generated_ids.append(next_id)
    remove_patch(model)
    remove_lm_head_patch(model)
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), steps


def strict_tag_score(text: str) -> float:
    return 1.0 if re.match(r"^<think>.+?</think>\s+<answer>.+?</answer>\s*$", text, re.S) else 0.0


def evaluate_query(model, tokenizer, cfg, system_prompt, query, dynamic_text, steps, format_mode):
    full_prompt = apply_chat(tokenizer, system_prompt, query)
    query_prompt = apply_chat(tokenizer, "", query)
    baseline_logits = logits_for_text(model, tokenizer, full_prompt, cfg.device, cfg.max_eval_prompt_tokens)
    query_logits = logits_for_text(model, tokenizer, query_prompt, cfg.device, cfg.max_eval_prompt_tokens)
    baseline_text = generate_text(model, tokenizer, full_prompt, cfg)
    return {
        "query": query,
        "kl_query_only_next_token": kl_divergence_from_logits(baseline_logits, query_logits),
        "kl_dynamic_mean_after_lm_head": float(sum(s["kl_after_lm_head"] for s in steps) / max(len(steps), 1)),
        "kl_dynamic_first_after_lm_head": float(steps[0]["kl_after_lm_head"]) if steps else None,
        "baseline_length_tokens": len(tokenizer.encode(baseline_text, add_special_tokens=False)),
        "dynamic_length_tokens": len(tokenizer.encode(dynamic_text, add_special_tokens=False)),
        "length_diff_tokens": len(tokenizer.encode(dynamic_text, add_special_tokens=False))
        - len(tokenizer.encode(baseline_text, add_special_tokens=False)),
        "bleu": token_bleu(baseline_text, dynamic_text),
        "char_bleu": char_bleu(baseline_text, dynamic_text),
        "char_f1": char_f1(baseline_text, dynamic_text),
        "sequence_similarity": seq_ratio(baseline_text, dynamic_text),
        "format_score": format_score(dynamic_text, format_mode),
        "baseline_format_score": format_score(baseline_text, format_mode),
        "strict_tag_score": strict_tag_score(dynamic_text),
        "baseline_strict_tag_score": strict_tag_score(baseline_text),
        "baseline": baseline_text,
        "dynamic": dynamic_text,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--system-prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1.0e-3)
    ap.add_argument("--lm-rank", type=int, default=64)
    ap.add_argument("--lm-scale", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-calib-tokens", type=int, default=160)
    ap.add_argument("--max-eval-prompt-tokens", type=int, default=224)
    ap.add_argument("--layer-start", type=int, default=28)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--format-mode", default="tags")
    ap.add_argument("--eval-limit", type=int, default=3)
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

    t0 = time.time()
    results = []
    for idx, query in enumerate(eval_queries):
        print(f"Dynamic token patch {idx+1}/{len(eval_queries)}: {query}", flush=True)
        dynamic_text, steps = generate_dynamic(
            model,
            tokenizer,
            cfg,
            system_prompt,
            query,
            selected_layers,
            args.rank,
            args.ridge,
            args.lm_rank,
            args.lm_scale,
            dtype,
            args.max_new_tokens,
        )
        row = evaluate_query(model, tokenizer, cfg, system_prompt, query, dynamic_text, steps, args.format_mode)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "query",
                        "kl_dynamic_mean_after_lm_head",
                        "char_bleu",
                        "char_f1",
                        "format_score",
                        "strict_tag_score",
                    ]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        results.append(row)

    keys = [
        "kl_query_only_next_token",
        "kl_dynamic_mean_after_lm_head",
        "kl_dynamic_first_after_lm_head",
        "length_diff_tokens",
        "bleu",
        "char_bleu",
        "char_f1",
        "sequence_similarity",
        "format_score",
        "baseline_format_score",
        "strict_tag_score",
        "baseline_strict_tag_score",
    ]
    averages = {k: float(sum(r[k] for r in results if r[k] is not None) / max(sum(r[k] is not None for r in results), 1)) for k in keys}
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "dynamic_token_patch",
        "system_prompt_file": args.system_prompt_file,
        "config": vars(args),
        "averages": averages,
        "results": results,
        "runtime_seconds": round(time.time() - t0, 3),
        "note": "Fits a fresh Delta(C, x+prefix) before each generated token; analysis baseline, not one static mergeable adapter.",
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out, "averages": averages, "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
