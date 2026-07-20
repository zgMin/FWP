#!/usr/bin/env python3
"""Query-dependent and short-trajectory thought patches.

This is a no-training evaluator for three variants:

* qdep: fit a separate mergeable patch Delta(C, x) for each evaluation query.
* qavg: fit per-prefix token patches Delta(C, x_t) for the first N teacher-forced
  generated prefixes and average those parameters into one query-specific patch.
* qtraj: fit one mergeable patch against all selected teacher-forced prefixes
  jointly, solving min_D sum_t ||D a_t(query+prefix_t) - target_t||^2.
"""

from __future__ import annotations

import argparse
import json
import math
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
    logits_for_text,
    logits_hidden_for_answer_positions,
    remove_lm_head_patch,
    remove_patch,
    seq_ratio,
    solve_ridge_factors,
    token_bleu,
)


PatchDict = Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]


def average_factorized_delta(factors: List[Tuple[torch.Tensor, torch.Tensor]], rank: int):
    """Average dense deltas B_i @ A_i without materializing the dense matrix."""
    if len(factors) == 1:
        b_full, a_full = factors[0]
        return compress_factorized_delta(b_full, a_full, rank)
    scale = 1.0 / len(factors)
    b_cat = torch.cat([b.float() * scale for b, _ in factors], dim=1)
    a_cat = torch.cat([a.float() for _, a in factors], dim=0)
    return compress_factorized_delta(b_cat, a_cat, rank)


def solve_avg_token_patch(a_cols: torch.Tensor, t_cols: torch.Tensor, ridge: float, rank: int):
    factors = []
    for i in range(a_cols.shape[1]):
        b_full, a_full = solve_ridge_factors(a_cols[:, i : i + 1], t_cols[:, i : i + 1], ridge)
        factors.append((b_full, a_full))
    return average_factorized_delta(factors, rank)


def solve_joint_token_patch(a_cols: torch.Tensor, t_cols: torch.Tensor, ridge: float, rank: int):
    """Fit one low-rank delta to all token/prefix constraints at once."""
    b_full, a_full = solve_ridge_factors(a_cols, t_cols, ridge)
    return compress_factorized_delta(b_full, a_full, rank)


def solve_token_patch(a_cols: torch.Tensor, t_cols: torch.Tensor, ridge: float, rank: int, mode: str):
    if mode == "qavg":
        return solve_avg_token_patch(a_cols, t_cols, ridge, rank)
    if mode in {"qdep", "qtraj"}:
        return solve_joint_token_patch(a_cols, t_cols, ridge, rank)
    raise ValueError(f"unknown patch mode: {mode}")


def make_prefix_pairs(
    tokenizer,
    model,
    cfg,
    system_prompt: str,
    query: str,
    prefix_count: int,
    max_answer_tokens: int,
    input_mode: str = "system",
    answer_text_override: str | None = None,
):
    if input_mode == "concat_user":
        full_prompt = apply_chat(tokenizer, "", system_prompt + "\n" + query)
    elif input_mode == "system":
        full_prompt = apply_chat(tokenizer, system_prompt, query)
    else:
        raise ValueError(f"unknown input_mode: {input_mode}")
    query_prompt = apply_chat(tokenizer, "", query)
    answer_text = answer_text_override if answer_text_override is not None else generate_text(model, tokenizer, full_prompt, cfg)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    n = max(1, min(prefix_count, len(answer_ids) + 1))
    pairs = []
    for t in range(n):
        prefix = tokenizer.decode(answer_ids[:t], skip_special_tokens=True)
        pairs.append((full_prompt + prefix, query_prompt + prefix))
    return full_prompt, query_prompt, answer_text, pairs


def fit_query_patch(
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
    lm_teacher_tokens: int,
    prefix_count: int,
    mode: str,
    dtype: torch.dtype,
    input_mode: str = "system",
    answer_text_override: str | None = None,
) -> Tuple[PatchDict, Tuple[torch.Tensor, torch.Tensor] | None, str, dict]:
    remove_patch(model)
    remove_lm_head_patch(model)
    full_prompt, query_prompt, answer_text, pairs = make_prefix_pairs(
        tokenizer,
        model,
        cfg,
        system_prompt,
        query,
        prefix_count,
        cfg.max_new_tokens,
        input_mode,
        answer_text_override,
    )

    full_acts_by_pair = []
    for full_text, _ in pairs:
        full_acts_by_pair.append(
            collect_layer_mlp_inputs(model, tokenizer, full_text, cfg.device, cfg.max_calib_tokens, selected_layers)
        )

    patches: PatchDict = {}
    patch_stats = []
    for layer_idx in selected_layers:
        remove_patch(model)
        install_patch(model, patches, cfg.device, dtype)
        records = []
        for _, query_text in pairs:
            records.append(
                collect_layer_mlp_inputs(model, tokenizer, query_text, cfg.device, cfg.max_calib_tokens, [layer_idx])[
                    layer_idx
                ]
            )

        layer = model.model.layers[layer_idx]
        patches[layer_idx] = {}

        # Attention output projection: Delta_o * o_in ~= attn_out_full - attn_out_query.
        attn_a_cols = torch.stack([r["attn_o_in"] for r in records], dim=1).float()
        attn_t_cols = torch.stack(
            [full_acts_by_pair[i][layer_idx]["attn_out"] - records[i]["attn_out"] for i in range(len(records))],
            dim=1,
        ).float()
        a_lr, b_lr, energy, fro_norm = solve_token_patch(attn_a_cols, attn_t_cols, ridge, rank, mode)
        patches[layer_idx]["attn_o_proj"] = (a_lr, b_lr)
        patch_stats.append({"layer": layer_idx, "projection": "attn_o_proj", "rank": int(a_lr.shape[0]), "energy": energy, "fro": fro_norm})

        # Recollect after attention patch before fitting MLP patches.
        remove_patch(model)
        install_patch(model, patches, cfg.device, dtype)
        records = []
        for _, query_text in pairs:
            records.append(
                collect_layer_mlp_inputs(model, tokenizer, query_text, cfg.device, cfg.max_calib_tokens, [layer_idx])[
                    layer_idx
                ]
            )

        a_cols = torch.stack([r["mlp_in"] for r in records], dim=1).float()
        delta_cols = torch.stack(
            [full_acts_by_pair[i][layer_idx]["mlp_in"] - records[i]["mlp_in"] for i in range(len(records))], dim=1
        ).float()
        for proj_name in ("gate_proj", "up_proj"):
            weight = getattr(layer.mlp, proj_name).weight.detach().float().cpu()
            targets = weight @ delta_cols
            a_lr, b_lr, energy, fro_norm = solve_token_patch(a_cols, targets, ridge, rank, mode)
            patches[layer_idx][proj_name] = (a_lr, b_lr)
            patch_stats.append({"layer": layer_idx, "projection": proj_name, "rank": int(a_lr.shape[0]), "energy": energy, "fro": fro_norm})

        remove_patch(model)
        install_patch(model, patches, cfg.device, dtype)
        down_inputs = []
        down_targets = []
        for i, (_, query_text) in enumerate(pairs):
            patched_acts = collect_layer_mlp_inputs(
                model, tokenizer, query_text, cfg.device, cfg.max_calib_tokens, [layer_idx]
            )[layer_idx]
            down_inputs.append(patched_acts["down_in"])
            down_targets.append(full_acts_by_pair[i][layer_idx]["mlp_out"] - patched_acts["mlp_out"])
        down_a_cols = torch.stack(down_inputs, dim=1).float()
        down_t_cols = torch.stack(down_targets, dim=1).float()
        a_lr, b_lr, energy, fro_norm = solve_token_patch(down_a_cols, down_t_cols, ridge, rank, mode)
        patches[layer_idx]["down_proj"] = (a_lr, b_lr)
        patch_stats.append({"layer": layer_idx, "projection": "down_proj", "rank": int(a_lr.shape[0]), "energy": energy, "fro": fro_norm})

    lm_head_patch = None
    if lm_teacher_tokens > 0:
        remove_patch(model)
        remove_lm_head_patch(model)
        full_answer = answer_text
        install_patch(model, patches, cfg.device, dtype)
        hidden_cols = []
        target_cols = []
        # Use the full teacher answer positions. These are already token-dependent
        # output-head constraints, complementary to the averaged internal patches.
        remove_patch(model)
        full_logits, _ = logits_hidden_for_answer_positions(
            model,
            tokenizer,
            full_prompt,
            full_answer,
            cfg.device,
            cfg.max_eval_prompt_tokens + cfg.max_new_tokens,
            lm_teacher_tokens,
        )
        install_patch(model, patches, cfg.device, dtype)
        query_logits, query_hidden = logits_hidden_for_answer_positions(
            model,
            tokenizer,
            query_prompt,
            full_answer,
            cfg.device,
            cfg.max_eval_prompt_tokens + cfg.max_new_tokens,
            lm_teacher_tokens,
        )
        n = min(full_logits.shape[0], query_logits.shape[0], query_hidden.shape[0])
        for pos in range(n):
            hidden_cols.append(query_hidden[pos])
            target_cols.append(full_logits[pos] - query_logits[pos])
        if hidden_cols:
            h_cols = torch.stack(hidden_cols, dim=1).float()
            t_cols = torch.stack(target_cols, dim=1).float()
            a_lr, b_lr, energy, fro_norm = solve_token_patch(h_cols, t_cols, ridge, lm_rank, mode)
            lm_head_patch = (a_lr, b_lr)
            patch_stats.append({"layer": "output", "projection": "lm_head", "rank": int(a_lr.shape[0]), "scale": lm_scale, "energy": energy, "fro": fro_norm})
    remove_patch(model)
    remove_lm_head_patch(model)
    meta = {"mode": mode, "prefix_count": len(pairs), "answer_text": answer_text, "patch_stats": patch_stats}
    return patches, lm_head_patch, full_prompt, meta


def strict_tag_score(text: str) -> float:
    return 1.0 if re.match(r"^<think>.+?</think>\s+<answer>.+?</answer>\s*$", text, re.S) else 0.0


def evaluate_query(model, tokenizer, cfg, system_prompt, query, patches, lm_head_patch, lm_scale, dtype, format_mode):
    remove_patch(model)
    remove_lm_head_patch(model)
    full_prompt = apply_chat(tokenizer, system_prompt, query)
    query_prompt = apply_chat(tokenizer, "", query)
    baseline_logits = logits_for_text(model, tokenizer, full_prompt, cfg.device, cfg.max_eval_prompt_tokens)
    baseline_text = generate_text(model, tokenizer, full_prompt, cfg)
    install_patch(model, patches, cfg.device, dtype)
    install_lm_head_patch(model, lm_head_patch, cfg.device, dtype, scale=lm_scale)
    patched_logits = logits_for_text(model, tokenizer, query_prompt, cfg.device, cfg.max_eval_prompt_tokens)
    patched_text = generate_text(model, tokenizer, query_prompt, cfg)
    remove_patch(model)
    remove_lm_head_patch(model)
    return {
        "query": query,
        "kl_next_token": kl_divergence_from_logits(baseline_logits, patched_logits),
        "baseline_length_tokens": len(tokenizer.encode(baseline_text, add_special_tokens=False)),
        "patched_length_tokens": len(tokenizer.encode(patched_text, add_special_tokens=False)),
        "length_diff_tokens": len(tokenizer.encode(patched_text, add_special_tokens=False))
        - len(tokenizer.encode(baseline_text, add_special_tokens=False)),
        "bleu": token_bleu(baseline_text, patched_text),
        "char_bleu": char_bleu(baseline_text, patched_text),
        "char_f1": char_f1(baseline_text, patched_text),
        "sequence_similarity": seq_ratio(baseline_text, patched_text),
        "format_score": format_score(patched_text, format_mode),
        "baseline_format_score": format_score(baseline_text, format_mode),
        "strict_tag_score": strict_tag_score(patched_text),
        "baseline_strict_tag_score": strict_tag_score(baseline_text),
        "baseline": baseline_text,
        "patched": patched_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    ap.add_argument("--system-prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-out-dir", required=True)
    ap.add_argument("--mode", choices=["qdep", "qavg", "qtraj"], default="qdep")
    ap.add_argument("--prefix-count", type=int, default=1)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1.0e-3)
    ap.add_argument("--lm-rank", type=int, default=64)
    ap.add_argument("--lm-scale", type=float, default=0.6)
    ap.add_argument("--lm-teacher-tokens", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-calib-tokens", type=int, default=160)
    ap.add_argument("--max-eval-prompt-tokens", type=int, default=224)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--format-mode", default="tags")
    ap.add_argument("--eval-limit", type=int, default=5)
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
    prefix_count = 1 if args.mode == "qdep" else max(1, args.prefix_count)
    for idx, query in enumerate(eval_queries):
        print(f"Fitting {args.mode} patch {idx+1}/{len(eval_queries)} prefixes={prefix_count}: {query}")
        patches, lm_head_patch, full_prompt, meta = fit_query_patch(
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
            args.lm_teacher_tokens,
            prefix_count,
            args.mode,
            dtype,
        )
        patch_path = patch_dir / f"{args.mode}_query{idx}.pt"
        torch.save(
            {
                "patches": patches,
                "lm_head_patch": lm_head_patch,
                "rank": args.rank,
                "lm_rank": args.lm_rank,
                "lm_scale": args.lm_scale,
                "ridge": args.ridge,
                "selected_layers": selected_layers,
                "mode": args.mode,
                "prefix_count": prefix_count,
                "query": query,
                "system_prompt": system_prompt,
                "note": "Query-dependent mergeable patch; merge with weight += B @ A.",
            },
            patch_path,
        )
        row = evaluate_query(
            model, tokenizer, cfg, system_prompt, query, patches, lm_head_patch, args.lm_scale, dtype, args.format_mode
        )
        row.update({"patch_bundle": str(patch_path), "fit_meta": meta})
        print(json.dumps({k: row[k] for k in ["query", "kl_next_token", "char_bleu", "char_f1", "format_score", "strict_tag_score"]}, ensure_ascii=False))
        results.append(row)

    keys = [
        "kl_next_token",
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
    averages = {k: float(sum(r[k] for r in results) / max(len(results), 1)) for k in keys}
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "prefix_count": prefix_count,
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
