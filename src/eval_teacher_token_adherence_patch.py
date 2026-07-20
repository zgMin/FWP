#!/usr/bin/env python3
"""Closed-form teacher-token adherence patch.

For each query, generate a teacher answer with the full prompt C + x.  Then run
query-only teacher forcing on x + y_<t and fit a mergeable lm_head update that
boosts the full-prompt teacher token y_t and suppresses the current query-only
top alternative.  No gradients or optimizer are used.
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
    logits_and_last_hidden,
    logits_for_text,
    logits_hidden_for_answer_positions,
    remove_lm_head_patch,
    remove_patch,
    seq_ratio,
    solve_ridge_factors,
    token_bleu,
)


PatchDict = Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]


def combine_lm_head(existing, existing_scale: float, anchor, anchor_scale: float):
    if existing is None:
        if anchor is None:
            return None
        a, b = anchor
        return a.contiguous(), (b.float() * anchor_scale).contiguous()
    if anchor is None:
        a, b = existing
        return a.contiguous(), (b.float() * existing_scale).contiguous()
    a0, b0 = existing
    a1, b1 = anchor
    return torch.cat([a0.float(), a1.float()], dim=0).contiguous(), torch.cat(
        [b0.float() * existing_scale, b1.float() * anchor_scale], dim=1
    ).contiguous()


def load_base_bundle(path: Path | None):
    if path is None:
        return {"patches": {}, "lm_head_patch": None, "lm_scale": 1.0}
    return torch.load(path, map_location="cpu", weights_only=False)


def strict_tag_score(text: str) -> float:
    return 1.0 if re.match(r"^<think>.+?</think>\s+<answer>.+?</answer>\s*$", text, re.S) else 0.0


@torch.no_grad()
def fit_teacher_token_anchor(
    model,
    tokenizer,
    cfg: PatchConfig,
    system_prompt: str,
    query: str,
    base_bundle: dict,
    teacher_tokens: int,
    rank: int,
    ridge: float,
    margin: float,
    negative_scale: float,
    max_boost: float,
    add_eos: bool,
    target_mode: str,
    delta_max_scale: float,
    top_k_delta: int,
    dtype: torch.dtype,
    input_mode: str = "system",
    teacher_text_override: str | None = None,
):
    remove_patch(model)
    remove_lm_head_patch(model)
    if input_mode == "concat_user":
        full_prompt = apply_chat(tokenizer, "", system_prompt + "\n" + query)
    elif input_mode == "system":
        full_prompt = apply_chat(tokenizer, system_prompt, query)
    else:
        raise ValueError(f"unknown input_mode: {input_mode}")
    query_prompt = apply_chat(tokenizer, "", query)
    teacher_text = teacher_text_override or generate_text(model, tokenizer, full_prompt, cfg)
    teacher_ids = tokenizer.encode(teacher_text, add_special_tokens=False)
    n_teacher = min(teacher_tokens, len(teacher_ids))
    if n_teacher == 0:
        return None, teacher_text, []

    full_logits, _ = logits_hidden_for_answer_positions(
        model,
        tokenizer,
        full_prompt,
        teacher_text,
        cfg.device,
        cfg.max_eval_prompt_tokens + cfg.max_new_tokens,
        n_teacher,
    )

    install_patch(model, base_bundle.get("patches", {}), cfg.device, dtype)
    install_lm_head_patch(
        model,
        base_bundle.get("lm_head_patch"),
        cfg.device,
        dtype,
        scale=float(base_bundle.get("lm_scale", 1.0)),
    )
    query_logits, query_hidden = logits_hidden_for_answer_positions(
        model,
        tokenizer,
        query_prompt,
        teacher_text,
        cfg.device,
        cfg.max_eval_prompt_tokens + cfg.max_new_tokens,
        n_teacher,
    )
    remove_patch(model)
    remove_lm_head_patch(model)

    hidden_cols = []
    target_cols = []
    rows = []
    vocab = model.config.vocab_size
    for pos in range(min(n_teacher, full_logits.shape[0], query_logits.shape[0], query_hidden.shape[0])):
        teacher_id = int(teacher_ids[pos])
        full_pos_logits = full_logits[pos].float()
        query_pos_logits = query_logits[pos].float()
        top_vals, top_ids = torch.topk(query_pos_logits, k=min(8, query_pos_logits.numel()))
        bad_id = int(top_ids[0].item())
        if bad_id == teacher_id and top_ids.numel() > 1:
            bad_id = int(top_ids[1].item())
        target = torch.zeros(vocab, dtype=torch.float32)
        teacher_logit = float(query_pos_logits[teacher_id].item())
        bad_logit = float(query_pos_logits[bad_id].item())
        full_teacher_logit = float(full_pos_logits[teacher_id].item())
        full_bad_logit = float(full_pos_logits[bad_id].item())
        if target_mode == "margin":
            boost = max(0.0, min(max_boost, margin - (teacher_logit - bad_logit)))
            if boost <= 0:
                continue
            target[teacher_id] += boost
            target[bad_id] -= negative_scale * boost
        elif target_mode == "two_delta":
            teacher_delta = full_teacher_logit - teacher_logit
            bad_delta = full_bad_logit - bad_logit
            if abs(teacher_delta) <= 1.0e-6 and abs(bad_delta) <= 1.0e-6:
                continue
            target[teacher_id] += teacher_delta
            target[bad_id] += bad_delta
            boost = None
        elif target_mode == "scaled_two_delta":
            teacher_delta = full_teacher_logit - teacher_logit
            bad_delta = full_bad_logit - bad_logit
            delta_gap = teacher_delta - bad_delta
            query_gap = teacher_logit - bad_logit
            scale = 1.0
            if delta_gap > 1.0e-6 and query_gap + delta_gap <= 0:
                scale = min(delta_max_scale, max(1.0, (-query_gap + 1.0e-3) / delta_gap))
            if abs(teacher_delta) <= 1.0e-6 and abs(bad_delta) <= 1.0e-6:
                continue
            target[teacher_id] += scale * teacher_delta
            target[bad_id] += scale * bad_delta
            boost = scale
        elif target_mode == "pair_gap":
            query_gap = teacher_logit - bad_logit
            full_gap = full_teacher_logit - full_bad_logit
            gap_delta = full_gap - query_gap
            if abs(gap_delta) <= 1.0e-6:
                continue
            target[teacher_id] += 0.5 * gap_delta
            target[bad_id] -= 0.5 * gap_delta
            boost = gap_delta
        elif target_mode == "topk_delta":
            k = min(top_k_delta, query_pos_logits.numel())
            full_top_ids = torch.topk(full_pos_logits, k=k).indices.tolist()
            query_top_ids = torch.topk(query_pos_logits, k=k).indices.tolist()
            ids = set(full_top_ids + query_top_ids + [teacher_id, bad_id])
            max_abs_delta = 0.0
            for tok_id in ids:
                delta = float(full_pos_logits[tok_id].item() - query_pos_logits[tok_id].item())
                target[int(tok_id)] += delta
                max_abs_delta = max(max_abs_delta, abs(delta))
            if max_abs_delta <= 1.0e-6:
                continue
            boost = None
        else:
            raise ValueError(f"unknown target_mode: {target_mode}")
        hidden_cols.append(query_hidden[pos].float())
        target_cols.append(target)
        rows.append(
            {
                "pos": pos,
                "teacher_id": teacher_id,
                "teacher_token": tokenizer.decode([teacher_id], skip_special_tokens=False),
                "bad_id": bad_id,
                "bad_token": tokenizer.decode([bad_id], skip_special_tokens=False),
                "boost": boost,
                "target_mode": target_mode,
                "teacher_logit": teacher_logit,
                "bad_logit": bad_logit,
                "full_teacher_logit": full_teacher_logit,
                "full_bad_logit": full_bad_logit,
                "teacher_delta": full_teacher_logit - teacher_logit,
                "bad_delta": full_bad_logit - bad_logit,
                "query_gap": teacher_logit - bad_logit,
                "full_gap": full_teacher_logit - full_bad_logit,
            }
        )

    if add_eos and teacher_ids:
        install_patch(model, base_bundle.get("patches", {}), cfg.device, dtype)
        install_lm_head_patch(
            model,
            base_bundle.get("lm_head_patch"),
            cfg.device,
            dtype,
            scale=float(base_bundle.get("lm_scale", 1.0)),
        )
        eos_logits, eos_hidden = logits_and_last_hidden(
            model,
            tokenizer,
            query_prompt + teacher_text,
            cfg.device,
            cfg.max_eval_prompt_tokens + cfg.max_new_tokens,
        )
        remove_patch(model)
        remove_lm_head_patch(model)
        logits = eos_logits[0].float()
        eos_id = int(tokenizer.eos_token_id)
        bad_id = int(torch.argmax(logits).item())
        if bad_id != eos_id:
            boost = max(0.0, min(max_boost, margin - (float(logits[eos_id]) - float(logits[bad_id]))))
            if boost > 0:
                target = torch.zeros(vocab, dtype=torch.float32)
                target[eos_id] += boost
                target[bad_id] -= negative_scale * boost
                # logits_and_last_hidden returns a single [hidden_size] vector.
                hidden_cols.append(eos_hidden.float())
                target_cols.append(target)
                rows.append(
                    {
                        "pos": "eos",
                        "teacher_id": eos_id,
                        "teacher_token": tokenizer.decode([eos_id], skip_special_tokens=False),
                        "bad_id": bad_id,
                        "bad_token": tokenizer.decode([bad_id], skip_special_tokens=False),
                        "boost": boost,
                        "teacher_logit": float(logits[eos_id]),
                        "bad_logit": float(logits[bad_id]),
                    }
                )

    if not hidden_cols:
        return None, teacher_text, rows

    h_cols = torch.stack(hidden_cols, dim=1).float()
    t_cols = torch.stack(target_cols, dim=1).float()
    b_full, a_full = solve_ridge_factors(h_cols, t_cols, ridge)
    a_lr, b_lr, energy, fro_norm = compress_factorized_delta(b_full, a_full, rank)
    rows.append({"summary": {"constraints": len(hidden_cols), "energy": energy, "fro_norm": fro_norm}})
    return (a_lr, b_lr), teacher_text, rows


def evaluate_query(model, tokenizer, cfg, system_prompt, query, bundle, dtype, format_mode):
    remove_patch(model)
    remove_lm_head_patch(model)
    full_prompt = apply_chat(tokenizer, system_prompt, query)
    query_prompt = apply_chat(tokenizer, "", query)
    baseline_logits = logits_for_text(model, tokenizer, full_prompt, cfg.device, cfg.max_eval_prompt_tokens)
    baseline_text = generate_text(model, tokenizer, full_prompt, cfg)
    install_patch(model, bundle.get("patches", {}), cfg.device, dtype)
    install_lm_head_patch(model, bundle.get("lm_head_patch"), cfg.device, dtype, scale=float(bundle.get("lm_scale", 1.0)))
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
    ap.add_argument("--base-patch-dir", default="")
    ap.add_argument("--base-patch-name", default="qtraj_query{idx}.pt")
    ap.add_argument("--teacher-tokens", type=int, default=32)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--negative-scale", type=float, default=0.5)
    ap.add_argument("--max-boost", type=float, default=12.0)
    ap.add_argument("--anchor-scale", type=float, default=1.0)
    ap.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--target-mode",
        choices=["margin", "two_delta", "scaled_two_delta", "pair_gap", "topk_delta"],
        default="margin",
    )
    ap.add_argument("--delta-max-scale", type=float, default=8.0)
    ap.add_argument("--top-k-delta", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-calib-tokens", type=int, default=160)
    ap.add_argument("--max-eval-prompt-tokens", type=int, default=224)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--format-mode", default="tags")
    ap.add_argument("--eval-limit", type=int, default=1)
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    cfg = PatchConfig(
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
    patch_dir = Path(args.patch_out_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = []
    for idx, query in enumerate(eval_queries):
        base_path = None
        if args.base_patch_dir:
            base_path = Path(args.base_patch_dir) / args.base_patch_name.format(idx=idx)
        base_bundle = load_base_bundle(base_path)
        print(f"Teacher-token adherence {idx+1}/{len(eval_queries)} base={base_path}: {query}", flush=True)
        anchor_patch, teacher_text, constraints = fit_teacher_token_anchor(
            model,
            tokenizer,
            cfg,
            system_prompt,
            query,
            base_bundle,
            args.teacher_tokens,
            args.rank,
            args.ridge,
            args.margin,
            args.negative_scale,
            args.max_boost,
            args.add_eos,
            args.target_mode,
            args.delta_max_scale,
            args.top_k_delta,
            dtype,
        )
        combined_lm = combine_lm_head(
            base_bundle.get("lm_head_patch"),
            float(base_bundle.get("lm_scale", 1.0)),
            anchor_patch,
            args.anchor_scale,
        )
        out_bundle = {
            "patches": base_bundle.get("patches", {}),
            "lm_head_patch": combined_lm,
            "lm_scale": 1.0,
            "mode": "teacher_token_adherence",
            "base_patch": str(base_path) if base_path else "",
            "query": query,
            "system_prompt": system_prompt,
            "teacher_text": teacher_text,
            "teacher_constraints": constraints,
            "note": "Closed-form lm_head teacher-token adherence patch; merge with lm_head.weight += B @ A.",
        }
        patch_path = patch_dir / f"teacher_token_query{idx}.pt"
        torch.save(out_bundle, patch_path)
        row = evaluate_query(model, tokenizer, cfg, system_prompt, query, out_bundle, dtype, args.format_mode)
        row.update({"patch_bundle": str(patch_path), "teacher_text": teacher_text, "constraints": constraints})
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ["query", "kl_next_token", "char_bleu", "char_f1", "format_score", "strict_tag_score"]
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
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
        "mode": "teacher_token_adherence",
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
