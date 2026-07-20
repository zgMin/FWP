#!/usr/bin/env python3
"""Evaluate descriptive prompt-to-weight behavior with aligned token KL.

KL is teacher-forced on the Full-prompt greedy trajectory y_1..y_T:
  KL(P_full(. | C,x,y_<t) || P_method(. | x,y_<t)).
This keeps every method aligned at every output token even when free-running
generations diverge. QTraj and teacher-token patches are closed-form and fixed
for the whole generation; no gradients or per-token weight updates are used.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_query_dependent_patch import fit_query_patch
from eval_teacher_token_adherence_patch import combine_lm_head, fit_teacher_token_anchor
from run_thought_patch_qwen import (
    PatchConfig,
    apply_chat,
    configure_chat_terminator,
    install_lm_head_patch,
    install_patch,
    make_inputs,
    remove_lm_head_patch,
    remove_patch,
)


METHODS = ["base_no_prompt", "full_prompt", "qtraj_teacher_margin", "qtraj_topk_delta"]


@torch.no_grad()
def generate_text_and_ids(model, tokenizer, text: str, cfg: PatchConfig) -> tuple[str, list[int], bool]:
    inputs = make_inputs(tokenizer, text, cfg.device, max_tokens=cfg.max_eval_prompt_tokens)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0, inputs["input_ids"].shape[1] :].tolist()
    terminated = bool(generated and generated[-1] == tokenizer.eos_token_id)
    visible = generated[:-1] if terminated else generated
    return tokenizer.decode(visible, skip_special_tokens=True).strip(), visible, terminated


@torch.no_grad()
def teacher_forced_logits(model, prefix_ids: list[int], teacher_ids: list[int], device: str) -> torch.Tensor:
    if not teacher_ids:
        return torch.empty(0, model.config.vocab_size, device=device, dtype=torch.float32)
    input_ids = torch.tensor([prefix_ids + teacher_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    start = len(prefix_ids) - 1
    return output.logits[0, start : start + len(teacher_ids), :].float()


@torch.no_grad()
def aligned_trajectory(
    full_logits: torch.Tensor,
    method_logits: torch.Tensor,
    teacher_ids: list[int],
    tokenizer,
    method: str,
    chunk_size: int = 8,
) -> tuple[list[dict], float]:
    if full_logits.shape != method_logits.shape or full_logits.shape[0] != len(teacher_ids):
        raise ValueError("aligned trajectory shapes do not match")
    rows: list[dict] = []
    for start in range(0, len(teacher_ids), chunk_size):
        end = min(start + chunk_size, len(teacher_ids))
        p_logits = full_logits[start:end]
        q_logits = method_logits[start:end]
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).clamp_min(0.0)
        token_ids = torch.tensor(teacher_ids[start:end], device=full_logits.device, dtype=torch.long)
        full_lp = log_p.gather(1, token_ids[:, None]).squeeze(1)
        method_lp = log_q.gather(1, token_ids[:, None]).squeeze(1)
        for offset, token_id in enumerate(teacher_ids[start:end]):
            rows.append(
                {
                    "method": method,
                    "step": start + offset,
                    "teacher_token_id": token_id,
                    "teacher_token": tokenizer.decode([token_id], skip_special_tokens=False),
                    "is_eos": token_id == tokenizer.eos_token_id,
                    "kl_full_to_method": float(kl[offset].item()),
                    "full_teacher_token_logprob": float(full_lp[offset].item()),
                    "method_teacher_token_logprob": float(method_lp[offset].item()),
                }
            )
    return rows, float(sum(row["kl_full_to_method"] for row in rows) / max(len(rows), 1))


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


def fit_anchor_bundle(model, tokenizer, cfg, prompt, query, base_bundle, teacher_text, mode, dtype, args):
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
        mode,
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
    return {"patches": base_bundle.get("patches", {}), "lm_head_patch": combined, "lm_scale": 1.0}, constraints


def load_rows(
    path: Path,
    shard_index: int,
    num_shards: int,
    limit: int | None,
    length_variants: list[str],
) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("length_variant") in length_variants]
    rows.sort(key=lambda row: row["pair_id"])
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
    parser.add_argument(
        "--length-variants",
        nargs="+",
        choices=["short", "medium_redundant", "long_redundant"],
        default=["short"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-calib-tokens", type=int, default=1024)
    parser.add_argument("--max-eval-prompt-tokens", type=int, default=1024)
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

    rows = load_rows(
        args.dataset,
        args.shard_index,
        args.num_shards,
        args.limit,
        args.length_variants,
    )
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
        for key in ("dataset", "shard_index", "num_shards", "prefix_count", "qtraj_lm_teacher_tokens", "teacher_tokens"):
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
        query = row["query"]
        full_prompt = apply_chat(tokenizer, "", prompt + "\n" + query)
        query_prompt = apply_chat(tokenizer, "", query)
        full_prefix_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        query_prefix_ids = tokenizer(query_prompt, add_special_tokens=False)["input_ids"]

        remove_patch(model)
        remove_lm_head_patch(model)
        full_text, full_visible_ids, full_eos = generate_text_and_ids(model, tokenizer, full_prompt, cfg)
        base_text, base_ids, base_eos = generate_text_and_ids(model, tokenizer, query_prompt, cfg)
        teacher_ids = [*full_visible_ids, *([tokenizer.eos_token_id] if full_eos else [])]
        full_logits = teacher_forced_logits(model, full_prefix_ids, teacher_ids, cfg.device)
        base_logits = teacher_forced_logits(model, query_prefix_ids, teacher_ids, cfg.device)
        base_trajectory, base_kl = aligned_trajectory(full_logits, base_logits, teacher_ids, tokenizer, "base_no_prompt")
        del base_logits

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

        margin_started = time.time()
        margin_bundle, margin_constraints = fit_anchor_bundle(
            model, tokenizer, cfg, prompt, query, base_bundle, full_text, "margin", dtype, args
        )
        install_bundle(model, margin_bundle, cfg, dtype)
        margin_text, margin_ids, margin_eos = generate_text_and_ids(model, tokenizer, query_prompt, cfg)
        margin_logits = teacher_forced_logits(model, query_prefix_ids, teacher_ids, cfg.device)
        margin_trajectory, margin_kl = aligned_trajectory(
            full_logits, margin_logits, teacher_ids, tokenizer, "qtraj_teacher_margin"
        )
        margin_seconds = time.time() - margin_started
        del margin_logits

        topk_started = time.time()
        topk_bundle, topk_constraints = fit_anchor_bundle(
            model, tokenizer, cfg, prompt, query, base_bundle, full_text, "topk_delta", dtype, args
        )
        install_bundle(model, topk_bundle, cfg, dtype)
        topk_text, topk_ids, topk_eos = generate_text_and_ids(model, tokenizer, query_prompt, cfg)
        topk_logits = teacher_forced_logits(model, query_prefix_ids, teacher_ids, cfg.device)
        topk_trajectory, topk_kl = aligned_trajectory(
            full_logits, topk_logits, teacher_ids, tokenizer, "qtraj_topk_delta"
        )
        topk_seconds = time.time() - topk_started
        del topk_logits, full_logits

        result = {
            **row,
            "outputs": {
                "base_no_prompt": base_text,
                "full_prompt": full_text,
                "qtraj_teacher_margin": margin_text,
                "qtraj_topk_delta": topk_text,
            },
            "output_token_counts": {
                "base_no_prompt": len(base_ids),
                "full_prompt": len(full_visible_ids),
                "qtraj_teacher_margin": len(margin_ids),
                "qtraj_topk_delta": len(topk_ids),
            },
            "terminated_with_eos": {
                "base_no_prompt": base_eos,
                "full_prompt": full_eos,
                "qtraj_teacher_margin": margin_eos,
                "qtraj_topk_delta": topk_eos,
            },
            "teacher_trajectory_token_count": len(teacher_ids),
            "trajectory_kl_mean": {
                "base_no_prompt": base_kl,
                "full_prompt": 0.0,
                "qtraj_teacher_margin": margin_kl,
                "qtraj_topk_delta": topk_kl,
            },
            "token_kl": {
                "base_no_prompt": base_trajectory,
                "full_prompt": [],
                "qtraj_teacher_margin": margin_trajectory,
                "qtraj_topk_delta": topk_trajectory,
            },
            "teacher_matches_separate_full_generation": fit_meta["answer_text"] == full_text,
            "constraint_counts": {
                "teacher_margin": len(margin_constraints),
                "topk_delta": len(topk_constraints),
            },
            "timing_seconds": {
                "qtraj_fit": round(qtraj_seconds, 3),
                "teacher_margin": round(margin_seconds, 3),
                "topk_delta": round(topk_seconds, 3),
                "total": round(time.time() - item_started, 3),
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
                    "progress": f"{len(results)}/{total_rows}",
                    "pair_id": row["pair_id"],
                    "tokens": len(teacher_ids),
                    "kl": result["trajectory_kl_mean"],
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
        "kl_definition": "mean_t KL(P_full(.|C,x,y_<t) || P_method(.|x,y_<t)) on Full-prompt greedy tokens, including EOS when generated",
        "selected_layers": selected_layers,
        "result_count": len(results),
        "runtime_seconds": round(time.time() - started, 3),
        "results": results,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(results), "runtime_seconds": report["runtime_seconds"]}))


if __name__ == "__main__":
    main()
