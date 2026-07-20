#!/usr/bin/env python3
"""Fit a no-training, mergeable lm_head logit-margin anchor patch.

The patch is fitted in closed form. For each calibration query, we first ask the
base model with the fixed system prompt for a teacher answer. Then, under the
query-only condition plus any existing mergeable patch bundle, we collect hidden
states along the teacher-forced answer prefix and solve a ridge least-squares
problem that boosts the teacher token logits by a margin.

The resulting lm_head update is saved as another (A, B) low-rank factor and can
be merged into model weights with: lm_head.weight += B @ A.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "你是一个回答风格固定的助手。请始终用中文回答，结构为："
    "先给一句不超过20字的结论，然后列出三条要点；语气简洁、克制、可执行。"
)

CALIB_QUERIES = [
    "如何快速判断一个创业想法是否值得继续做？",
    "给我一个学习强化学习的两周计划。",
    "怎样把一篇论文读得更有效率？",
    "解释一下什么是过拟合，并给出避免方法。",
    "我想提升英文写作，应该每天练什么？",
    "如何设计一个可靠的A/B实验？",
    "团队开会总是低效，怎么改善？",
    "请给出一个健身新手的入门建议。",
    "怎样排查线上服务突然变慢的问题？",
    "如何准备一次技术分享？",
    "什么情况下应该使用缓存？",
    "请给出一个阅读源码的步骤。",
    "如何写出更清晰的产品需求文档？",
    "怎么判断一个模型评测是否可信？",
    "我想减少拖延，给我可执行建议。",
    "解释一下数据库索引的作用。",
    "如何给一个新项目设计里程碑？",
    "怎样判断一个开源库是否适合生产使用？",
    "请给出一个周末整理房间的计划。",
    "如何降低一次代码重构的风险？",
    "解释一下什么是梯度爆炸。",
    "怎样写一封清晰的工作周报？",
    "如何准备机器学习岗位面试？",
    "请给出一个减少手机使用时间的方法。",
    "如何评估一个数据集的质量？",
    "服务接口经常超时，应该怎么排查？",
    "怎样给初学者解释大语言模型？",
    "如何安排一次有效的一对一沟通？",
    "请给出一个学习Linux命令的路线。",
    "怎样判断一个需求是否值得做？",
    "如何提高代码评审的效率？",
    "解释一下什么是缓存穿透。",
]

EVAL_QUERIES = [
    "如何把一个长prompt压缩成可复用的模型参数？",
    "给我一个排查GPU显存爆掉的流程。",
    "怎样评估一个LLM Agent是否真的有用？",
    "请解释LoRA为什么能用很少参数微调模型。",
    "我需要一个每天30分钟的数学复习计划。",
]


def apply_chat(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = f"<|system|>\n{system_prompt}\n" if system_prompt else ""
    return f"{prefix}<|user|>\n{user_text}\n<|assistant|>\n"


def add_delta(linear, pair, scale: float = 1.0):
    if pair is None:
        return
    a, b = pair
    delta = (b.float() @ a.float()).to(device=linear.weight.device, dtype=linear.weight.dtype)
    linear.weight.data.add_(scale * delta)


def merge_bundle_into_model(model, bundle: Dict):
    for layer_idx, layer_patches in bundle.get("patches", {}).items():
        layer = model.model.layers[int(layer_idx)]
        for proj_name, pair in layer_patches.items():
            add_delta(getattr(layer.mlp, proj_name), pair)
    add_delta(model.lm_head, bundle.get("lm_head_patch"), float(bundle.get("lm_scale", 1.0)))


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


@torch.no_grad()
def logits_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True)
    return out.logits[:, -1, :].float().cpu(), out.hidden_states[-1][:, -1, :].float().cpu()


def solve_ridge_factors(a_cols: torch.Tensor, t_cols: torch.Tensor, ridge: float) -> Tuple[torch.Tensor, torch.Tensor]:
    gram = a_cols.t() @ a_cols
    reg = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    right = torch.linalg.solve(gram + reg, a_cols.t())
    return t_cols.contiguous(), right.contiguous()


def compress_factorized_delta(b_full: torch.Tensor, a_full: torch.Tensor, rank: int):
    q_b, r_b = torch.linalg.qr(b_full.float(), mode="reduced")
    q_a, r_a = torch.linalg.qr(a_full.float().t(), mode="reduced")
    core = r_b @ r_a.t()
    u, s, vh = torch.linalg.svd(core.float(), full_matrices=False)
    r = min(rank, s.numel())
    b = (q_b @ u[:, :r]) * s[:r].unsqueeze(0)
    a = vh[:r, :] @ q_a.t()
    kept = float((s[:r].square().sum() / s.square().sum().clamp_min(1.0e-12)).item())
    return a.contiguous(), b.contiguous(), kept


def char_bleu(reference: str, candidate: str) -> float:
    ref = [c for c in reference.strip() if not c.isspace()]
    cand = [c for c in candidate.strip() if not c.isspace()]
    if not ref or not cand:
        return 0.0
    return float(sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method1))


def char_f1(a: str, b: str) -> float:
    ca = list(a)
    cb = list(b)
    if not ca or not cb:
        return 0.0
    matcher = difflib.SequenceMatcher(a=ca, b=cb)
    matches = sum(size for _, _, size in matcher.get_matching_blocks())
    p = matches / max(len(cb), 1)
    r = matches / max(len(ca), 1)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def format_score(text: str) -> Dict[str, float]:
    clean = text.strip()
    checks = {
        "starts_with_conclusion": float(clean.startswith("结论")),
        "has_keypoints": float("要点" in clean),
        "has_1": float(("1." in clean) or ("1、" in clean) or ("1．" in clean)),
        "has_2": float(("2." in clean) or ("2、" in clean) or ("2．" in clean)),
        "has_3": float(("3." in clean) or ("3、" in clean) or ("3．" in clean)),
    }
    checks["format_score"] = sum(checks.values()) / len(checks)
    return checks


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    lp = F.log_softmax(p_logits, dim=-1)
    lq = F.log_softmax(q_logits, dim=-1)
    return float((lp.exp() * (lp - lq)).sum(dim=-1).mean().item())


def combine_lm_head(existing, existing_scale: float, anchor, anchor_scale: float):
    if existing is None:
        if anchor_scale == 1.0:
            return anchor
        a, b = anchor
        return a, b * anchor_scale
    a0, b0 = existing
    a1, b1 = anchor
    return torch.cat([a0, a1], dim=0), torch.cat([b0 * existing_scale, b1 * anchor_scale], dim=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    p.add_argument("--base-patch", default="/root/zgm/thoughtpatch_qwen25/outputs/thought_patch_format_teacher_lm64_scale1.pt")
    p.add_argument("--out", default="/root/zgm/thoughtpatch_qwen25/outputs/report_logit_margin_anchor.json")
    p.add_argument("--patch-out", default="/root/zgm/thoughtpatch_qwen25/outputs/thought_patch_logit_margin_anchor.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--calib-limit", type=int, default=32)
    p.add_argument("--teacher-tokens", type=int, default=16)
    p.add_argument("--force-prefix-text", default="", help="If set, fit this fixed prefix instead of teacher answer tokens.")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--ridge", type=float, default=0.1)
    p.add_argument("--margin", type=float, default=4.0)
    p.add_argument("--max-boost", type=float, default=12.0)
    p.add_argument("--anchor-scale", type=float, default=0.75)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=23)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

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

    base_bundle = torch.load(args.base_patch, map_location="cpu", weights_only=False)

    # Baseline teachers before merging any patch.
    teacher_rows = []
    for i, q in enumerate(CALIB_QUERIES[: args.calib_limit], 1):
        full = apply_chat(tokenizer, SYSTEM_PROMPT, q)
        teacher = generate_text(model, tokenizer, full, args.max_new_tokens)
        teacher_rows.append({"query": q, "teacher": teacher})
        print(f"teacher {i}/{args.calib_limit}: {q[:32]} -> {teacher[:42].replace(chr(10), ' | ')}")

    baseline_cache = []
    for q in EVAL_QUERIES:
        full = apply_chat(tokenizer, SYSTEM_PROMPT, q)
        full_logits, _ = logits_hidden(model, **tokenizer(full, return_tensors="pt").to(model.device))
        baseline = generate_text(model, tokenizer, full, args.max_new_tokens)
        baseline_cache.append((q, full_logits, baseline))

    merge_bundle_into_model(model, base_bundle)

    hidden_cols = []
    target_cols = []
    vocab = model.config.vocab_size
    for row_idx, row in enumerate(teacher_rows, 1):
        query_prompt = apply_chat(tokenizer, "", row["query"])
        prompt_ids = tokenizer(query_prompt, add_special_tokens=False).input_ids
        source_text = args.force_prefix_text if args.force_prefix_text else row["teacher"]
        answer_ids = tokenizer(source_text, add_special_tokens=False).input_ids[: args.teacher_tokens]
        for pos, target_id in enumerate(answer_ids):
            ids = torch.tensor([prompt_ids + answer_ids[:pos]], dtype=torch.long, device=model.device)
            mask = torch.ones_like(ids)
            logits, hidden = logits_hidden(model, ids, mask)
            current = logits[0]
            top_val, top_id = torch.max(current, dim=0)
            target_val = current[target_id]
            boost = float(torch.clamp(top_val - target_val + args.margin, min=0.0, max=args.max_boost).item())
            if boost <= 1.0e-6:
                continue
            col = torch.zeros(vocab, dtype=torch.float32)
            col[target_id] = boost
            if int(top_id.item()) != int(target_id):
                col[int(top_id.item())] = -0.35 * boost
            hidden_cols.append(hidden[0])
            target_cols.append(col)
        print(f"anchor rows {row_idx}/{len(teacher_rows)} collected")

    h_cols = torch.stack(hidden_cols, dim=1).float()
    t_cols = torch.stack(target_cols, dim=1).float()
    b_full, a_full = solve_ridge_factors(h_cols, t_cols, args.ridge)
    a_anchor, b_anchor, energy = compress_factorized_delta(b_full, a_full, args.rank)
    anchor_patch = (a_anchor.cpu(), b_anchor.cpu())

    combined = dict(base_bundle)
    combined["lm_head_patch"] = combine_lm_head(
        base_bundle.get("lm_head_patch"),
        float(base_bundle.get("lm_scale", 1.0)),
        anchor_patch,
        args.anchor_scale,
    )
    combined["lm_scale"] = 1.0
    combined["logit_margin_anchor"] = {
        "rank": args.rank,
        "ridge": args.ridge,
        "margin": args.margin,
        "max_boost": args.max_boost,
        "anchor_scale": args.anchor_scale,
        "teacher_tokens": args.teacher_tokens,
        "columns": len(hidden_cols),
        "svd_energy_kept": energy,
    }
    torch.save(combined, args.patch_out)

    # Merge anchor into the already base-patched in-memory model.
    add_delta(model.lm_head, anchor_patch, args.anchor_scale)

    results = []
    for q, full_logits, baseline in baseline_cache:
        query_prompt = apply_chat(tokenizer, "", q)
        query_logits, _ = logits_hidden(model, **tokenizer(query_prompt, return_tensors="pt").to(model.device))
        patched = generate_text(model, tokenizer, query_prompt, args.max_new_tokens)
        baseline_tokens = tokenizer.encode(baseline, add_special_tokens=False)
        patched_tokens = tokenizer.encode(patched, add_special_tokens=False)
        results.append(
            {
                "query": q,
                "kl_next_token_full_prompt_vs_anchor_query": kl_divergence(full_logits, query_logits),
                "length_diff_tokens": len(patched_tokens) - len(baseline_tokens),
                "char_bleu": char_bleu(baseline, patched),
                "char_f1": char_f1(baseline, patched),
                "baseline_format": format_score(baseline),
                "patched_format": format_score(patched),
                "baseline_full_prompt_output": baseline,
                "patched_query_only_output": patched,
            }
        )
        print(f"eval {q[:32]} KL={results[-1]['kl_next_token_full_prompt_vs_anchor_query']:.4f}")

    averages = {}
    for key in ["kl_next_token_full_prompt_vs_anchor_query", "length_diff_tokens", "char_bleu", "char_f1"]:
        averages[key] = float(sum(r[key] for r in results) / len(results))
    averages["baseline_format_score"] = float(sum(r["baseline_format"]["format_score"] for r in results) / len(results))
    averages["patched_format_score"] = float(sum(r["patched_format"]["format_score"] for r in results) / len(results))

    report = {
        "method": "no-training closed-form logit-margin anchor on top of mergeable thought patch",
        "base_patch": args.base_patch,
        "patch_bundle": args.patch_out,
        "config": vars(args),
        "anchor_columns": len(hidden_cols),
        "svd_energy_kept": energy,
        "teacher_rows": teacher_rows,
        "averages": averages,
        "results": results,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out, "patch": args.patch_out, "averages": averages}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
