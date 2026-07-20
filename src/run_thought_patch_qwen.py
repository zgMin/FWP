#!/usr/bin/env python3
"""Static thought-patch experiment for Qwen2.5-style decoder models.

This is a no-training implementation inspired by "Learning without training:
The implicit dynamics of in-context learning" (arXiv:2507.16003v4).

For a fixed prompt C, it approximates the context-conditioned forward pass
T(C, x) with a query-only forward pass T_{W+Delta}(x).  Delta is computed by
closed-form ridge least squares from calibration queries, then truncated to a
low-rank adapter and injected into each layer's MLP gate/up projections.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclasses.dataclass
class PatchConfig:
    rank: int = 8
    ridge: float = 1.0e-3
    max_calib_tokens: int = 96
    max_eval_prompt_tokens: int = 160
    max_new_tokens: int = 64
    temperature: float = 0.0
    device: str = "cuda:0"
    dtype: str = "bfloat16"


class LowRankPatchedLinear(nn.Module):
    """Wrap a Linear layer with a frozen low-rank additive update.

    The forward computes base(x) + scale * ((x @ A.T) @ B.T), where
    A has shape [rank, in_features] and B has shape [out_features, rank].
    """

    def __init__(self, base: nn.Linear, a: torch.Tensor, b: torch.Tensor, scale: float = 1.0):
        super().__init__()
        self.base = base
        self.register_buffer("thought_a", a.contiguous())
        self.register_buffer("thought_b", b.contiguous())
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        update = torch.matmul(torch.matmul(x, self.thought_a.t()), self.thought_b.t())
        return y + self.scale * update.to(dtype=y.dtype)

    def merge(self) -> nn.Linear:
        rows = self.thought_b.shape[0]
        chunk = 8192
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            delta = torch.matmul(self.thought_b[start:end].float(), self.thought_a.float()).to(
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            )
            self.base.weight.data[start:end].add_(self.scale * delta)
        return self.base


def get_mlp_input(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, store: dict, key: int):
    # Qwen2 MLP forward input is hidden_states after post_attention_layernorm.
    item = store.setdefault(key, {})
    item["mlp_in"] = inputs[0].detach()
    item["mlp_out"] = output.detach()


def get_down_input(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, store: dict, key: int):
    item = store.setdefault(key, {})
    item["down_in"] = inputs[0].detach()


def get_attn_output(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output, store: dict, key: int):
    item = store.setdefault(key, {})
    attn_out = output[0] if isinstance(output, tuple) else output
    item["attn_out"] = attn_out.detach()


def get_attn_o_input(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, store: dict, key: int):
    item = store.setdefault(key, {})
    item["attn_o_in"] = inputs[0].detach()


def get_attn_q_input(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, store: dict, key: int):
    item = store.setdefault(key, {})
    item["attn_q_in"] = inputs[0].detach()
    item["attn_q_out"] = output.detach()


def get_attn_v_input(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, store: dict, key: int):
    item = store.setdefault(key, {})
    item["attn_v_in"] = inputs[0].detach()
    item["attn_v_out"] = output.detach()


def attention_projection_owner(layer: nn.Module) -> tuple[nn.Module, str]:
    """Return the sequence-mixer module and its output projection attribute."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn, "o_proj"
    if hasattr(layer, "linear_attn"):
        return layer.linear_attn, "out_proj"
    raise AttributeError(f"unsupported decoder layer attention module: {type(layer).__name__}")


def last_token_vec(hidden: torch.Tensor, token_count: int) -> torch.Tensor:
    # Works with batch size 1. The last non-padding token is the final token.
    return hidden[0, token_count - 1].detach().float().cpu()


def apply_chat(tokenizer, system_prompt: str, user_text: str) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = f"<|system|>\n{system_prompt}\n" if system_prompt else ""
    return f"{prefix}<|user|>\n{user_text}\n<|assistant|>\n"


def configure_chat_terminator(model: nn.Module, tokenizer) -> int:
    """Select the model's chat-turn terminator for generation and EOS anchors."""
    configured = getattr(model.generation_config, "eos_token_id", None)
    eos_ids = list(configured) if isinstance(configured, (list, tuple)) else [configured]
    eos_ids = [int(token_id) for token_id in eos_ids if token_id is not None]
    tokenizer_eos = tokenizer.eos_token_id
    if tokenizer_eos is not None:
        tokenizer_eos = int(tokenizer_eos)
    preferred = tokenizer_eos
    if len(eos_ids) > 1 and tokenizer_eos in eos_ids:
        preferred = eos_ids[-1]
    elif preferred is None and eos_ids:
        preferred = eos_ids[-1]
    if preferred is None:
        raise ValueError("model and tokenizer do not define an EOS token")
    if preferred != tokenizer_eos:
        token = tokenizer.convert_ids_to_tokens(preferred)
        if token is None:
            raise ValueError(f"cannot resolve EOS token id {preferred}")
        tokenizer.eos_token = token
    return preferred


def make_inputs(tokenizer, text: str, device: str, max_tokens: int | None = None):
    tok = tokenizer(text, return_tensors="pt", truncation=max_tokens is not None, max_length=max_tokens)
    return {k: v.to(device) for k, v in tok.items()}


@torch.no_grad()
def collect_layer_mlp_inputs(
    model: nn.Module,
    tokenizer,
    text: str,
    device: str,
    max_tokens: int | None,
    layer_indices: Iterable[int] | None = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    stores: Dict[int, Dict[str, torch.Tensor]] = {}
    handles = []
    wanted = set(layer_indices) if layer_indices is not None else None
    for i, layer in enumerate(model.model.layers):
        if wanted is not None and i not in wanted:
            continue
        attn_owner, out_proj_name = attention_projection_owner(layer)
        handles.append(attn_owner.register_forward_hook(lambda m, inp, out, i=i: get_attn_output(m, inp, out, stores, i)))
        if hasattr(attn_owner, "q_proj"):
            handles.append(attn_owner.q_proj.register_forward_hook(lambda m, inp, out, i=i: get_attn_q_input(m, inp, out, stores, i)))
        if hasattr(attn_owner, "v_proj"):
            handles.append(attn_owner.v_proj.register_forward_hook(lambda m, inp, out, i=i: get_attn_v_input(m, inp, out, stores, i)))
        handles.append(getattr(attn_owner, out_proj_name).register_forward_hook(lambda m, inp, out, i=i: get_attn_o_input(m, inp, out, stores, i)))
        handles.append(layer.mlp.register_forward_hook(lambda m, inp, out, i=i: get_mlp_input(m, inp, out, stores, i)))
        handles.append(layer.mlp.down_proj.register_forward_hook(lambda m, inp, out, i=i: get_down_input(m, inp, out, stores, i)))
    try:
        inputs = make_inputs(tokenizer, text, device, max_tokens=max_tokens)
        model(**inputs, use_cache=False)
        token_count = int(inputs["attention_mask"].sum().item())
        out = {}
        for i, values in stores.items():
            out[i] = {name: last_token_vec(hidden, token_count) for name, hidden in values.items()}
        return out
    finally:
        for h in handles:
            h.remove()


def solve_ridge_factors(a_cols: torch.Tensor, t_cols: torch.Tensor, ridge: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve min_D ||D A - T||_F^2 + ridge ||D||_F^2 as low-rank factors.

    a_cols: [d_in, k], t_cols: [d_out, k].
    Returns B, A such that dense Delta = B @ A, with rank at most k.
    """
    gram = a_cols.t() @ a_cols
    reg = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    # Delta = T A^T (A A^T + ridge I)^-1
    #       = T (A^T A + ridge I)^-1 A^T.
    # The second form only solves a KxK system.
    try:
        right = torch.linalg.solve(gram + reg, a_cols.t())  # [k, d_in]
    except torch._C._LinAlgError:
        # With nearly duplicate trajectory columns, a small ridge can vanish at
        # float32 precision relative to A^T A. Solve the identical KxK system in
        # float64, then use a Hermitian pseudo-inverse only as a final fallback.
        a64 = a_cols.double()
        system64 = a64.t() @ a64
        system64 = system64 + ridge * torch.eye(system64.shape[0], dtype=torch.float64)
        try:
            right = torch.linalg.solve(system64, a64.t()).to(a_cols.dtype)
        except torch._C._LinAlgError:
            right = (torch.linalg.pinv(system64, hermitian=True) @ a64.t()).to(a_cols.dtype)
    return t_cols.contiguous(), right.contiguous()


def solve_input_space_factors(a_cols: torch.Tensor, delta_cols: torch.Tensor, ridge: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit M for x -> x + Mx; a projection W then receives mergeable delta W M."""
    return solve_ridge_factors(a_cols, delta_cols, ridge)


def projection_patch_from_input_map(weight: torch.Tensor, b_m: torch.Tensor, a_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b = weight.float().cpu() @ b_m.float()
    return a_m.contiguous(), b.contiguous()


def compress_factorized_delta(b_full: torch.Tensor, a_full: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """Compress Delta=B_full@A_full to rank r without materializing full SVD."""
    q_b, r_b = torch.linalg.qr(b_full.float(), mode="reduced")
    q_a, r_a = torch.linalg.qr(a_full.float().t(), mode="reduced")
    core = r_b @ r_a.t()
    u, s, vh = torch.linalg.svd(core.float(), full_matrices=False)
    r = min(rank, s.numel())
    dense_energy = float(s.square().sum().item())
    s_r = s[:r]
    b = (q_b @ u[:, :r]) * s_r.unsqueeze(0)
    a = vh[:r, :] @ q_a.t()
    kept = float((s_r.square().sum() / s.square().sum().clamp_min(1.0e-12)).item())
    return a.contiguous(), b.contiguous(), kept, math.sqrt(max(dense_energy, 0.0))


def install_patch(model: nn.Module, patches: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]], device: str, dtype: torch.dtype):
    for layer_idx, layer_patches in patches.items():
        layer = model.model.layers[layer_idx]
        for proj_name, (a_cpu, b_cpu) in layer_patches.items():
            if proj_name == "attn_o_proj":
                owner, attr_name = attention_projection_owner(layer)
            elif proj_name == "attn_q_proj":
                owner = layer.self_attn
                attr_name = "q_proj"
            elif proj_name == "attn_v_proj":
                owner = layer.self_attn
                attr_name = "v_proj"
            else:
                owner = layer.mlp
                attr_name = proj_name
            base = getattr(owner, attr_name)
            if isinstance(base, LowRankPatchedLinear):
                base = base.base
            wrapped = LowRankPatchedLinear(base, a_cpu.to(device=device, dtype=dtype), b_cpu.to(device=device, dtype=dtype))
            setattr(owner, attr_name, wrapped)


def remove_patch(model: nn.Module):
    for layer in model.model.layers:
        attn_owner, out_proj_name = attention_projection_owner(layer)
        mod = getattr(attn_owner, out_proj_name)
        if isinstance(mod, LowRankPatchedLinear):
            setattr(attn_owner, out_proj_name, mod.base)
        for attr_name in ("q_proj", "v_proj"):
            if not hasattr(attn_owner, attr_name):
                continue
            mod = getattr(attn_owner, attr_name)
            if isinstance(mod, LowRankPatchedLinear):
                setattr(attn_owner, attr_name, mod.base)
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(layer.mlp, proj_name)
            if isinstance(mod, LowRankPatchedLinear):
                setattr(layer.mlp, proj_name, mod.base)


def merge_installed_patches(model: nn.Module) -> int:
    merged = 0
    for layer in model.model.layers:
        attn_owner, out_proj_name = attention_projection_owner(layer)
        mod = getattr(attn_owner, out_proj_name)
        if isinstance(mod, LowRankPatchedLinear):
            setattr(attn_owner, out_proj_name, mod.merge())
            merged += 1
        for attr_name in ("q_proj", "v_proj"):
            if not hasattr(attn_owner, attr_name):
                continue
            mod = getattr(attn_owner, attr_name)
            if isinstance(mod, LowRankPatchedLinear):
                setattr(attn_owner, attr_name, mod.merge())
                merged += 1
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(layer.mlp, proj_name)
            if isinstance(mod, LowRankPatchedLinear):
                setattr(layer.mlp, proj_name, mod.merge())
                merged += 1
    if isinstance(model.lm_head, LowRankPatchedLinear):
        model.lm_head = model.lm_head.merge()
        merged += 1
    return merged


def install_lm_head_patch(
    model: nn.Module,
    patch: Tuple[torch.Tensor, torch.Tensor] | None,
    device: str,
    dtype: torch.dtype,
    scale: float = 1.0,
):
    if patch is None:
        return
    base = model.lm_head
    if isinstance(base, LowRankPatchedLinear):
        base = base.base
    a_cpu, b_cpu = patch
    model.lm_head = LowRankPatchedLinear(
        base,
        a_cpu.to(device=device, dtype=dtype),
        b_cpu.to(device=device, dtype=dtype),
        scale=scale,
    )


def remove_lm_head_patch(model: nn.Module):
    if isinstance(model.lm_head, LowRankPatchedLinear):
        model.lm_head = model.lm_head.base


def token_bleu(reference: str, candidate: str) -> float:
    ref = reference.strip().split()
    cand = candidate.strip().split()
    if not ref or not cand:
        return 0.0
    smoother = SmoothingFunction().method1
    return float(sentence_bleu([ref], cand, smoothing_function=smoother))


def char_bleu(reference: str, candidate: str) -> float:
    ref = [c for c in reference.strip() if not c.isspace()]
    cand = [c for c in candidate.strip() if not c.isspace()]
    if not ref or not cand:
        return 0.0
    smoother = SmoothingFunction().method1
    return float(sentence_bleu([ref], cand, smoothing_function=smoother))


def char_f1(a: str, b: str) -> float:
    ca = list(a)
    cb = list(b)
    if not ca or not cb:
        return 0.0
    matcher = difflib.SequenceMatcher(a=ca, b=cb)
    matches = sum(size for _, _, size in matcher.get_matching_blocks())
    precision = matches / max(len(cb), 1)
    recall = matches / max(len(ca), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def seq_ratio(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(a=a, b=b).ratio())


def format_score(text: str, mode: str) -> float:
    if mode == "tags":
        think_open = text.find("<think>")
        think_close = text.find("</think>")
        answer_open = text.find("<answer>")
        answer_close = text.find("</answer>")
        checks = [
            think_open == 0,
            think_close > think_open,
            answer_open > think_close,
            answer_close > answer_open,
            bool(text[think_open + 7 : think_close].strip()) if think_open >= 0 and think_close > think_open else False,
            bool(text[answer_open + 8 : answer_close].strip()) if answer_open >= 0 and answer_close > answer_open else False,
        ]
        return sum(checks) / len(checks)
    if mode == "repeat_tags":
        first_think = text.find("<think>")
        second_think = text.find("<think>", first_think + len("<think>")) if first_think >= 0 else -1
        first_answer = text.find("<answer>", second_think + len("<think>")) if second_think >= 0 else -1
        second_answer = text.find("<answer>", first_answer + len("<answer>")) if first_answer >= 0 else -1
        checks = [
            first_think == 0,
            second_think > first_think,
            first_answer > second_think,
            second_answer > first_answer,
            bool(text[first_think + 7 : second_think].strip()) if first_think >= 0 and second_think > first_think else False,
            bool(text[first_answer + 8 : second_answer].strip()) if first_answer >= 0 and second_answer > first_answer else False,
        ]
        return sum(checks) / len(checks)
    if mode == "json":
        try:
            obj = json.loads(text.strip())
        except json.JSONDecodeError:
            return 0.0
        checks = [
            isinstance(obj, dict),
            set(obj.keys()) == {"thought", "answer"} if isinstance(obj, dict) else False,
            isinstance(obj.get("thought"), str) and bool(obj.get("thought", "").strip()) if isinstance(obj, dict) else False,
            isinstance(obj.get("answer"), str) and bool(obj.get("answer", "").strip()) if isinstance(obj, dict) else False,
        ]
        return sum(checks) / len(checks)
    if mode == "conclusion3":
        checks = [
            bool(re.match(r"^\s*结论", text)),
            "要点" in text,
            bool(re.search(r"(?:^|[\n\s|])(?:1[\.、．）：:]|要点\s*1)", text)),
            bool(re.search(r"(?:^|[\n\s|])(?:2[\.、．）：:]|要点\s*2)", text)),
            bool(re.search(r"(?:^|[\n\s|])(?:3[\.、．）：:]|要点\s*3)", text)),
        ]
        return sum(checks) / len(checks)
    return 0.0


@torch.no_grad()
def next_token_kl(model: nn.Module, tokenizer, full_text: str, query_text: str, device: str, max_tokens: int) -> float:
    remove_patch(model)
    full_inputs = make_inputs(tokenizer, full_text, device, max_tokens=max_tokens)
    logits_full = model(**full_inputs, use_cache=False).logits[:, -1, :].float()
    p = F.log_softmax(logits_full, dim=-1)

    # The caller should have installed the patch before this function returns
    # patched logits. We temporarily leave the model unpatched only for full C+x.
    raise RuntimeError("next_token_kl should not be called directly")


@torch.no_grad()
def logits_for_text(model: nn.Module, tokenizer, text: str, device: str, max_tokens: int) -> torch.Tensor:
    inputs = make_inputs(tokenizer, text, device, max_tokens=max_tokens)
    try:
        out = model(**inputs, use_cache=False, logits_to_keep=1)
    except TypeError:
        out = model(**inputs, use_cache=False)
    return out.logits[:, -1, :].float().cpu()


@torch.no_grad()
def logits_and_last_hidden(model: nn.Module, tokenizer, text: str, device: str, max_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = make_inputs(tokenizer, text, device, max_tokens=max_tokens)
    out = model(**inputs, use_cache=False, output_hidden_states=True)
    token_count = int(inputs["attention_mask"].sum().item())
    hidden = out.hidden_states[-1][0, token_count - 1].detach().float().cpu()
    logits = out.logits[:, -1, :].detach().float().cpu()
    return logits, hidden


@torch.no_grad()
def logits_hidden_for_answer_positions(
    model: nn.Module,
    tokenizer,
    prompt_text: str,
    answer_text: str,
    device: str,
    max_tokens: int,
    max_answer_positions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompt_inputs = make_inputs(tokenizer, prompt_text, device, max_tokens=max_tokens)
    full_inputs = make_inputs(tokenizer, prompt_text + answer_text, device, max_tokens=max_tokens)
    prefix_len = int(prompt_inputs["attention_mask"].sum().item())
    full_len = int(full_inputs["attention_mask"].sum().item())
    n_pos = max(0, min(max_answer_positions, full_len - prefix_len))
    if n_pos == 0:
        return torch.empty(0, model.config.vocab_size), torch.empty(0, model.config.hidden_size)
    out = model(**full_inputs, use_cache=False, output_hidden_states=True)
    # Position prefix_len - 1 predicts the first answer token.
    positions = torch.arange(prefix_len - 1, prefix_len - 1 + n_pos, device=out.logits.device)
    logits = out.logits[0, positions, :].detach().float().cpu()
    hidden = out.hidden_states[-1][0, positions, :].detach().float().cpu()
    return logits, hidden


def kl_divergence_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum(dim=-1).mean().item())


@torch.no_grad()
def generate_text(model, tokenizer, text: str, cfg: PatchConfig) -> str:
    inputs = make_inputs(tokenizer, text, cfg.device, max_tokens=cfg.max_eval_prompt_tokens)
    do_sample = cfg.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = cfg.temperature
    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def build_default_sets() -> Tuple[str, List[str], List[str]]:
    system_prompt = (
        "你是一个回答风格固定的助手。请始终用中文回答，结构为："
        "先给一句不超过20字的结论，然后列出三条要点；语气简洁、克制、可执行。"
    )
    calib_queries = [
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
    eval_queries = [
        "如何把一个长prompt压缩成可复用的模型参数？",
        "给我一个排查GPU显存爆掉的流程。",
        "怎样评估一个LLM Agent是否真的有用？",
        "请解释LoRA为什么能用很少参数微调模型。",
        "我需要一个每天30分钟的数学复习计划。",
    ]
    return system_prompt, calib_queries, eval_queries


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    p.add_argument("--out", default="/root/zgm/thoughtpatch_qwen25/outputs/report.json")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--ridge", type=float, default=1.0e-3)
    p.add_argument("--calib-limit", type=int, default=16)
    p.add_argument("--max-calib-tokens", type=int, default=128)
    p.add_argument("--max-eval-prompt-tokens", type=int, default=192)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--patch-layers", default="all", help="'all' or comma separated layer indices")
    p.add_argument("--fit-mode", choices=["sequential", "independent"], default="sequential")
    p.add_argument("--include-attn-o", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-attn-q", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-attn-v", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--include-down", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--multiplicative", action=argparse.BooleanOptionalAction, default=False, help="Fit input-space M and merge W@M into gate/up projections.")
    p.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lm-rank", type=int, default=None, help="Rank for lm_head patch; defaults to --rank.")
    p.add_argument("--lm-scale", type=float, default=1.0, help="Scale applied when installing/merging lm_head patch.")
    p.add_argument("--lm-teacher-tokens", type=int, default=0, help="Use this many full-prompt generated answer positions for lm_head fitting; 0 uses next-token only.")
    p.add_argument("--lm-prefix-focus-tokens", type=int, default=0, help="Repeat the first N teacher positions for lm_head fitting.")
    p.add_argument("--lm-prefix-repeat", type=int, default=1, help="How many times to repeat focused teacher positions.")
    p.add_argument("--merge-before-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--patch-out", default="/root/zgm/thoughtpatch_qwen25/outputs/thought_patch.pt")
    p.add_argument("--system-prompt", default=None, help="Override the default system prompt.")
    p.add_argument("--system-prompt-file", default=None, help="Read the system prompt override from a UTF-8 file.")
    p.add_argument("--format-mode", choices=["none", "conclusion3", "tags", "repeat_tags", "json"], default="none")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

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

    t0 = time.time()
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

    system_prompt, calib_queries, eval_queries = build_default_sets()
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    calib_queries = calib_queries[: args.calib_limit]

    if args.patch_layers == "all":
        selected_layers = list(range(len(model.model.layers)))
    else:
        selected_layers = [int(x) for x in args.patch_layers.split(",") if x.strip()]
    selected_layers = sorted(selected_layers)

    print(f"Loaded model with {len(model.model.layers)} layers. Fitting layers: {selected_layers}")
    print(f"Calibration queries: {len(calib_queries)}, rank={args.rank}, ridge={args.ridge}")
    lm_rank = args.lm_rank if args.lm_rank is not None else args.rank

    # Collect full-context activations once. In sequential mode, query-only
    # activations are re-collected layer by layer after previous patches have
    # already been installed, matching the induction in the paper's Appendix C.
    full_acts_by_query: List[Dict[int, Dict[str, torch.Tensor]]] = []
    for idx, q in enumerate(calib_queries, 1):
        full = apply_chat(tokenizer, system_prompt, q)
        full_acts_by_query.append(
            collect_layer_mlp_inputs(model, tokenizer, full, args.device, args.max_calib_tokens, selected_layers)
        )
        print(f"  collected full-context calibration {idx}/{len(calib_queries)}: {q[:36]}")

    per_layer_a: Dict[int, List[torch.Tensor]] = {i: [] for i in selected_layers}
    per_layer_delta: Dict[int, List[torch.Tensor]] = {i: [] for i in selected_layers}
    per_layer_records: Dict[int, List[Dict[str, torch.Tensor]]] = {i: [] for i in selected_layers}
    if args.fit_mode == "independent":
        for idx, q in enumerate(calib_queries, 1):
            query_only = apply_chat(tokenizer, "", q)
            a_only = collect_layer_mlp_inputs(model, tokenizer, query_only, args.device, args.max_calib_tokens, selected_layers)
            for layer_idx in selected_layers:
                per_layer_records[layer_idx].append(a_only[layer_idx])
                per_layer_a[layer_idx].append(a_only[layer_idx]["mlp_in"])
                per_layer_delta[layer_idx].append(
                    full_acts_by_query[idx - 1][layer_idx]["mlp_in"] - a_only[layer_idx]["mlp_in"]
                )
            print(f"  collected query-only calibration {idx}/{len(calib_queries)}: {q[:36]}")

    # Fit factorized Delta for gate/up, optionally followed by a mergeable
    # down_proj patch that directly matches the full-context MLP output.
    patches: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    patch_stats = []
    for layer_idx in selected_layers:
        if args.fit_mode == "sequential":
            per_layer_a[layer_idx] = []
            per_layer_delta[layer_idx] = []
            per_layer_records[layer_idx] = []
            remove_patch(model)
            install_patch(model, patches, args.device, dtype)
            for idx, q in enumerate(calib_queries, 1):
                query_only = apply_chat(tokenizer, "", q)
                patched_acts = collect_layer_mlp_inputs(
                    model,
                    tokenizer,
                    query_only,
                    args.device,
                    args.max_calib_tokens,
                    [layer_idx],
                )
                per_layer_records[layer_idx].append(patched_acts[layer_idx])
                per_layer_a[layer_idx].append(patched_acts[layer_idx]["mlp_in"])
                per_layer_delta[layer_idx].append(
                    full_acts_by_query[idx - 1][layer_idx]["mlp_in"] - patched_acts[layer_idx]["mlp_in"]
                )

        layer = model.model.layers[layer_idx]
        patches[layer_idx] = {}
        if args.include_attn_q:
            q_a_cols = torch.stack([r["attn_q_in"] for r in per_layer_records[layer_idx]], dim=1).float()
            q_t_cols = torch.stack(
                [
                    full_acts_by_query[i][layer_idx]["attn_q_out"] - per_layer_records[layer_idx][i]["attn_q_out"]
                    for i in range(len(per_layer_records[layer_idx]))
                ],
                dim=1,
            ).float()
            b_full, a_full = solve_ridge_factors(q_a_cols, q_t_cols, args.ridge)
            a_lr, b_lr, energy, fro_norm = compress_factorized_delta(b_full, a_full, args.rank)
            patches[layer_idx]["attn_q_proj"] = (a_lr, b_lr)
            pred = (b_lr @ a_lr) @ q_a_cols
            mse = float(F.mse_loss(pred, q_t_cols).item())
            weight = layer.self_attn.q_proj.weight.detach()
            patch_stats.append(
                {
                    "layer": layer_idx,
                    "projection": "attn_q_proj",
                    "dense_shape": [int(weight.shape[0]), int(weight.shape[1])],
                    "rank": int(a_lr.shape[0]),
                    "svd_energy_kept": energy,
                    "calibration_mse_rank_truncated": mse,
                    "factorized_delta_fro_norm": fro_norm,
                }
            )
            if args.fit_mode == "sequential":
                remove_patch(model)
                install_patch(model, patches, args.device, dtype)
                per_layer_a[layer_idx] = []
                per_layer_delta[layer_idx] = []
                per_layer_records[layer_idx] = []
                for idx, q in enumerate(calib_queries, 1):
                    query_only = apply_chat(tokenizer, "", q)
                    patched_acts = collect_layer_mlp_inputs(
                        model,
                        tokenizer,
                        query_only,
                        args.device,
                        args.max_calib_tokens,
                        [layer_idx],
                    )
                    per_layer_records[layer_idx].append(patched_acts[layer_idx])
                    per_layer_a[layer_idx].append(patched_acts[layer_idx]["mlp_in"])
                    per_layer_delta[layer_idx].append(
                        full_acts_by_query[idx - 1][layer_idx]["mlp_in"] - patched_acts[layer_idx]["mlp_in"]
                    )

        if args.include_attn_v:
            v_a_cols = torch.stack([r["attn_v_in"] for r in per_layer_records[layer_idx]], dim=1).float()
            v_t_cols = torch.stack(
                [
                    full_acts_by_query[i][layer_idx]["attn_v_out"] - per_layer_records[layer_idx][i]["attn_v_out"]
                    for i in range(len(per_layer_records[layer_idx]))
                ],
                dim=1,
            ).float()
            b_full, a_full = solve_ridge_factors(v_a_cols, v_t_cols, args.ridge)
            a_lr, b_lr, energy, fro_norm = compress_factorized_delta(b_full, a_full, args.rank)
            patches[layer_idx]["attn_v_proj"] = (a_lr, b_lr)
            pred = (b_lr @ a_lr) @ v_a_cols
            mse = float(F.mse_loss(pred, v_t_cols).item())
            weight = layer.self_attn.v_proj.weight.detach()
            patch_stats.append(
                {
                    "layer": layer_idx,
                    "projection": "attn_v_proj",
                    "dense_shape": [int(weight.shape[0]), int(weight.shape[1])],
                    "rank": int(a_lr.shape[0]),
                    "svd_energy_kept": energy,
                    "calibration_mse_rank_truncated": mse,
                    "factorized_delta_fro_norm": fro_norm,
                }
            )
            if args.fit_mode == "sequential":
                remove_patch(model)
                install_patch(model, patches, args.device, dtype)
                per_layer_a[layer_idx] = []
                per_layer_delta[layer_idx] = []
                per_layer_records[layer_idx] = []
                for idx, q in enumerate(calib_queries, 1):
                    query_only = apply_chat(tokenizer, "", q)
                    patched_acts = collect_layer_mlp_inputs(
                        model,
                        tokenizer,
                        query_only,
                        args.device,
                        args.max_calib_tokens,
                        [layer_idx],
                    )
                    per_layer_records[layer_idx].append(patched_acts[layer_idx])
                    per_layer_a[layer_idx].append(patched_acts[layer_idx]["mlp_in"])
                    per_layer_delta[layer_idx].append(
                        full_acts_by_query[idx - 1][layer_idx]["mlp_in"] - patched_acts[layer_idx]["mlp_in"]
                    )

        if args.include_attn_o:
            attn_a_cols = torch.stack([r["attn_o_in"] for r in per_layer_records[layer_idx]], dim=1).float()
            attn_t_cols = torch.stack(
                [
                    full_acts_by_query[i][layer_idx]["attn_out"] - per_layer_records[layer_idx][i]["attn_out"]
                    for i in range(len(per_layer_records[layer_idx]))
                ],
                dim=1,
            ).float()
            b_full, a_full = solve_ridge_factors(attn_a_cols, attn_t_cols, args.ridge)
            a_lr, b_lr, energy, fro_norm = compress_factorized_delta(b_full, a_full, args.rank)
            patches[layer_idx]["attn_o_proj"] = (a_lr, b_lr)
            pred = (b_lr @ a_lr) @ attn_a_cols
            mse = float(F.mse_loss(pred, attn_t_cols).item())
            weight = layer.self_attn.o_proj.weight.detach()
            patch_stats.append(
                {
                    "layer": layer_idx,
                    "projection": "attn_o_proj",
                    "dense_shape": [int(weight.shape[0]), int(weight.shape[1])],
                    "rank": int(a_lr.shape[0]),
                    "svd_energy_kept": energy,
                    "calibration_mse_rank_truncated": mse,
                    "factorized_delta_fro_norm": fro_norm,
                }
            )
            if args.fit_mode == "sequential":
                remove_patch(model)
                install_patch(model, patches, args.device, dtype)
                per_layer_a[layer_idx] = []
                per_layer_delta[layer_idx] = []
                per_layer_records[layer_idx] = []
                for idx, q in enumerate(calib_queries, 1):
                    query_only = apply_chat(tokenizer, "", q)
                    patched_acts = collect_layer_mlp_inputs(
                        model,
                        tokenizer,
                        query_only,
                        args.device,
                        args.max_calib_tokens,
                        [layer_idx],
                    )
                    per_layer_records[layer_idx].append(patched_acts[layer_idx])
                    per_layer_a[layer_idx].append(patched_acts[layer_idx]["mlp_in"])
                    per_layer_delta[layer_idx].append(
                        full_acts_by_query[idx - 1][layer_idx]["mlp_in"] - patched_acts[layer_idx]["mlp_in"]
                    )

        a_cols = torch.stack(per_layer_a[layer_idx], dim=1).float()
        delta_cols = torch.stack(per_layer_delta[layer_idx], dim=1).float()
        input_map_patch = None
        if args.multiplicative:
            b_m, a_m = solve_input_space_factors(a_cols, delta_cols, args.ridge)
            input_map_patch = compress_factorized_delta(b_m, a_m, args.rank)
        for proj_name in ("gate_proj", "up_proj"):
            weight = getattr(layer.mlp, proj_name).weight.detach().float().cpu()
            targets = weight @ delta_cols
            if input_map_patch is None:
                b_full, a_full = solve_ridge_factors(a_cols, targets, args.ridge)
                a_lr, b_lr, energy, fro_norm = compress_factorized_delta(b_full, a_full, args.rank)
            else:
                a_m_lr, b_m_lr, energy, fro_norm = input_map_patch
                a_lr, b_lr = projection_patch_from_input_map(weight, b_m_lr, a_m_lr)
            patches[layer_idx][proj_name] = (a_lr, b_lr)
            pred = (b_lr @ a_lr) @ a_cols
            mse = float(F.mse_loss(pred, targets).item())
            patch_stats.append(
                {
                    "layer": layer_idx,
                    "projection": proj_name,
                    "multiplicative_input_map": bool(args.multiplicative),
                    "dense_shape": [int(weight.shape[0]), int(weight.shape[1])],
                    "rank": int(a_lr.shape[0]),
                    "svd_energy_kept": energy,
                    "calibration_mse_rank_truncated": mse,
                    "factorized_delta_fro_norm": fro_norm,
                }
            )

        if args.include_down:
            # Re-run this layer with gate/up already installed, then fit a
            # mergeable output projection patch:
            # Delta_down * down_in_query ~= mlp_out_full - mlp_out_query.
            remove_patch(model)
            install_patch(model, patches, args.device, dtype)
            down_inputs = []
            down_targets = []
            for idx, q in enumerate(calib_queries, 1):
                query_only = apply_chat(tokenizer, "", q)
                patched_acts = collect_layer_mlp_inputs(
                    model,
                    tokenizer,
                    query_only,
                    args.device,
                    args.max_calib_tokens,
                    [layer_idx],
                )[layer_idx]
                down_inputs.append(patched_acts["down_in"])
                down_targets.append(full_acts_by_query[idx - 1][layer_idx]["mlp_out"] - patched_acts["mlp_out"])
            down_a_cols = torch.stack(down_inputs, dim=1).float()
            down_t_cols = torch.stack(down_targets, dim=1).float()
            b_full, a_full = solve_ridge_factors(down_a_cols, down_t_cols, args.ridge)
            a_lr, b_lr, energy, fro_norm = compress_factorized_delta(b_full, a_full, args.rank)
            patches[layer_idx]["down_proj"] = (a_lr, b_lr)
            pred = (b_lr @ a_lr) @ down_a_cols
            mse = float(F.mse_loss(pred, down_t_cols).item())
            weight = layer.mlp.down_proj.weight.detach()
            patch_stats.append(
                {
                    "layer": layer_idx,
                    "projection": "down_proj",
                    "dense_shape": [int(weight.shape[0]), int(weight.shape[1])],
                    "rank": int(a_lr.shape[0]),
                    "svd_energy_kept": energy,
                    "calibration_mse_rank_truncated": mse,
                    "factorized_delta_fro_norm": fro_norm,
                }
            )

        if args.fit_mode == "sequential":
            remove_patch(model)
            install_patch(model, patches, args.device, dtype)
        print(f"  fitted layer {layer_idx}")

    lm_head_patch = None
    if args.include_lm_head:
        remove_patch(model)
        remove_lm_head_patch(model)
        full_logits_by_query = []
        teacher_answers = []
        for q in calib_queries:
            full = apply_chat(tokenizer, system_prompt, q)
            if args.lm_teacher_tokens > 0:
                teacher_answers.append(generate_text(model, tokenizer, full, cfg))
            else:
                full_logits_by_query.append(logits_for_text(model, tokenizer, full, args.device, args.max_calib_tokens).squeeze(0))

        install_patch(model, patches, args.device, dtype)
        hidden_cols = []
        target_cols = []
        for idx, q in enumerate(calib_queries, 1):
            query_only = apply_chat(tokenizer, "", q)
            if args.lm_teacher_tokens > 0:
                full = apply_chat(tokenizer, system_prompt, q)
                answer = teacher_answers[idx - 1]
                remove_patch(model)
                full_logits, _ = logits_hidden_for_answer_positions(
                    model,
                    tokenizer,
                    full,
                    answer,
                    args.device,
                    args.max_eval_prompt_tokens + args.max_new_tokens,
                    args.lm_teacher_tokens,
                )
                install_patch(model, patches, args.device, dtype)
                query_logits, query_hidden = logits_hidden_for_answer_positions(
                    model,
                    tokenizer,
                    query_only,
                    answer,
                    args.device,
                    args.max_eval_prompt_tokens + args.max_new_tokens,
                    args.lm_teacher_tokens,
                )
                n = min(full_logits.shape[0], query_logits.shape[0], query_hidden.shape[0])
                for pos in range(n):
                    repeats = args.lm_prefix_repeat if pos < args.lm_prefix_focus_tokens else 1
                    for _ in range(max(1, repeats)):
                        hidden_cols.append(query_hidden[pos])
                        target_cols.append(full_logits[pos] - query_logits[pos])
            else:
                query_logits, query_hidden = logits_and_last_hidden(
                    model,
                    tokenizer,
                    query_only,
                    args.device,
                    args.max_calib_tokens,
                )
                hidden_cols.append(query_hidden)
                target_cols.append(full_logits_by_query[idx - 1] - query_logits.squeeze(0))
        h_cols = torch.stack(hidden_cols, dim=1).float()
        t_cols = torch.stack(target_cols, dim=1).float()
        b_full, a_full = solve_ridge_factors(h_cols, t_cols, args.ridge)
        a_lr, b_lr, energy, fro_norm = compress_factorized_delta(b_full, a_full, lm_rank)
        lm_head_patch = (a_lr, b_lr)
        pred = (b_lr @ a_lr) @ h_cols
        mse = float(F.mse_loss(pred, t_cols).item())
        weight = model.lm_head.base.weight.detach() if isinstance(model.lm_head, LowRankPatchedLinear) else model.lm_head.weight.detach()
        patch_stats.append(
            {
                "layer": "output",
                "projection": "lm_head",
                "dense_shape": [int(weight.shape[0]), int(weight.shape[1])],
                "rank": int(a_lr.shape[0]),
                "scale": args.lm_scale,
                "svd_energy_kept": energy,
                "calibration_mse_rank_truncated": mse,
                "factorized_delta_fro_norm": fro_norm,
            }
        )
        remove_patch(model)
        print("  fitted lm_head distribution correction")

    patch_out = Path(args.patch_out)
    patch_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "patches": patches,
            "lm_head_patch": lm_head_patch,
            "rank": args.rank,
            "lm_rank": lm_rank,
            "lm_scale": args.lm_scale,
            "ridge": args.ridge,
            "selected_layers": selected_layers,
            "include_attn_o": args.include_attn_o,
            "include_attn_q": args.include_attn_q,
            "include_attn_v": args.include_attn_v,
            "include_down": args.include_down,
            "multiplicative": args.multiplicative,
            "include_lm_head": args.include_lm_head,
            "lm_teacher_tokens": args.lm_teacher_tokens,
            "lm_prefix_focus_tokens": args.lm_prefix_focus_tokens,
            "lm_prefix_repeat": args.lm_prefix_repeat,
            "fit_mode": args.fit_mode,
            "model": args.model,
            "note": "Merge with weight += B @ A for each saved (A, B) pair.",
        },
        patch_out,
    )

    # Evaluate distribution and generation differences.
    baseline_cache = []
    remove_patch(model)
    remove_lm_head_patch(model)
    for q in eval_queries:
        full = apply_chat(tokenizer, system_prompt, q)
        query_only = apply_chat(tokenizer, "", q)
        full_logits = logits_for_text(model, tokenizer, full, args.device, args.max_eval_prompt_tokens)
        baseline_text = generate_text(model, tokenizer, full, cfg)
        baseline_cache.append((q, query_only, full_logits, baseline_text))

    install_patch(model, patches, args.device, dtype)
    install_lm_head_patch(model, lm_head_patch, args.device, dtype, scale=args.lm_scale)
    merged_linear_count = 0
    if args.merge_before_eval:
        merged_linear_count = merge_installed_patches(model)
        print(f"  merged {merged_linear_count} low-rank adapters into Linear weights")

    results = []
    for q, query_only, full_logits, baseline_text in baseline_cache:
        patched_logits = logits_for_text(model, tokenizer, query_only, args.device, args.max_eval_prompt_tokens)
        patched_text = generate_text(model, tokenizer, query_only, cfg)

        kl = kl_divergence_from_logits(full_logits, patched_logits)
        baseline_ids = tokenizer.encode(baseline_text, add_special_tokens=False)
        patched_ids = tokenizer.encode(patched_text, add_special_tokens=False)
        results.append(
            {
                "query": q,
                "kl_next_token_full_prompt_vs_patched_query": kl,
                "baseline_length_chars": len(baseline_text),
                "patched_length_chars": len(patched_text),
                "length_diff_chars": len(patched_text) - len(baseline_text),
                "baseline_length_tokens": len(baseline_ids),
                "patched_length_tokens": len(patched_ids),
                "length_diff_tokens": len(patched_ids) - len(baseline_ids),
                "bleu": token_bleu(baseline_text, patched_text),
                "char_bleu": char_bleu(baseline_text, patched_text),
                "char_f1": char_f1(baseline_text, patched_text),
                "sequence_similarity": seq_ratio(baseline_text, patched_text),
                "format_score": format_score(patched_text, args.format_mode),
                "baseline_full_prompt_output": baseline_text,
                "patched_query_only_output": patched_text,
            }
        )
        print(f"  evaluated: {q[:36]} KL={kl:.4f}")

    avg = {}
    metric_keys = [
        "kl_next_token_full_prompt_vs_patched_query",
        "length_diff_chars",
        "length_diff_tokens",
        "bleu",
        "char_bleu",
        "char_f1",
        "sequence_similarity",
        "format_score",
    ]
    for key in metric_keys:
        avg[key] = float(sum(r[key] for r in results) / max(len(results), 1))

    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "environment": "thoughtpatch-qwen25",
        "method": "closed-form ridge static thought patch; mergeable SVD-truncated low-rank weight deltas on Qwen attention/MLP/lm_head projections",
        "config": dataclasses.asdict(cfg),
        "fit_mode": args.fit_mode,
        "include_attn_o": args.include_attn_o,
        "include_attn_q": args.include_attn_q,
        "include_attn_v": args.include_attn_v,
        "include_down": args.include_down,
        "multiplicative": args.multiplicative,
        "include_lm_head": args.include_lm_head,
        "lm_rank": lm_rank,
        "lm_scale": args.lm_scale,
        "lm_teacher_tokens": args.lm_teacher_tokens,
        "lm_prefix_focus_tokens": args.lm_prefix_focus_tokens,
        "lm_prefix_repeat": args.lm_prefix_repeat,
        "merge_before_eval": args.merge_before_eval,
        "merged_linear_count": merged_linear_count,
        "patch_bundle": str(patch_out),
        "system_prompt_converted": system_prompt,
        "calibration_queries": calib_queries,
        "eval_queries": eval_queries,
        "selected_layers": selected_layers,
        "patch_stats": patch_stats,
        "averages": avg,
        "results": results,
        "runtime_seconds": round(time.time() - t0, 3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "averages": avg, "runtime_seconds": report["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
