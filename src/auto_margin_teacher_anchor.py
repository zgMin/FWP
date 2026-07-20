#!/usr/bin/env python3
"""Hyperparameter-free teacher-token boundary patch.

Given a fixed QTraj bundle and a Full-prompt greedy trajectory, this module
finds the minimum-Frobenius-norm lm_head update that makes every teacher token
win against the competitors encountered on the query-only teacher path.  The
small convex dual is solved numerically; model parameters are never optimized
with gradients.  The returned B @ A update is fixed for the whole generation
and can be merged into lm_head.weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from transformers import LogitsProcessor, LogitsProcessorList

from eval_teacher_token_adherence_patch import combine_lm_head
from run_thought_patch_qwen import (
    PatchConfig,
    install_lm_head_patch,
    install_patch,
    merge_installed_patches,
    remove_lm_head_patch,
    remove_patch,
)


@dataclass
class BoundaryConstraint:
    position: int
    teacher_id: int
    competitor_id: int
    rhs: float
    numerical_gap: float


def _dtype_ulp(value: float, dtype: torch.dtype) -> float:
    """Return one representable step near value in the injection dtype."""
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        dtype = torch.float32
    scalar = torch.tensor(abs(value), dtype=dtype)
    toward = torch.tensor(float("inf"), dtype=dtype)
    step = float((torch.nextafter(scalar, toward) - scalar).float().item())
    floor = float(torch.finfo(dtype).eps) * max(abs(value), 1.0)
    return max(step, floor, float(torch.finfo(torch.float32).eps))


def _numerical_gap(teacher_logit: float, competitor_logit: float, dtype: torch.dtype) -> float:
    return max(_dtype_ulp(teacher_logit, dtype), _dtype_ulp(competitor_logit, dtype))


@torch.no_grad()
def teacher_path_logits_hidden(
    model,
    prefix_ids: list[int],
    teacher_ids: list[int],
    device: str,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and final hidden states that predict exact teacher IDs."""
    if not teacher_ids:
        return (
            torch.empty(0, model.config.vocab_size, dtype=torch.float32),
            torch.empty(0, model.config.hidden_size, dtype=torch.float32),
        )
    if len(teacher_ids) >= max_tokens:
        raise ValueError("teacher trajectory must leave room for at least one prefix token")
    kept_prefix = prefix_ids[-(max_tokens - len(teacher_ids)) :]
    if not kept_prefix:
        raise ValueError("query prefix is empty")
    ids = torch.tensor([kept_prefix + teacher_ids], device=device, dtype=torch.long)
    mask = torch.ones_like(ids)
    output = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True)
    start = len(kept_prefix) - 1
    end = start + len(teacher_ids)
    return (
        output.logits[0, start:end].detach().float().cpu(),
        output.hidden_states[-1][0, start:end].detach().float().cpu(),
    )


class _ForcedTeacherProcessor(LogitsProcessor):
    def __init__(self, teacher_ids: list[int]):
        self.teacher_ids = teacher_ids
        self.logits: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        position = len(self.logits)
        self.logits.append(scores.detach().float().cpu())
        forced = torch.full_like(scores, torch.finfo(scores.dtype).min)
        forced[:, self.teacher_ids[position]] = 0
        return forced


@torch.no_grad()
def cached_teacher_path_logits_hidden(
    model,
    prefix_ids: list[int],
    teacher_ids: list[int],
    device: str,
    max_tokens: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect the exact cache-enabled path used by model.generate.

    A logits processor forces the Full-prompt teacher IDs while preserving the
    unmodified scores and lm_head inputs seen at each generation step.
    """
    if not teacher_ids:
        return (
            torch.empty(0, model.config.vocab_size, dtype=torch.float32),
            torch.empty(0, model.config.hidden_size, dtype=torch.float32),
        )
    if len(prefix_ids) > max_tokens:
        prefix_ids = prefix_ids[-max_tokens:]
    ids = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    mask = torch.ones_like(ids)
    processor = _ForcedTeacherProcessor(teacher_ids)
    hidden: list[torch.Tensor] = []

    def capture_hidden(_module, inputs, _output):
        hidden.append(inputs[0][0, -1].detach().float().cpu())

    handle = model.lm_head.register_forward_hook(capture_hidden)
    try:
        model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=len(teacher_ids),
            do_sample=False,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_id,
            logits_processor=LogitsProcessorList([processor]),
        )
    finally:
        handle.remove()
    if len(processor.logits) != len(teacher_ids) or len(hidden) != len(teacher_ids):
        raise RuntimeError(
            f"cached teacher path length mismatch: logits={len(processor.logits)}, "
            f"hidden={len(hidden)}, teacher={len(teacher_ids)}"
        )
    return torch.cat(processor.logits, dim=0), torch.stack(hidden, dim=0)


def _constraint_gram(constraints: list[BoundaryConstraint], hidden: torch.Tensor) -> torch.Tensor:
    positions = torch.tensor([item.position for item in constraints], dtype=torch.long)
    h = hidden[positions].double()
    h_gram = h @ h.t()
    k = len(constraints)
    token_gram = torch.zeros(k, k, dtype=torch.float64)
    for i, left in enumerate(constraints):
        for j, right in enumerate(constraints):
            token_gram[i, j] = (
                int(left.teacher_id == right.teacher_id)
                - int(left.teacher_id == right.competitor_id)
                - int(left.competitor_id == right.teacher_id)
                + int(left.competitor_id == right.competitor_id)
            )
    gram = token_gram * h_gram
    return 0.5 * (gram + gram.t())


def solve_minimum_norm_dual(
    constraints: list[BoundaryConstraint],
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve max r^T alpha - 1/2 alpha^T G alpha, alpha >= 0."""
    if not constraints:
        return torch.empty(0, dtype=torch.float64), {"iterations": 0, "kkt_residual": 0.0}
    gram = _constraint_gram(constraints, hidden)
    rhs = torch.tensor([item.rhs for item in constraints], dtype=torch.float64)
    g = gram.numpy()
    r = rhs.numpy()

    def objective(alpha: np.ndarray) -> float:
        return float(0.5 * alpha @ g @ alpha - r @ alpha)

    def gradient(alpha: np.ndarray) -> np.ndarray:
        return g @ alpha - r

    scale = max(float(np.max(np.abs(r))), float(np.max(np.abs(g))), 1.0)
    solver_tol = np.sqrt(np.finfo(np.float64).eps) * scale
    result = minimize(
        objective,
        np.zeros(len(constraints), dtype=np.float64),
        jac=gradient,
        bounds=[(0.0, None)] * len(constraints),
        method="L-BFGS-B",
        options={
            "ftol": np.finfo(np.float64).eps,
            "gtol": solver_tol,
            "maxiter": max(100, 20 * len(constraints)),
            "maxls": 40,
        },
    )
    alpha = torch.from_numpy(np.maximum(result.x, 0.0)).double()
    grad = gram @ alpha - rhs
    active = alpha > np.sqrt(np.finfo(np.float64).eps)
    stationarity = float(grad[active].abs().max().item()) if active.any() else 0.0
    dual_feasibility = float((-grad[~active]).clamp_min(0).max().item()) if (~active).any() else 0.0
    kkt_residual = max(stationarity, dual_feasibility)
    if not result.success and kkt_residual > 100.0 * solver_tol:
        raise RuntimeError(f"auto-margin dual solve failed: {result.message}; KKT={kkt_residual:.3e}")
    return alpha, {
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "kkt_residual": kkt_residual,
        "solver_tolerance": solver_tol,
        "solver_message": str(result.message),
    }


def factors_from_dual(
    constraints: list[BoundaryConstraint],
    hidden: torch.Tensor,
    alpha: torch.Tensor,
    vocab_size: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, int]:
    threshold = np.sqrt(np.finfo(np.float64).eps)
    active = [index for index, value in enumerate(alpha.tolist()) if value > threshold]
    if not active:
        return None, 0
    a = torch.stack([hidden[constraints[index].position].float() for index in active], dim=0)
    b = torch.zeros(vocab_size, len(active), dtype=torch.float32)
    for column, index in enumerate(active):
        item = constraints[index]
        weight = float(alpha[index].item())
        b[item.teacher_id, column] += weight
        b[item.competitor_id, column] -= weight
    return (a.contiguous(), b.contiguous()), len(active)


def _bundle_with_anchor(base_bundle: dict, anchor) -> dict:
    combined = combine_lm_head(
        base_bundle.get("lm_head_patch"),
        float(base_bundle.get("lm_scale", 1.0)),
        anchor,
        1.0,
    )
    return {
        "patches": base_bundle.get("patches", {}),
        "lm_head_patch": combined,
        "lm_scale": 1.0,
    }


def _install_bundle(model, bundle: dict, cfg: PatchConfig, dtype: torch.dtype) -> None:
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


@torch.no_grad()
def merge_bundle_into_model(model, bundle: dict, cfg: PatchConfig, dtype: torch.dtype) -> dict[str, Any]:
    """Permanently merge a bundle, safely untying the output head if needed."""
    remove_patch(model)
    remove_lm_head_patch(model)
    input_embeddings = model.get_input_embeddings()
    tied = model.lm_head.weight.data_ptr() == input_embeddings.weight.data_ptr()
    if tied:
        old_head = model.lm_head
        new_head = nn.Linear(old_head.in_features, old_head.out_features, bias=old_head.bias is not None)
        new_head = new_head.to(device=old_head.weight.device, dtype=old_head.weight.dtype)
        new_head.weight = nn.Parameter(old_head.weight.detach().clone(), requires_grad=False)
        if old_head.bias is not None:
            new_head.bias = nn.Parameter(old_head.bias.detach().clone(), requires_grad=False)
        model.lm_head = new_head
        model.config.tie_word_embeddings = False
    _install_bundle(model, bundle, cfg, dtype)
    merged_modules = merge_installed_patches(model)
    return {
        "untied_lm_head": tied,
        "merged_modules": merged_modules,
        "input_output_weights_still_tied": (
            model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        ),
    }


@torch.no_grad()
def fit_auto_margin_teacher_anchor(
    model,
    cfg: PatchConfig,
    query_prefix_ids: list[int],
    teacher_ids: list[int],
    base_bundle: dict,
    dtype: torch.dtype,
) -> tuple[dict, dict[str, Any]]:
    """Return a fixed bundle whose greedy query path follows teacher_ids.

    Each round adds only the currently violated teacher-vs-argmax boundaries.
    If BF16 injection changes a boundary through rounding, the same constraint
    is tightened by the observed deficit and solved again.
    """
    if not teacher_ids:
        return _bundle_with_anchor(base_bundle, None), {
            "converged": True,
            "teacher_tokens": 0,
            "constraints": [],
            "active_constraints": 0,
            "rounds": [],
        }
    eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[-1]
    if eos_token_id is None:
        eos_token_id = model.config.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[-1]
    if eos_token_id is None:
        raise ValueError("model does not define an EOS token")

    _install_bundle(model, base_bundle, cfg, dtype)
    baseline_logits, hidden = cached_teacher_path_logits_hidden(
        model,
        query_prefix_ids,
        teacher_ids,
        cfg.device,
        cfg.max_eval_prompt_tokens,
        int(eos_token_id),
    )
    remove_patch(model)
    remove_lm_head_patch(model)

    constraints_by_key: dict[tuple[int, int, int], BoundaryConstraint] = {}
    anchor = None
    rounds: list[dict[str, Any]] = []
    solver_meta: dict[str, Any] = {"iterations": 0, "kkt_residual": 0.0}
    active_count = 0
    converged = False

    # At most one genuinely new competitor can be introduced per position per
    # round; this cap is a termination guard, not a method hyperparameter.
    for round_index in range(len(teacher_ids) + 2):
        candidate_bundle = _bundle_with_anchor(base_bundle, anchor)
        _install_bundle(model, candidate_bundle, cfg, dtype)
        current_logits, _ = cached_teacher_path_logits_hidden(
            model,
            query_prefix_ids,
            teacher_ids,
            cfg.device,
            cfg.max_eval_prompt_tokens,
            int(eos_token_id),
        )
        remove_patch(model)
        remove_lm_head_patch(model)

        added = 0
        tightened = 0
        violations = []
        for position, teacher_id in enumerate(teacher_ids):
            competitor_id = int(torch.argmax(current_logits[position]).item())
            if competitor_id == int(teacher_id):
                continue
            teacher_logit = float(current_logits[position, teacher_id].item())
            competitor_logit = float(current_logits[position, competitor_id].item())
            gap = _numerical_gap(teacher_logit, competitor_logit, dtype)
            key = (position, int(teacher_id), competitor_id)
            baseline_rhs = float(
                baseline_logits[position, competitor_id].item() - baseline_logits[position, teacher_id].item() + gap
            )
            observed_deficit = competitor_logit - teacher_logit + gap
            if key not in constraints_by_key:
                constraints_by_key[key] = BoundaryConstraint(
                    position=position,
                    teacher_id=int(teacher_id),
                    competitor_id=competitor_id,
                    rhs=baseline_rhs,
                    numerical_gap=gap,
                )
                added += 1
            else:
                item = constraints_by_key[key]
                strengthened = max(item.rhs, item.rhs + observed_deficit)
                if strengthened > item.rhs:
                    item.rhs = strengthened
                    item.numerical_gap = max(item.numerical_gap, gap)
                    tightened += 1
            violations.append(
                {
                    "position": position,
                    "teacher_id": int(teacher_id),
                    "competitor_id": competitor_id,
                    "observed_gap_teacher_minus_competitor": teacher_logit - competitor_logit,
                }
            )

        rounds.append(
            {
                "round": round_index,
                "violations": len(violations),
                "added_constraints": added,
                "tightened_constraints": tightened,
            }
        )
        if not violations:
            converged = True
            break
        if added == 0 and tightened == 0:
            break

        constraints = list(constraints_by_key.values())
        alpha, solver_meta = solve_minimum_norm_dual(constraints, hidden)
        anchor, active_count = factors_from_dual(constraints, hidden, alpha, model.config.vocab_size)

    final_bundle = _bundle_with_anchor(base_bundle, anchor)
    constraints = list(constraints_by_key.values())
    metadata = {
        "converged": converged,
        "teacher_tokens": len(teacher_ids),
        "constraint_count": len(constraints),
        "active_constraints": active_count,
        "effective_rank_upper_bound": active_count,
        "anchor_parameter_count": active_count * (model.config.hidden_size + model.config.vocab_size),
        "rounds": rounds,
        "solver": solver_meta,
        "constraints": [item.__dict__ for item in constraints],
        "merge_rule": "lm_head.weight += B @ A",
    }
    return final_bundle, metadata
