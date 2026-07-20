#!/usr/bin/env python3
"""Merge a saved thought-patch bundle into a Hugging Face causal LM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_delta(linear, pair, scale=1.0):
    if pair is None:
        return
    a, b = pair
    delta = (b.float() @ a.float()).to(device=linear.weight.device, dtype=linear.weight.dtype)
    linear.weight.data.add_(scale * delta)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
    p.add_argument("--patch", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": args.device},
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    bundle = torch.load(args.patch, map_location="cpu", weights_only=False)
    patches = bundle.get("patches", {})
    for layer_idx, layer_patches in patches.items():
        layer = model.model.layers[int(layer_idx)]
        for proj_name, pair in layer_patches.items():
            if proj_name == "attn_o_proj":
                add_delta(layer.self_attn.o_proj, pair)
            else:
                add_delta(getattr(layer.mlp, proj_name), pair)
    add_delta(model.lm_head, bundle.get("lm_head_patch"), scale=float(bundle.get("lm_scale", 1.0)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"saved merged model to {out}")


if __name__ == "__main__":
    main()
