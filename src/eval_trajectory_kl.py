import argparse, json, math, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL = [
    "如何把一个长prompt压缩成可复用的模型参数？",
    "给我一个排查GPU显存爆掉的流程。",
    "怎样评估一个LLM Agent是否真的有用？",
    "请解释LoRA为什么能用很少参数微调模型。",
    "我需要一个每天30分钟的数学复习计划。",
]


def chat(tok, sys, q):
    messages = []
    if sys:
        messages.append({"role": "system", "content": sys})
    messages.append({"role": "user", "content": q})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def add(linear, pair, scale=1.0):
    if pair is None:
        return
    a, b = pair
    for s in range(0, b.shape[0], 8192):
        e = min(s + 8192, b.shape[0])
        linear.weight.data[s:e].add_(scale * (b[s:e].float() @ a.float()).to(linear.weight.device, linear.weight.dtype))


def merge(model, bundle, scale_override=None):
    for li, lps in bundle.get("patches", {}).items():
        layer = model.model.layers[int(li)]
        for name, pair in lps.items():
            if name == "attn_o_proj":
                target = layer.self_attn.o_proj
            elif name == "attn_q_proj":
                target = layer.self_attn.q_proj
            elif name == "attn_v_proj":
                target = layer.self_attn.v_proj
            else:
                target = getattr(layer.mlp, name)
            add(target, pair, 1.0)
    add(model.lm_head, bundle.get("lm_head_patch"), float(scale_override if scale_override is not None else bundle.get("lm_scale", 1.0)))


@torch.no_grad()
def generate_ids(model, tok, text, max_new):
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return out[0, inp["input_ids"].shape[1] :].detach().cpu()


@torch.no_grad()
def step_logits(model, ids):
    attn = torch.ones_like(ids)
    try:
        out = model(input_ids=ids, attention_mask=attn, use_cache=False, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=ids, attention_mask=attn, use_cache=False)
    return out.logits[:, -1, :].float().cpu()


def kl_logits(p, q):
    lp = F.log_softmax(p, -1)
    lq = F.log_softmax(q, -1)
    return float((lp.exp() * (lp - lq)).sum(-1).item())


def top_info(logits, tok, target_id):
    probs = torch.softmax(logits[0], dim=-1)
    topv, topi = torch.topk(probs, 5)
    return {
        "target": tok.decode([int(target_id)], skip_special_tokens=False),
        "target_prob": float(probs[int(target_id)].item()),
        "top": [
            {"token": tok.decode([int(i)], skip_special_tokens=False), "id": int(i), "prob": float(v)}
            for v, i in zip(topv, topi)
        ],
    }


def summarize(values):
    if not values:
        return {}
    sv = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "max": max(values),
        "p50": sv[len(sv) // 2],
        "p90": sv[min(len(sv) - 1, math.ceil(len(sv) * 0.9) - 1)],
        "first": values[0],
        "n": len(values),
    }


ap = argparse.ArgumentParser()
ap.add_argument("--patch", required=True)
ap.add_argument("--system-prompt-file", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--scale", type=float, default=None)
ap.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-new", type=int, default=80)
ap.add_argument("--max-steps", type=int, default=80)
args = ap.parse_args()

system = open(args.system_prompt_file, encoding="utf-8").read()
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map={"": args.device},
    trust_remote_code=True,
    local_files_only=True,
)
model.eval()

baseline_records = []
for q in EVAL:
    full_text = chat(tok, system, q)
    query_text = chat(tok, "", q)
    full_prompt_ids = tok(full_text, return_tensors="pt").input_ids[0]
    query_prompt_ids = tok(query_text, return_tensors="pt").input_ids[0]
    answer_ids = generate_ids(model, tok, full_text, args.max_new)[: args.max_steps]
    full_logits_by_pos = []
    prefix = []
    for pos, target in enumerate(answer_ids.tolist()):
        full_ids = torch.tensor([full_prompt_ids.tolist() + prefix], device=model.device)
        full_logits_by_pos.append(step_logits(model, full_ids))
        prefix.append(target)
    baseline_records.append((q, full_prompt_ids, query_prompt_ids, answer_ids, tok.decode(answer_ids, skip_special_tokens=True), full_logits_by_pos))

bundle = torch.load(args.patch, map_location="cpu", weights_only=False)
merge(model, bundle, args.scale)

all_values = []
results = []
for q, full_prompt_ids, query_prompt_ids, answer_ids, baseline_text, full_logits_by_pos in baseline_records:
    per = []
    prefix = []
    for pos, target in enumerate(answer_ids.tolist()):
        patch_ids = torch.tensor([query_prompt_ids.tolist() + prefix], device=model.device)
        fl = full_logits_by_pos[pos]
        ql = step_logits(model, patch_ids)
        val = kl_logits(fl, ql)
        all_values.append(val)
        item = {
            "pos": pos,
            "token_id": int(target),
            "token": tok.decode([int(target)], skip_special_tokens=False),
            "kl": val,
        }
        if pos < 8 or val > 1.0 or item["token"].strip() in {"<think>", "</think>", "<answer>", "</answer>"}:
            item["full"] = top_info(fl, tok, target)
            item["patched"] = top_info(ql, tok, target)
        per.append(item)
        prefix.append(target)
    results.append({"query": q, "baseline": baseline_text, "summary": summarize([x["kl"] for x in per]), "kl_by_pos": per})

report = {"overall": summarize(all_values), "results": results, "patch": args.patch, "scale": args.scale}
open(args.out, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps({"out": args.out, "overall": report["overall"]}, ensure_ascii=False, indent=2))
