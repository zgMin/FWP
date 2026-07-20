import argparse, difflib, json, re, torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
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
        delta = (b[s:e].float() @ a.float()).to(linear.weight.device, linear.weight.dtype)
        linear.weight.data[s:e].add_(scale * delta)


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
    scale = float(scale_override if scale_override is not None else bundle.get("lm_scale", 1.0))
    add(model.lm_head, bundle.get("lm_head_patch"), scale)


def generate(model, tok, text, max_new):
    inp = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, inp["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def logits(model, tok, text):
    inp = tok(text, return_tensors="pt").to(model.device)
    try:
        out = model(**inp, use_cache=False, logits_to_keep=1)
    except TypeError:
        out = model(**inp, use_cache=False)
    return out.logits[:, -1, :].float().cpu()


def kl(p, q):
    lp = F.log_softmax(p, -1)
    lq = F.log_softmax(q, -1)
    return float((lp.exp() * (lp - lq)).sum(-1).item())


def char_f1(a, b):
    m = difflib.SequenceMatcher(a=list(a), b=list(b))
    matches = sum(size for *_, size in m.get_matching_blocks())
    precision = matches / max(len(b), 1)
    recall = matches / max(len(a), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def char_bleu(a, b):
    ref = [c for c in a if not c.isspace()]
    cand = [c for c in b if not c.isspace()]
    return 0.0 if not ref or not cand else float(sentence_bleu([ref], cand, smoothing_function=SmoothingFunction().method1))


def format_score(text, mode):
    if mode == "conclusion3":
        checks = [
            bool(re.match(r"^\s*结论", text)),
            "要点" in text,
            bool(re.search(r"(?:^|[\n\s|])(?:1[\.、．）：:]|要点\s*1)", text)),
            bool(re.search(r"(?:^|[\n\s|])(?:2[\.、．）：:]|要点\s*2)", text)),
            bool(re.search(r"(?:^|[\n\s|])(?:3[\.、．）：:]|要点\s*3)", text)),
        ]
        return sum(checks) / len(checks)
    if mode == "tags":
        to = text.find("<think>")
        tc = text.find("</think>")
        ao = text.find("<answer>")
        ac = text.find("</answer>")
        checks = [
            to == 0,
            tc > to,
            ao > tc,
            ac > ao,
            bool(text[to + 7 : tc].strip()) if to >= 0 and tc > to else False,
            bool(text[ao + 8 : ac].strip()) if ao >= 0 and ac > ao else False,
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
    if mode == "tags_strict":
        return 1.0 if re.match(r"^<think>.+?</think>\s*<answer>.+?</answer>\s*$", text, re.S) else 0.0
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
    return 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--patch", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--system-prompt-file", required=True)
ap.add_argument("--format-mode", default="none", choices=["none", "conclusion3", "tags", "repeat_tags", "tags_strict", "json"])
ap.add_argument("--scale", type=float, default=None)
ap.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-new", type=int, default=80)
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

baseline = []
for q in EVAL:
    full = chat(tok, system, q)
    baseline.append((q, logits(model, tok, full), generate(model, tok, full, args.max_new)))

bundle = torch.load(args.patch, map_location="cpu", weights_only=False)
merge(model, bundle, args.scale)

results = []
for q, full_logits, baseline_text in baseline:
    query_only = chat(tok, "", q)
    patched_text = generate(model, tok, query_only, args.max_new)
    patched_logits = logits(model, tok, query_only)
    results.append(
        {
            "query": q,
            "kl": kl(full_logits, patched_logits),
            "char_bleu": char_bleu(baseline_text, patched_text),
            "char_f1": char_f1(baseline_text, patched_text),
            "format_score": format_score(patched_text, args.format_mode),
            "baseline_format_score": format_score(baseline_text, args.format_mode),
            "tokdiff": len(tok.encode(patched_text, add_special_tokens=False)) - len(tok.encode(baseline_text, add_special_tokens=False)),
            "baseline": baseline_text,
            "patched": patched_text,
        }
    )

avg = {k: sum(r[k] for r in results) / len(results) for k in ["kl", "char_bleu", "char_f1", "format_score", "baseline_format_score", "tokdiff"]}
open(args.out, "w", encoding="utf-8").write(json.dumps({"averages": avg, "results": results, "scale": args.scale}, ensure_ascii=False, indent=2))
print(json.dumps({"out": args.out, "averages": avg}, ensure_ascii=False, indent=2))
