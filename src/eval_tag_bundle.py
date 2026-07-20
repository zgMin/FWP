import argparse, json, re, difflib, torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL = [
    "如何把一个长prompt压缩成可复用的模型参数？",
    "给我一个排查GPU显存爆掉的流程。",
    "怎样评估一个LLM Agent是否真的有用？",
    "请解释LoRA为什么能用很少参数微调模型。",
    "我需要一个每天30分钟的数学复习计划。",
]


def chat(tok, sys, q):
    m = []
    if sys:
        m.append({"role": "system", "content": sys})
    m.append({"role": "user", "content": q})
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


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


def gen(model, tok, text, max_new):
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
    mt = sum(s for *_, s in m.get_matching_blocks())
    p = mt / max(len(b), 1)
    r = mt / max(len(a), 1)
    return 0 if p + r == 0 else 2 * p * r / (p + r)


def char_bleu(a, b):
    r = [c for c in a if not c.isspace()]
    c = [x for x in b if not x.isspace()]
    return 0 if not r or not c else float(sentence_bleu([r], c, smoothing_function=SmoothingFunction().method1))


def tag_score(s):
    to = s.find("<think>")
    tc = s.find("</think>")
    ao = s.find("<answer>")
    ac = s.find("</answer>")
    checks = [
        to == 0,
        tc > to,
        ao > tc,
        ac > ao,
        bool(s[to + 7 : tc].strip()) if to >= 0 and tc > to else False,
        bool(s[ao + 8 : ac].strip()) if ao >= 0 and ac > ao else False,
    ]
    return sum(checks) / len(checks), checks


ap = argparse.ArgumentParser()
ap.add_argument("--patch", required=True)
ap.add_argument("--scale", type=float, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--system-prompt-file", required=True)
ap.add_argument("--model", default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--max-new", type=int, default=96)
args = ap.parse_args()

system = open(args.system_prompt_file, encoding="utf-8").read()
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
tok.pad_token = tok.pad_token or tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map={"": args.device},
    trust_remote_code=True,
    local_files_only=True,
)
base.eval()

bas = []
for q in EVAL:
    full = chat(tok, system, q)
    bt = gen(base, tok, full, args.max_new)
    bas.append((q, logits(base, tok, full), bt, tag_score(bt)[0]))

bundle = torch.load(args.patch, map_location="cpu", weights_only=False)
merge(base, bundle, args.scale)
res = []
for q, fl, bt, bfmt in bas:
    qo = chat(tok, "", q)
    pt = gen(base, tok, qo, args.max_new)
    ql = logits(base, tok, qo)
    sf, checks = tag_score(pt)
    res.append(
        {
            "query": q,
            "kl": kl(fl, ql),
            "char_bleu": char_bleu(bt, pt),
            "char_f1": char_f1(bt, pt),
            "tag_score": sf,
            "tag_checks": checks,
            "baseline_tag_score": bfmt,
            "tokdiff": len(tok.encode(pt, add_special_tokens=False)) - len(tok.encode(bt, add_special_tokens=False)),
            "baseline": bt,
            "patched": pt,
        }
    )

avg = {k: sum(x[k] for x in res) / len(res) for k in ["kl", "char_bleu", "char_f1", "tag_score", "baseline_tag_score", "tokdiff"]}
open(args.out, "w", encoding="utf-8").write(json.dumps({"scale": args.scale, "averages": avg, "results": res}, ensure_ascii=False, indent=2))
print(json.dumps({"scale": args.scale, "averages": avg, "out": args.out}, ensure_ascii=False, indent=2))
