import argparse,json,torch,difflib,re
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction
SYSTEM="你是一个回答风格固定的助手。请始终用中文回答，结构为：先给一句不超过20字的结论，然后列出三条要点；语气简洁、克制、可执行。"
CALIB=["如何快速判断一个创业想法是否值得继续做？","给我一个学习强化学习的两周计划。","怎样把一篇论文读得更有效率？","解释一下什么是过拟合，并给出避免方法。","我想提升英文写作，应该每天练什么？","如何设计一个可靠的A/B实验？","团队开会总是低效，怎么改善？","请给出一个健身新手的入门建议。","怎样排查线上服务突然变慢的问题？","如何准备一次技术分享？","什么情况下应该使用缓存？","请给出一个阅读源码的步骤。","如何写出更清晰的产品需求文档？","怎么判断一个模型评测是否可信？","我想减少拖延，给我可执行建议。","解释一下数据库索引的作用。","如何给一个新项目设计里程碑？","怎样判断一个开源库是否适合生产使用？","请给出一个周末整理房间的计划。","如何降低一次代码重构的风险？","解释一下什么是梯度爆炸。","怎样写一封清晰的工作周报？","如何准备机器学习岗位面试？","请给出一个减少手机使用时间的方法。","如何评估一个数据集的质量？","服务接口经常超时，应该怎么排查？","怎样给初学者解释大语言模型？","如何安排一次有效的一对一沟通？","请给出一个学习Linux命令的路线。","怎样判断一个需求是否值得做？","如何提高代码评审的效率？","解释一下什么是缓存穿透。"]
EVAL=["如何把一个长prompt压缩成可复用的模型参数？","给我一个排查GPU显存爆掉的流程。","怎样评估一个LLM Agent是否真的有用？","请解释LoRA为什么能用很少参数微调模型。","我需要一个每天30分钟的数学复习计划。"]
def chat(tok,sys,q):
 m=[]
 if sys:m.append({"role":"system","content":sys})
 m.append({"role":"user","content":q})
 return tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
def add(linear,pair,scale=1.0):
 if pair is None:return
 a,b=pair
 for s in range(0,b.shape[0],8192):
  e=min(s+8192,b.shape[0])
  linear.weight.data[s:e].add_(scale*(b[s:e].float()@a.float()).to(linear.weight.device,linear.weight.dtype))
def merge(model,bundle):
 for li,lps in bundle.get("patches",{}).items():
  layer=model.model.layers[int(li)]
  for name,pair in lps.items():
   if name=="attn_o_proj": target=layer.self_attn.o_proj
   elif name=="attn_q_proj": target=layer.self_attn.q_proj
   elif name=="attn_v_proj": target=layer.self_attn.v_proj
   else: target=getattr(layer.mlp,name)
   add(target,pair,1.0)
 add(model.lm_head,bundle.get("lm_head_patch"),float(bundle.get("lm_scale",1.0)))
def gen_ids(model,tok,text,max_new):
 inp=tok(text,return_tensors="pt").to(model.device)
 out=model.generate(**inp,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
 return inp["input_ids"][0].tolist(), out[0,inp["input_ids"].shape[1]:].tolist()
def gen_text(model,tok,text,max_new):
 _,ids=gen_ids(model,tok,text,max_new); return tok.decode(ids,skip_special_tokens=True).strip()
def logits_hidden_for_prefix(model,ids):
 x=torch.tensor([ids],device=model.device); m=torch.ones_like(x)
 store={}
 def grab(_module,_inputs,output):
  store["h"]=output[:, -1, :].detach().float().cpu()
 handle=model.model.norm.register_forward_hook(grab)
 try:
  try: out=model(input_ids=x,attention_mask=m,use_cache=False,logits_to_keep=1)
  except TypeError: out=model(input_ids=x,attention_mask=m,use_cache=False)
 finally:
  handle.remove()
 return out.logits[:,-1,:].float().cpu(), store["h"]
def solve(a,t,ridge):
 gram=a.t()@a; right=torch.linalg.solve(gram+ridge*torch.eye(gram.shape[0]),a.t()); return t,right
def compress(b,a,rank):
 qb,rb=torch.linalg.qr(b.float(),mode="reduced"); qa,ra=torch.linalg.qr(a.float().t(),mode="reduced"); core=rb@ra.t(); u,s,vh=torch.linalg.svd(core.float(),full_matrices=False); r=min(rank,s.numel()); return (vh[:r,:]@qa.t()).contiguous(), ((qb@u[:,:r])*s[:r].unsqueeze(0)).contiguous()
def cb(a,b):
 r=[c for c in a if not c.isspace()]; c=[x for x in b if not x.isspace()]; return 0 if not r or not c else float(sentence_bleu([r],c,smoothing_function=SmoothingFunction().method1))
def cf(a,b):
 m=difflib.SequenceMatcher(a=list(a),b=list(b)); mt=sum(s for *_,s in m.get_matching_blocks()); p=mt/max(len(b),1); r=mt/max(len(a),1); return 0 if p+r==0 else 2*p*r/(p+r)
def fmt(s):
 return sum([bool(re.match(r"^\s*结论",s)),"要点" in s,bool(re.search(r"(?:^|[\n\s|])(?:1[\.、．）：:]|要点\s*1)",s)),bool(re.search(r"(?:^|[\n\s|])(?:2[\.、．）：:]|要点\s*2)",s)),bool(re.search(r"(?:^|[\n\s|])(?:3[\.、．）：:]|要点\s*3)",s))])/5
def kl(p,q):
 lp=F.log_softmax(p,-1); lq=F.log_softmax(q,-1); return float((lp.exp()*(lp-lq)).sum(-1).item())
ap=argparse.ArgumentParser(); ap.add_argument("--patch",default="/root/zgm/thoughtpatch_qwen25/outputs/thought_patch_attn_o_format_teacher_lm64_scale1.pt"); ap.add_argument("--out",default="/root/zgm/thoughtpatch_qwen25/outputs/report_anti_repeat.json"); ap.add_argument("--patch-out",default="/root/zgm/thoughtpatch_qwen25/outputs/thought_patch_anti_repeat.pt"); ap.add_argument("--model",default="/root/zgm/e2pse/models/Qwen2.5-3B-Instruct"); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--rank",type=int,default=32); ap.add_argument("--ridge",type=float,default=0.1); ap.add_argument("--penalty",type=float,default=6.0); ap.add_argument("--max-new",type=int,default=64); ap.add_argument("--calib-limit",type=int,default=12); args=ap.parse_args()
tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token
model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map={"":args.device},trust_remote_code=True,local_files_only=True); model.eval(); bundle=torch.load(args.patch,map_location="cpu",weights_only=False); merge(model,bundle)
H=[]; T=[]; bad=[]
for q in CALIB[:args.calib_limit]:
 prompt_ids, ans=gen_ids(model,tok,chat(tok,"",q),args.max_new); seen_y=0
 for i,t in enumerate(ans):
  piece=tok.decode([t],skip_special_tokens=False)
  is_bad=(i>0 and t==ans[i-1]) or (piece.strip() in ["1","１"] and i>0) or (piece in ["要点"] and seen_y>=1)
  if piece=="要点": seen_y+=1
  if is_bad:
   _,h=logits_hidden_for_prefix(model,prompt_ids+ans[:i]); col=torch.zeros(model.config.vocab_size); col[t]=-args.penalty; H.append(h[0]); T.append(col); bad.append(piece)
print("bad cols",len(H),bad[:30])
if H:
 a=torch.stack(H,dim=1).float(); t=torch.stack(T,dim=1).float(); bfull,afull=solve(a,t,args.ridge); A,B=compress(bfull,afull,args.rank)
 old=bundle.get("lm_head_patch"); bundle["lm_head_patch"]=(torch.cat([old[0],A.cpu()],dim=0),torch.cat([old[1]*float(bundle.get("lm_scale",1.0)),B.cpu()],dim=1)) if old is not None else (A.cpu(),B.cpu()); bundle["lm_scale"]=1.0; bundle["anti_repeat"]={"cols":len(H),"rank":args.rank,"penalty":args.penalty}; add(model.lm_head,(A,B),1.0)
torch.save(bundle,args.patch_out)
# baseline full prompt model separate
base=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,device_map={"":args.device},trust_remote_code=True,local_files_only=True); base.eval()
res=[]
for q in EVAL:
 full=chat(tok,SYSTEM,q); qo=chat(tok,"",q); bl=gen_text(base,tok,full,args.max_new); pl=gen_text(model,tok,qo,args.max_new); fl=logits_hidden_for_prefix(base,tok(full,add_special_tokens=False).input_ids)[0]; ql=logits_hidden_for_prefix(model,tok(qo,add_special_tokens=False).input_ids)[0]
 res.append({"query":q,"kl":kl(fl,ql),"char_bleu":cb(bl,pl),"char_f1":cf(bl,pl),"fmt":fmt(pl),"tokdiff":len(tok.encode(pl,add_special_tokens=False))-len(tok.encode(bl,add_special_tokens=False)),"baseline":bl,"patched":pl})
avg={k:sum(x[k] for x in res)/len(res) for k in ["kl","char_bleu","char_f1","fmt","tokdiff"]}
open(args.out,"w",encoding="utf-8").write(json.dumps({"averages":avg,"results":res,"bad_cols":len(H),"patch":args.patch_out},ensure_ascii=False,indent=2)); print(json.dumps({"averages":avg,"bad_cols":len(H),"out":args.out,"patch":args.patch_out},ensure_ascii=False,indent=2))
