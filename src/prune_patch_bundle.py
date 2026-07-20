import argparse, torch
ap=argparse.ArgumentParser(); ap.add_argument("--src",required=True); ap.add_argument("--dst",required=True); ap.add_argument("--keep",required=True); ap.add_argument("--drop-lm",action="store_true")
args=ap.parse_args(); b=torch.load(args.src,map_location="cpu",weights_only=False)
keep=set()
for part in args.keep.split(","):
    if "-" in part:
        a,c=map(int,part.split("-")); keep.update(range(a,c+1))
    elif part.strip(): keep.add(int(part))
b["patches"]={k:v for k,v in b.get("patches",{}).items() if int(k) in keep}
if args.drop_lm: b["lm_head_patch"]=None
b["layer_selection"]={"keep":sorted(keep),"drop_lm":args.drop_lm,"source":args.src}
torch.save(b,args.dst); print(args.dst, "layers", len(b["patches"]), "drop_lm", args.drop_lm)
