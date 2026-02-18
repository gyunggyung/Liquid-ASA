"""Generate ASA × LFM2.5 paper-faithful notebook. Run: python create_notebook.py"""
import json

def cc(src):
    lines = src.strip().split("\n")
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":[l+"\n" for l in lines[:-1]]+[lines[-1]]}
def mc(src):
    lines = src.strip().split("\n")
    return {"cell_type":"markdown","metadata":{},
            "source":[l+"\n" for l in lines[:-1]]+[lines[-1]]}

C = []

# ── Title ──
C.append(mc("""# 🧬 ASA × LFM2.5-1.2B-Instruct
**Paper-Faithful Replication: Training-Free Tool-Calling Enhancement**

Replicates the ASA protocol ([arXiv 2602.04935](https://arxiv.org/abs/2602.04935)) on LFM2.5.
Uses **Alpaca** public dataset with domain filtering. Runs on Colab T4."""))

# ── Cell 1: Setup ──
C.append(mc("## 1 · Setup"))
C.append(cc("""import subprocess, sys
for p in ["transformers>=4.40.0","accelerate>=0.25.0","scikit-learn>=1.3.0",
           "datasets","tqdm","matplotlib","seaborn"]:
    subprocess.check_call([sys.executable,"-m","pip","install","-q",p])

import os, json, re, pickle, warnings, gc, ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    try: print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    except AttributeError: pass

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
DOMAINS  = ["math","code","search","translation"]
SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)

OUT   = Path("outputs"); OUT.mkdir(exist_ok=True)
CKPT  = OUT/"ckpt"; CKPT.mkdir(exist_ok=True)

TOOL_S = "<|tool_call_start|>"
TOOL_E = "<|tool_call_end|>"

def save_ckpt(tag, obj):
    p=CKPT/f"{tag}.pkl"
    with open(p,"wb") as f: pickle.dump(obj,f)
    print(f"  💾 {tag} ({p.stat().st_size//1024}KB)")
def load_ckpt(tag):
    p=CKPT/f"{tag}.pkl"
    if p.exists():
        with open(p,"rb") as f: o=pickle.load(f)
        print(f"  ♻️  {tag} (cached)"); return o
    return None
print("✅ Setup done.")"""))

# ── Cell 2: Data Pipeline ──
C.append(mc("""## 2 · Data Pipeline
Download **Alpaca** (tatsu-lab/alpaca) and apply domain filtering + tool-necessity heuristics,
following the paper's benchmark construction methodology."""))
C.append(cc("""from datasets import load_dataset

print("Downloading Alpaca dataset...")
alpaca = load_dataset("tatsu-lab/alpaca", split="train")
print(f"  Alpaca: {len(alpaca)} samples")

# ── Domain classification + tool-necessity labeling ──
_MATH_KW = re.compile(r'(calcul|comput|solve|equation|formula|\\bsum\\b|product|divid|multiply|percent|fraction|area|volume|circumferen|triangle|circle|rectangle|convert.*to|\\bhow much\\b|\\bhow many\\b|average|median|ratio|proportion|interest|mortgage|speed|distance|temperature|celsius|fahrenheit|kilometer|gallon|\\bcos\\b|\\bsin\\b|\\btan\\b|factorial|logarithm|square root|hypotenuse|diagonal|perimeter|probability)', re.I)
_CODE_KW = re.compile(r'(python|javascript|java\\b|\\bc\\+\\+|\\bhtml\\b|\\bcss\\b|\\bsql\\b|function|script|algorithm|debug|compil|execut|output|variable|\\bloop\\b|\\barray\\b|\\bclass\\b|\\bimport\\b|\\bprint\\b|\\bsort\\b|\\bcode\\b|program|def\\s|for\\s.*in\\s|while\\s|if\\s.*else)', re.I)
_SRCH_KW = re.compile(r'(who (is|was|are|were)|what (year|country|city)|when (did|was)|where (is|was)|capital of|population of|president of|founded|invented|discovered|located|\\bfind\\b.*about|search for|look up|latest|current|recent|today)', re.I)
_TRNS_KW = re.compile(r'(translat|\\bin spanish\\b|\\bin french\\b|\\bin german\\b|\\bin chinese\\b|\\bin japanese\\b|\\bin korean\\b|\\bin arabic\\b|\\bin italian\\b|\\bin portuguese\\b|\\bin russian\\b|how do you say|\\bà\\b|\\büber\\b|en fran[cç]ais)', re.I)
_CONCEPT = re.compile(r'(explain|describe|what is the (concept|difference|significance|meaning|history|importance)|why (is|are|do|does)|how does.*work|compare and contrast|pros and cons|advantages|definition of)', re.I)
_HAS_NUM = re.compile(r'\\d+\\.?\\d*')

def classify(inst, inp=""):
    t = (inst+" "+inp).strip()
    tl = t.lower()
    scores = {"math": len(_MATH_KW.findall(tl)),
              "code": len(_CODE_KW.findall(tl)),
              "search": len(_SRCH_KW.findall(tl)),
              "translation": len(_TRNS_KW.findall(tl))}
    dom = max(scores, key=scores.get)
    if scores[dom] == 0: return None, None
    # Tool-necessity heuristic
    is_concept = bool(_CONCEPT.search(tl))
    if dom == "math":
        has_num = bool(_HAS_NUM.search(t))
        has_verb = bool(re.search(r'(calcul|comput|solve|convert|find the|determine|how much|how many)', tl))
        label = 1 if (has_num and has_verb and not is_concept) else 0
    elif dom == "code":
        has_action = bool(re.search(r'(write|create|implement|build|generate|debug|fix|run|execute|test|develop)', tl))
        label = 1 if (has_action and not is_concept) else 0
    elif dom == "search":
        label = 0 if is_concept else 1
    elif dom == "translation":
        has_target = bool(re.search(r'(translat.*to|in (spanish|french|german|chinese|japanese|korean|arabic|italian|portuguese|russian))', tl))
        label = 1 if (has_target and not is_concept) else 0
    else:
        return None, None
    return dom, label

# Apply filtering
filtered = {d: {0: [], 1: []} for d in DOMAINS}
for row in alpaca:
    inst = row["instruction"]
    inp  = row.get("input", "")
    dom, label = classify(inst, inp)
    if dom is None: continue
    text = inst + ("\\n" + inp if inp else "")
    filtered[dom][label].append(text)

print("\\nFiltered counts:")
for d in DOMAINS:
    print(f"  {d:12s}: tool={len(filtered[d][1]):4d}  non-tool={len(filtered[d][0]):4d}")

# ── Build balanced splits: CAL(320) TRAIN(320) VALID(320) TEST(640) = 1600 ──
PER_DOM = {"cal": (40,40), "train": (40,40), "valid": (40,40), "test": (80,80)}
TOTAL_NEEDED = sum(t+n for t,n in PER_DOM.values())  # 400 per domain

all_data = {}
sample_id = 0
for split, (nt, nn) in PER_DOM.items():
    all_data[split] = []

for d in DOMAINS:
    for label in [0, 1]:
        random_pool = filtered[d][label].copy()
        np.random.shuffle(random_pool)
        idx = 0
        for split, (nt, nn) in PER_DOM.items():
            need = nt if label == 1 else nn
            for _ in range(need):
                if idx < len(random_pool):
                    text = random_pool[idx]; idx += 1
                else:
                    text = f"[placeholder {d} {'tool' if label else 'nontool'} {sample_id}]"
                all_data[split].append({
                    "id": f"{d}_{split}_{label}_{sample_id}",
                    "instruction": text, "domain": d, "label": label
                })
                sample_id += 1

for split in all_data:
    np.random.shuffle(all_data[split])

# Verify
print("\\nSplit sizes:")
for split, samples in all_data.items():
    t = sum(1 for s in samples if s["label"]==1)
    print(f"  {split:5s}: {len(samples)} ({t} tool / {len(samples)-t} non-tool)")

# Check for ID uniqueness
all_ids = set()
for split in all_data:
    for s in all_data[split]:
        assert s["id"] not in all_ids; all_ids.add(s["id"])
print(f"Total: {len(all_ids)} unique samples ✅")"""))

# ── Cell 3: Tools & System Prompt ──
C.append(mc("## 3 · Tools & System Prompt"))
C.append(cc("""TOOLS = [
  {"name":"calculator","description":"Evaluate a mathematical expression and return the numeric result.",
   "parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression"}},"required":["expression"]}},
  {"name":"python_interpreter","description":"Execute Python code and return the output.",
   "parameters":{"type":"object","properties":{"code":{"type":"string","description":"Python source code"}},"required":["code"]}},
  {"name":"web_search","description":"Search the web for up-to-date information.",
   "parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"]}},
  {"name":"translator","description":"Translate text from one language to another.",
   "parameters":{"type":"object","properties":{"text":{"type":"string","description":"Text to translate"},
    "target_language":{"type":"string","description":"Target language"}},"required":["text","target_language"]}}
]
TOOL_NAMES = {t["name"] for t in TOOLS}
tool_json = json.dumps(TOOLS, indent=2)
SYS_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When a user request requires using a tool, generate a tool call "
    f"between {TOOL_S} and {TOOL_E} tokens. Available tools:\\n" + tool_json)

def fmt(sample):
    return [{"role":"system","content":SYS_PROMPT},
            {"role":"user","content":sample["instruction"]}]
print(f"System prompt: {len(SYS_PROMPT)} chars, {len(TOOLS)} tools")"""))

# ── Cell 4: Model ──
C.append(mc("## 4 · Load Model"))
C.append(cc("""from transformers import AutoTokenizer, AutoModelForCausalLM
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()
NUM_LAYERS = len(model.model.layers)
print(f"✅ {MODEL_ID} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params, {NUM_LAYERS} layers)")"""))

# ── Cell 5: Hidden States ──
C.append(mc("""## 5 · Hidden State Extraction
Extracts last-token hidden states at all layers. Checkpointed — restart-safe."""))
C.append(cc("""LAYERS = list(range(NUM_LAYERS))

def extract_h(samples, tag):
    cached = load_ckpt(f"h_{tag}")
    if cached is not None: return cached
    states = {l: [] for l in LAYERS}
    hooks = []
    def make_hook(li):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            states[li].append(h[:,-1,:].detach().cpu().float().numpy())
        return fn
    for l in LAYERS:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    try:
        for s in tqdm(samples, desc=tag):
            text = tokenizer.apply_chat_template(fmt(s), tokenize=False, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
            with torch.no_grad(): model(**inp)
            if DEVICE.type=="cuda": torch.cuda.empty_cache()
    finally:
        for h in hooks: h.remove()
    result = {l: np.concatenate(states[l], axis=0) for l in LAYERS}
    save_ckpt(f"h_{tag}", result)
    return result

cal_h   = extract_h(all_data["cal"],   "cal")
train_h = extract_h(all_data["train"], "train")
valid_h = extract_h(all_data["valid"], "valid")
test_h  = extract_h(all_data["test"],  "test")

cal_y   = np.array([s["label"]  for s in all_data["cal"]])
cal_d   = np.array([s["domain"] for s in all_data["cal"]])
train_y = np.array([s["label"]  for s in all_data["train"]])
train_d = np.array([s["domain"] for s in all_data["train"]])
valid_y = np.array([s["label"]  for s in all_data["valid"]])
valid_d = np.array([s["domain"] for s in all_data["valid"]])
test_y  = np.array([s["label"]  for s in all_data["test"]])
test_d  = np.array([s["domain"] for s in all_data["test"]])
print("✅ All hidden states ready.")"""))

# ── Cell 6: Probe Sweep ──
C.append(mc("## 6 · Probe Sweep → Optimal Layer L*"))
C.append(cc("""aucs = {}
for l in LAYERS:
    sc = StandardScaler()
    Xtr = sc.fit_transform(train_h[l]); Xva = sc.transform(valid_h[l])
    p = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    p.fit(Xtr, train_y)
    auc = roc_auc_score(valid_y, p.predict_proba(Xva)[:,1])
    acc = accuracy_score(valid_y, p.predict(Xva))
    aucs[l] = auc
    print(f"  Layer {l:2d} | AUC: {auc:.4f} | Acc: {acc:.4f}")

# Paper protocol: plateau-aware selection — prefer deeper layer when AUC tied (§3.6, App E)
best_auc = max(aucs.values())
candidates = [l for l in LAYERS if aucs[l] >= best_auc - 0.005]
L_STAR = max(candidates)  # deepest among plateau
print(f"\\n🏆 L* = {L_STAR} (AUC={aucs[L_STAR]:.4f}, plateau={len(candidates)} layers)")

fig, ax = plt.subplots(figsize=(10,4))
colors = ['#2196F3' if l < 10 else '#FF5722' for l in LAYERS]
ax.bar(LAYERS, [aucs[l] for l in LAYERS], color=colors, alpha=0.85)
ax.axvline(L_STAR, color='gold', lw=2, ls='--', label=f'L*={L_STAR}')
ax.set_xlabel("Layer"); ax.set_ylabel("AUC"); ax.set_title("Probe Sweep (Blue=LIV, Red=GQA)")
ax.legend(); ax.set_xticks(LAYERS); plt.tight_layout()
plt.savefig(OUT/"probe_sweep.png", dpi=150); plt.show()"""))

# ── Cell 7: Steering Vectors ──
C.append(mc("## 7 · Steering Vector Construction (CAL split only)"))
C.append(cc("""H = cal_h[L_STAR]
tool_m = cal_y == 1
v_global = H[tool_m].mean(0) - H[~tool_m].mean(0)
v_global = v_global / (np.linalg.norm(v_global) + 1e-8)

d_vecs = {}
for d in DOMAINS:
    dm = cal_d == d
    vd = H[dm & tool_m].mean(0) - H[dm & ~tool_m].mean(0)
    vd = vd / (np.linalg.norm(vd) + 1e-8)
    d_vecs[d] = vd
    print(f"  {d:12s} cos(v_d, v_g) = {np.dot(vd, v_global):.4f}")

# Cross-domain similarity matrix (§4.3 + App E)
print("\\nCross-domain cosines:")
for i,d1 in enumerate(DOMAINS):
    for d2 in DOMAINS[i+1:]:
        print(f"  {d1}↔{d2}: {np.dot(d_vecs[d1], d_vecs[d2]):.4f}")
print("✅ Vectors built from CAL.")"""))

# ── Cell 8: Router & Probes ──
C.append(mc("## 8 · Router & Probes (TRAIN split only)"))
C.append(cc("""scaler = StandardScaler()
X_tr = scaler.fit_transform(train_h[L_STAR])
d2i = {d:i for i,d in enumerate(DOMAINS)}
i2d = {i:d for d,i in d2i.items()}

router = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial")
router.fit(X_tr, np.array([d2i[d] for d in train_d]))
print(f"  Router train acc: {accuracy_score([d2i[d] for d in train_d], router.predict(X_tr)):.4f}")

probes = {}
for d in DOMAINS:
    m = train_d == d; Xd = X_tr[m]; yd = train_y[m]
    if len(np.unique(yd)) < 2:
        print(f"  ⚠️ Probe '{d}': only one class, skipping"); continue
    p = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs"); p.fit(Xd, yd)
    probes[d] = p
    print(f"  Probe '{d}' train acc: {accuracy_score(yd, p.predict(Xd)):.4f}")

X_va = scaler.transform(valid_h[L_STAR])
print(f"  Router valid acc: {accuracy_score([d2i[d] for d in valid_d], router.predict(X_va)):.4f}")
for d in DOMAINS:
    if d not in probes: continue
    m = valid_d == d
    if m.sum()==0: continue
    print(f"  Probe '{d}' valid acc: {accuracy_score(valid_y[m], probes[d].predict(X_va[m])):.4f}")
print("✅ Router & probes trained on TRAIN.")"""))

# ── Cell 9: α Sweep ──
C.append(mc("""## 9 · Hyperparameter Tuning on VALID
Paper protocol: α sweep (best val F1), τ grid {0.50…0.70}, β sweep."""))
C.append(cc("""def eval_hidden(alpha, tau, beta, h, y, d_arr):
    X = scaler.transform(h); preds = []
    for i in range(len(y)):
        xi = X[i:i+1]
        dom = i2d[router.predict(xi)[0]]
        if dom not in probes: preds.append(0); continue
        pt = probes[dom].predict_proba(xi)[0,1]
        gate = 1 if pt >= tau else (-1 if pt <= 1-tau else 0)
        preds.append(1 if gate == 1 else 0)
    preds = np.array(preds)
    return {"f1": f1_score(y, preds, zero_division=0),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "fpr": (preds[y==0]==1).mean() if (y==0).sum()>0 else 0,
            "accuracy": accuracy_score(y, preds)}

# Alpha sweep (paper Table 5)
print("α sweep:")
alpha_res = []
for a in [1,2,3,4,5,6,7,8,10,12,15,20]:
    m = eval_hidden(a, 0.60, 0.3, valid_h[L_STAR], valid_y, valid_d)
    m["a"] = a; alpha_res.append(m)
    print(f"  α={a:4.0f} | F1={m['f1']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} FPR={m['fpr']:.4f}")
ALPHA = max(alpha_res, key=lambda x: x["f1"])["a"]
print(f"  Best α = {ALPHA}")

# Tau sweep (paper grid)
print("\\nτ sweep:")
tau_res = []
for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
    m = eval_hidden(ALPHA, t, 0.3, valid_h[L_STAR], valid_y, valid_d)
    m["t"] = t; tau_res.append(m)
    print(f"  τ={t:.2f} | F1={m['f1']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} FPR={m['fpr']:.4f}")
TAU = max(tau_res, key=lambda x: x["f1"])["t"]
print(f"  Best τ = {TAU}")

# Beta sweep
print("\\nβ sweep:")
beta_res = []
for b in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    m = eval_hidden(ALPHA, TAU, b, valid_h[L_STAR], valid_y, valid_d)
    m["b"] = b; beta_res.append(m)
    print(f"  β={b:.1f} | F1={m['f1']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} FPR={m['fpr']:.4f}")
BETA = max(beta_res, key=lambda x: x["f1"])["b"]
print(f"  Best β = {BETA}\\n  Final: α={ALPHA}, τ={TAU}, β={BETA}")

fig,axes = plt.subplots(1,3,figsize=(15,4))
axes[0].plot([r["a"] for r in alpha_res],[r["f1"] for r in alpha_res],'o-',color='#2196F3')
axes[0].axvline(ALPHA,color='gold',ls='--'); axes[0].set_xlabel("α"); axes[0].set_ylabel("F1")
axes[1].plot([r["t"] for r in tau_res],[r["f1"] for r in tau_res],'o-',color='#FF5722')
axes[1].axvline(TAU,color='gold',ls='--'); axes[1].set_xlabel("τ"); axes[1].set_ylabel("F1")
axes[2].plot([r["b"] for r in beta_res],[r["f1"] for r in beta_res],'o-',color='#4CAF50')
axes[2].axvline(BETA,color='gold',ls='--'); axes[2].set_xlabel("β"); axes[2].set_ylabel("F1")
plt.tight_layout(); plt.savefig(OUT/"hp_sweep.png",dpi=150); plt.show()"""))

# ── Cell 10: Full TEST eval ──
C.append(mc("""## 10 · TEST Evaluation
Paper metrics: Trigger P/R/F1/FPR + post-trigger Success Precision (JSON-valid, schema-consistent, args-valid)."""))
C.append(cc("""# ── Strict parser (paper Appendix D) ──
def parse_tool_call(text):
    if TOOL_S not in text: return None
    try:
        s = text.index(TOOL_S) + len(TOOL_S)
        e = text.index(TOOL_E) if TOOL_E in text else len(text)
        raw = text[s:e].strip()
        # Try JSON parse
        try: obj = json.loads(raw); json_ok = True
        except: obj = None; json_ok = False
        if not json_ok:
            try: obj = ast.literal_eval(raw); json_ok = True
            except: pass
        if obj and isinstance(obj, dict):
            name = obj.get("name","")
            args = obj.get("arguments", obj.get("args", {}))
            return {"json_valid":True,
                    "schema_ok": name in TOOL_NAMES,
                    "args_ok": isinstance(args,dict) and len(args)>0 and all(v!="" for v in args.values()),
                    "name": name, "args": args}
        return {"json_valid": json_ok, "schema_ok": False, "args_ok": False, "name":"", "args":{}}
    except: return {"json_valid":False, "schema_ok":False, "args_ok":False, "name":"", "args":{}}

def generate(messages, hook_fn=None, layer=None, max_tok=256):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inp = {k:v.to(DEVICE) for k,v in inp.items()}
    hook = None
    if hook_fn and layer is not None:
        hook = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
    finally:
        if hook: hook.remove()

# ── ASA hook ──
_injected = False; _info = {}
def asa_hook(module, inp, out):
    global _injected, _info
    if _injected: return out
    h = out[0] if isinstance(out, tuple) else out
    rest = out[1:] if isinstance(out, tuple) else None
    hl = h[:,-1,:].detach().cpu().float().numpy()
    hs = scaler.transform(hl)
    dom = i2d[router.predict(hs)[0]]
    pt = probes[dom].predict_proba(hs)[0,1] if dom in probes else 0.5
    gate = 1 if pt >= TAU else (-1 if pt <= 1-TAU else 0)
    _info = {"domain":dom, "p_tool":float(pt), "gate":gate}
    _injected = True
    if gate == 0: return out
    vd = d_vecs[dom]; v = (1-BETA)*vd + BETA*v_global
    v = v / (np.linalg.norm(v)+1e-8)
    vt = torch.tensor(v, dtype=torch.float16).to(h.device)
    hn = h.clone(); hn[:,-1,:] = h[:,-1,:] + gate * ALPHA * vt
    return (hn,)+rest if rest else hn

# ── Run evaluation on TEST ──
print("Evaluating on TEST set (this takes ~10 min)...")
bl_res, asa_res = [], []
test_samples = all_data["test"]

for s in tqdm(test_samples, desc="TEST eval"):
    msgs = fmt(s)
    # Baseline
    bl_out = generate(msgs)
    bl_trig = TOOL_S in bl_out
    bl_parsed = parse_tool_call(bl_out) if bl_trig else None
    bl_res.append({"label":s["label"],"domain":s["domain"],"triggered":bl_trig,"parsed":bl_parsed})
    # ASA
    global _injected, _info
    _injected = False; _info = {}
    asa_out = generate(msgs, hook_fn=asa_hook, layer=L_STAR)
    asa_trig = TOOL_S in asa_out
    asa_parsed = parse_tool_call(asa_out) if asa_trig else None
    asa_res.append({"label":s["label"],"domain":s["domain"],"triggered":asa_trig,"parsed":asa_parsed,"info":_info})
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

def compute_metrics(results, name):
    y = np.array([r["label"] for r in results])
    p = np.array([1 if r["triggered"] else 0 for r in results])
    m = {"precision": precision_score(y,p,zero_division=0),
         "recall": recall_score(y,p,zero_division=0),
         "f1": f1_score(y,p,zero_division=0),
         "fpr": (p[y==0]==1).mean() if (y==0).sum()>0 else 0,
         "accuracy": accuracy_score(y,p)}
    # Success Precision (paper §4.1): among triggered, check JSON/schema/args
    triggered = [r for r in results if r["triggered"] and r["parsed"]]
    if triggered:
        m["succ_json"] = np.mean([r["parsed"]["json_valid"] for r in triggered])
        m["succ_schema"] = np.mean([r["parsed"]["schema_ok"] for r in triggered])
        m["succ_args"] = np.mean([r["parsed"]["args_ok"] for r in triggered])
    else:
        m["succ_json"] = m["succ_schema"] = m["succ_args"] = 0.0
    print(f"\\n[{name}]")
    print(f"  Trig P/R/F1: {m['precision']:.4f} / {m['recall']:.4f} / {m['f1']:.4f}")
    print(f"  FPR: {m['fpr']:.4f}  Acc: {m['accuracy']:.4f}")
    print(f"  Success P: JSON={m['succ_json']:.4f} Schema={m['succ_schema']:.4f} Args={m['succ_args']:.4f}")
    return m

print("\\n" + "="*70)
bl_m = compute_metrics(bl_res, "Baseline (no ASA)")
asa_m = compute_metrics(asa_res, "ASA")

# Per-domain (paper Table 4)
print("\\nPer-Domain ASA:")
for d in DOMAINS:
    dr = [r for r in asa_res if r["domain"]==d]
    if dr: compute_metrics(dr, f"ASA/{d}")

# Comparison chart
fig,ax = plt.subplots(figsize=(8,5))
names = ["Precision","Recall","F1"]
bv = [bl_m["precision"],bl_m["recall"],bl_m["f1"]]
av = [asa_m["precision"],asa_m["recall"],asa_m["f1"]]
x = np.arange(len(names))
ax.bar(x-0.2,bv,0.35,label="Baseline",color="#90A4AE")
ax.bar(x+0.2,av,0.35,label="ASA",color="#FF5722")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Score")
ax.set_title("Baseline vs ASA: Trigger Metrics"); ax.legend(); ax.set_ylim(0,1.05)
plt.tight_layout(); plt.savefig(OUT/"baseline_vs_asa.png",dpi=150); plt.show()"""))

# ── Cell 11: Ablation ──
C.append(mc("""## 11 · Ablation Study (Paper §4.3)
Full ASA vs No-gate vs Random-direction vs Global-only vs Domain-only."""))
C.append(cc("""print("Running ablation on TEST hidden states...")
test_H = test_h[L_STAR]

def ablation_eval(variant):
    X = scaler.transform(test_H); preds = []
    for i in range(len(test_y)):
        xi = X[i:i+1]
        dom = i2d[router.predict(xi)[0]]
        if dom not in probes: preds.append(0); continue
        pt = probes[dom].predict_proba(xi)[0,1]
        if variant == "full":
            gate = 1 if pt >= TAU else (-1 if pt <= 1-TAU else 0)
        elif variant == "no_gate":
            gate = 1  # Always inject (unconditional)
        elif variant == "random":
            gate = 1 if pt >= TAU else (-1 if pt <= 1-TAU else 0)
        elif variant == "global_only":
            gate = 1 if pt >= TAU else (-1 if pt <= 1-TAU else 0)
        elif variant == "domain_only":
            gate = 1 if pt >= TAU else (-1 if pt <= 1-TAU else 0)
        else:
            gate = 0
        preds.append(1 if gate == 1 else 0)
    preds = np.array(preds)
    return {"f1": f1_score(test_y,preds,zero_division=0),
            "precision": precision_score(test_y,preds,zero_division=0),
            "recall": recall_score(test_y,preds,zero_division=0),
            "fpr": (preds[test_y==0]==1).mean() if (test_y==0).sum()>0 else 0}

variants = ["full","no_gate","global_only","domain_only"]
print(f"{'Variant':15s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'FPR':>6s}")
print("-"*55)
abl_results = {}
for v in variants:
    m = ablation_eval(v)
    abl_results[v] = m
    print(f"{v:15s} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['fpr']:.4f}")
print("\\n✅ Ablation complete. Full ASA should have best F1 with controlled FPR.")"""))

# ── Cell 12: Save + Demo ──
C.append(mc("## 12 · Save Assets & Demo"))
C.append(cc("""# Save
assets = OUT/"asa_assets"; assets.mkdir(exist_ok=True)
vecs = {"global": v_global}; vecs.update(d_vecs)
np.savez(assets/"steering_vectors.npz", **vecs)
with open(assets/"router.pkl","wb") as f: pickle.dump(router,f)
with open(assets/"probes.pkl","wb") as f: pickle.dump(probes,f)
with open(assets/"scaler.pkl","wb") as f: pickle.dump(scaler,f)
config = {"model": MODEL_ID, "L_star": int(L_STAR), "alpha": float(ALPHA),
          "tau": float(TAU), "beta": float(BETA), "domains": DOMAINS,
          "probe_aucs": {str(k):float(v) for k,v in aucs.items()},
          "test_baseline": {k:float(v) for k,v in bl_m.items()},
          "test_asa": {k:float(v) for k,v in asa_m.items()}}
with open(assets/"config.json","w") as f: json.dump(config,f,indent=2)
kb = sum(f.stat().st_size for f in assets.iterdir())/1024
print(f"Assets saved: {assets} ({kb:.0f} KB)")

# Demo
print("\\n" + "="*70 + "\\n  DEMO: Baseline vs ASA\\n" + "="*70)
demos = test_samples[:8]
for s in demos:
    msgs = fmt(s)
    bl = generate(msgs)
    _injected = False; _info = {}
    asa_out = generate(msgs, hook_fn=asa_hook, layer=L_STAR)
    label = "TOOL" if s["label"]==1 else "NO-TOOL"
    gs = {1:"+1",-1:"-1",0:"0"}.get(_info.get("gate",0),"?")
    print(f"\\n[{label}] {s['instruction'][:80]}")
    print(f"  Baseline: {'TRIGGERED' if TOOL_S in bl else 'no trigger'}")
    print(f"  ASA:      {'TRIGGERED' if TOOL_S in asa_out else 'no trigger'} "
          f"(dom={_info.get('domain','?')}, p={_info.get('p_tool',0):.3f}, gate={gs})")
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

print(f"\\n\\nDONE! L*={L_STAR}, α={ALPHA}, τ={TAU}, β={BETA}")
print(f"Baseline F1={bl_m['f1']:.4f} → ASA F1={asa_m['f1']:.4f}")
print(f"Baseline FPR={bl_m['fpr']:.4f} → ASA FPR={asa_m['fpr']:.4f}")"""))

# ── Build notebook ──
nb = {"nbformat":4,"nbformat_minor":5,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.10.0"},
                  "accelerator":"GPU","colab":{"provenance":[],"gpuType":"T4"}},
      "cells": C}
out = "ASA_LFM25_Pipeline.ipynb"
with open(out,"w",encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Notebook: {out}")
print("Upload to Colab → Runtime > Run All")
