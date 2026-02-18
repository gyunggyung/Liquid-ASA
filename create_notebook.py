"""
Run this script to generate the Colab notebook (.ipynb).
Usage: python create_notebook.py
Output: ASA_LFM25_Pipeline.ipynb (upload to Colab directly)
"""
import json

def code_cell(source: str, cell_id: str = None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().split("\n") if "\n" not in source else
                  [line + "\n" for line in source.strip().split("\n")[:-1]] + [source.strip().split("\n")[-1]]
    }

def md_cell(source: str):
    lines = source.strip().split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]]
    }

cells = []

# ── Title ──
cells.append(md_cell("""# 🧬 ASA × LFM2.5-1.2B-Instruct
**Training-Free Tool-Calling Enhancement via Activation Steering**

This notebook runs the complete ASA pipeline on a Colab T4 GPU.
Each cell can be run independently — stop and resume anytime."""))

# ── Cell 1: Setup ──
cells.append(md_cell("## 1. Setup"))
cells.append(code_cell("""!pip install -q transformers>=4.40.0 accelerate>=0.25.0 scikit-learn>=1.3.0 tqdm matplotlib seaborn
!git clone https://github.com/gyunggyung/Liquid-ASA.git 2>/dev/null || echo "Already cloned"
%cd /content/Liquid-ASA
!ls data/"""))

# ── Cell 2: Imports & Config ──
cells.append(md_cell("## 2. Imports & Config"))
cells.append(code_cell("""import os, json, pickle, warnings, gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    try:
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except AttributeError:
        print("VRAM: (unknown)")

# Config
MODEL_ID  = "LiquidAI/LFM2.5-1.2B-Instruct"
DOMAINS   = ["math", "code", "search", "translation"]
LAYERS    = list(range(16))
GQA_START = 10
TOOL_S    = "<|tool_call_start|>"
TOOL_E    = "<|tool_call_end|>"
SEED      = 42

DATA_DIR = Path("data")
OUT_DIR  = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR = Path("outputs/ckpt"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA, TAU, BETA = 6.0, 0.60, 0.3
np.random.seed(SEED); torch.manual_seed(SEED)

def save_ckpt(name, obj):
    p = CKPT_DIR / f"{name}.pkl"
    with open(p, "wb") as f: pickle.dump(obj, f)
    print(f"  Saved: {name} ({p.stat().st_size/1024:.0f} KB)")

def load_ckpt(name):
    p = CKPT_DIR / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f: obj = pickle.load(f)
        print(f"  Cached: {name}")
        return obj
    return None

print("Config ready.")"""))

# ── Cell 3: Load Data ──
cells.append(md_cell("## 3. Load Data"))
cells.append(code_cell("""def load_json(path):
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

tools      = load_json(DATA_DIR / "tools.json")
cal_data   = load_json(DATA_DIR / "cal_data.json")
train_data = load_json(DATA_DIR / "train_data.json")
valid_data = load_json(DATA_DIR / "valid_data.json")
test_data  = load_json(DATA_DIR / "test_data.json")

tool_json = json.dumps(tools, indent=2)
SYS_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When a user request requires using a tool, generate a tool call "
    f"between {TOOL_S} and {TOOL_E} tokens. Available tools:\\n" + tool_json
)

def fmt(sample):
    return [{"role":"system","content":SYS_PROMPT}, {"role":"user","content":sample["instruction"]}]

# Verify splits
all_ids = set()
for name, ds in [("CAL",cal_data),("TRAIN",train_data),("VALID",valid_data),("TEST",test_data)]:
    ids = {s["id"] for s in ds}
    assert not (ids & all_ids), f"Overlap in {name}!"
    all_ids |= ids
    t = sum(1 for s in ds if s["label"]==1)
    print(f"  {name:5s}: {len(ds)} samples ({t} tool / {len(ds)-t} non-tool)")
print("Data OK - no overlap between splits")"""))

# ── Cell 4: Load Model ──
cells.append(md_cell("## 4. Load Model"))
cells.append(code_cell("""from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)
model.eval()
print(f"Loaded: {MODEL_ID} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")"""))

# ── Cell 5: Extract Hidden States ──
cells.append(md_cell("""## 5. Extract Hidden States
> This is the slowest step (~15 min on T4). Results are checkpointed — if you restart, cached data is loaded instantly."""))
cells.append(code_cell("""def extract(samples, tag):
    cached = load_ckpt(f"h_{tag}")
    if cached is not None: return cached
    states = {l: [] for l in LAYERS}
    hooks = []
    def make_hook(li):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            states[li].append(h[:, -1, :].detach().cpu().float().numpy())
        return fn
    for l in LAYERS:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
    try:
        for s in tqdm(samples, desc=tag):
            text = tokenizer.apply_chat_template(fmt(s), tokenize=False, add_generation_prompt=True)
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
            with torch.no_grad(): model(**inp)
            if device.type == "cuda": torch.cuda.empty_cache()
    finally:
        for h in hooks: h.remove()
    result = {l: np.concatenate(states[l], axis=0) for l in LAYERS}
    save_ckpt(f"h_{tag}", result)
    return result

cal_h   = extract(cal_data,   "cal")
train_h = extract(train_data, "train")
valid_h = extract(valid_data, "valid")
test_h  = extract(test_data,  "test")

cal_y,   cal_d   = np.array([s["label"] for s in cal_data]),   np.array([s["domain"] for s in cal_data])
train_y, train_d = np.array([s["label"] for s in train_data]), np.array([s["domain"] for s in train_data])
valid_y, valid_d = np.array([s["label"] for s in valid_data]), np.array([s["domain"] for s in valid_data])
test_y,  test_d  = np.array([s["label"] for s in test_data]),  np.array([s["domain"] for s in test_data])
print("All hidden states ready!")"""))

# ── Cell 6: Probe Sweep ──
cells.append(md_cell("## 6. Probe Sweep → Find Optimal Layer"))
cells.append(code_cell("""aucs = {}
for l in LAYERS:
    sc = StandardScaler()
    Xtr = sc.fit_transform(train_h[l])
    Xva = sc.transform(valid_h[l])
    p = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    p.fit(Xtr, train_y)
    auc = roc_auc_score(valid_y, p.predict_proba(Xva)[:, 1])
    acc = accuracy_score(valid_y, p.predict(Xva))
    aucs[l] = auc
    tag = "GQA" if l >= GQA_START else "LIV"
    print(f"  Layer {l:2d} ({tag}) | AUC: {auc:.4f} | Acc: {acc:.4f}")

L_STAR = max(aucs, key=aucs.get)
print(f"\\nBest layer: L{L_STAR} (AUC = {aucs[L_STAR]:.4f})")

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#2196F3' if l < GQA_START else '#FF5722' for l in LAYERS]
ax.bar(LAYERS, [aucs[l] for l in LAYERS], color=colors, alpha=0.85)
ax.axvline(x=L_STAR, color='gold', linewidth=2, linestyle='--', label=f'L*={L_STAR}')
ax.set_xlabel("Layer"); ax.set_ylabel("AUC"); ax.set_title("Probe Sweep (Blue=LIV, Red=GQA)")
ax.legend(); ax.set_xticks(LAYERS); plt.tight_layout()
plt.savefig(OUT_DIR / "probe_sweep.png", dpi=150)
plt.show()"""))

# ── Cell 7: Steering Vectors ──
cells.append(md_cell("## 7. Build Steering Vectors"))
cells.append(code_cell("""H = cal_h[L_STAR]
tool_mask = cal_y == 1
v_global = H[tool_mask].mean(0) - H[~tool_mask].mean(0)
v_global = v_global / (np.linalg.norm(v_global) + 1e-8)

domain_vecs = {}
for d in DOMAINS:
    dm = cal_d == d
    vd = H[dm & tool_mask].mean(0) - H[dm & ~tool_mask].mean(0)
    vd = vd / (np.linalg.norm(vd) + 1e-8)
    domain_vecs[d] = vd
    print(f"  {d:12s} | cos(v_d, v_global) = {np.dot(vd, v_global):.4f}")
print("Vectors built.")"""))

# ── Cell 8: Router & Probes ──
cells.append(md_cell("## 8. Train Router & Probes"))
cells.append(code_cell("""scaler = StandardScaler()
X_tr = scaler.fit_transform(train_h[L_STAR])
d2i = {d: i for i, d in enumerate(DOMAINS)}
i2d = {i: d for d, i in d2i.items()}

router = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial")
router.fit(X_tr, np.array([d2i[d] for d in train_d]))
print(f"  Router train acc: {accuracy_score(np.array([d2i[d] for d in train_d]), router.predict(X_tr)):.4f}")

probes = {}
for d in DOMAINS:
    m = train_d == d
    p = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    p.fit(X_tr[m], train_y[m])
    probes[d] = p
    print(f"  Probe '{d}' train acc: {accuracy_score(train_y[m], p.predict(X_tr[m])):.4f}")

X_va = scaler.transform(valid_h[L_STAR])
print(f"  Router valid acc: {accuracy_score([d2i[d] for d in valid_d], router.predict(X_va)):.4f}")
for d in DOMAINS:
    m = valid_d == d
    print(f"  Probe '{d}' valid acc: {accuracy_score(valid_y[m], probes[d].predict(X_va[m])):.4f}")"""))

# ── Cell 9: Hyperparameter Tuning ──
cells.append(md_cell("## 9. Hyperparameter Tuning (α, τ, β)"))
cells.append(code_cell("""def eval_fast(alpha, tau, beta, h, y, d_arr):
    preds = []
    X = scaler.transform(h)
    for i in range(len(y)):
        xi = X[i:i+1]
        dom = i2d[router.predict(xi)[0]]
        pt = probes[dom].predict_proba(xi)[0, 1]
        gate = 1 if pt >= tau else (-1 if pt <= 1-tau else 0)
        preds.append(1 if gate == 1 else 0)
    preds = np.array(preds)
    return {
        "f1": f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "fpr": (preds[y==0]==1).mean() if (y==0).sum()>0 else 0,
        "accuracy": accuracy_score(y, preds),
    }

print("Alpha sweep:")
best_f1, best_alpha = 0, ALPHA
for a in [1,2,3,4,5,6,7,8,10,12]:
    m = eval_fast(a, TAU, BETA, valid_h[L_STAR], valid_y, valid_d)
    print(f"  a={a:5.1f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | FPR={m['fpr']:.4f}")
    if m["f1"] > best_f1: best_f1, best_alpha = m["f1"], a
ALPHA = best_alpha; print(f"  Best alpha = {ALPHA}")

print("\\nTau sweep:")
best_f1, best_tau = 0, TAU
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    m = eval_fast(ALPHA, t, BETA, valid_h[L_STAR], valid_y, valid_d)
    print(f"  t={t:.2f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | FPR={m['fpr']:.4f}")
    if m["f1"] > best_f1: best_f1, best_tau = m["f1"], t
TAU = best_tau; print(f"  Best tau = {TAU}")

print("\\nBeta sweep:")
best_f1, best_beta = 0, BETA
for b in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    m = eval_fast(ALPHA, TAU, b, valid_h[L_STAR], valid_y, valid_d)
    print(f"  b={b:.1f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | FPR={m['fpr']:.4f}")
    if m["f1"] > best_f1: best_f1, best_beta = m["f1"], b
BETA = best_beta; print(f"  Best beta = {BETA}")
print(f"\\nFinal: alpha={ALPHA}, tau={TAU}, beta={BETA}")"""))

# ── Cell 10: Evaluation ──
cells.append(md_cell("## 10. Evaluation on TEST"))
cells.append(code_cell("""test_m = eval_fast(ALPHA, TAU, BETA, test_h[L_STAR], test_y, test_d)
print("Overall:")
for k, v in test_m.items(): print(f"  {k:10s}: {v:.4f}")

print("\\nPer-Domain:")
for d in DOMAINS:
    m = test_d == d
    dm = eval_fast(ALPHA, TAU, BETA, test_h[L_STAR][m], test_y[m], test_d[m])
    print(f"  {d:12s} | F1={dm['f1']:.4f} | Prec={dm['precision']:.4f} | Rec={dm['recall']:.4f} | FPR={dm['fpr']:.4f}")"""))

# ── Cell 11: Generation Demo ──
cells.append(md_cell("## 11. Generation Demo (Baseline vs ASA)"))
cells.append(code_cell("""def generate(messages, hook_fn=None, layer=None, max_tokens=256):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inp = {k: v.to(device) for k, v in inp.items()}
    hook = None
    if hook_fn and layer is not None:
        hook = model.model.layers[layer].register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
    finally:
        if hook: hook.remove()
    return gen

_injected = False
_info = {}

def asa_hook(module, inp, out):
    global _injected, _info
    if _injected: return out
    h = out[0] if isinstance(out, tuple) else out
    rest = out[1:] if isinstance(out, tuple) else None
    hl = h[:, -1, :].detach().cpu().float().numpy()
    hs = scaler.transform(hl)
    dom = i2d[router.predict(hs)[0]]
    pt = probes[dom].predict_proba(hs)[0, 1]
    gate = 1 if pt >= TAU else (-1 if pt <= 1-TAU else 0)
    _info = {"domain": dom, "p_tool": float(pt), "gate": gate}
    _injected = True
    if gate == 0: return out
    vd = domain_vecs[dom]
    v = (1-BETA)*vd + BETA*v_global
    v = v / (np.linalg.norm(v) + 1e-8)
    vt = torch.tensor(v, dtype=torch.float16).to(h.device)
    hn = h.clone()
    hn[:, -1, :] = h[:, -1, :] + gate * ALPHA * vt
    return (hn,)+rest if rest else hn

demos = [s for s in test_data if s["label"]==1][:4] + [s for s in test_data if s["label"]==0][:4]
for s in demos:
    msgs = fmt(s)
    bl = generate(msgs)
    bl_t = TOOL_S in bl
    _injected = False; _info = {}
    asa = generate(msgs, hook_fn=asa_hook, layer=L_STAR)
    asa_t = TOOL_S in asa
    label = "TOOL" if s["label"]==1 else "NO-TOOL"
    gs = {1:"+1", -1:"-1", 0:"0"}.get(_info.get("gate",0), "?")
    print(f"[{label}] {s['instruction'][:70]}")
    print(f"  Baseline: {'triggered' if bl_t else 'no trigger'}")
    print(f"  ASA:      {'triggered' if asa_t else 'no trigger'} (dom={_info.get('domain','?')}, p={_info.get('p_tool',0):.3f}, gate={gs})")
    print()
    if device.type == "cuda": torch.cuda.empty_cache()"""))

# ── Cell 12: Save ──
cells.append(md_cell("## 12. Save Assets"))
cells.append(code_cell("""assets_dir = OUT_DIR / "asa_assets"; assets_dir.mkdir(exist_ok=True)
vecs = {"global": v_global}; vecs.update(domain_vecs)
np.savez(assets_dir / "steering_vectors.npz", **vecs)
with open(assets_dir / "router.pkl", "wb") as f: pickle.dump(router, f)
with open(assets_dir / "probes.pkl", "wb") as f: pickle.dump(probes, f)
with open(assets_dir / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
config = {"model_id": MODEL_ID, "best_layer": int(L_STAR), "alpha": ALPHA, "tau": TAU, "beta": BETA,
          "domains": DOMAINS, "test_metrics": {k: float(v) for k,v in test_m.items()}}
with open(assets_dir / "config.json", "w") as f: json.dump(config, f, indent=2)
total_kb = sum(f.stat().st_size for f in assets_dir.iterdir()) / 1024
print(f"Assets saved to {assets_dir} ({total_kb:.0f} KB)")
print(f"\\nDONE! L*={L_STAR}, alpha={ALPHA}, tau={TAU}, beta={BETA}")
print(f"Test F1={test_m['f1']:.4f}, FPR={test_m['fpr']:.4f}")"""))

# ── Build notebook ──
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": cells,
}

out_path = "ASA_LFM25_Pipeline.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created: {out_path}")
print(f"Upload this file to Google Colab and click Runtime > Run All")
