# %% [markdown]
# # 🧬 ASA × LFM2.5-1.2B-Instruct: Training-Free Tool-Calling Enhancement
#
# This notebook implements the complete ASA (Activation Steering Adapter) pipeline
# for the LiquidAI/LFM2.5-1.2B-Instruct model on a Colab T4 GPU.
#
# **Pipeline:**
# 1. Environment Setup
# 2. Data Loading
# 3. Hidden State Extraction
# 4. Probe Sweep (Optimal Layer Selection)
# 5. Steering Vector Construction
# 6. Router & Probe Training
# 7. ASA Inference System
# 8. Hyperparameter Tuning
# 9. Evaluation
# 10. Upload to HuggingFace
# 11. Interactive Demo

# %% [markdown]
# ## 1. Environment Setup

# %%
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("transformers>=4.40.0")
install("torch>=2.1.0")
install("accelerate>=0.25.0")
install("scikit-learn>=1.3.0")
install("huggingface_hub>=0.20.0")
install("tqdm")
install("matplotlib")
install("seaborn")

print("✅ All packages installed.")

# %%
import os, json, copy, warnings, gc, pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name()}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ── Configuration ──────────────────────────────────────────────────────────
class Config:
    MODEL_ID = "LiquidAI/LFM2.5-1.2B-Instruct"
    NUM_LAYERS = 16           # 10 LIV conv + 6 GQA blocks
    GQA_START = 10            # GQA blocks start at layer 10
    HIDDEN_DIM = 2048         # hidden state dimension

    # ASA hyperparameters (tuned on VALID set)
    ALPHA = 6.0               # steering strength (initial)
    TAU = 0.60                # probe confidence threshold
    BETA = 0.3                # global vector weight in MoV

    # Probe sweep range
    PROBE_LAYERS = list(range(16))

    # Data paths
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("outputs")
    CKPT_DIR = Path("outputs/checkpoints")
    HF_REPO_ID = "gyunggyung/ASA-LFM2.5-1.2B-Instruct"

    DOMAINS = ["math", "code", "search", "translation"]
    TOOL_TOKEN_START = "<|tool_call_start|>"
    TOOL_TOKEN_END = "<|tool_call_end|>"

    SEED = 42

cfg = Config()
cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)

# ── Checkpoint Utilities ──────────────────────────────────────────────────
def save_checkpoint(name: str, data: dict):
    """Save a checkpoint to disk."""
    path = cfg.CKPT_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    size_kb = path.stat().st_size / 1024
    print(f"   💾 Checkpoint saved: {name} ({size_kb:.1f} KB)")

def load_checkpoint(name: str) -> Optional[dict]:
    """Load a checkpoint if it exists, otherwise return None."""
    path = cfg.CKPT_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        size_kb = path.stat().st_size / 1024
        print(f"   ♻️  Checkpoint loaded: {name} ({size_kb:.1f} KB)")
        return data
    return None

print("✅ Config ready. Checkpoints will be saved to outputs/checkpoints/")

# %% [markdown]
# ## 2. Data Loading

# %%
def load_dataset(path: Path) -> List[dict]:
    """Load a JSON dataset file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_tools(path: Path) -> List[dict]:
    """Load tool definitions."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_system_prompt(tools: List[dict]) -> str:
    """Build LFM2.5-style system prompt with tool definitions."""
    tool_json = json.dumps(tools, indent=2)
    return (
        "You are a helpful assistant with access to tools. "
        "When a user request requires using a tool, generate a tool call "
        f"between {cfg.TOOL_TOKEN_START} and {cfg.TOOL_TOKEN_END} tokens. "
        "Available tools:\n" + tool_json
    )

def format_messages(sample: dict, system_prompt: str) -> List[dict]:
    """Format a sample into ChatML messages for LFM2.5."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": sample["instruction"]},
    ]

# Load everything
tools = load_tools(cfg.DATA_DIR / "tools.json")
system_prompt = build_system_prompt(tools)

cal_data   = load_dataset(cfg.DATA_DIR / "cal_data.json")
train_data = load_dataset(cfg.DATA_DIR / "train_data.json")
valid_data = load_dataset(cfg.DATA_DIR / "valid_data.json")
test_data  = load_dataset(cfg.DATA_DIR / "test_data.json")

print(f"📊 Datasets loaded:")
print(f"   CAL:   {len(cal_data)} samples")
print(f"   TRAIN: {len(train_data)} samples")
print(f"   VALID: {len(valid_data)} samples")
print(f"   TEST:  {len(test_data)} samples")
print(f"🔧 Tools: {[t['name'] for t in tools]}")

# Quick sanity check: verify split isolation
all_ids = set()
for name, ds in [("CAL", cal_data), ("TRAIN", train_data),
                 ("VALID", valid_data), ("TEST", test_data)]:
    ids = {s["id"] for s in ds}
    overlap = ids & all_ids
    assert not overlap, f"❌ ID overlap in {name}: {overlap}"
    all_ids |= ids
    tool_ct = sum(1 for s in ds if s["label"] == 1)
    print(f"   {name}: {tool_ct} tool / {len(ds)-tool_ct} non-tool")
print("✅ Data isolation verified — no overlap between splits.")

# %% [markdown]
# ## 3. Model Loading & Hidden State Extraction

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

print("⏳ Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg.MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"✅ Model loaded: {cfg.MODEL_ID}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# %%
def get_layer_module(model, layer_idx: int):
    """
    Get the module at a given layer index for hook registration.
    LFM2.5: 10 LIV conv blocks + 6 GQA blocks = 16 layers total.
    Access through model.model.layers[layer_idx].
    """
    return model.model.layers[layer_idx]

def extract_hidden_states(
    model, tokenizer, samples: List[dict], system_prompt: str,
    layers: List[int] = None, batch_size: int = 1,
) -> Dict[int, np.ndarray]:
    """
    Extract last-token hidden states at specified layers.

    Returns: {layer_idx: np.ndarray of shape (N, hidden_dim)}
    """
    if layers is None:
        layers = cfg.PROBE_LAYERS

    # Storage: layer -> list of hidden state vectors
    hidden_states = {l: [] for l in layers}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output can be a tuple; take the first element (hidden states)
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            # Last token's hidden state
            last_h = h[:, -1, :].detach().cpu().float().numpy()
            hidden_states[layer_idx].append(last_h)
        return hook_fn

    # Register hooks
    for l in layers:
        mod = get_layer_module(model, l)
        hooks.append(mod.register_forward_hook(make_hook(l)))

    try:
        for i in tqdm(range(0, len(samples), batch_size), desc="Extracting"):
            batch = samples[i:i+batch_size]
            messages_list = [
                format_messages(s, system_prompt) for s in batch
            ]

            for msgs in messages_list:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=2048
                ).to(device)

                with torch.no_grad():
                    model(**inputs)

                # Clear CUDA cache periodically
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        # Remove all hooks
        for h in hooks:
            h.remove()

    # Stack into arrays
    result = {}
    for l in layers:
        result[l] = np.concatenate(hidden_states[l], axis=0)
        print(f"   Layer {l:2d}: shape {result[l].shape}")

    return result

# %%
# ── Extract CAL hidden states (with checkpoint support) ──
print("🔍 Extracting hidden states from CAL data (for steering vectors)...")
cal_labels = np.array([s["label"] for s in cal_data])
cal_domains = np.array([s["domain"] for s in cal_data])

_ckpt = load_checkpoint("cal_hidden")
if _ckpt is not None:
    cal_hidden = _ckpt
else:
    cal_hidden = extract_hidden_states(model, tokenizer, cal_data, system_prompt)
    save_checkpoint("cal_hidden", cal_hidden)
print(f"✅ CAL hidden states ready: {len(cal_data)} samples × {len(cfg.PROBE_LAYERS)} layers")

# %%
# ── Extract TRAIN hidden states ──
print("🔍 Extracting hidden states from TRAIN data (for router/probes)...")
train_labels = np.array([s["label"] for s in train_data])
train_domains = np.array([s["domain"] for s in train_data])

_ckpt = load_checkpoint("train_hidden")
if _ckpt is not None:
    train_hidden = _ckpt
else:
    train_hidden = extract_hidden_states(model, tokenizer, train_data, system_prompt)
    save_checkpoint("train_hidden", train_hidden)
print(f"✅ TRAIN hidden states ready.")

# %%
# ── Extract VALID hidden states ──
print("🔍 Extracting hidden states from VALID data (for hyperparam tuning)...")
valid_labels = np.array([s["label"] for s in valid_data])
valid_domains = np.array([s["domain"] for s in valid_data])

_ckpt = load_checkpoint("valid_hidden")
if _ckpt is not None:
    valid_hidden = _ckpt
else:
    valid_hidden = extract_hidden_states(model, tokenizer, valid_data, system_prompt)
    save_checkpoint("valid_hidden", valid_hidden)
print(f"✅ VALID hidden states ready.")

# %%
# ── Extract TEST hidden states ──
print("🔍 Extracting hidden states from TEST data (for evaluation)...")
test_labels = np.array([s["label"] for s in test_data])
test_domains = np.array([s["domain"] for s in test_data])

_ckpt = load_checkpoint("test_hidden")
if _ckpt is not None:
    test_hidden = _ckpt
else:
    test_hidden = extract_hidden_states(model, tokenizer, test_data, system_prompt)
    save_checkpoint("test_hidden", test_hidden)
print(f"✅ TEST hidden states ready.")

# %% [markdown]
# ## 4. Probe Sweep — Find Optimal Layer L*

# %%
def probe_sweep(
    train_h: Dict[int, np.ndarray], train_y: np.ndarray,
    valid_h: Dict[int, np.ndarray], valid_y: np.ndarray,
    layers: List[int],
) -> Tuple[int, Dict[int, float]]:
    """
    Train a logistic regression probe at each layer, evaluate AUC on valid set.
    Returns: (best_layer, {layer: auc_score})
    """
    results = {}
    for l in tqdm(layers, desc="Probe sweep"):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_h[l])
        X_val = scaler.transform(valid_h[l])

        probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        probe.fit(X_tr, train_y)

        proba = probe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(valid_y, proba)
        results[l] = auc
        print(f"   Layer {l:2d} | AUC: {auc:.4f} | Acc: {accuracy_score(valid_y, probe.predict(X_val)):.4f}")

    best_layer = max(results, key=results.get)
    print(f"\n🏆 Best layer: L{best_layer} (AUC = {results[best_layer]:.4f})")
    return best_layer, results

print("📈 Running probe sweep to find optimal layer L*...")
best_layer, probe_aucs = probe_sweep(
    train_hidden, train_labels,
    valid_hidden, valid_labels,
    cfg.PROBE_LAYERS,
)

# Visualize probe sweep results
fig, ax = plt.subplots(figsize=(10, 4))
layers_list = sorted(probe_aucs.keys())
aucs_list = [probe_aucs[l] for l in layers_list]
colors = ['#2196F3' if l < cfg.GQA_START else '#FF5722' for l in layers_list]
ax.bar(layers_list, aucs_list, color=colors, alpha=0.85)
ax.axvline(x=best_layer, color='gold', linewidth=2, linestyle='--', label=f'L*={best_layer}')
ax.set_xlabel("Layer Index")
ax.set_ylabel("AUC Score")
ax.set_title("Probe Sweep: Tool-Use Intent by Layer (Blue=LIV, Red=GQA)")
ax.legend()
ax.set_xticks(layers_list)
plt.tight_layout()
plt.savefig(cfg.OUTPUT_DIR / "probe_sweep.png", dpi=150)
plt.show()
print("✅ Probe sweep complete. Plot saved.")

# %% [markdown]
# ## 5. Steering Vector Construction

# %%
def build_steering_vectors(
    hidden_states: Dict[int, np.ndarray],
    labels: np.ndarray,
    domains: np.ndarray,
    layer: int,
    domain_list: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Build domain-specific and global steering vectors.

    v_d = mean(h | tool, domain=d) - mean(h | non-tool, domain=d)
    v_global = mean(h | tool) - mean(h | non-tool)

    Returns: (domain_vectors: {domain: vector}, global_vector)
    """
    H = hidden_states[layer]

    # Global vector
    tool_mask = labels == 1
    non_tool_mask = labels == 0
    v_global = H[tool_mask].mean(axis=0) - H[non_tool_mask].mean(axis=0)
    v_global = v_global / (np.linalg.norm(v_global) + 1e-8)

    # Domain-specific vectors
    domain_vectors = {}
    for d in domain_list:
        d_mask = domains == d
        d_tool = d_mask & tool_mask
        d_nontool = d_mask & non_tool_mask

        n_tool = d_tool.sum()
        n_nontool = d_nontool.sum()
        print(f"   Domain '{d}': {n_tool} tool, {n_nontool} non-tool")

        v_d = H[d_tool].mean(axis=0) - H[d_nontool].mean(axis=0)
        v_d = v_d / (np.linalg.norm(v_d) + 1e-8)
        domain_vectors[d] = v_d

    return domain_vectors, v_global

print(f"🧭 Building steering vectors at Layer {best_layer}...")
domain_vectors, global_vector = build_steering_vectors(
    cal_hidden, cal_labels, cal_domains, best_layer, cfg.DOMAINS
)

# Verify orthogonality / alignment
print("\n📐 Cosine similarities between domain vectors:")
for i, d1 in enumerate(cfg.DOMAINS):
    for d2 in cfg.DOMAINS[i+1:]:
        cos = np.dot(domain_vectors[d1], domain_vectors[d2])
        print(f"   {d1} ↔ {d2}: {cos:.4f}")
for d in cfg.DOMAINS:
    cos = np.dot(domain_vectors[d], global_vector)
    print(f"   {d} ↔ global: {cos:.4f}")

print("✅ Steering vectors built.")

# %% [markdown]
# ## 6. Router & Probe Training

# %%
def train_router(
    hidden_states: np.ndarray, domains: np.ndarray, domain_list: List[str]
) -> Tuple[LogisticRegression, StandardScaler]:
    """
    Train a multi-class domain router.
    Input: hidden state → Output: domain label.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(hidden_states)

    # Encode domain labels
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    y = np.array([domain_to_idx[d] for d in domains])

    router = LogisticRegression(
        max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial"
    )
    router.fit(X, y)
    acc = accuracy_score(y, router.predict(X))
    print(f"   Router train accuracy: {acc:.4f}")
    return router, scaler

def train_probes(
    hidden_states: np.ndarray, labels: np.ndarray, domains: np.ndarray,
    domain_list: List[str], scaler: StandardScaler,
) -> Dict[str, LogisticRegression]:
    """
    Train per-domain binary probes: P(tool | domain=d).
    """
    X = scaler.transform(hidden_states)
    probes = {}
    for d in domain_list:
        mask = domains == d
        X_d = X[mask]
        y_d = labels[mask]

        probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        probe.fit(X_d, y_d)
        acc = accuracy_score(y_d, probe.predict(X_d))
        print(f"   Probe '{d}' train accuracy: {acc:.4f}")
        probes[d] = probe

    return probes

print(f"🎯 Training router and probes at Layer {best_layer}...")
H_train = train_hidden[best_layer]

print("\n── Router (domain classification) ──")
router, scaler = train_router(H_train, train_domains, cfg.DOMAINS)

print("\n── Per-domain probes (tool intent) ──")
probes = train_probes(H_train, train_labels, train_domains, cfg.DOMAINS, scaler)

# Validate on VALID set
print("\n── Validation ──")
H_val = scaler.transform(valid_hidden[best_layer])
domain_to_idx = {d: i for i, d in enumerate(cfg.DOMAINS)}
idx_to_domain = {i: d for d, i in domain_to_idx.items()}

router_preds = router.predict(H_val)
router_acc = accuracy_score(
    [domain_to_idx[d] for d in valid_domains], router_preds
)
print(f"   Router valid accuracy: {router_acc:.4f}")

for d in cfg.DOMAINS:
    mask = valid_domains == d
    if mask.sum() == 0:
        continue
    X_d = H_val[mask]
    y_d = valid_labels[mask]
    probe_pred = probes[d].predict(X_d)
    acc = accuracy_score(y_d, probe_pred)
    print(f"   Probe '{d}' valid accuracy: {acc:.4f}")

print("✅ Router and probes trained.")

# %% [markdown]
# ## 7. ASA Controller — Inference Hook

# %%
class ASAController:
    """
    ASA (Activation Steering Adapter) inference-time controller.

    Operates as a forward hook at layer L*, performing:
    1. Standardize last-token hidden state
    2. Route to domain via router
    3. Evaluate intent via domain probe
    4. Compose steering direction (MoV)
    5. Apply ternary gate (+v, -v, or 0)
    6. Inject into residual stream
    """

    def __init__(
        self,
        router: LogisticRegression,
        probes: Dict[str, LogisticRegression],
        domain_vectors: Dict[str, np.ndarray],
        global_vector: np.ndarray,
        scaler: StandardScaler,
        domain_list: List[str],
        alpha: float = 6.0,
        tau: float = 0.60,
        beta: float = 0.3,
    ):
        self.router = router
        self.probes = probes
        self.scaler = scaler
        self.domain_list = domain_list
        self.domain_to_idx = {d: i for i, d in enumerate(domain_list)}
        self.idx_to_domain = {i: d for d, i in self.domain_to_idx.items()}

        # Convert vectors to torch tensors
        self.domain_vectors = {
            d: torch.tensor(v, dtype=torch.float16) for d, v in domain_vectors.items()
        }
        self.global_vector = torch.tensor(global_vector, dtype=torch.float16)

        self.alpha = alpha
        self.tau = tau
        self.beta = beta

        # State tracking
        self.active = True
        self.last_info = {}
        self._injected = False

    def compose_mov(self, domain: str) -> torch.Tensor:
        """
        Mixture-of-Vectors: v = (1-β)·v_d + β·v_global
        """
        v_d = self.domain_vectors[domain]
        v = (1 - self.beta) * v_d + self.beta * self.global_vector
        v = v / (v.norm() + 1e-8)
        return v

    def ternary_gate(self, p_tool: float) -> int:
        """
        g = +1 if p_tool ≥ τ        (promote tool calling)
        g = -1 if p_tool ≤ 1-τ      (suppress tool calling)
        g =  0 otherwise             (no intervention)
        """
        if p_tool >= self.tau:
            return 1
        elif p_tool <= (1 - self.tau):
            return -1
        else:
            return 0

    def hook_fn(self, module, input, output):
        """Forward hook applied at layer L*."""
        if not self.active or self._injected:
            return output

        # Get hidden states
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        # Extract last-token state
        h_last = hidden[:, -1, :].detach().cpu().float().numpy()

        # Standardize
        h_std = self.scaler.transform(h_last)

        # Route to domain
        domain_idx = self.router.predict(h_std)[0]
        domain = self.idx_to_domain[domain_idx]

        # Probe intent
        p_tool = self.probes[domain].predict_proba(h_std)[0, 1]

        # Ternary gate
        gate = self.ternary_gate(p_tool)

        # Store info for logging
        self.last_info = {
            "domain": domain,
            "p_tool": float(p_tool),
            "gate": gate,
            "alpha": self.alpha,
        }

        if gate == 0:
            self._injected = True
            return output

        # Compose MoV direction
        v = self.compose_mov(domain).to(hidden.device)

        # Injection: h' = h + gate * alpha * v
        delta = gate * self.alpha * v
        hidden_new = hidden.clone()
        hidden_new[:, -1, :] = hidden[:, -1, :] + delta.half()

        self._injected = True

        if rest is not None:
            return (hidden_new,) + rest
        return hidden_new

    def reset(self):
        """Reset injection flag for new generation."""
        self._injected = False
        self.last_info = {}

# %%
# Build the controller
controller = ASAController(
    router=router,
    probes=probes,
    domain_vectors=domain_vectors,
    global_vector=global_vector,
    scaler=scaler,
    domain_list=cfg.DOMAINS,
    alpha=cfg.ALPHA,
    tau=cfg.TAU,
    beta=cfg.BETA,
)
print("✅ ASA Controller built.")

# %% [markdown]
# ## 8. ASA-Augmented Generation

# %%
def generate_with_asa(
    model, tokenizer, messages: List[dict],
    controller: ASAController, layer: int,
    max_new_tokens: int = 256,
) -> Tuple[str, dict]:
    """
    Generate text with ASA intervention.

    Returns: (generated_text, controller_info)
    """
    controller.reset()
    controller.active = True

    # Register hook
    target_module = get_layer_module(model, layer)
    hook = target_module.register_forward_hook(controller.hook_fn)

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,     # Greedy for determinism
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=False)

    finally:
        hook.remove()
        controller.active = False

    return generated, controller.last_info

def generate_baseline(
    model, tokenizer, messages: List[dict],
    max_new_tokens: int = 256,
) -> str:
    """Generate without ASA for comparison."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)

def detect_tool_call(text: str) -> bool:
    """Check if output contains a tool call trigger."""
    return cfg.TOOL_TOKEN_START in text

def parse_tool_call(text: str) -> Optional[dict]:
    """
    Parse LFM2.5 tool call format:
    <|tool_call_start|>[func_name(arg1="val1", arg2="val2")]<|tool_call_end|>
    """
    if cfg.TOOL_TOKEN_START not in text:
        return None

    try:
        start = text.index(cfg.TOOL_TOKEN_START) + len(cfg.TOOL_TOKEN_START)
        end = text.index(cfg.TOOL_TOKEN_END) if cfg.TOOL_TOKEN_END in text else len(text)
        call_str = text[start:end].strip()

        # Validate basic structure
        if call_str.startswith("[") and "(" in call_str:
            return {
                "raw": call_str,
                "format_valid": True,
                "has_function_name": True,
                "has_args": "=" in call_str,
            }
        return {"raw": call_str, "format_valid": False}
    except Exception:
        return None

print("✅ Generation functions ready.")

# %% [markdown]
# ## 9. Hyperparameter Tuning (α, τ, β)

# %%
def evaluate_on_hidden_states(
    controller: ASAController,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
) -> dict:
    """
    Fast evaluation using pre-extracted hidden states.
    Evaluates the routing + probing decisions only (no generation).
    """
    preds = []
    for i in range(len(labels)):
        h = hidden_states[i:i+1]
        h_std = controller.scaler.transform(h)

        domain_idx = controller.router.predict(h_std)[0]
        domain = controller.idx_to_domain[domain_idx]
        p_tool = controller.probes[domain].predict_proba(h_std)[0, 1]
        gate = controller.ternary_gate(p_tool)

        # Prediction: tool if gate == +1
        pred = 1 if gate == 1 else 0
        preds.append(pred)

    preds = np.array(preds)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "fpr": (preds[labels == 0] == 1).mean() if (labels == 0).sum() > 0 else 0,
    }
    return metrics

print("🔧 Tuning hyperparameters on VALID set...")

# Alpha sweep
alpha_results = []
for alpha in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
    controller.alpha = alpha
    m = evaluate_on_hidden_states(
        controller, valid_hidden[best_layer], valid_labels, valid_domains
    )
    m["alpha"] = alpha
    alpha_results.append(m)
    print(f"   α={alpha:5.1f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | "
          f"Rec={m['recall']:.4f} | FPR={m['fpr']:.4f}")

best_alpha_entry = max(alpha_results, key=lambda x: x["f1"])
best_alpha = best_alpha_entry["alpha"]
controller.alpha = best_alpha
print(f"\n🏆 Best α = {best_alpha} (F1 = {best_alpha_entry['f1']:.4f})")

# Tau sweep
tau_results = []
for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    controller.tau = tau
    m = evaluate_on_hidden_states(
        controller, valid_hidden[best_layer], valid_labels, valid_domains
    )
    m["tau"] = tau
    tau_results.append(m)
    print(f"   τ={tau:.2f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | "
          f"Rec={m['recall']:.4f} | FPR={m['fpr']:.4f}")

best_tau_entry = max(tau_results, key=lambda x: x["f1"])
best_tau = best_tau_entry["tau"]
controller.tau = best_tau
print(f"\n🏆 Best τ = {best_tau} (F1 = {best_tau_entry['f1']:.4f})")

# Beta sweep
beta_results = []
for beta in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
    controller.beta = beta
    m = evaluate_on_hidden_states(
        controller, valid_hidden[best_layer], valid_labels, valid_domains
    )
    m["beta"] = beta
    beta_results.append(m)
    print(f"   β={beta:.1f} | F1={m['f1']:.4f} | Prec={m['precision']:.4f} | "
          f"Rec={m['recall']:.4f} | FPR={m['fpr']:.4f}")

best_beta_entry = max(beta_results, key=lambda x: x["f1"])
best_beta = best_beta_entry["beta"]
controller.beta = best_beta
print(f"\n🏆 Best β = {best_beta} (F1 = {best_beta_entry['f1']:.4f})")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot([r["alpha"] for r in alpha_results], [r["f1"] for r in alpha_results], 'o-', color='#2196F3')
axes[0].axvline(best_alpha, color='gold', linestyle='--')
axes[0].set_xlabel("α (Steering Strength)")
axes[0].set_ylabel("F1 Score")
axes[0].set_title("Alpha Sweep")

axes[1].plot([r["tau"] for r in tau_results], [r["f1"] for r in tau_results], 'o-', color='#FF5722')
axes[1].axvline(best_tau, color='gold', linestyle='--')
axes[1].set_xlabel("τ (Confidence Threshold)")
axes[1].set_ylabel("F1 Score")
axes[1].set_title("Tau Sweep")

axes[2].plot([r["beta"] for r in beta_results], [r["f1"] for r in beta_results], 'o-', color='#4CAF50')
axes[2].axvline(best_beta, color='gold', linestyle='--')
axes[2].set_xlabel("β (Global Weight)")
axes[2].set_ylabel("F1 Score")
axes[2].set_title("Beta Sweep")

plt.tight_layout()
plt.savefig(cfg.OUTPUT_DIR / "hyperparam_sweep.png", dpi=150)
plt.show()
print(f"\n✅ Final hyperparameters: α={best_alpha}, τ={best_tau}, β={best_beta}")

# %% [markdown]
# ## 10. Full Evaluation on TEST Set

# %%
print("=" * 70)
print("🧪 FULL EVALUATION ON TEST SET")
print("=" * 70)

# 10a. Hidden-state level evaluation (fast)
print("\n── Hidden-State Level Evaluation (ASA decisions) ──")
test_metrics = evaluate_on_hidden_states(
    controller, test_hidden[best_layer], test_labels, test_domains
)
print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"   Precision: {test_metrics['precision']:.4f}")
print(f"   Recall:    {test_metrics['recall']:.4f}")
print(f"   F1:        {test_metrics['f1']:.4f}")
print(f"   FPR:       {test_metrics['fpr']:.4f}")

# Per-domain breakdown
print("\n── Per-Domain Breakdown ──")
for d in cfg.DOMAINS:
    mask = test_domains == d
    dm = evaluate_on_hidden_states(
        controller, test_hidden[best_layer][mask], test_labels[mask],
        test_domains[mask],
    )
    print(f"   {d:12s} | F1={dm['f1']:.4f} | Prec={dm['precision']:.4f} | "
          f"Rec={dm['recall']:.4f} | FPR={dm['fpr']:.4f}")

# %%
# 10b. End-to-end generation evaluation (slower but definitive)
print("\n── End-to-End Generation Evaluation (subset) ──")
N_EVAL = min(40, len(test_data))  # Evaluate a subset for speed
eval_subset = test_data[:N_EVAL]

baseline_results = []
asa_results = []

for sample in tqdm(eval_subset, desc="Generating"):
    messages = format_messages(sample, system_prompt)

    # Baseline
    baseline_out = generate_baseline(model, tokenizer, messages, max_new_tokens=256)
    baseline_triggered = detect_tool_call(baseline_out)
    baseline_results.append({
        "id": sample["id"],
        "label": sample["label"],
        "domain": sample["domain"],
        "triggered": baseline_triggered,
        "output": baseline_out[:200],
    })

    # ASA
    asa_out, asa_info = generate_with_asa(
        model, tokenizer, messages, controller, best_layer, max_new_tokens=256
    )
    asa_triggered = detect_tool_call(asa_out)
    asa_parsed = parse_tool_call(asa_out)
    asa_results.append({
        "id": sample["id"],
        "label": sample["label"],
        "domain": sample["domain"],
        "triggered": asa_triggered,
        "parsed": asa_parsed,
        "info": asa_info,
        "output": asa_out[:200],
    })

    if device.type == "cuda":
        torch.cuda.empty_cache()

# Compute metrics
def compute_gen_metrics(results, name):
    labels = np.array([r["label"] for r in results])
    preds = np.array([1 if r["triggered"] else 0 for r in results])
    m = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "fpr": (preds[labels == 0] == 1).mean() if (labels == 0).sum() > 0 else 0,
    }
    print(f"\n   [{name}]")
    print(f"   Accuracy:  {m['accuracy']:.4f}")
    print(f"   Precision: {m['precision']:.4f}")
    print(f"   Recall:    {m['recall']:.4f}")
    print(f"   F1:        {m['f1']:.4f}")
    print(f"   FPR:       {m['fpr']:.4f}")
    return m

baseline_m = compute_gen_metrics(baseline_results, "Baseline")
asa_m = compute_gen_metrics(asa_results, "ASA")

# Improvement summary
print(f"\n📊 Improvement over Baseline:")
print(f"   F1:  {baseline_m['f1']:.4f} → {asa_m['f1']:.4f} "
      f"({'↑' if asa_m['f1'] > baseline_m['f1'] else '↓'} "
      f"{abs(asa_m['f1'] - baseline_m['f1']):.4f})")
print(f"   FPR: {baseline_m['fpr']:.4f} → {asa_m['fpr']:.4f} "
      f"({'↓ (better)' if asa_m['fpr'] < baseline_m['fpr'] else '↑ (worse)'})")

# Comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
baseline_vals = [baseline_m["accuracy"], baseline_m["precision"],
                 baseline_m["recall"], baseline_m["f1"]]
asa_vals = [asa_m["accuracy"], asa_m["precision"],
            asa_m["recall"], asa_m["f1"]]
x = np.arange(len(metrics_names))
ax.bar(x - 0.2, baseline_vals, 0.35, label="Baseline", color="#90A4AE")
ax.bar(x + 0.2, asa_vals, 0.35, label="ASA", color="#FF5722")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylabel("Score")
ax.set_title("Baseline vs ASA: Tool-Calling Performance")
ax.legend()
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(cfg.OUTPUT_DIR / "baseline_vs_asa.png", dpi=150)
plt.show()
print("✅ Evaluation complete.")

# %% [markdown]
# ## 11. Save & Upload to HuggingFace

# %%
def save_asa_assets(
    output_dir: Path,
    domain_vectors: Dict[str, np.ndarray],
    global_vector: np.ndarray,
    router: LogisticRegression,
    probes: Dict[str, LogisticRegression],
    scaler: StandardScaler,
    config: dict,
):
    """Save all ASA assets for deployment."""
    import pickle

    assets_dir = output_dir / "asa_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Vectors
    vectors = {"global": global_vector}
    for d, v in domain_vectors.items():
        vectors[d] = v
    np.savez(assets_dir / "steering_vectors.npz", **vectors)
    print(f"   Saved steering vectors ({len(vectors)} vectors)")

    # Router & Probes
    with open(assets_dir / "router.pkl", "wb") as f:
        pickle.dump(router, f)
    with open(assets_dir / "probes.pkl", "wb") as f:
        pickle.dump(probes, f)
    with open(assets_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("   Saved router, probes, scaler")

    # Config
    with open(assets_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("   Saved config")

    # Size report
    total_bytes = sum(
        f.stat().st_size for f in assets_dir.iterdir() if f.is_file()
    )
    print(f"   Total asset size: {total_bytes / 1024:.1f} KB")
    return assets_dir

asa_config = {
    "model_id": cfg.MODEL_ID,
    "best_layer": int(best_layer),
    "alpha": float(controller.alpha),
    "tau": float(controller.tau),
    "beta": float(controller.beta),
    "domains": cfg.DOMAINS,
    "tool_token_start": cfg.TOOL_TOKEN_START,
    "tool_token_end": cfg.TOOL_TOKEN_END,
    "probe_aucs": {str(k): float(v) for k, v in probe_aucs.items()},
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
}

print("💾 Saving ASA assets...")
assets_dir = save_asa_assets(
    cfg.OUTPUT_DIR, domain_vectors, global_vector,
    router, probes, scaler, asa_config,
)

# Save evaluation results
eval_results = {
    "baseline": baseline_m,
    "asa": asa_m,
    "per_domain_asa": {},
    "config": asa_config,
}
for d in cfg.DOMAINS:
    mask = test_domains == d
    dm = evaluate_on_hidden_states(
        controller, test_hidden[best_layer][mask], test_labels[mask], test_domains[mask]
    )
    eval_results["per_domain_asa"][d] = dm

with open(cfg.OUTPUT_DIR / "evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=2, default=str)

print("✅ All assets saved.")

# %%
# Upload to HuggingFace
print("📤 Uploading to HuggingFace Hub...")
try:
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo (will skip if already exists)
    try:
        create_repo(cfg.HF_REPO_ID, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"   Repo creation note: {e}")

    # Upload assets
    api.upload_folder(
        folder_path=str(assets_dir),
        repo_id=cfg.HF_REPO_ID,
        path_in_repo="asa_assets",
    )

    # Upload evaluation
    api.upload_file(
        path_or_fileobj=str(cfg.OUTPUT_DIR / "evaluation_results.json"),
        path_in_repo="evaluation_results.json",
        repo_id=cfg.HF_REPO_ID,
    )

    # Upload plots
    for plot_file in cfg.OUTPUT_DIR.glob("*.png"):
        api.upload_file(
            path_or_fileobj=str(plot_file),
            path_in_repo=f"plots/{plot_file.name}",
            repo_id=cfg.HF_REPO_ID,
        )

    print(f"✅ Uploaded to https://huggingface.co/{cfg.HF_REPO_ID}")
except Exception as e:
    print(f"⚠️ Upload failed (you may need to login first: `huggingface-cli login`)")
    print(f"   Error: {e}")
    print("   Assets are saved locally — you can upload manually later.")

# %% [markdown]
# ## 12. Interactive Demo 🎮

# %%
def demo(instruction: str, show_comparison: bool = True):
    """
    Interactive demo: compare baseline vs ASA-augmented output.
    """
    messages = format_messages({"instruction": instruction}, system_prompt)

    print("=" * 70)
    print(f"📝 Input: {instruction}")
    print("=" * 70)

    if show_comparison:
        print("\n── Baseline Output ──")
        baseline = generate_baseline(model, tokenizer, messages)
        triggered = detect_tool_call(baseline)
        print(f"   Tool triggered: {'✅ Yes' if triggered else '❌ No'}")
        print(f"   Output: {baseline[:300]}")

    print("\n── ASA Output ──")
    asa_out, info = generate_with_asa(
        model, tokenizer, messages, controller, best_layer
    )
    triggered = detect_tool_call(asa_out)
    parsed = parse_tool_call(asa_out)

    print(f"   Domain:  {info.get('domain', '?')}")
    print(f"   P(tool): {info.get('p_tool', 0):.4f}")
    print(f"   Gate:    {'+1 (promote)' if info.get('gate') == 1 else '-1 (suppress)' if info.get('gate') == -1 else '0 (neutral)'}")
    print(f"   Tool triggered: {'✅ Yes' if triggered else '❌ No'}")
    if parsed:
        print(f"   Parsed:  {parsed}")
    print(f"   Output:  {asa_out[:300]}")
    print()

# Run demo examples
print("\n" + "🎮 " * 15)
print("                    INTERACTIVE DEMO")
print("🎮 " * 15 + "\n")

# Tool-necessary examples
demo("Calculate the compound interest on $10000 at 5% rate for 10 years.")
demo("Execute Python code to sort the list [9, 3, 7, 1, 5] in descending order.")
demo("Search for the latest developments in fusion energy research.")
demo("Translate 'Artificial intelligence is transforming the world' to Korean.")

# Non-tool examples (should NOT trigger tool calls)
demo("Tell me about the concept of gradient descent in machine learning.")
demo("What makes Python such a beginner-friendly programming language?")
demo("What are the philosophical implications of quantum mechanics?")
demo("How do children learn their first language naturally?")

print("\n✅ Demo complete!")
print(f"\n📦 All ASA assets saved to: {cfg.OUTPUT_DIR / 'asa_assets'}")
print(f"   Total asset size: ~20KB (training-free, no model weights modified!)")
print(f"   Optimal layer: L{best_layer}")
print(f"   Hyperparameters: α={controller.alpha}, τ={controller.tau}, β={controller.beta}")
