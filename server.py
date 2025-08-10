#!/usr/bin/env python3
# server.py — Sentinel-C Sidecar with Dark UI + Per-Model Latency Compare (fast FC)

import os, time, random
from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# C backend (built from src/my_module.c + src/nn.c)
import frameworkc as fc

# Shared helpers
try:
    from helpers import Featurizer, find_spans, flags_from_spans, redact, train_risk
except ImportError as e:
    raise ImportError(
        "server.py needs helpers.py (Featurizer, find_spans, flags_from_spans, redact, train_risk). "
        "Create helpers.py using the snippet I provided earlier."
    ) from e

# ── Optional baselines (controlled by env) ─────────────────────────
TORCH_ON = os.getenv("SC_TORCH", "0") == "1"
SK_ON    = os.getenv("SC_SK", "0")    == "1"

try:
    if TORCH_ON:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        TORCH_OK = True
    else:
        TORCH_OK = False
except Exception:
    TORCH_OK = False

try:
    if SK_ON:
        from sklearn.neural_network import MLPClassifier
        SK_OK = True
    else:
        SK_OK = False
except Exception:
    SK_OK = False

# -------------------- Config --------------------
DIM   = int(os.getenv("SC_DIM", "4096"))
NHID  = int(os.getenv("SC_NHID","128"))
EPOCH = int(os.getenv("SC_EPOCH","6"))  # Framework-C warmup
LR    = float(os.getenv("SC_LR","0.01"))

# Baseline knobs
SHADOW_N     = int(os.getenv("SC_SHADOW_N", "240")) # synthetic train size
TORCH_EPOCHS = int(os.getenv("SC_TORCH_E", "3"))
SK_MAXIT     = int(os.getenv("SC_SK_MAXIT", "30"))

# Recommended to avoid oversubscription on tiny batches
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "1")

fe = Featurizer(dim=DIM, ngrams=(3,4,5))
NIPS = DIM + 6

print(">> Bootstrapping Framework-C model…")
t0 = time.time()
net = train_risk(fe, nips=NIPS, nhid=NHID, lr=LR, epochs=EPOCH)
print(f">> Framework-C ready in {time.time()-t0:.1f}s")

# Warm-up to stabilize allocator/caches
_ = fc.predict_batch(net, np.zeros((1, NIPS), dtype=np.float32))

# --- Fast-path helpers for Framework-C (float32 + contiguous + 1×d batch) ---
def _vec32(fe: Featurizer, text: str, flags) -> np.ndarray:
    x = fe.vectorize(text, flags)
    if not (isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags.c_contiguous):
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return x

def _predict1_fc(net, x_row: np.ndarray) -> np.ndarray:
    x1 = x_row[None, :]  # view, no copy
    return np.asarray(fc.predict_batch(net, x1), dtype=np.float32)[0]

# ───────────────────────────────────────────────────────────────────
# Baseline helpers: tiny synthetic set that matches our detectors
# ───────────────────────────────────────────────────────────────────
def _synth_samples(n: int) -> List[str]:
    pos_patterns = [
        "My card 4111 1111 1111 1111 please don't leak",
        "Email me at bob@example.com about the invoice",
        "OpenAI key sk-abcDEF1234567890abcDEF1234567890",
        "Send SSN 123-45-6789 to HR now",
        "IBAN TR12 3456 7890 1234 5678 9012 34 keep private",
    ]
    neg_patterns = [
        "Schedule a meeting tomorrow at 10am.",
        "Let’s discuss the contract next week.",
        "The weather is great for a walk.",
        "Please review the draft report by Friday.",
        "I enjoyed the workshop on algorithms.",
    ]
    out = []
    for _ in range(n):
        s = random.choice(pos_patterns if random.random() < 0.5 else neg_patterns)
        out.append(s)
    return out

def _make_xy(fe: Featurizer, n: int):
    texts = _synth_samples(n)
    X = np.zeros((n, NIPS), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i, t in enumerate(texts):
        spans = find_spans(t)
        flags = flags_from_spans(spans)
        X[i] = _vec32(fe, t, flags)
        y[i] = 1 if len(spans) > 0 else 0
    return X, y

# ── Torch baseline (optional) ───────────────────────────────────────
_torch_model: Optional["nn.Module"] = None
def _torch_build_train():
    global _torch_model
    if not TORCH_OK: return
    class TorchMLP(nn.Module):
        def __init__(self, nips, nhid, nops=2):
            super().__init__()
            self.fc1 = nn.Linear(nips, nhid, bias=True)
            self.fc2 = nn.Linear(nhid, nops, bias=True)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    model = TorchMLP(NIPS, NHID, 2).to("cpu")
    X, y = _make_xy(fe, SHADOW_N)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 64
    model.train()
    for _ in range(TORCH_EPOCHS):
        for i in range(0, len(X_t), bs):
            xb, yb = X_t[i:i+bs], y_t[i:i+bs]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
    _torch_model = model.eval()

def _torch_predict_prob(x_vec: np.ndarray) -> float:
    assert _torch_model is not None
    with torch.no_grad():
        t = torch.from_numpy(x_vec.astype(np.float32))[None, :]
        logits = _torch_model(t).cpu().numpy().reshape(-1)
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    return float(p[1])

# ── scikit-learn baseline (optional) ────────────────────────────────
_sk_model: Optional["MLPClassifier"] = None
def _sk_build_train():
    global _sk_model
    if not SK_OK: return
    X, y = _make_xy(fe, SHADOW_N)
    clf = MLPClassifier(hidden_layer_sizes=(NHID,),
                        activation="relu",
                        solver="adam",
                        alpha=0.0,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        max_iter=SK_MAXIT,
                        n_iter_no_change=10,
                        random_state=42,
                        verbose=False)
    clf.fit(X, y)
    _sk_model = clf

def _sk_predict_prob(x_vec: np.ndarray) -> float:
    assert _sk_model is not None
    p = _sk_model.predict_proba(x_vec.reshape(1, -1))[0]
    return float(p[1])

# Build baselines if enabled
if TORCH_OK:
    print(">> Building Torch baseline…")
    t1 = time.time()
    _torch_build_train()
    print(f">> Torch baseline ready in {time.time()-t1:.1f}s")
if SK_OK:
    print(">> Building scikit-learn baseline…")
    t1 = time.time()
    _sk_build_train()
    print(f">> Sklearn baseline ready in {time.time()-t1:.1f}s")

# -------------------- Core gate --------------------
def gate(text: str) -> Dict[str, Any]:
    # detectors + features (outside model timing)
    spans = find_spans(text)
    flags = flags_from_spans(spans)
    x = _vec32(fe, text, flags)

    # Framework-C (model-only latency, fast single-sample path)
    t_fc0 = time.perf_counter_ns()
    o_fc = _predict1_fc(net, x)
    fc_ms = (time.perf_counter_ns() - t_fc0) / 1e6
    prob_needs_fc = float(o_fc[1])

    # Baselines: measure per-call latency on the same vector
    comp = {
        "frameworkc": {"latency_ms": round(fc_ms, 3), "prob_needs": prob_needs_fc}
    }

    if TORCH_OK and _torch_model is not None:
        t2 = time.perf_counter_ns()
        p = _torch_predict_prob(x)
        comp["torch"] = {"latency_ms": round((time.perf_counter_ns()-t2)/1e6, 3),
                         "prob_needs": p}

    if SK_OK and _sk_model is not None:
        t3 = time.perf_counter_ns()
        p = _sk_predict_prob(x)
        comp["sklearn"] = {"latency_ms": round((time.perf_counter_ns()-t3)/1e6, 3),
                           "prob_needs": p}

    # Policy: force redaction if detectors fired
    if spans:
        cls, conf, reason = "needs_redaction", max(prob_needs_fc, 0.99), "detectors"
    else:
        cls, conf, reason = ("needs_redaction", prob_needs_fc, "mlp") if prob_needs_fc >= 0.60 else ("safe", 1.0-prob_needs_fc, "mlp")

    return {
        "class": cls,
        "prob": conf,
        "reason": reason,
        "redacted": redact(text, spans),
        "spans": spans,
        "latency_ms": round(fc_ms, 3),   # model-only latency
        "models": comp
    }

# -------------------- API --------------------
app = FastAPI(title="Sentinel-C Gate", version="0.3")

class GateIn(BaseModel):
    text: str

class GateBatchIn(BaseModel):
    texts: List[str]

@app.get("/health")
def health(): return {"ok": True}

@app.post("/v1/gate")
def gate_one(req: GateIn): return gate(req.text)

# Faster batch: single FC batched call; per-item results
@app.post("/v1/gate/batch")
def gate_batch(req: GateBatchIn):
    items = req.texts
    spans_list, flags_list, rows = [], [], []
    for t in items:
        sp = find_spans(t)
        spans_list.append(sp)
        flags = flags_from_spans(sp)
        flags_list.append(flags)
        rows.append(_vec32(fe, t, flags))
    X = np.stack(rows, axis=0) if rows else np.zeros((0, NIPS), dtype=np.float32)

    t0 = time.perf_counter_ns()
    O = np.asarray(fc.predict_batch(net, X), dtype=np.float32) if len(items) else np.zeros((0,2), dtype=np.float32)
    fc_total_ms = (time.perf_counter_ns() - t0) / 1e6
    per_item_ms = round(fc_total_ms / max(1, len(items)), 3)

    out = []
    for i, text in enumerate(items):
        prob_needs = float(O[i, 1])
        spans = spans_list[i]
        if spans:
            cls, conf, reason = "needs_redaction", max(prob_needs, 0.99), "detectors"
        else:
            cls, conf, reason = ("needs_redaction", prob_needs, "mlp") if prob_needs >= 0.60 else ("safe", 1.0-prob_needs, "mlp")
        out.append({
            "class": cls,
            "prob": conf,
            "reason": reason,
            "redacted": redact(text, spans),
            "spans": spans,
            "latency_ms": per_item_ms,
            "models": {"frameworkc": {"latency_ms": per_item_ms, "prob_needs": prob_needs}}
        })
    return out

# -------------------- Dark UI @ "/" --------------------
HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sentinel-C — PII Redaction & Risk Gate</title>
<style>
  :root{
    --bg:#0a0f1a; --bg2:#0b1223; --card:#0f1629; --ink:#e6eefb; --muted:#9fb1cc;
    --border:#1a2746; --primary:#6ea8fe; --accent:#7ef0c1; --danger:#ff8aa1; --warn:#ffd166;
    --chip:#1a2746; --chip-safe:#133b2a; --chip-need:#3c1f2a;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:radial-gradient(1200px 600px at 10% -20%, #0e1a31 0,transparent 60%),linear-gradient(#0a0f1a,#0b1223);color:var(--ink);font:14px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Inter,Arial}
  header{display:flex;gap:14px;align-items:center;padding:18px;border-bottom:1px solid var(--border);position:sticky;top:0;background:linear-gradient(180deg,rgba(11,18,35,.9),rgba(11,18,35,.6))}
  .logo{width:36px;height:36px;border-radius:12px;background:linear-gradient(135deg,#304b93,#1e2f57);display:grid;place-items:center;font-weight:800}
  .wrap{max-width:1100px;margin:22px auto;padding:0 18px}
  .grid{display:grid;gap:14px;grid-template-columns:1fr;align-items:start}
  @media(min-width:1000px){.grid{grid-template-columns:1.3fr .7fr}}
  .card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.25)}
  h1{font-size:18px;margin:0}
  .muted{color:var(--muted)}
  textarea{width:100%;min-height:140px;background:#0a1224;border:1px solid var(--border);border-radius:12px;color:var(--ink);padding:12px;resize:vertical}
  .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}
  button{background:#2f63d1;border:none;color:#fff;padding:10px 14px;border-radius:12px;font-weight:600;cursor:pointer}
  button.secondary{background:#18233f;color:#cfe0ff}
  button:disabled{opacity:.6;cursor:not-allowed}
  .badge{display:inline-flex;gap:8px;align-items:center;padding:6px 10px;border-radius:999px;background:var(--chip);font-weight:700;letter-spacing:.2px}
  .badge.safe{background:var(--chip-safe);color:#a7f3d0}
  .badge.need{background:var(--chip-need);color:#ffc5cf}
  .pill{background:var(--chip);padding:6px 8px;border-radius:999px}
  .result{white-space:pre-wrap;background:#0a1122;border:1px solid var(--border);border-radius:12px;padding:10px}
  .table{width:100%;border-collapse:collapse}
  .table th,.table td{padding:8px 10px;border-bottom:1px solid var(--border);text-align:left}
  .table th{color:#a7b8d8}
  .hl{background:#2a3b1f;border-bottom:1px dashed #78d28e;padding:0 2px;border-radius:2px}
  .bar{height:10px;width:100%;background:#131f3a;border-radius:999px;overflow:hidden}
  .bar span{display:block;height:100%;width:0;background:linear-gradient(90deg,#5aa2ff,#76f3c8);transition:width .35s ease}
  .gauge{--p:0; width:56px;height:56px;border-radius:50%;background:
      conic-gradient(#76f3c8 calc(var(--p)*1%), #1b2747 0),
      radial-gradient(closest-side,#0f1629 76%,transparent 78% 100%,#0000 0),
      conic-gradient(#23325a 0 100%);display:grid;place-items:center;font-weight:800;color:#c9e7ff}
  .chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
  .chip{background:#15213f;border:1px solid var(--border);color:#bcd0ff;padding:6px 10px;border-radius:999px;cursor:pointer}
  footer{padding:10px 18px;color:#91a8cc}

  /* NEW: model compare styles */
  .mc-row{display:grid;grid-template-columns:120px 90px 1fr 90px;gap:10px;align-items:center;margin:8px 0}
  .mc-name{font-weight:700}
  .mc-bar{height:10px;background:#131f3a;border-radius:999px;overflow:hidden}
  .mc-bar > span{display:block;height:100%;width:0;background:linear-gradient(90deg,#6ea8fe,#7ef0c1);transition:width .35s}
  .muted-small{color:#9fb1cc;font-size:12px}
</style>
</head>
<body>
<header>
  <div class="logo">SC</div>
  <div>
    <h1>Sentinel-C — PII Redaction & Risk Gate</h1>
    <div class="muted">Local, CPU-only C MLP · Great for Copilot/RAG sidecar</div>
  </div>
</header>

<div class="wrap grid">
  <section class="card">
    <div class="muted">Input</div>
    <textarea id="input" placeholder="Paste text… e.g. My card 4111 1111 1111 1111 please don’t leak"></textarea>
    <div class="chips" id="examples">
      <span class="chip">Email me at alice@contoso.com about the contract.</span>
      <span class="chip">Schedule a meeting tomorrow at 10am.</span>
      <span class="chip">My card 4111 1111 1111 1111 please don't leak</span>
      <span class="chip">OpenAI key sk-abcDEF1234567890abcDEF1234567890</span>
    </div>
    <div class="row">
      <button id="analyze">Analyze</button>
      <button class="secondary" id="batch" title="Each line becomes a request">Analyze Batch</button>
      <span id="status" class="muted">Ready.</span>
    </div>

    <div class="row" style="margin-top:12px">
      <div class="gauge" id="gauge" title="Confidence">0</div>
      <span id="badge" class="badge">—</span>
      <span id="lat" class="muted">latency: —</span>
      <span id="reason" class="muted">reason: —</span>
    </div>

    <div style="margin-top:12px;">
      <div class="muted">Probability needs_redaction</div>
      <div class="bar"><span id="probbar"></span></div>
    </div>

    <div style="margin-top:12px;">
      <div class="muted">Redacted</div>
      <div class="result" id="redacted">—</div>
    </div>

    <div style="margin-top:12px;">
      <div class="muted">Spans</div>
      <div id="spans" class="result" style="background:#0d1426;">—</div>
    </div>

    <!-- NEW: Model comparison -->
    <div style="margin-top:16px;">
      <div class="muted">Model Comparison (per-input inference)</div>
      <div id="mcWrap" class="muted-small">Framework-C only. Enable Torch/Sklearn baselines via env vars.</div>
    </div>
  </section>

  <aside class="card">
    <div class="muted" style="margin-bottom:8px;">Batch Results</div>
    <div id="batchPanel" class="muted">Paste multiple lines and click <b>Analyze Batch</b>.</div>
  </aside>
</div>

<footer class="wrap">All inference runs locally. No data leaves your machine.</footer>

<script>
const $ = s => document.querySelector(s);
const input = $('#input'), probbar = $('#probbar'), gauge = $('#gauge');
const badge = $('#badge'), lat = $('#lat'), reason = $('#reason'), statusEl=$('#status');
const red = $('#redacted'), spansEl = $('#spans'), batchPanel = $('#batchPanel');
const mcWrap = $('#mcWrap');

function busy(v){ $('#analyze').disabled = v; $('#batch').disabled = v; statusEl.textContent = v ? 'Working…' : 'Ready.'; }
function esc(s){ return s.replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m])); }
function paintBadge(cls){
  badge.classList.remove('safe','need');
  if (cls==='needs_redaction'){ badge.textContent='NEEDS REDACTION'; badge.classList.add('need'); }
  else { badge.textContent='SAFE'; badge.classList.add('safe'); }
}
function highlight(redacted, spans){
  if(!spans || spans.length===0) return esc(redacted);
  let out='', cur=0;
  for(const [s,e,t] of spans){
    out += esc(redacted.slice(cur, s));
    out += '<span class="hl" title="'+t+'">'+esc(redacted.slice(s,e))+'</span>';
    cur = e;
  }
  out += esc(redacted.slice(cur));
  return out;
}
function setGauge(prob, cls){
  const p = Math.round(100 * (cls==='needs_redaction' ? prob : (1-prob)));
  probbar.style.width = p+'%';
  gauge.style.setProperty('--p', p);
  gauge.textContent = p;
}
async function postJSON(url, body){
  const r = await fetch(url,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error('HTTP '+r.status);
  return r.json();
}

// render per-model comparison with deltas vs Framework-C
function renderModelCompare(models){
  if(!models || !models.frameworkc){
    mcWrap.textContent = 'No comparison data (enable SC_TORCH=1 and/or SC_SK=1).';
    return;
  }
  const entries = Object.entries(models);
  const fc = models.frameworkc;
  const maxMs = Math.max(...entries.map(([_,v])=>v.latency_ms));
  const bestMs = Math.min(...entries.map(([_,v])=>v.latency_ms));

  const rows = entries.map(([name, v])=>{
    const nice = name === 'frameworkc' ? 'Framework‑C' : (name === 'torch' ? 'PyTorch' : 'scikit‑learn');
    const w = maxMs > 0 ? Math.round(100 * v.latency_ms / maxMs) : 0;

    const delta = v.latency_ms - fc.latency_ms;
    const deltaStr = (delta >= 0 ? '+' : '−') + Math.abs(delta).toFixed(3) + ' ms';

    let rel = v.latency_ms / fc.latency_ms;
    let relStr = '';
    if (name === 'frameworkc') {
      relStr = 'baseline';
    } else if (rel >= 1) {
      relStr = (rel).toFixed(2) + '× slower';
    } else {
      relStr = (1/rel).toFixed(2) + '× faster';
    }

    const prob = (100 * (v.prob_needs ?? 0)).toFixed(1) + '%';
    const isBest = v.latency_ms === bestMs;
    const star = isBest ? ' ⭐' : '';

    return `<div class="mc-row" title="${nice}">
      <div class="mc-name">${nice}${star}</div>
      <div>${v.latency_ms.toFixed(3)} ms</div>
      <div class="mc-bar"><span style="width:${w}%"></span></div>
      <div>${deltaStr} · ${relStr} · p₁=${prob}</div>
    </div>`;
  }).join('');

  mcWrap.innerHTML = rows;
}

$('#analyze').addEventListener('click', async ()=>{
  const text = input.value.trim(); if(!text) return;
  busy(true);
  try{
    const r = await postJSON('/v1/gate', {text});
    paintBadge(r.class);
    setGauge(r.prob, r.class);
    red.innerHTML = highlight(r.redacted, r.spans);
    spansEl.innerHTML = r.spans?.length ? r.spans.map(([s,e,t])=>`[${s}, ${e}, ${t}]`).join('  ') : '—';
    lat.textContent = 'latency: ' + r.latency_ms + ' ms';
    reason.textContent = 'reason: ' + r.reason;
    renderModelCompare(r.models);
  }catch(err){
    statusEl.textContent = 'Error: ' + err.message;
  }finally{
    busy(false);
  }
});

$('#batch').addEventListener('click', async ()=>{
  const lines = input.value.split('\n').map(s=>s.trim()).filter(Boolean);
  if(lines.length===0) return;
  busy(true); batchPanel.textContent = 'Working…';
  try{
    const res = await postJSON('/v1/gate/batch', {texts: lines});
    const rows = res.map((r,i)=>{
      const p = Math.round(100*(r.class==='needs_redaction'?r.prob:(1-r.prob)));
      return `<tr>
        <td>${i+1}</td>
        <td><span class="badge ${r.class==='needs_redaction'?'need':'safe'}">${r.class}</span></td>
        <td>${r.latency_ms} ms</td>
        <td>${p}%</td>
        <td>${r.spans?.length||0}</td>
        <td class="result" style="background:#0a1122">${highlight(r.redacted, r.spans)}</td>
      </tr>`;
    }).join('');
    batchPanel.innerHTML = `<table class="table">
      <thead><tr><th>#</th><th>class</th><th>latency</th><th>conf</th><th>#spans</th><th>redacted</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }catch(err){
    batchPanel.textContent = 'Error: ' + err.message;
  }finally{
    busy(false);
  }
});

document.querySelectorAll('#examples .chip').forEach(chip=>{
  chip.addEventListener('click', ()=>{ input.value = chip.textContent; });
});
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
