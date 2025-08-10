# helpers.py — shared logic for Sentinel-C (detailed, faster, deterministic)

from __future__ import annotations
import re, random, time, math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Dict, Iterable, Sequence, Optional
import numpy as np

# =============================================================================
#                               DETECTORS
# =============================================================================

# Tighter patterns (avoid pathological backtracking, fewer FPs on short text)
EMAIL_RE = re.compile(
    r'\b[a-zA-Z0-9._%+\-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,24}\b'
)
# Intl phone (loose but bounded), e.g., +1 415-555-1212, (415) 555 1212, 0212...
PHONE_RE = re.compile(
    r'(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{2,4}[\s.\-]?\d{2,4}'
)
# 13–19 digits with separators; Luhn filter applied after
CARD_RE  = re.compile(r'\b(?:\d[ \-]?){13,19}\b')
# JWT: three base64url segments
JWT_RE   = re.compile(r'\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b')
# IBAN (basic format check; mod-97 verified below)
IBAN_RE  = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b')
# Example “sk-” key (keep generic)
API_RE   = re.compile(r'\b(sk-[A-Za-z0-9]{20,})\b')

# ---- Validators --------------------------------------------------------------

def luhn_ok(s: str) -> bool:
    ds = [ord(c) - 48 for c in s if '0' <= c <= '9']
    if not (13 <= len(ds) <= 19): 
        return False
    chk = 0
    # Double every second digit from the right
    for i, d in enumerate(reversed(ds)):
        if (i & 1) == 1:
            d = d * 2
            if d > 9: d -= 9
        chk += d
    return (chk % 10) == 0

def iban_ok(s: str) -> bool:
    s = s.replace(" ", "").upper()
    if not (15 <= len(s) <= 34): 
        return False
    # Move first 4 chars to end, convert letters to numbers A=10..Z=35
    t = s[4:] + s[:4]
    acc = 0
    for ch in t:
        v = ord(ch) - 55 if 'A' <= ch <= 'Z' else (ord(ch) - 48)
        acc = (acc * 10 + v) % 97
    return acc == 1

# ---- Span utils --------------------------------------------------------------

Span = Tuple[int, int, str]  # (start, end, type)

# Priority: if overlapping, keep the more specific/stronger type
SPAN_PRIORITY = {
    "credit_card": 90,
    "iban":        85,
    "api_key":     80,
    "jwt":         70,
    "email":       60,
    "phone":       50,
}

def _merge_spans(spans: List[Span]) -> List[Span]:
    """Merge overlapping spans; keep the higher-priority type."""
    if not spans:
        return []
    spans.sort(key=lambda x: (x[0], -x[1]))
    out: List[List[object]] = []
    for s, e, t in spans:
        if not out or s > out[-1][1]:
            out.append([s, e, t])
        else:
            # overlap: extend, pick higher priority type
            out[-1][1] = max(out[-1][1], e)
            if SPAN_PRIORITY.get(t, 0) > SPAN_PRIORITY.get(out[-1][2], 0):
                out[-1][2] = t
    return [(int(s), int(e), str(t)) for s, e, t in out]

def find_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    # Emails
    for m in EMAIL_RE.finditer(text):
        spans.append((m.start(), m.end(), "email"))
    # Cards (Luhn‑filtered)
    for m in CARD_RE.finditer(text):
        seg = text[m.start():m.end()]
        if luhn_ok(seg):
            spans.append((m.start(), m.end(), "credit_card"))
    # IBAN (mod‑97)
    for m in IBAN_RE.finditer(text):
        seg = text[m.start():m.end()]
        if iban_ok(seg):
            spans.append((m.start(), m.end(), "iban"))
    # Phones (bound digits)
    for m in PHONE_RE.finditer(text):
        digits = sum(1 for c in text[m.start():m.end()] if c.isdigit())
        if 10 <= digits <= 14:
            spans.append((m.start(), m.end(), "phone"))
    # JWT + API keys
    for m in JWT_RE.finditer(text):
        spans.append((m.start(), m.end(), "jwt"))
    for m in API_RE.finditer(text):
        spans.append((m.start(), m.end(), "api_key"))
    return _merge_spans(spans)

# ---- Redaction ---------------------------------------------------------------

REDACT_TOKENS = {
    "email":       "█",
    "credit_card": "•",
    "iban":        "•",
    "phone":       "•",
    "jwt":         "×",
    "api_key":     "×",
    "default":     "*",
}

def redact(text: str, spans: Sequence[Span], mask_char: Optional[str] = None) -> str:
    """Type‑aware redaction; defaults to per‑type mask, else mask_char."""
    if not spans:
        return text
    out = []
    last = 0
    for s, e, t in spans:
        out.append(text[last:s])
        char = mask_char or REDACT_TOKENS.get(t, REDACT_TOKENS["default"])
        out.append(char * (e - s))
        last = e
    out.append(text[last:])
    return "".join(out)

# ---- Flags (stable order!) ---------------------------------------------------

FLAG_ORDER = ("has_email","has_card","has_phone","has_iban","has_jwt","has_api")

def flags_from_spans(spans: Sequence[Span]) -> Dict[str, int]:
    f = {k: 0 for k in FLAG_ORDER}
    for _, _, t in spans:
        if   t == "email":       f["has_email"] = 1
        elif t == "credit_card": f["has_card"]  = 1
        elif t == "phone":       f["has_phone"] = 1
        elif t == "iban":        f["has_iban"]  = 1
        elif t == "jwt":         f["has_jwt"]   = 1
        elif t == "api_key":     f["has_api"]   = 1
    return f

# =============================================================================
#                               FEATURIZER
# =============================================================================

@dataclass
class Featurizer:
    dim: int = 4096
    ngrams: Tuple[int, ...] = (3, 4, 5)

    @staticmethod
    @lru_cache(maxsize=1_000_000)  # tiny per‑process cache helps short inputs
    def _h_cached(chunk: bytes) -> int:
        # 32‑bit FNV‑1a (stable across runs)
        h = 2166136261
        for b in chunk:
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def vectorize(self, text: str, flags: Dict[str, int]) -> np.ndarray:
        # allocate once; float32 + row‑major for C backend
        x = np.zeros(self.dim + len(FLAG_ORDER), dtype=np.float32)
        t = text.lower().encode("utf-8", "ignore")
        L = len(t)

        for n in self.ngrams:
            if L < n:
                continue
            # n‑gram hashing + count
            for i in range(L - n + 1):
                h = self._h_cached(t[i:i+n]) % self.dim
                x[h] += 1.0

        # L2 normalize hashed part (avoid div by zero)
        norm = float(np.linalg.norm(x[:self.dim])) or 1.0
        x[:self.dim] /= norm

        # Append flags in FIXED order (stable feature layout!)
        base = self.dim
        for j, k in enumerate(FLAG_ORDER):
            x[base + j] = float(flags.get(k, 0))
        return x

# =============================================================================
#                           SYNTH DATA & TRAINING
# =============================================================================

SAFE_TEMPL = (
    "Let's schedule a meeting tomorrow at 10am.",
    "Can you summarize this document for me?",
    "Where is the latest roadmap file?",
    "Draft a polite email to the client about the delay.",
    "What is the office wifi name?",
    "Remind me to submit the timesheet.",
)
PII_TEMPL = (
    "My email is {email}, reach me there.",
    "Card: {card} exp 12/28",
    "Phone: {phone}, call me after 6.",
    "IBAN: {iban} please wire the refund.",
    "JWT: {jwt} keep it safe.",
    "OpenAI key {api} do not leak.",
)

# ---- Random generators (deterministic if seed set before) -------------------

def _r_email(rng: random.Random) -> str:
    doms  = ("contoso.com","fabrikam.org","outlook.com","gmail.com")
    name  = ''.join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7))
    return f"{name}@{rng.choice(doms)}"

def _r_card(rng: random.Random) -> str:
    # Generate a Visa 16 with valid Luhn
    base = [4] + [rng.randrange(10) for _ in range(14)]
    s = 0
    for i, d in enumerate(reversed(base)):
        if (i & 1) == 0: s += d
        else:
            x = d * 2
            s += (x - 9) if x > 9 else x
    check = (10 - (s % 10)) % 10
    return ''.join(map(str, base + [check]))

def _r_phone(rng: random.Random) -> str:
    return f"+90 {rng.randrange(100,999)} {rng.randrange(100,999)} {rng.randrange(10,99)}"

def _r_iban(rng: random.Random) -> str:
    return "TR" + str(rng.randrange(10,99)) + ''.join(rng.choice("0123456789") for _ in range(22))

def _r_jwt(rng: random.Random) -> str:
    chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    s=lambda n: ''.join(rng.choice(chars) for _ in range(n))
    return f"{s(16)}.{s(32)}.{s(32)}"

def _r_api(rng: random.Random) -> str:
    chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "sk-" + ''.join(rng.choice(chars) for _ in range(32))

def _make_dataset(n: int, fe: Featurizer, seed: int = 42):
    """Balanced synthetic dataset; returns (texts, X, Y_onehot, y_idx)."""
    rng = random.Random(seed)
    n_pos = n // 2
    n_neg = n - n_pos

    texts: List[str] = []
    labels: List[int] = []

    for _ in range(n_neg):
        texts.append(rng.choice(SAFE_TEMPL))
        labels.append(0)

    for _ in range(n_pos):
        t = rng.choice(PII_TEMPL).format(
            email=_r_email(rng), card=_r_card(rng), phone=_r_phone(rng),
            iban=_r_iban(rng), jwt=_r_jwt(rng), api=_r_api(rng)
        )
        texts.append(t); labels.append(1)

    # Shuffle deterministically
    order = list(range(n))
    rng.shuffle(order)
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]

    # Featurize
    X = np.zeros((n, fe.dim + len(FLAG_ORDER)), dtype=np.float32)
    Y = np.zeros((n, 2), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    for i, t in enumerate(texts):
        sp = find_spans(t)
        fl = flags_from_spans(sp)
        X[i] = fe.vectorize(t, fl)
        y[i] = labels[i]
        Y[i, labels[i]] = 1.0

    return texts, X, Y, y

# ---- Training with Framework‑C ----------------------------------------------

def train_risk(
    fe: Featurizer,
    nips: int,
    nhid: int = 128,
    lr: float = 0.01,
    epochs: int = 6,
    batch: int = 256,
    seed: int = 42,
    early_stop_patience: int = 4,
) -> "fc.Network":
    """
    Trains a tiny binary MLP in Framework‑C on synthetic data produced to align with
    the detectors. Keeps the old printed logs for compatibility.
    """
    import frameworkc as fc  # local import to avoid import cycles

    texts, X, Y, y = _make_dataset(6000, fe, seed=seed)

    # Split
    idx = np.arange(len(texts))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    tr = idx[:int(0.8 * len(idx))]
    va = idx[int(0.8 * len(idx)):]

    Xtr, Ytr = X[tr], Y[tr]
    Xva, yva = X[va], y[va]

    net = fc.build(int(nips), int(nhid), 2, int(seed))  # 2 classes
    best_acc = -1.0
    best_snapshot = None
    bad_epochs = 0

    # Precompute slices for batches (lower Python overhead)
    def batch_slices(n: int, bs: int) -> List[slice]:
        return [slice(i, min(i+bs, n)) for i in range(0, n, bs)]

    sl_tr = batch_slices(len(Xtr), batch)

    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        Xsh, Ysh = Xtr[order], Ytr[order]

        # Train epoch
        for sl in sl_tr:
            xb = Xsh[sl]; yb = Ysh[sl]
            try:
                fc.train_batch(net, xb, yb, float(lr))
            except Exception:
                # Scalar fallback (should be rare)
                for i in range(sl.start, sl.stop):
                    fc.train_one(net, Xsh[i], Ysh[i], float(lr))

        # Validation — use batched predict for speed
        O = np.asarray(fc.predict_batch(net, Xva), dtype=np.float32)
        acc = float((O.argmax(axis=1) == yva).mean())
        print(f"epoch {ep+1:02d} | val acc={acc:.3f}")

        # Track best
        if acc > best_acc:
            best_acc = acc
            best_snapshot = fc.clone(net) if hasattr(fc, "clone") else net
            bad_epochs = 0
        else:
            bad_epochs += 1
            if early_stop_patience and bad_epochs >= early_stop_patience:
                # Early stop; keep best
                if best_snapshot is not None and hasattr(fc, "assign"):
                    fc.assign(net, best_snapshot)
                break

    return net
