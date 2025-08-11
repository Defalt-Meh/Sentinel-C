# helpers.py — shared logic for Sentinel-C (stress-hardened, deterministic)
#
# As one who loves both Truth and proofs, I approach this text as I would a catechism:
# each line a proposition, each function a doctrine. I yearn for certainty,
# yet the world (and Unicode) are fallen; therefore we hedge our regex,
# we map our indices carefully, and we test as if on the eve of judgment.

from __future__ import annotations
import re, random, time, math, unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple, Dict, Iterable, Sequence, Optional
import numpy as np

# =============================================================================
#                           UNICODE & OBFUSCATION
# =============================================================================
# If text is Babel, normalization is Pentecost. We prepare the string for discernment:
# strip the invisible tempters, fold confusable letters to their plainer forms,
# and reveal the obfuscations that masquerade as piety ("[at]" and "(dot)").

# Zero-width & soft characters commonly used to hide PII
_ZW = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",  # ZWSP, ZWNJ, ZWJ, WJ, BOM
    "\u00ad",  # soft hyphen — the serpent that cleaves words without sound
}
# Minimal confusables map (keep conservative to avoid over-normalizing)
# We do not presume to rewrite the created order entirely; only enough to see clearly.
_CONFUSABLES = {
    "œ": "oe", "Œ": "OE",
    "ø": "o",  "Ø": "O",
    "ß": "ss",
    "ð": "d",  "Ð": "D",
    "þ": "th", "Þ": "Th",
    "ł": "l",  "Ł": "L",
    "ı": "i",  "İ": "I",
    "ş": "s",  "Ş": "S",
    "ğ": "g",  "Ğ": "G",
    "ç": "c",  "Ç": "C",
    "á": "a", "à": "a", "ä": "a", "â": "a", "ã": "a", "å": "a",
    "Á": "A", "À": "A", "Ä": "A", "Â": "A", "Ã": "A", "Å": "A",
    "é": "e", "è": "e", "ë": "e", "ê": "e",
    "É": "E", "È": "E", "Ë": "E", "Ê": "E",
    "í": "i", "ì": "i", "ï": "i", "î": "i",
    "Í": "I", "Ì": "I", "Ï": "I", "Î": "I",
    "ó": "o", "ò": "o", "ö": "o", "ô": "o", "õ": "o",
    "Ó": "O", "Ò": "O", "Ö": "O", "Ô": "O", "Õ": "O",
    "ú": "u", "ù": "u", "ü": "u", "û": "u",
    "Ú": "U", "Ù": "U", "Ü": "U", "Û": "U",
    "ñ": "n", "Ñ": "N",
}

# The euphemisms that cloak addresses in modest attire; we convert them to plain truth.
_OBFUSCATION_PATTERNS = [
    (re.compile(r"\[\s*at\s*\]|\(\s*at\s*\)|\s+at\s+", re.I), "@"),
    (re.compile(r"\[\s*dot\s*\]|\(\s*dot\s*\)|\s+dot\s+", re.I), "."),
]

def _fold_confusables(s: str) -> str:
    # A gentle homily: restore letters to their simpler equivalents.
    return "".join(_CONFUSABLES.get(ch, ch) for ch in s)

def _strip_zero_width(s: str) -> str:
    # Cast out those characters which stumble the faithful tokenizer.
    return "".join(ch for ch in s if ch not in _ZW)

def _collapse_ws(s: str) -> str:
    # Keep newlines (for liturgical structure), but tame unruly whitespace to a single space.
    return re.sub(r"[ \t\r\f\v]+", " ", s)

def _apply_deobfuscations(s: str) -> str:
    # Replace the coy periphrases with forthright symbols; charity rejoices in the truth.
    out = s
    for pat, repl in _OBFUSCATION_PATTERNS:
        out = pat.sub(repl, out)
    return out

def normalize_for_detection(text: str) -> Tuple[str, List[int]]:
    """
    Return (normalized_text, idx_map), a concordance from the purified text
    back to its original scripture. We proceed in five movements:
    NFKC → strip zero-width/soft → fold confusables → deobfuscate → collapse whitespace.
    As Augustine warned, zeal without knowledge errs; thus we keep an index map.
    """
    idx_map: List[int] = []
    # Step 1: NFKC — like a council uniting schismatic glyphs under one creed.
    t_nfkc = unicodedata.normalize("NFKC", text)
    # Build initial map 1:1 — here I admit uncertainty: NFKC can reorder in rare cases.
    map1 = list(range(len(text)))
    # Step 2: strip zero-width / soft hyphen — removing stones from the pilgrim’s path.
    t1 = []
    map2 = []
    for j, ch in enumerate(t_nfkc):
        if ch in _ZW:
            continue
        t1.append(ch)
        # heuristic: map to nearest original index (bounded by len-1).
        # I confess, this is a pastoral accommodation, not an axiom.
        map2.append(min(j, len(text)-1))
    s1 = "".join(t1)
    # Step 3: fold confusables (length may change → map per char).
    # The Body is one, though its members many; we track each limb faithfully.
    t2 = []
    map3 = []
    for k, ch in enumerate(s1):
        repl = _CONFUSABLES.get(ch, ch)
        for _ in repl:
            t2.append(_)
            map3.append(map2[k])
    s2 = "".join(t2)
    # Step 4: deobfuscate [at]/(dot)/spelled tokens — the veils are lifted.
    s3 = _apply_deobfuscations(s2)
    # Mapping through regex replacements: we adopt a conservative alignment,
    # preferring humility to hubris. (A full Levenshtein backmap would be scholastic excess here.)
    def approximate_map(a: str, b: str, prev_map: List[int]) -> List[int]:
        i = j = 0
        out: List[int] = []
        while j < len(b):
            if i < len(a) and a[i] == b[j]:
                out.append(prev_map[i]); i += 1; j += 1
            else:
                # Seek until we find agreement — as monks debating until the bell tolls.
                while i < len(a) and (a[i] != b[j]):
                    i += 1
                if i < len(a):
                    out.append(prev_map[i]); i += 1; j += 1
                else:
                    # fallback to last known or beginning — a concession to human frailty.
                    out.append(prev_map[-1] if prev_map else 0); j += 1
        return out

    map4 = approximate_map(s2, s3, map3)
    # Step 5: collapse horizontal whitespace — many spaces, one utterance.
    s4 = _collapse_ws(s3)
    map5 = approximate_map(s3, s4, map4)

    return s4, map5

# =============================================================================
#                               DETECTORS
# =============================================================================
# Here our canons of detection: crafted to avoid both Pharisaic rigor (overreach)
# and laxity (neglect). Each regex pronounces on externals; validators probe the heart.

# Emails: ASCII-local + dotted domain. International text is normalized beforehand.
EMAIL_RE = re.compile(r'\b[a-zA-Z0-9._%+\-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,24}\b')

# Intl-ish phone; bounded length; later filtered by digit count and reserved ranges.
# We fear the scandal of false positives; thus digits are counted and certain fakes absolved.
PHONE_RE = re.compile(
    r'(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{2,4}[\s.\-]?\d{2,4}'
)

# 13–19 digits with separators; Luhn validated afterward — faith without works is dead.
CARD_RE  = re.compile(r'\b(?:\d[ \-]?){13,19}\b')

# JWT: allow optional whitespace around dots (handles line breaks and wrapped text).
# Mercy for the copy-pasted.
JWT_RE   = re.compile(r'\b[A-Za-z0-9_-]{8,}\s*\.\s*[A-Za-z0-9_-]{8,}\s*\.\s*[A-Za-z0-9_-]{8,}\b')

# IBAN (format only here; mod-97 below). Orthodoxy in both liturgy and checksum.
IBAN_RE  = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b')

# API keys: keep a general sk-… plus a few common prefixes without overreach.
# We resist the temptation to universalize; specificity is a virtue.
API_RE   = re.compile(r'\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{21})\b')

# ---- Validators --------------------------------------------------------------
# These are our examinations of conscience: Luhn and mod-97, simple yet searching.

def luhn_ok(s: str) -> bool:
    ds = [ord(c) - 48 for c in s if '0' <= c <= '9']
    if not (13 <= len(ds) <= 19):
        return False
    chk = 0
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
    t = s[4:] + s[:4]
    acc = 0
    for ch in t:
        v = ord(ch) - 55 if 'A' <= ch <= 'Z' else (ord(ch) - 48)
        acc = (acc * 10 + v) % 97
    return acc == 1

# ---- Span utils --------------------------------------------------------------
# Spans are our annotations in the margin of the sacred text: where the PII dwells,
# and of what species. If two annotations overlap, the weightier doctrine prevails.

Span = Tuple[int, int, str]  # (start, end, type)

SPAN_PRIORITY = {
    "credit_card": 90,
    "iban":        85,
    "api_key":     80,
    "jwt":         70,
    "email":       60,
    "phone":       50,
}

def _merge_spans(spans: List[Span]) -> List[Span]:
    if not spans: return []
    spans.sort(key=lambda x: (x[0], -x[1]))
    out: List[List[object]] = []
    for s, e, t in spans:
        if not out or s > out[-1][1]:
            out.append([s, e, t])  # a new paragraph in our commentary
        else:
            out[-1][1] = max(out[-1][1], e)
            if SPAN_PRIORITY.get(t, 0) > SPAN_PRIORITY.get(out[-1][2], 0):
                out[-1][2] = t  # the graver matter rules when doctrines conflict
    return [(int(s), int(e), str(t)) for s, e, t in out]

# --- Helpers for phone false-positives (US 555 test range, obvious dummies) ---
# Even the just make mistakes; these exemptions spare us needless scandal.
_555_TEST_RE = re.compile(r'\b555[\s.\-]?(?:01\d{2}|000[\s.\-]?\d{4})\b')  # 555-0100..0199 and 555-000-xxxx

def _project_spans(spans_norm: List[Span], idx_map: List[int], text_len: int) -> List[Span]:
    """Project spans from normalized string back to original indices.
    Our zeal must not overreach; projection guards the innocent text outside the span.
    """
    out: List[Span] = []
    for s, e, t in spans_norm:
        s0 = idx_map[min(max(s,0), len(idx_map)-1)]
        e0 = idx_map[min(max(e-1,0), len(idx_map)-1)] + 1
        s0 = max(0, min(s0, text_len))
        e0 = max(0, min(e0, text_len))
        if e0 > s0:
            out.append((s0, e0, t))
    return out

def _find_spans_one(text_norm: str) -> List[Span]:
    # A single liturgy over normalized text: regex rites, then moral tests (Luhn, mod-97).
    spans: List[Span] = []
    # Emails
    for m in EMAIL_RE.finditer(text_norm):
        spans.append((m.start(), m.end(), "email"))
    # Cards (Luhn-filtered)
    for m in CARD_RE.finditer(text_norm):
        seg = text_norm[m.start():m.end()]
        if luhn_ok(seg):
            spans.append((m.start(), m.end(), "credit_card"))
    # IBAN (mod-97)
    for m in IBAN_RE.finditer(text_norm):
        seg = text_norm[m.start():m.end()]
        if iban_ok(seg):
            spans.append((m.start(), m.end(), "iban"))
    # Phones (digit bounds + reserved/test guard)
    for m in PHONE_RE.finditer(text_norm):
        seg = text_norm[m.start():m.end()]
        digits = sum(1 for c in seg if c.isdigit())
        if 10 <= digits <= 14 and not _555_TEST_RE.search(seg):
            spans.append((m.start(), m.end(), "phone"))
    # JWT + API keys
    for m in JWT_RE.finditer(text_norm):
        spans.append((m.start(), m.end(), "jwt"))
    for m in API_RE.finditer(text_norm):
        spans.append((m.start(), m.end(), "api_key"))
    return spans

def find_spans(text: str) -> List[Span]:
    """
    We test two canonical variants:
      v0: normalized (NFKC + strip zero-width + deobfuscate + collapse ws)
      v1: v0 with explicit ' dot '→'.' to catch coy addresses.
    Then we project back, lest we redact beyond what justice demands.
    """
    v0, map0 = normalize_for_detection(text)
    spans0 = _find_spans_one(v0)

    # Variant v1: try to catch "john dot doe @ example dot com"
    v1 = re.sub(r"\s+dot\s+", ".", v0, flags=re.I)
    map1 = list(map0)  # approximation: ordinarily unchanged; I acknowledge a small leap of faith.
    spans1 = _find_spans_one(v1) if v1 != v0 else []

    # Project back and merge — unity without confusion, distinction without division.
    proj = _project_spans(spans0, map0, len(text))
    if spans1:
        proj += _project_spans(spans1, map1, len(text))
    return _merge_spans(proj)

# ---- Redaction ---------------------------------------------------------------
# Charity covers a multitude of sins; redaction covers a multitude of tokens.
# Each kind receives its proper vestment (mask character); defaults exist for the uncertain.

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
    # We do not tear the garment of the sentence; only the offending span is veiled.
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
# A confession checklist for the classifier: did we see email? card? a token of Babel?
# Stable order ensures our feature space is not tossed by every wind of change.

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
# Words may lie; distributions rarely do. We cast text into hashed n-gram space
# and append our moral flags. Normalization keeps pride at bay.

@dataclass
class Featurizer:
    dim: int = 4096
    ngrams: Tuple[int, ...] = (3, 4, 5)

    @staticmethod
    @lru_cache(maxsize=1_000_000)
    def _h_cached(chunk: bytes) -> int:
        # 32-bit FNV-1a — a humble hash, yet faithful in little and much.
        h = 2166136261
        for b in chunk:
            h ^= b
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def vectorize(self, text: str, flags: Dict[str, int]) -> np.ndarray:
        # We begin empty, as from dust, and accumulate counts which we later normalize.
        x = np.zeros(self.dim + len(FLAG_ORDER), dtype=np.float32)
        t = text.lower().encode("utf-8", "ignore")
        L = len(t)
        for n in self.ngrams:
            if L < n: continue
            for i in range(L - n + 1):
                h = self._h_cached(t[i:i+n]) % self.dim
                x[h] += 1.0
        # L2 normalization — grace to proportion zeal.
        norm = float(np.linalg.norm(x[:self.dim])) or 1.0
        x[:self.dim] /= norm
        # Append flags in FIXED order — the creed after the homily.
        base = self.dim
        for j, k in enumerate(FLAG_ORDER):
            x[base + j] = float(flags.get(k, 0))
        return x

# =============================================================================
#                           SYNTH DATA & TRAINING
# =============================================================================
# We fashion a parable dataset: some texts harmless as doves, others serpentine with PII.
# The classifier is catechized upon these until it discerns good from evil at a glance.

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

def _r_email(rng: random.Random) -> str:
    # The fictive saints of corporate domains; their intercession is harmless.
    doms  = ("contoso.com","fabrikam.org","outlook.com","gmail.com")
    name  = ''.join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(7))
    return f"{name}@{rng.choice(doms)}"

def _r_card(rng: random.Random) -> str:
    # A card whose digits are immaculate with respect to Luhn.
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
    # Let us speak Turkish (+90), for we are among friends.
    return f"+90 {rng.randrange(100,999)} {rng.randrange(100,999)} {rng.randrange(10,99)}"

def _r_iban(rng: random.Random) -> str:
    # Render unto Caesar a TR-format string; its checksum shall be tested later.
    return "TR" + str(rng.randrange(10,99)) + ''.join(rng.choice("0123456789") for _ in range(22))

def _r_jwt(rng: random.Random) -> str:
    # A triune token: header.payload.signature — and we forgive its line breaks.
    chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    s=lambda n: ''.join(rng.choice(chars) for _ in range(n))
    return f"{s(16)}.{s(32)}.{s(32)}"

def _r_api(rng: random.Random) -> str:
    # Keys that open no heaven, but do tempt logging.
    chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "sk-" + ''.join(rng.choice(chars) for _ in range(32))

def _make_dataset(n: int, fe: Featurizer, seed: int = 42):
    # A balanced homily: half virtue (safe), half vice (PII).
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
    # Shuffle — lest the model be tempted by order rather than substance.
    order = list(range(n))
    rng.shuffle(order)
    texts  = [texts[i] for i in order]
    labels = [labels[i] for i in order]
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
    We instruct a modest MLP, not unlike a novice in the cloister:
    simple habits (2 classes), small hidden life, steady learning rate.
    Early stopping prevents ascetic excess.
    """
    import frameworkc as fc
    texts, X, Y, y = _make_dataset(6000, fe, seed=seed)
    idx = np.arange(len(texts))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    tr = idx[:int(0.8 * len(idx))]
    va = idx[int(0.8 * len(idx)):]
    Xtr, Ytr = X[tr], Y[tr]
    Xva, yva = X[va], y[va]
    net = fc.build(int(nips), int(nhid), 2, int(seed))  # 2 classes — yes and no, as in any catechism.
    best_acc = -1.0
    best_snapshot = None
    bad_epochs = 0

    def batch_slices(n: int, bs: int) -> List[slice]:
        # A rhythm for training — like the Hours — dividing the day into faithful portions.
        return [slice(i, min(i+bs, n)) for i in range(0, n, bs)]

    sl_tr = batch_slices(len(Xtr), batch)

    for ep in range(epochs):
        order = rng.permutation(len(Xtr))
        Xsh, Ysh = Xtr[order], Ytr[order]
        for sl in sl_tr:
            xb = Xsh[sl]; yb = Ysh[sl]
            try:
                fc.train_batch(net, xb, yb, float(lr))
            except Exception:
                # When communal prayer fails, the hermit continues alone: one sample at a time.
                for i in range(sl.start, sl.stop):
                    fc.train_one(net, Xsh[i], Ysh[i], float(lr))
        O = np.asarray(fc.predict_batch(net, Xva), dtype=np.float32)
        acc = float((O.argmax(axis=1) == yva).mean())
        print(f"epoch {ep+1:02d} | val acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_snapshot = fc.clone(net) if hasattr(fc, "clone") else net
            bad_epochs = 0
        else:
            bad_epochs += 1
            if early_stop_patience and bad_epochs >= early_stop_patience:
                if best_snapshot is not None and hasattr(fc, "assign"):
                    fc.assign(net, best_snapshot)
                break
    return net
