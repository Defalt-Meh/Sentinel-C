#!/usr/bin/env python3
import sys, json, argparse, time
import numpy as np

try:
    import frameworkc as fc
except ImportError:
    print("Build the C extension first:\n"
          "  python -m pip install -U numpy\n"
          '  FWC_RELU_HID=1 FWC_NATIVE=1 python setup.py build_ext --inplace')
    sys.exit(1)

from helpers import Featurizer, find_spans, flags_from_spans, redact, train_risk

def _vec32(fe: Featurizer, text: str, flags) -> np.ndarray:
    """Featurize → float32, C-contiguous (best path for C backend)."""
    x = fe.vectorize(text, flags)
    # Ensure dtype/layout; avoid copies when already correct
    if not (isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags.c_contiguous):
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    return x

def _predict1_fast(net, x_row: np.ndarray) -> np.ndarray:
    """Single-sample inference via batch kernel on a 1×d slice (no Python loop)."""
    # x_row is shape (d,), wrap as (1, d) without copying
    x1 = x_row[None, :]
    return np.asarray(fc.predict_batch(net, x1), dtype=np.float32)[0]

def classify_and_redact(net, fe, text: str):
    spans = find_spans(text)
    flags = flags_from_spans(spans)
    x = _vec32(fe, text, flags)

    t0 = time.perf_counter_ns()
    o = _predict1_fast(net, x)
    dt_ms = (time.perf_counter_ns() - t0) / 1e6

    prob_needs = float(o[1])
    if spans:
        cls, conf, reason = "needs_redaction", max(prob_needs, 0.99), "detectors"
    else:
        cls, conf, reason = ("needs_redaction", prob_needs, "mlp") if prob_needs >= 0.60 else ("safe", 1.0 - prob_needs, "mlp")
    return {
        "class": cls,
        "prob": conf,
        "reason": reason,
        "redacted": redact(text, spans),
        "spans": spans,
        "latency_ms": round(dt_ms, 3),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--nhid", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    fe = Featurizer(dim=args.dim, ngrams=(3, 4, 5))
    nips = args.dim + 6
    print(">> Training tiny risk classifier (binary)...")
    t0 = time.time()
    net = train_risk(fe, nips=nips, nhid=args.nhid, epochs=args.epochs, lr=args.lr)
    print(f">> Train done in {time.time() - t0:.1f}s\n")

    # Warm-up a few inferences to stabilize alloc/caches
    _ = classify_and_redact(net, fe, "Warmup once.")
    _ = classify_and_redact(net, fe, "Warmup twice.")

    samples = [
        "Email me at alice@contoso.com about the contract.",
        "Schedule a meeting tomorrow at 10am.",
        "My card 4111 1111 1111 1111 please don't leak",
        "OpenAI key sk-abcDEF1234567890abcDEF1234567890",
    ]
    print("=== Samples ===")
    for s in samples:
        print(json.dumps(classify_and_redact(net, fe, s), ensure_ascii=False))

    print("\nEnter text (Ctrl-D to quit):")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        print(json.dumps(classify_and_redact(net, fe, line), ensure_ascii=False))

if __name__ == "__main__":
    # Optional: reduce oversubscription for lower latency
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        # Don’t override if user already set them
        import os
        os.environ.setdefault(k, "1")
    main()
