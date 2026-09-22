"""Calibration utilities: temperature clamping, ECE, confidence.

Adapted from laya/common.py (Apache-2.0, Convai Innovations) — see NOTICE.md.
`fit_temperature` is a narde extension (laya fits temperatures inside its training
notebook; see docs/divergence.md).
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from narde.prompts import QTYPE_NAMES

# A fitted temperature below 1 sharpens the logits instead of softening them. The shipped
# `choice:11+` bucket is 0.1006, which multiplies them ~10x: a 0.24 top probability is published as
# 0.99, so a caller gating on confidence is told a coin flip is a certainty. No honest calibration
# needs to sharpen this hard, so refuse to apply one that does.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):    # NaN / inf
        return 1.0
    return min(hi, max(lo, t))


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k)."""
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


def temp_bucket(qtype: int, k: int) -> str:
    """Bucket key used by per-bucket temperatures, e.g. 'choice:3-5', 'noul:2'."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def fit_temperature(
    logits: np.ndarray,
    target: np.ndarray,
    qtype: int = 0,
    k: int = 2,
    lo: float = TEMP_MIN,
    hi: float = TEMP_MAX,
    n_samples_min: int = 10,
) -> float:
    """narde extension: golden-section NLL temperature fit on held-out logits.

    Mirrors the refit step laya runs after fine-tuning (see its training notebook).
    Returns 1.0 when fewer than `n_samples_min` samples are available.
    """
    del qtype, k  # bucket handled by caller; keep signature stable
    if len(logits) < n_samples_min:
        return 1.0

    def nll(temp: float) -> float:
        t = clamp_temperature(temp, lo, hi)
        z = logits / t
        z = z - z.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p = p / p.sum(axis=-1, keepdims=True)
        p = np.clip(p, 1e-12, 1.0)
        return float(-(target * np.log(p)).sum(axis=-1).mean())

    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(d)
    return clamp_temperature((a + b) / 2, lo, hi)


__all__ = [
    "TEMP_MIN",
    "TEMP_MAX",
    "clamp_temperature",
    "ece_score",
    "confidence_from_probs",
    "temp_bucket",
    "fit_temperature",
]
