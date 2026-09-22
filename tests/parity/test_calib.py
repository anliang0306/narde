"""calib parity: narde.calib must match laya.common calibration helpers exactly."""
import math

import numpy as np

import pf
from laya.common import (
    clamp_temperature as L_clamp,
    confidence_from_probs as L_conf,
    ece_score as L_ece,
    temp_bucket as L_bucket,
)
from narde import calib as N


def test_clamp_temperature_parity():
    cases = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -1.0, float("nan"), float("inf"),
             float("-inf"), "abc", None, 3.14159, np.float64(0.9), "0.7"]
    for t in cases:
        assert L_clamp(t) == N.clamp_temperature(t), f"clamp differs on {t!r}"
    assert N.clamp_temperature(0.1, lo=0.3, hi=4.0) == L_clamp(0.1, 0.3, 4.0)


def test_temp_bucket_parity():
    for qt in range(3):
        for k in (1, 2, 3, 5, 6, 10, 11, 50):
            assert L_bucket(qt, k) == N.temp_bucket(qt, k), (qt, k)
    # boundary checks
    assert N.temp_bucket(0, 2) == "choice:2"
    assert N.temp_bucket(0, 3) == "choice:3-5"
    assert N.temp_bucket(0, 11) == "choice:11+"
    assert N.temp_bucket(1, 5) == "score:3-5"
    assert N.temp_bucket(2, 2) == "noul:2"


def test_confidence_from_probs_parity():
    rng = np.random.default_rng(42)
    for k in (1, 2, 3, 4, 8):
        p = rng.dirichlet(np.ones(k) * 2)
        # pad to a longer vector to exercise the p[:k] slice
        p_long = np.concatenate([p, np.zeros(10 - k)]) if k < 10 else p
        assert L_conf(p_long, k) == N.confidence_from_probs(p_long, k), k
    assert N.confidence_from_probs(np.array([0.7, 0.3]), 1) == 1.0
    assert L_conf(np.array([0.5, 0.5]), 2) == N.confidence_from_probs(np.array([0.5, 0.5]), 2)
    # degenerate: single option, k=1
    assert N.confidence_from_probs(np.array([1.0]), 1) == 1.0


def test_ece_score_parity():
    rng = np.random.default_rng(7)
    conf = rng.uniform(0, 1, 500)
    correct = (rng.uniform(0, 1, 500) < conf).astype(float)
    assert L_ece(conf, correct) == N.ece_score(conf, correct)
    for bins in (2, 5, 10, 15, 40):
        assert L_ece(conf, correct, bins=bins) == N.ece_score(conf, correct, bins=bins)
    # empty -> nan
    assert math.isnan(L_ece(np.array([]), np.array([])))
    assert math.isnan(N.ece_score(np.array([]), np.array([])))
    # all-confident correct
    assert N.ece_score(np.ones(50), np.ones(50)) == 0.0


def test_narde_extension_fit_temperature_runs():
    # narde-only helper: must return a valid temperature in [TEMP_MIN, TEMP_MAX]
    import torch

    g = torch.Generator().manual_seed(3)
    logits = torch.randn(64, 3, generator=g)
    target = torch.zeros(64, 3)
    target[torch.arange(64), torch.randint(0, 3, (64,), generator=g)] = 1.0
    t = N.fit_temperature(logits.numpy(), target.numpy())
    assert N.TEMP_MIN <= t <= N.TEMP_MAX
    assert N.fit_temperature(logits.numpy()[:3], target[:3]) == 1.0  # < n_samples_min
