"""DecisionModel + training-math parity."""
import numpy as np
import torch

import pf
from laya.common import amp_dtype as L_amp, proper_reward as L_pr, td_lambda_targets as L_td
from narde.model import amp_dtype as N_amp, proper_reward as N_pr, td_lambda_targets as N_td


def _batch(n=3, L=12, kmax=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 1899, (n, L), generator=g)
    att = torch.ones(n, L, dtype=torch.long)
    att[1, 6:] = 0
    mpos = torch.randint(0, L, (n, kmax), generator=g)
    mmask = torch.ones(n, kmax, dtype=torch.bool)
    mmask[0, 3:] = False
    qtype = torch.tensor([0, 1, 2][:n])
    return ids, att, mpos, mmask, qtype


def test_forward_parity():
    lm, nm, _ = pf.shared_decision_models()
    ids, att, mpos, mmask, qt = _batch(seed=1)
    with torch.no_grad():
        llog, lact = lm(ids, att, mpos, mmask, qt)
        nlog, nact = nm(ids, att, mpos, mmask, qt)
    assert pf.approx_eq(llog, nlog), f"logits diverge: { (llog - nlog).abs().max() }"
    assert pf.approx_eq(lact, nact), f"act_logits diverge: { (lact - nact).abs().max() }"
    assert llog.shape == nlog.shape and lact.shape == nact.shape


def test_forward_parity_masked_detach():
    lm, nm, _ = pf.shared_decision_models()
    ids, att, mpos, mmask, qt = _batch(n=2, kmax=5, seed=2)
    with torch.no_grad():
        l1, _ = lm(ids, att, mpos, mmask, qt, detach_encoder=True)
        n1, _ = nm(ids, att, mpos, mmask, qt, detach_encoder=True)
    assert pf.approx_eq(l1, n1)


def test_single_option_no_crash_parity():
    # regression (laya #96): a single-marker question must not crash topk(2)
    lm, nm, _ = pf.shared_decision_models()
    ids, att = torch.randint(1000, 1899, (1, 8)), torch.ones(1, 8, dtype=torch.long)
    mpos, mmask = torch.tensor([[0]]), torch.ones(1, 1, dtype=torch.bool)
    qt = torch.tensor([0])
    with torch.no_grad():
        ll, la = lm(ids, att, mpos, mmask, qt)
        nl, na = nm(ids, att, mpos, mmask, qt)
    assert ll.shape == (1, 1) and torch.isfinite(ll).all()
    assert pf.approx_eq(ll, nl) and pf.approx_eq(la, na)


def test_no_head_parity():
    lm, nm = pf.shared_decision_models_no_head()
    ids, att, mpos, mmask, qt = _batch(n=2, L=8, kmax=2, seed=3)
    with torch.no_grad():
        ll, la = lm(ids, att, mpos, mmask, qt)
        nl, na = nm(ids, att, mpos, mmask, qt)
    assert pf.approx_eq(ll, nl) and pf.approx_eq(la, na)


def test_proper_reward_parity():
    g = torch.Generator().manual_seed(11)
    q = torch.rand(5, 6, generator=g)
    q = q / q.sum(-1, keepdim=True)
    target = torch.zeros(5, 6)
    target[:, 3] = 1.0
    target[0] = torch.softmax(torch.rand(6, generator=g), -1)  # a soft target
    qtype = torch.tensor([0, 1, 2, 1, 0])
    mask = torch.ones(5, 6, dtype=torch.long)
    mask[4, 4:] = 0
    assert pf.approx_eq(L_pr(q, target, qtype, mask), N_pr(q, target, qtype, mask))
    # custom weights
    assert pf.approx_eq(L_pr(q, target, qtype, mask, w_sph=0.2, w_rps=0.5, log_floor=-5.0),
                       N_pr(q, target, qtype, mask, w_sph=0.2, w_rps=0.5, log_floor=-5.0))


def test_td_lambda_targets_parity():
    p_true = torch.tensor([0.2, 0.5, 0.9, 0.1, 0.7])
    batch = {
        "target": torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.0, 1.0], [0.5, 0.5]]),
        "ep_group": torch.tensor([0, 0, 0, 1, -1]),
        "ep_step": torch.tensor([0, 1, 2, 0, 0]),
    }
    for lam in (0.0, 0.5, 1.0):
        assert pf.approx_eq(L_td(p_true, batch, lam), N_td(p_true, batch, lam))
    # no groups -> passthrough
    assert torch.equal(L_td(p_true, {"target": batch["target"]}), N_td(p_true, {"target": batch["target"]}))


def test_amp_dtype_parity():
    assert L_amp("bf16") == N_amp("bf16") == torch.bfloat16
    assert L_amp("fp16") == N_amp("fp16") == torch.float16
    assert L_amp(None) == N_amp(None)
