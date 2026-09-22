"""Prompt-protocol parity: narde.prompts must be byte-for-byte laya.common."""
import json

import pf
from laya.common import (
    build_sequence as L_build,
    collate_items as L_collate,
    render_criterion as L_rc,
    render_options as L_ro,
    serialize_state as L_ss,
)
from narde.prompts import (
    build_sequence as N_build,
    collate_items as N_collate,
    render_criterion as N_rc,
    render_options as N_ro,
    serialize_state as N_ss,
)


def test_serialize_state_parity():
    for s in pf.STATES:
        assert L_ss(s) == N_ss(s), f"serialize_state diverged on {s!r}"


def test_render_criterion_parity():
    vals = ["plain", 0, False, None, 3.14, {"a": 1, "b": [2, 3]}, [1, 2], {"nested": {"x": "中文值"}}]
    for v in vals:
        assert L_rc(v) == N_rc(v), f"render_criterion diverged on {v!r}"


def test_render_options_parity():
    for q in pf.Q_INT:
        assert L_ro(q) == N_ro(q), f"render_options diverged on {q!r}"


def test_build_sequence_parity_sweep():
    n = 0
    for state, q, kw in pf.all_prompt_cases():
        lid, lmk = L_build(pf.TOK, state, q, **kw)
        nid, nmk = N_build(pf.TOK, state, q, **kw)
        assert lid == nid, f"ids diverge: state={state!r} q={q} kw={kw}\nlaya={lid}\nnarde={nid}"
        assert lmk == nmk, f"markers diverge: state={state!r} q={q} kw={kw}"
        n += 1
    assert n >= 100, f"expected a large sweep, only {n} cases ran"


def test_collate_items_parity():
    import torch

    items_groups = []
    batch = []
    for i, (state, q, kw) in enumerate(pf.all_prompt_cases()):
        if i % 7 == 0:
            continue
        ids, markers = L_build(pf.TOK, state, q, **kw)
        batch.append({"ids": ids, "markers": markers, "qtype": i % 3, "label": i % 4, "qid": f"q{i}"})
        if i % 3 == 0:
            batch[-1]["target"] = [0.0, 1.0][: len(markers)]
        items_groups.append(list(batch[-1:]) )

    lg = L_collate(items_groups, 0)
    ng = N_collate(items_groups, 0)
    assert set(lg.keys()) == set(ng.keys())
    for k, v in lg.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, ng[k]), f"collate tensor {k} diverged"
        elif k == "meta":
            assert v == ng[k]
        else:
            assert v == ng[k]

    assert L_collate([[]], 0) is None
    assert N_collate([[]], 0) is None
