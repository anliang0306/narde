"""shortlist parity: ranking, passthrough, error behaviour must match laya."""
import numpy as np
import pf  # noqa: F401  (bootstrap sys.path)
from laya.shortlist import embed_fn_from_agent, predict_shortlist as L_ps, shortlist_choice as L_sc
from narde.shortlist import embed_fn_from_agent as N_efa, predict_shortlist as N_ps, shortlist_choice as N_sc


def _embed(texts):
    """Deterministic bag-of-char vectors; stable across both packages."""
    vecs = []
    for t in texts:
        v = np.zeros(32, dtype=np.float64)
        for i, ch in enumerate(t):
            v[i % 32] += ord(ch)
        n = np.linalg.norm(v)
        vecs.append(v / n if n else v)
    return np.asarray(vecs)


def test_shortlist_choice_parity():
    state = "my invoice payment failed and I need it refunded quickly"
    criteria = {f"cat{i:02d}": f"category {i} about payments invoices and refunds and billing" for i in range(12)}
    for k in (1, 3, 20):
        l = L_sc(state, criteria, _embed, k=k, instructions="pick the category")
        n = N_sc(state, criteria, _embed, k=k, instructions="pick the category")
        assert l == n, (k, l, n)
    # passthrough: k >= n -> original order, embed_fn NOT called
    calls = {"n": 0}

    def counting_embed(texts):
        calls["n"] += 1
        return _embed(texts)

    l = L_sc("x", ["a", "b"], counting_embed, k=5)
    assert l == ["a", "b"] and calls["n"] == 0
    assert N_sc("x", ["a", "b"], counting_embed, k=5) == ["a", "b"] and calls["n"] == 0

    # list-form criteria
    l = L_sc(state, [f"cat{i}" for i in range(12)], _embed, k=4)
    assert l == N_sc(state, [f"cat{i}" for i in range(12)], _embed, k=4)


def test_shortlist_error_parity():
    import laya.shortlist as Ls
    import narde.shortlist as Ns

    cases = [
        lambda m: m.shortlist_choice("s", {"a": 1}, _embed, k=0),
        lambda m: m.shortlist_choice("s", {"a": 1}, _embed, k=True),
        lambda m: m.shortlist_choice("s", {}, _embed, k=2),
        lambda m: m.shortlist_choice("s", {"a": 1, "a": 2} or {"a": 1}, _embed, k=2),  # dup not possible in dict
        lambda m: m.shortlist_choice("s", 42, _embed, k=2),
        lambda m: m.shortlist_choice("s", ["a", "b"], 5, k=1),
        lambda m: m.predict_shortlist(object(), "s", {"q": {"type": "choice", "criteria": {"a": 1, "b": 2}}}, _embed, k=1),
    ]
    for i, c in enumerate(cases):
        for exc in (ValueError, TypeError):
            try:
                c(Ls)
                l_exc = None
            except (ValueError, TypeError) as e:
                l_exc = e
            try:
                c(Ns)
                n_exc = None
            except (ValueError, TypeError) as e:
                n_exc = e
            assert type(l_exc) is type(n_exc) or (l_exc is None and n_exc is None), (i, l_exc, n_exc)
            if l_exc is not None:
                assert str(l_exc) == str(n_exc), (i, str(l_exc), str(n_exc))


class _FakeAgent:
    """Records the questions it receives and returns a fixed dict (like system_one)."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions, **kw):
        self.calls.append(questions)
        return {"model": "fake", "answers": {q: {"n": len(d.get("criteria") or [])} for q, d in questions.items()}}


def test_predict_shortlist_parity():
    la, na = _FakeAgent(), _FakeAgent()
    questions = {
        "cat": {"type": "choice", "instructions": "pick", "criteria": {f"cat{i:02d}": f"description {i}" for i in range(10)}},
        "urgent": {"type": "noul", "instructions": "urgent?"},
        "severity": {"type": "score", "instructions": "how bad", "criteria": ["low", "high"]},
    }
    state = {"message": "my payment failed twice"}
    ll = L_ps(la, state, questions, _embed, k=3)
    nn = N_ps(na, state, questions, _embed, k=3)
    assert ll["shortlist"] == nn["shortlist"], (ll["shortlist"], nn["shortlist"])
    assert ll["answers"] == nn["answers"]
    # reduced criteria passed to the agent must be identical
    assert la.calls[0] == na.calls[0]
    assert len(la.calls[0]["cat"]["criteria"]) == 3

    # passthrough: k >= n_labels -> embed_fn not called, questions untouched
    calls = {"n": 0}

    def counting(texts):
        calls["n"] += 1
        return _embed(texts)

    la2, na2 = _FakeAgent(), _FakeAgent()
    small = {"cat": {"type": "choice", "criteria": {"a": 1, "b": 2}}}
    ll = L_ps(la2, "s", small, counting, k=5)
    nn = N_ps(na2, "s", small, counting, k=5)
    assert calls["n"] == 0
    assert ll == nn
    assert ll["shortlist"]["cat"]["passthrough"] is True


def test_embed_fn_from_agent_parity():
    lm, nm, _ = pf.shared_decision_models()
    import torch

    class Agent:
        def __init__(self, model):
            self.tok = pf.BatchTok()
            self.model = model
            self.device = torch.device("cpu")

    texts = ["please refund my duplicate charge", "the invoice was wrong", "x"]
    efa_l = embed_fn_from_agent(Agent(lm))
    efa_n = N_efa(Agent(nm))
    al, an = efa_l(texts), efa_n(texts)
    assert al.shape == an.shape == (3, 32)
    assert np.allclose(al, an, atol=1e-6)
    # validation parity
    import laya.shortlist as Ls
    import narde.shortlist as Ns
    for ml, bs in ((0, 32), (1, 0), ("10", 32), (10, "8")):
        try:
            Ls.embed_fn_from_agent(Agent(lm), max_length=ml, batch_size=bs)
            le = None
        except ValueError as e:
            le = str(e)
        try:
            Ns.embed_fn_from_agent(Agent(nm), max_length=ml, batch_size=bs)
            ne = None
        except ValueError as e:
            ne = str(e)
        assert le == ne, (ml, bs, le, ne)
