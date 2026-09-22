"""Shared fixtures for the laya <-> narde parity suite.

Everything here is dependency-light and deterministic so the suite runs offline
(no HF downloads, no GPU). Importing this module also puts both `src/` and
`reference/laya/` on sys.path so `import laya` / `import narde` both work whether the
suite is driven by pytest or by run.py.
"""
import contextlib
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_ROOT, "src")
_LAYA = os.path.join(_ROOT, "reference", "laya")

for _p in (_SRC, _LAYA, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Keep the process quiet and single-threaded for reproducibility on CPU.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


# ----------------------------------------------------------------------------
# Deterministic word-level mock tokenizer (word -> stable id, no special ids).
# Shared by laya.* and narde.* so build_sequence inputs/outputs are directly
# comparable. Python's builtin hash() is salted per-process, so use FNV instead.
# ----------------------------------------------------------------------------
def _w2id(word: str) -> int:
    h = 2166136261
    for ch in word:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return 1000 + (h % 900)  # word ids live in [1000, 1899)


class BatchTok:
    """Batching mock tokenizer: returns fixed-width (B, 12) tensors with pad zeros."""

    def __call__(self, texts, **kw):
        import torch

        rows = []
        for t in texts:
            toks = [_w2id(w) % 1500 + 1 for w in str(t).split()][:8]
            rows.append((toks + [0] * 12)[:12])  # fixed width 12
        ids = torch.tensor(rows, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class MockTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103
    pad_token_id = 0
    mask_token = "[MASK]"
    pad_token = "[PAD]"
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    vocab_size = 2048

    def __call__(self, text, *a, **kw):
        return {"input_ids": [_w2id(w) for w in str(text).split()]}


TOK = MockTokenizer()


# ----------------------------------------------------------------------------
# Sample battery. `states` are raw states; `qs_int` are *internal* question dicts
# (keys t/ins/crit) used directly against build_sequence on both sides.
# ----------------------------------------------------------------------------
LONG_STATE = " ".join(["lorem ipsum dolor sit amet"] * 60)  # forces truncation

STATES = [
    "The invoice for March was charged twice, please refund.",
    {"message": "Mein Konto wurde zweimal belastet", "sender": "a@b.com"},
    ["I was charged twice", {"note": "urgent", "n": 2}],
    "",
    "123 456 789",
    "please refund [MASK] the extra charge now",
    LONG_STATE,
    {"a": {"b": ["x", {"c": 1}]}, "msg": "hello world"},
]

Q_INT = [
    {"t": "choice", "ins": "What does the customer want?", "crit": {"refund": "money back", "other": "anything else"}},
    {"t": "choice", "ins": "Category?", "crit": {"a": "x", "b": 0, "c": False, "d": None, "e": ""}},
    {"t": "choice", "ins": "Deep rubric", "crit": {"k": {"desc": "nested", "w": 3}, "j": ["list", "crit"]}},
    {"t": "score", "ins": "How urgent is it?", "crit": ["low", "medium", "high"]},
    {"t": "score", "ins": "Severity", "crit": [{"u": 1}, "plain text"]},
    {"t": "noul", "ins": "Is it time sensitive?", "crit": None},
    {"t": "noul", "ins": "Valid?", "crit": {"false": "no reason", "true": "yes reason"}},
    {"t": "noul", "ins": "One-sided", "crit": {"true": "only true is described"}},
]

# (max_len, head_max_len) sweeps, including tiny budgets that force the re-truncate path.
PARAMS = [
    (512, 192), (1024, 256), (64, 32), (32, 16), (16, 8),
]


def all_prompt_cases():
    """Yield (state, q_int, kwargs) combos that both sides must map identically."""
    import random
    rng = random.Random(1234)
    for state in STATES:
        for q in Q_INT:
            for (ml, hml) in PARAMS:
                yield state, dict(q), {"max_len": ml, "head_max_len": hml}
                # alternate ordering / truncation
                yield state, dict(q), {"max_len": ml, "head_max_len": hml,
                                       "option_order": list(reversed(range(len(_opt_count(q)))))}
                yield state, dict(q), {"max_len": ml, "head_max_len": hml, "truncate_left": True}
    # high-cardinality choice to force the per-option re-truncation branch
    big = {"t": "choice", "ins": "pick one", "crit": {f"opt{rng.randint(0,999)}": f"description number {i} words here" for i in range(14)}}
    for state in STATES[:3]:
        for ml, hml in PARAMS:
            yield state, big, {"max_len": ml, "head_max_len": hml}


def _opt_count(q):
    crit = q.get("crit")
    if q["t"] == "noul":
        return [0, 1]
    if crit is None:
        return []
    return range(len(crit))


# ----------------------------------------------------------------------------
# Tiny BERT factory + a shared DecisionModel with identical weights.
# ----------------------------------------------------------------------------
def tiny_bert(seed: int = 7):
    import torch
    from transformers import BertConfig, BertModel
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=2048, hidden_size=32, num_hidden_layers=1,
                     num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=128, hidden_dropout_prob=0.0,
                     attention_probs_dropout_prob=0.0)
    return BertModel(cfg).eval()


def shared_decision_models():
    """Return (laya_model, narde_model, state_dict) with identical weights.

    The narde model is loaded from the laya model's state_dict, which doubles as a
    key/shape compatibility check between the two architectures.
    """
    from laya.common import DecisionModel as LayaDM
    from narde.model import DecisionModel as NardeDM
    laya_m = LayaDM(tiny_bert(), head_layers=2, n_act=2).eval()
    narde_m = NardeDM(tiny_bert(), head_layers=2, n_act=2)
    narde_m.load_state_dict(laya_m.state_dict(), strict=True)
    narde_m.eval()
    return laya_m, narde_m, laya_m.state_dict()


def shared_decision_models_no_head():
    from laya.common import DecisionModel as LayaDM
    from narde.model import DecisionModel as NardeDM
    laya_m = LayaDM(tiny_bert(9), head_layers=0, n_act=2).eval()
    narde_m = NardeDM(tiny_bert(9), head_layers=0, n_act=2)
    narde_m.load_state_dict(laya_m.state_dict(), strict=True)
    narde_m.eval()
    return laya_m, narde_m


# ----------------------------------------------------------------------------
# Tiny fake agent for embed_fn_from_agent parity (batched tokenizer + real encoder).
# ----------------------------------------------------------------------------
def fake_embed_agent():
    import torch
    enc = tiny_bert(5).eval()

    class BatchTok:
        def __call__(self, texts, **kw):
            import torch

            rows = []
            for t in texts:
                toks = [_w2id(w) % 1500 + 1 for w in str(t).split()][:8]
                rows.append(toks + [0] * (12 - len(toks)))  # fixed width 12
            ids = torch.tensor(rows, dtype=torch.long)
            mask = (ids != 0).long()
            return {"input_ids": ids, "attention_mask": mask}

    ag = types.SimpleNamespace(tok=BatchTok(), device=torch.device("cpu"),
                               model=types.SimpleNamespace(encoder=enc))
    return ag


# ----------------------------------------------------------------------------
# Tiny exception helper so the suite runs without pytest.
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def expect_raises(exc_type):
    try:
        yield
    except exc_type:
        return
    except Exception as e:  # wrong exception type
        raise AssertionError(f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc_type.__name__} to be raised, but nothing was raised")


def approx_eq(a, b, tol=1e-6):
    import torch
    return torch.allclose(a, b, atol=tol, rtol=tol)
