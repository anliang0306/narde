"""agent parity: Agent._to_internal / _fix_tokenizer_config / system_one match laya."""
import json
import os
import shutil

import torch

import pf  # noqa: F401  (bootstrap sys.path)
from laya.agent import Agent as LAgent, _fix_tokenizer_config as L_fix, _verify_compatibility as L_verify
from narde.agent import Agent as NAgent, _fix_tokenizer_config as N_fix, _verify_compatibility as N_verify


def test_to_internal_parity():
    cases = [
        {"type": "choice", "instructions": "which", "criteria": ["a", "b"]},
        {"type": "choice", "instructions": "which", "criteria": {"x": 0, "y": False, "z": None, "w": ""}},
        {"type": "score", "instructions": "rate", "criteria": ["a", "b", "c"]},
        {"type": "noul", "instructions": "yes?"},
        {"type": "noul", "instructions": "yes?", "criteria": {"true": "t"}},
        {"type": "choice", "instructions": {"a": 1}, "criteria": ["k"]},  # non-str instructions
    ]
    for q in cases:
        assert LAgent._to_internal(dict(q)) == NAgent._to_internal(dict(q)), q


def _run_fix(cfg):
    # NOTE: system temp dir is not writable in the sandbox -> use a workspace-local scratch dir
    import shutil
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scratch")
    out = {}
    for mod in (L_fix, N_fix):
        d = os.path.join(base, mod.__module__.split(".")[0])
        shutil.rmtree(d, ignore_errors=True)
        tok_dir = os.path.join(d, "tokenizer")
        os.makedirs(tok_dir)
        path = os.path.join(tok_dir, "tokenizer_config.json")
        with open(path, "w") as f:
            json.dump(cfg, f)
        mod(d)
        with open(path) as f:
            out[mod.__module__.split(".")[0]] = json.load(f)
    shutil.rmtree(base, ignore_errors=True)
    return out


def test_fix_tokenizer_config_parity():
    variants = [
        {"tokenizer_class": "TokenizersBackend", "backend": "x", "is_local": True,
         "extra_special_tokens": ["[A]", "[B]"]},
        {"tokenizer_class": "LlamaTokenizer", "extra_special_tokens": "single"},
        {"extra_special_tokens": [None, 7]},
    ]
    for cfg in variants:
        r = _run_fix(json.loads(json.dumps(cfg)))
        assert r["laya"] == r["narde"], (cfg, r)


def test_verify_compatibility_parity():
    lm, nm, sd = pf.shared_decision_models()
    cfg = {"encoder": "x", "head_layers": 2, "act_costs": {"a": 1}}
    L_verify(lm, cfg, sd, "x")
    N_verify(nm, cfg, sd, "x")  # both should pass

    # missing prefix
    bad = {k: v for k, v in sd.items() if not k.startswith("scorer.")}
    for m, ver in ((lm, L_verify), (nm, N_verify)):
        try:
            ver(m, cfg, bad, "x"); assert False
        except ValueError as e:
            assert "'scorer.'" in str(e)

    # shape mismatch
    bad2 = dict(sd)
    bad2["type_emb.weight"] = sd["type_emb.weight"].new_full((9, 16), 0.0)
    for m, ver in ((lm, L_verify), (nm, N_verify)):
        try:
            ver(m, cfg, bad2, "x"); assert False
        except ValueError as e:
            assert "type_emb.weight" in str(e)

    # missing key (remove a key whose prefix still has siblings, so the
    # prefix check passes and the missing-key check fires)
    bad3 = {k: v for k, v in sd.items() if k != "scorer.3.weight"}
    for m, ver in ((lm, L_verify), (nm, N_verify)):
        try:
            ver(m, cfg, bad3, "x"); assert False
        except ValueError as e:
            assert "scorer.3.weight" in str(e)

    # missing config keys
    for m, ver in ((lm, L_verify), (nm, N_verify)):
        try:
            ver(m, {}, sd, "x"); assert False
        except ValueError as e:
            assert "encoder" in str(e) and "head_layers" in str(e)


def _mk_agent(cls, model, cfg):
    a = object.__new__(cls)
    a.tok = pf.TOK
    a.model = model
    a.cfg = cfg
    a.device = torch.device("cpu")
    a.dtype = torch.float32
    a.temperature = [1.0, 1.0, 1.0]
    a.temperature_raw = [1.0, 1.0, 1.0]
    a.temperature_by_options = {}
    a.temperature_by_options_raw = {}
    return a


def test_system_one_end_to_end_parity():
    lm, nm, _ = pf.shared_decision_models()
    cfg = {"encoder": "mock", "head_layers": 2, "act_costs": {"act": 1},
           "max_len": 64, "head_max_len": 40}
    la = _mk_agent(LAgent, lm, cfg)
    na = _mk_agent(NAgent, nm, cfg)

    questions = {
        "cat": {"type": "choice", "instructions": "pick a category",
                "criteria": {"a": "alpha", "b": "beta", "c": "gamma"}},
        "sev": {"type": "score", "instructions": "rate severity",
                "criteria": ["low", "mid", "high"]},
        "urgent": {"type": "noul", "instructions": "is this urgent?"},
    }
    state = {"message": "please refund my double charge, it is urgent", "amount": 42}

    lr = la.system_one(state, questions)
    nr = na.system_one(state, questions)

    assert lr["answers"] == nr["answers"], (lr["answers"], nr["answers"])
    assert lr["usage"] == nr["usage"]
    assert nr["model"] == "narde-rl-agent"
    # structure checks
    assert set(nr["answers"]["cat"]) == {"type", "choice", "probabilities", "confidence", "action"}
    assert set(nr["answers"]["sev"]) == {"type", "score", "legend", "probabilities", "confidence", "action"}
    assert set(nr["answers"]["urgent"]) == {"type", "noul", "confidence", "action"}
