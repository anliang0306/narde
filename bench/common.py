"""Shared helpers for the narde bench harnesses.

Every harness can run in two modes:

* **real** — `--model NAME=SOURCE` where SOURCE is an HF repo id
  (``convaiinnovations/laya``) or a local checkpoint dir. Downloads happen on
  demand; nothing is vendored into this repo.
* **tiny** — ``--tiny``: a seeded 1-layer BERT + deterministic mock tokenizer.
  Fully offline. It exercises the *exact narde code path* (build_sequence ->
  collate -> DecisionModel forward -> answer formatting) so the harnesses stay
  verifiable in CI / air-gapped environments. Tiny-mode numbers measure the
  pipeline, NOT a real checkpoint, and every report is labeled accordingly.

Methodology mirrors laya's research harnesses
(`reference/laya/research/scripts/bench_latency.py`, ``bench_local.py``) so
real-checkpoint numbers remain comparable.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

RESULTS_DIR = os.path.join(_REPO, "bench", "results")


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def env_meta(mode: str, model: str | None) -> dict:
    """Report header: timestamp, hardware, library versions, narde version, git sha."""
    import torch

    import narde

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "narde": narde.__version__,
        "git_sha": git_sha(),
        "mode": mode,
        "model": model,
    }
    try:
        import psutil  # optional enrichment only
        meta["cpu_count"] = psutil.cpu_count(logical=False)
        meta["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return meta


def timed(fn, warmup: int = 3, reps: int = 15) -> dict:
    """Wall-clock latency of fn() in ms: p50/p95/p99/mean/min/max over `reps` runs."""
    import numpy as np

    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts, float)
    return {
        "p50_ms": round(float(np.percentile(ts, 50)), 2),
        "p95_ms": round(float(np.percentile(ts, 95)), 2),
        "p99_ms": round(float(np.percentile(ts, 99)), 2),
        "mean_ms": round(float(np.mean(ts)), 2),
        "min_ms": round(float(np.min(ts)), 2),
        "max_ms": round(float(np.max(ts)), 2),
    }


def write_report(name: str, payload: dict) -> str:
    ensure_results_dir()
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Tiny offline agent (mock tokenizer + seeded tiny BERT, real narde code path)
# ---------------------------------------------------------------------------

def _mock_tokenizer():
    """Deterministic word-level mock tokenizer.

    Mirrors the fixture in tests/parity/pf.py: word ids via FNV-1a in
    [101, 500], cls/sep/mask ids 101/102/103, pad id 0, vocab_size 512.
    """

    class MockTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        mask_token_id = 103
        pad_token_id = 0
        mask_token = "[MASK]"
        pad_token = "[PAD]"
        vocab_size = 512

        def __call__(self, text, *a, **kw):
            ids = []
            for w in str(text).split():
                h = 2166136261
                for ch in w:
                    h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
                ids.append(101 + (h % 400))
            return {"input_ids": ids}

    return MockTokenizer()


def build_tiny_agent(seed: int = 13, hidden: int = 32, layers: int = 1):
    """Build an offline Agent stand-in: real narde.agent.Agent, seeded tiny encoder.

    The Agent is constructed without __init__ (no download path) and filled with
    the same attributes Agent.__init__ would set -- identical technique to the
    parity suite -- so ``system_one`` runs the production code path end to end.
    """
    import torch
    from transformers import BertConfig, BertModel

    from narde.agent import Agent
    from narde.model import DecisionModel

    torch.manual_seed(seed)
    cfg = BertConfig(
        vocab_size=512,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=2,
        intermediate_size=4 * hidden,
        max_position_embeddings=512,
    )
    model = DecisionModel(BertModel(cfg), head_layers=2, n_act=2, dropout=0.0)
    model.eval()

    agent = object.__new__(Agent)
    agent.tok = _mock_tokenizer()
    agent.model = model
    agent.cfg = {
        "encoder": "tiny-bert-seed%d" % seed,
        "head_layers": 2,
        "act_costs": {"act": 1},
        "max_len": 512,
        "head_max_len": 192,
    }
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_raw = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.temperature_by_options_raw = {}
    return agent, model


TINY_MODEL_TAG = "tiny-bert"


# Sample states shared with laya's latency harness (English + Hindi), so a
# real-checkpoint narde run is directly comparable to laya's published numbers.
STATE_EN = {"ticket": {"subject": "Payout failing", "messages": [{
    "from": "customer",
    "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. "
            "Please help ASAP. " * 6}]}}
STATE_HI = {"ticket": {"subject": "भउगतान विफल", "messages": [{
    "from": "customer",
    "text": "मएरा भउगतान तीन दिनो से विफल हो रहा है, कृपया तुरन्त मदद करेन ।" * 6}]}}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "payments", "technical": "bugs and integrations",
                         "sales": "pricing"}}


def qs(n: int) -> dict:
    """n questions alternating noul/choice (same shape as laya's bench)."""
    return {("q%d" % i): (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}
