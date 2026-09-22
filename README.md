# narde

Non-autoregressive, System-1 **decision engine** — a clean-room re-implementation of
[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0). It answers typed
questions (`choice` / `score` / `noul`) over any state (text, email, ticket, JSON)
in a **single forward pass**, is CPU-friendly, multilingual, and emits calibrated
probabilities. A `Router` picks the right checkpoint per request.

This repo is a **from-scratch re-implementation** of the Laya engine, not a fork.
The upstream is kept read-only under `reference/laya/` as an oracle for
byte-compatible prompts and numeric parity. We reference its *architecture*, not
its weights.

## Status

The full engine (prompts, model, router, lang, calibration, shortlist, presets,
email, agent) is implemented as a **1:1 port of laya v0.3.5** and is pinned by a
**differential parity suite** in `tests/parity/` (36 checks, green). Each test runs
the *same* input through both `reference/laya` and `narde` and asserts identical
outputs (exact token lists, `allclose` tensors, equal dicts) — that is the
executable definition of "100% replication" of the code/behavior layer. Model
weights are laya's Apache-2.0 checkpoints (loaded, not re-created). See
`docs/divergence.md` for the few intentional deltas. Domain fine-tune (M6) and
bench harness (M7) remain open. Details in `TASKS.md` / `AGENTS.md`.

## Quickstart

```bash
# Packaging smoke gate (bare env, pydantic only — no torch):
python tests/test_smoke.py            # or: pytest -q tests/test_smoke.py

# Full laya<->narde differential suite (offline, CPU, needs [engine] stack):
pip install -e ".[engine]"
python tests/parity/run.py            # or: pytest -q tests/parity
```

> CPU-first: laya's checkpoints are ModernBERT/mmBERT encoders + a 2-layer decision
> head. Expect ~200–500 ms/question on CPU (vs ~33 ms on a T4). That is expected;
> the goal of v1 is correctness and byte-compatible prompting, not raw CPU speed.

## Layout

```
src/narde/            engine (prompts, model, router, lang, calib,
                      shortlist, presets, email, agent, settings)
tests/test_smoke.py   bare-venv packaging green-bar (pydantic only, no torch)
tests/parity/         laya<->narde differential suite (36 checks) + run.py
bench/                latency + quality harnesses (need [engine], M7)
docs/                 architecture-notes · api-contract · divergence · bench-log
reference/laya/       read-only upstream oracle (gitignored, not committed)
```

## License / provenance

MIT for narde's own code. Anything adapted from Laya is Apache-2.0 and is called
out in `NOTICE.md` and per-file headers. Upstream weights are **not** vendored.
