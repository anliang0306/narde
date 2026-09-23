# narde

[![CI](https://github.com/anliang0306/narde/actions/workflows/ci.yml/badge.svg)](https://github.com/anliang0306/narde/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

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
**differential parity suite** in `tests/parity/` (40 checks, green). Each test runs
the *same* input through both `reference/laya` and `narde` and asserts identical
outputs (exact token lists, `allclose` tensors, equal dicts) — that is the
executable definition of "100% replication" of the code/behavior layer. Model
weights are laya's Apache-2.0 checkpoints (loaded, not re-created). See
`docs/divergence.md` for the few intentional deltas. A benchmark skeleton
(`bench/`) runs offline in `--tiny` mode. Domain fine-tune (M6) still needs HF
weights + GPU. Details in `TASKS.md` / `AGENTS.md`.

## Quickstart

```bash
# Packaging smoke gate (bare env, pydantic only — no torch):
python tests/test_smoke.py            # or: pytest -q tests/test_smoke.py

# Full laya<->narde differential suite (offline, CPU, needs [engine] stack).
# It compares against the read-only oracle at reference/laya/ (gitignored);
# CI fetches it pinned to a commit SHA:
pip install -e ".[engine]"
python tests/parity/run.py            # or: pytest -q tests/parity
```

Both suites are dependency-light and run **offline**: the parity suite uses a
seeded 1-layer BERT and a deterministic mock tokenizer, so nothing is downloaded.

> CPU-first: laya's checkpoints are ModernBERT/mmBERT encoders + a 2-layer decision
> head. Expect ~200–500 ms/question on CPU (vs ~33 ms on a T4). That is expected;
> the goal of v1 is correctness and byte-compatible prompting, not raw CPU speed.

## Layout

```
src/narde/            engine (prompts, model, router, lang, calib,
                      shortlist, presets, email, agent, settings)
tests/test_smoke.py   bare-venv packaging green-bar (pydantic only, no torch)
tests/parity/         laya<->narde differential suite (40 checks) + run.py
bench/                latency.py + quality.py + canary.jsonl + results/
docs/                 architecture-notes · api-contract · divergence · bench-log
.github/workflows/    ci.yml (parity + bare-venv smoke)
LICENSE               Apache-2.0 (upstream-compatible)
NOTICE.md             per-module attribution to laya v0.3.5
PAPER.md              research paper: differential parity testing as the
                      executable definition of "100% replication" (narde/laya)
reference/laya/       read-only upstream oracle (gitignored, not committed)
```

## Benchmarks (M7 skeleton)

Two offline-runnable harnesses under `bench/` (both support a `--tiny` mode that
uses a seeded 1-layer BERT + mock tokenizer, so they run with no network):

```bash
# Latency: detection overhead, raw system_one at 1/5/10/50 questions,
# router hot/cold path, mixed-language workload.
python bench/latency.py --tiny                 # offline pipeline numbers
python bench/latency.py --model english=convaiinnovations/laya \
    --model multilingual=convaiinnovations/laya:multilingual   # real checkpoints

# Quality: accuracy / ECE / calibration on a JSONL dataset.
python bench/quality.py --tiny --canary        # offline pipeline sanity check
python bench/quality.py --model convaiinnovations/laya --dataset mydata.jsonl
```

Results are written to `bench/results/{latency,quality}.json` and each run is
logged in `docs/bench-log.md`. `--tiny` numbers measure the *pipeline*, not a
real checkpoint — treat them as smoke tests, not quality claims.

## Continuous integration

`.github/workflows/ci.yml` enforces exactly the invariant this repo is built
around — so the "CI-checkable" claim in `PAPER.md` §4.6 is literal, not aspirational:

| Job | What it proves |
|---|---|
| `parity` | the **40-check** laya↔narde differential suite is green via **both** runners (`tests/parity/run.py` *and* `pytest`), against the oracle pinned to commit `4170897` (laya v0.3.5) |
| `smoke` | `import narde` still works on a **bare** venv — pydantic only, **no torch** — on Python 3.10 (the declared floor) and 3.12 |

Two details worth knowing:

- **The oracle is pinned by commit SHA, not by a tag.** Upstream laya publishes no
  git tags, and a drifting oracle would make "40/40 green" meaningless. CI asserts
  the fetched SHA equals the pin and that the tree reports `version = "0.3.5"`.
- **The `smoke` job asserts `torch` is absent** rather than merely importing
  `narde`. That separation is the actual invariant being protected: the engine
  stack must stay behind the `[engine]` extra, so `import narde` never drags in torch.

## License / provenance

**Apache-2.0** — see [`LICENSE`](LICENSE). narde adapts code from
[Laya](https://github.com/NandhaKishorM/laya), which is itself Apache-2.0, so the
combined work is Apache-2.0 as well. Per-file attribution is in the header of each
`src/narde/*.py` and is summarised in [`NOTICE.md`](NOTICE.md).

Upstream model weights are **not** vendored or retrained: inference loads the
Apache-2.0 `convaiinnovations/laya` checkpoints on demand.
