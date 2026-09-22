# narde — Agent Rules (AGENTS.md)

You are building **narde**, a re-implementation of the [Laya](https://github.com/NandhaKishorM/laya)
engine — a non-autoregressive System-1 decision engine that answers typed questions
(`choice`, `score`, `noul`) over any state (text, email, ticket, JSON) in a **single forward
pass**. It is CPU-friendly, multilingual, and produces calibrated probabilities.

> Reference: `reference/laya/` (read-only, do NOT modify). All prompts, weights, and outputs
> in this repo are ours. The upstream is Apache-2.0.

## Golden Invariants (never violate)

1. **Byte-compatible prompt protocol.** The byte sequence fed to the encoder MUST be
   identical to laya for the same `(state, questions)` input. See `docs/api-contract.md`
   §Prompt format. Any change to token layout must be re-validated against the reference.

2. **Determinism.** `predict(..., temperature=0.0)` is the only path we rely on for evals.
   Greedy (argmax) on CPU is the deterministic reference. Random sampling is *optional*
   and off by default.

3. **No silent truncation.** If state exceeds the context budget, either raise or (if a
   policy says truncate) set a `truncated: true` flag on the result. Never drop context silently.

4. **Calibration is a first-class concern.** Raw logits are NOT confidence. `confidence`
   must come from a calibrated source (temperature scaling or isotonic), tracked per bucket.
   Report ECE in every eval run.

5. **Apache-2.0 provenance.** Any code adapted from laya carries its attribution in
   `NOTICE.md` and a per-file header. We do not vendor their weights.

## Model Facts (from laya v0.3.5 — verify against reference before changing)

- Encoders: `ModernBERT-large` (English, 512 ctx) and `mmBERT-base` (multilingual, 1024 ctx,
  RoPE up to 8192). Do **not** retrain the encoder for v1.
- `head_layers` = 2 transformer layers on top of encoder; `type_emb` over 3 qtypes;
  a scalar `scorer` per [MASK] slot; an `act_head` over `[pooled_cls; top1; top1-top2; entropy; k/255]`.
- Prompt skeleton (must match byte-for-byte):
  `[CLS] {type} {instructions} [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] {state} [SEP]`
- Each option is prefixed with a `[MASK]` and capped at 48 tokens; if the option block
  overflows `head_max_len`, options are re-truncated to `max(4, (head_max_len - 16) // n_options)`.
- `noul` is the binary (2-option) primitive; `choice` and `score` are categorical / ordinal.

## Repo Layout (enforced)

```
src/narde/
  __init__.py      # public surface (lazy imports; no torch at import time)
  model.py         # DecisionModel (encoder + decision heads)
  prompts.py       # build_sequence / render_options / serialize_state
  router.py        # Router + RouteDecision
  lang.py          # script + language detection (pure Python, no deps)
  shortlist.py     # embed + top-k + second-pass
  calib.py         # temperature fit + ECE
  presets.py       # router/guard/moderation/triage/email question packs
  settings.py      # pydantic Settings (env-overridable, prefix NARDE_)
tests/             # pytest; tests/test_smoke.py is the green-bar gate
docs/              # architecture-notes.md, api-contract.md, bench-log.md
bench/             # latency + quality harnesses (run only when [engine] installed)
```

`src/narde/__init__.py` must import ONLY stdlib+pydantic at import time so that
`import narde` works without torch. Heavy imports live in `narde.model` / `narde.agent`
behind lazy imports.

## Tooling Gates

- `ruff check .` clean; `ruff format --check .` clean.
- `pytest -q` green. `tests/test_smoke.py` must pass on a **bare** venv
  (pydantic only) — it is the "did I break packaging?" canary.
- Any new public symbol gets a doctest or unit test in the same commit.

## Change Protocol

1. Read the relevant section of `docs/architecture-notes.md` first.
2. Change code + tests in the same commit.
3. Run the gate: `ruff check . && pytest -q && (bench if GPU present)`.
4. Log the run in `docs/bench-log.md` with date, git sha, device, numbers.
