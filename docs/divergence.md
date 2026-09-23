# narde vs laya — Intentional Divergences

narde aims for **byte- and behavior-identical** parity with [laya v0.3.5]
(`reference/laya/`, Apache-2.0). The differential suite in `tests/parity/`
asserts that laya and narde produce identical outputs for identical inputs.
This document lists the **intentional** places where narde deliberately differs.
Everything else must stay a mirror — any new divergence must be added here and
guarded by a test.

## Behavioral (observable) deltas

| Area | laya | narde | Rationale |
|------|------|-------|-----------|
| `system_one`/`predict` response | `"model": "laya-rl-agent"` | `"model": "narde-rl-agent"` | Branding. Parity tests assert exact match on `answers`/`usage` and compare this field separately. |
| Log/warning prefixes | `[laya] …` | `[narde] …` | Branding in CPU-fallback / clamp warnings. |
| Package version | `0.3.5` | `0.1.0` | Independent release line. |
| Checkpoint source | downloads `convaiinnovations/laya` | **same** repo id | Weights are Apache-2.0 model artifacts, not code. narde does not retrain or re-vendor them; `build_model`/`Agent` load the identical laya checkpoints. Re-creating weights from a prompt is out of scope (see TASKS.md M6 for the optional RLCD fine-tune). |

## Structural deltas (not observable through the public API)

- **`src/narde/__init__.py` is torch-free.** laya's `__init__` imports the heavy
  modules eagerly; narde defers all heavy symbols behind PEP 562 `__getattr__`,
  so `import narde` works in a bare venv with only pydantic installed.
  `tests/test_smoke.py` pins this.
- **`settings.py` (pydantic `BaseSettings`, prefix `NARDE_`)** is a narde-only
  convenience layer for deployment (device, LRU, batch knobs). laya has no
  settings module. It is not part of the parity surface.
- **`calib.fit_temperature`** is a narde-only helper (golden-section 1-D NLL
  temperature fit). laya performs the equivalent refit inside its training
  notebook rather than exposing a public function. Parity tests only assert it
  runs and returns a valid clamped float.

## Mirrored quirks (intentionally NOT "fixed")

- `Router.route()`'s typed-decisions workflow branch sets
  `repo = self.models["typed-decisions"]` (the raw `(bundle, subfolder)` tuple,
  not a repo id); laya does the same and narde mirrors it byte-for-byte.
- `noul` is the upstream spelling (binary no-output-label) — kept verbatim.
- `Agent._to_internal` short-key mapping (`type→t`, `instructions→ins`,
  `criteria→crit`) is preserved exactly, since `build_sequence` consumes the
  internal form.

## What parity does NOT cover

- **Model weights.** Parity proves *code* equivalence, not *checkpoint*
  equivalence. To exercise real inference, download the laya checkpoints
  (`convaiinnovations/laya`, subfolders `multilingual` and `typed-decisions`)
  or retrain via M6. The differential tests use a shared seeded tiny-BERT pair
  loaded with `strict=True` into both `DecisionModel` classes, which also
  proves architecture key/shape compatibility.
- **Network behavior.** `snapshot_download`, HF hub caching, and OOM→CPU
  fallback paths are not exercised offline; they are mirrored from laya and
  covered by the structural review + laya's own `test_download.py`.

## Verifying

```bash
# Full differential suite (40 checks), offline, CPU:
python tests/parity/run.py

# With pytest installed:
pytest -q tests/parity tests/test_smoke.py
```

Any new behavioral delta must: (1) be documented in this file, (2) have a
differential test that asserts the exact expected difference, and (3) keep
`tests/parity/run.py` green.
