# NOTICE

narde is a re-implementation of the Laya decision engine. It contains code adapted
from the following open-source project:

## Laya (laya)

- Copyright (c) 2024 Convai Innovations
- Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
- Upstream: https://github.com/NandhaKishorM/laya
- Local reference oracle: `reference/laya/` (v0.3.5, read-only, not distributed here)

A copy of the Apache-2.0 license is available at:
https://www.apache.org/licenses/LICENSE-2.0

The following narde modules are adapted from laya source files (see the per-file
attribution header at the top of each file):

| narde module | adapted from laya v0.3.5 file |
|--------------|-------------------------------|
| `src/narde/prompts.py`   | `laya/common.py` (sequence construction, option rendering, collation) |
| `src/narde/model.py`     | `laya/common.py` (`DecisionModel`, `build_model`, `proper_reward`, `td_lambda_targets`, `amp_dtype`) |
| `src/narde/calib.py`     | `laya/common.py` (temperature clamping, ECE, confidence, `temp_bucket`) + narde-only `fit_temperature` |
| `src/narde/agent.py`     | `laya/agent.py` (`Agent`, checkpoint loading, `system_one`) |
| `src/narde/router.py`    | `laya/router.py` (`Router`, `RouteDecision`, name normalisation, LRU) |
| `src/narde/lang.py`      | `laya/lang.py` (script + Latin-language detection) |
| `src/narde/shortlist.py` | `laya/shortlist.py` (embedding shortlist, `predict_shortlist`) |
| `src/narde/presets.py`   | `laya/presets.py` (question packs) |
| `src/narde/email.py`     | `laya/email.py` (email cleaning + state) |

## Model weights

narde does **not** vendor or retrain model weights. Inference uses the Apache-2.0
checkpoints published by Convai Innovations (`convaiinnovations/laya`), downloaded
on demand. See `docs/divergence.md` ("What parity does NOT cover").

## Original narde code

All other code (packaging, `settings.py`, the differential test suite under
`tests/parity/`, docs) is original to narde and released under the Apache-2.0
license as well.
