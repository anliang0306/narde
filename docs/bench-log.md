# bench-log

Append-only record of benchmark runs. Newest at the bottom. `--tiny` numbers
measure the **narde pipeline** with a seeded 1-layer BERT + mock tokenizer — they
are a smoke / regression baseline, NOT a quality or real-latency claim. Real
checkpoint numbers need network + (for laya's encoders) the HF download.

---

## 2026-07-24 — baseline (`--tiny`, offline)

- commit: `b693127` · device: CPU · torch `2.12.0+cpu`, threads=1 · narde `0.1.0`
- model: `tiny-bert` (hidden=32, 1 layer, 2 heads, vocab 512, seeded 13)

### `bench/latency.py --tiny`
| section | metric |
|---|---|
| detection: english | p50 0.42 ms, p95 0.75 ms |
| detection: hindi | p50 2.61 ms, p95 3.48 ms |
| detection: short english | p50 0.01 ms |
| detection: large json (200 rows) | p50 3.45 ms, p95 4.55 ms |
| raw system_one 1q | p50 11.80 ms |
| raw system_one 5q | p50 17.61 ms (3.52 ms/q) |
| raw system_one 10q | p50 23.05 ms (2.31 ms/q) |
| raw system_one 50q | p50 59.84 ms (1.20 ms/q) |
| routing sections | skipped (needs ≥2 real models) |

Observation: per-question cost amortizes well as batch grows (11.8 → 1.2 ms/q),
matching laya's T4 scaling shape — confirms the collate/batched-forward path is
intact.

### `bench/quality.py --tiny --canary`
| scope | acc | ece |
|---|---|---|
| overall (24 q) | 0.2917 | 0.1015 |
| choice (7) | 0.1429 | 0.1208 |
| noul (11) | 0.4545 | 0.0571 |
| score (6) | 0.1667 | 0.1605 |

Expectation: near-chance (choice 4-way ≈ 0.25, noul 2-way ≈ 0.5) since weights are
random — this validates the **pipeline**, not the model. noul ≈ 0.45 and choice
≈ 0.14 are consistent with chance + tiny-model noise. Real quality numbers come
from a `--model convaiinnovations/laya` run (M7, network required).
