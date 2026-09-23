"""narde inference-latency benchmark.

Measures what a caller actually pays:
  1. language-detection overhead (pure Python, no model)
  2. raw Agent.system_one latency at 1/5/10/50 questions (+ cold load)
  3. Router hot path (target model already resident)          [needs >= 2 models]
  4. Router cold path (max_loaded=1 forces a swap)            [needs >= 2 models]
  5. mixed-language workload at several max_loaded settings   [needs >= 2 models]

Methodology mirrors reference/laya/research/scripts/bench_latency.py.

Usage:
  python bench/latency.py --model english=convaiinnovations/laya
                          --model multilingual=convaiinnovations/laya:multilingual
  python bench/latency.py --tiny        # fully offline: seeded 1-layer BERT

``NAME=SOURCE``: SOURCE is an HF repo id or a local checkpoint dir; an optional
``:subfolder`` picks one checkpoint out of a bundle repo. Results go to
bench/results/latency.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def parse_models(pairs: list[str]) -> dict:
    """Parse NAME=SOURCE[:subfolder] pairs into a spec dict."""
    models = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--model entries must be NAME=SOURCE, got {p!r}")
        name, source = p.split("=", 1)
        subfolder = None
        # repo_id:subfolder  (local paths with : are rare; prefer repo syntax)
        if ":" in source and not source[1] == ":":  # not a Windows drive letter
            source, subfolder = source.split(":", 1)
        models[name] = {"source": source, "subfolder": subfolder}
    return models


def load_agent(spec: dict):
    """Load a narde Agent from a spec, returning (agent, load_seconds)."""
    import narde

    t0 = time.time()
    agent = narde.load(spec["source"], subfolder=spec["subfolder"])
    agent.model.eval()
    return agent, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="narde latency benchmark")
    ap.add_argument("--model", action="append", default=[],
                    help="NAME=SOURCE (HF repo id or local dir, optional :subfolder)")
    ap.add_argument("--tiny", action="store_true",
                    help="offline mode: seeded 1-layer BERT, no download")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(common.RESULTS_DIR, "latency.json"))
    a = ap.parse_args()

    res = {}

    # ---------------------------------------------------------------- setup
    if a.tiny:
        agent, _ = common.build_tiny_agent()
        res["meta"] = common.env_meta("tiny", common.TINY_MODEL_TAG)
        res["cold_load_ms"] = {"tiny": 0.0}
        print(f"[latency] TINY offline mode: {common.TINY_MODEL_TAG} "
              f"(no real checkpoint; pipeline numbers only)", flush=True)
    else:
        specs = parse_models(a.model or ["english=convaiinnovations/laya"])
        res["meta"] = common.env_meta("real", json.dumps(specs, ensure_ascii=False))
        agents, load_ms = {}, {}
        for name, spec in specs.items():
            ag, secs = load_agent(spec)
            agents[name] = ag
            load_ms[name] = round(secs * 1000, 1)
            print(f"[load] {name:<16s} {load_ms[name]:>8.0f} ms  ({spec['source']})", flush=True)
        res["cold_load_ms"] = load_ms
        agent = None  # per-model loop below uses agents dict

    # ------------------------------------------------------- 1. detection
    print("\n=== 1. language-detection overhead (no model) ===", flush=True)
    from narde.lang import analyse as _analyse

    det_states = {
        "english": common.STATE_EN,
        "hindi": common.STATE_HI,
        "short english": {"m": "refund me"},
        "large json": {"rows": [{"id": i, "text": "some ticket body here"} for i in range(200)]},
    }
    det = {}
    for label, st in det_states.items():
        det[label] = common.timed(lambda s=st: _analyse(s), warmup=20, reps=200)
        print(f"   {label:<14s} p50 {det[label]['p50_ms']:>8.3f} ms  "
              f"p95 {det[label]['p95_ms']:>8.3f} ms", flush=True)
    res["detection_overhead"] = det

    # ------------------------------------------------------- 2. raw latency
    print("\n=== 2. raw Agent.system_one latency (no routing) ===", flush=True)
    if a.tiny:
        raw = {}
        for n in (1, 5, 10, 50):
            q = common.qs(n)
            r = common.timed(lambda qq=q: agent.system_one(common.STATE_EN, qq),
                             warmup=a.warmup, reps=a.reps)
            r["ms_per_question"] = round(r["p50_ms"] / n, 2)
            raw[f"{n}_questions"] = r
            print(f"   {n:>2d}q  p50 {r['p50_ms']:>8.2f} ms  p95 {r['p95_ms']:>8.2f} ms  "
                  f"{r['ms_per_question']:>6.2f} ms/q", flush=True)
        res["raw_latency"] = {"tiny": raw}
    else:
        raw = {}
        for name, ag in agents.items():
            per = {}
            for n in (1, 5, 10, 50):
                q = common.qs(n)
                r = common.timed(lambda qq=q, aa=ag: aa.system_one(common.STATE_EN, qq),
                                 warmup=a.warmup, reps=a.reps)
                r["ms_per_question"] = round(r["p50_ms"] / n, 2)
                per[f"{n}_questions"] = r
                print(f"   {name:<16s} {n:>2d}q  p50 {r['p50_ms']:>8.1f} ms  "
                      f"p95 {r['p95_ms']:>8.1f} ms  {r['ms_per_question']:>6.2f} ms/q", flush=True)
            raw[name] = per
        res["raw_latency"] = raw

    # ------------------------------------------------------- 3-5. routing
    if not a.tiny and len(agents) >= 2:
        from narde.router import Router

        models = {name: spec["source"] for name, spec in specs.items()}

        # section 3: hot path
        cap = max(3, len(models))
        rr = Router(models=models, device="cpu", max_loaded=cap)
        rr.predict(common.STATE_EN, common.qs(5))
        rr.predict(common.STATE_HI, common.qs(5))
        print("\n=== 3. Router hot path (target model resident) ===", flush=True)
        hot = {}
        for label, st in (("english", common.STATE_EN), ("hindi", common.STATE_HI)):
            for n in (1, 10):
                q = common.qs(n)
                k = f"{label}_{n}q"
                hot[k] = common.timed(lambda s=st, qq=q: rr.predict(s, qq), warmup=2, reps=10)
                hot[k]["ms_per_question"] = round(hot[k]["p50_ms"] / n, 2)
                print(f"   {label:<10s} {n:>2d}q  p50 {hot[k]['p50_ms']:>8.1f} ms  "
                      f"({hot[k]['ms_per_question']:.2f} ms/q)", flush=True)
        base = raw[list(models)[0]]["10_questions"]["p50_ms"]
        en10 = hot.get("english_10q")
        if en10:
            res["routing_overhead_hot_ms"] = round(en10["p50_ms"] - base, 2)
            print(f"   routing overhead when hot: "
                  f"{res['routing_overhead_hot_ms']:+.2f} ms on a 10-question English call",
                  flush=True)
        res["router_hot"] = hot
        rr.unload(); del rr; gc.collect()

        # section 4: cold path
        r1 = Router(models=models, device="cpu", max_loaded=1)
        r1.predict(common.STATE_EN, common.qs(5))
        print("\n=== 4. Router cold path (max_loaded=1, swap on every flip) ===", flush=True)
        swap = []
        for i in range(4):
            st = common.STATE_HI if i % 2 == 0 else common.STATE_EN
            t0 = time.time()
            r1.predict(st, common.qs(10))
            swap.append((time.time() - t0) * 1000)
        res["router_cold_swap"] = {"samples_ms": [round(x, 1) for x in swap],
                                   "median_ms": round(statistics.median(swap), 1)}
        print(f"   swap calls (10q): {[round(x) for x in swap]}  "
              f"median {res['router_cold_swap']['median_ms']:.0f} ms", flush=True)
        r1.unload(); del r1; gc.collect()

        # section 5: mixed workload
        print("\n=== 5. mixed-language workload, 100 calls x 5 questions ===", flush=True)
        rng = random.Random(13)
        mix = {}
        for share in (0.0, 0.1, 0.3, 0.5):
            stream = [common.STATE_HI if rng.random() < share else common.STATE_EN
                      for _ in range(100)]
            for cap2, tag in ((1, "max_loaded=1"), (3, "max_loaded=3")):
                rrm = Router(models=models, device="cpu", max_loaded=cap2)
                rrm.predict(common.STATE_EN, common.qs(5))
                rrm.predict(common.STATE_HI, common.qs(5))
                q = common.qs(5)
                t0 = time.time()
                for st in stream:
                    rrm.predict(st, q)
                el = time.time() - t0
                key = f"{int(share*100)}%_non_english/{tag}"
                mix[key] = {"total_s": round(el, 2),
                            "mean_ms_per_call": round(el * 1000 / len(stream), 1),
                            "calls_per_s": round(len(stream) / el, 1)}
                print(f"   {key:<32s} {mix[key]['mean_ms_per_call']:>6.1f} ms/call   "
                      f"{mix[key]['calls_per_s']:>5.1f} calls/s", flush=True)
                rrm.unload(); del rrm; gc.collect()
        res["mixed_workload"] = mix
    else:
        res["routing"] = ("skipped: needs >=2 real models "
                          "(use --model NAME=SOURCE twice; --tiny is pipeline-only)")

    out_path = common.write_report(os.path.basename(a.out), res)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
