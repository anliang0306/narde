"""narde quality benchmark: accuracy / ECE / calibration on typed-decision datasets.

Dataset format (JSONL), one object per case:
  {
    "id": "c01",
    "workflow": "email",                      // optional grouping key
    "state": {...} or "text",                 // arbitrary state passed to the agent
    "questions": {"qid": {"type": "choice"|"score"|"noul", "instructions": ..., "criteria": ...}},
    "gold": {"qid": {"label": ...}}           // choice: a criteria key; noul: "true"/"false";
                                              // score: level index (int or numeric string)
  }

Metrics mirror laya's research harness (reference/laya/research/scripts/bench_local.py::metrics):
  accuracy, macro_f1, ECE (15 bins), brier, nll, mean_confidence, acc_at_50_coverage.

Usage:
  python bench/quality.py --model convaiinnovations/laya --dataset my_data.jsonl
  python bench/quality.py --model convaiinnovations/laya:typed-decisions --dataset d.jsonl
  python bench/quality.py --tiny --canary        # offline pipeline sanity check
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def load_jsonl(path: str) -> list:
    cases = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i + 1}: invalid JSON: {e}")
    return cases


def gold_index(qdef: dict, gold: dict) -> int:
    """Map a gold label onto the option index implied by the question type."""
    t = qdef["type"]
    label = str(gold["label"])
    if t == "noul":
        # render order is [false, true]
        return 1 if label.lower() in ("true", "1") else 0
    if t == "choice":
        return list(qdef["criteria"].keys()).index(label)
    # score
    n = len(qdef.get("criteria") or [])
    idx = int(float(label))
    return max(0, min(n - 1, idx))


def probs_vector(qdef: dict, ans: dict) -> np.ndarray:
    """Probability vector aligned to option order, read from the agent answer.

    Mirrors the Agent answer contract: choice probabilities keyed by criteria
    label, score probabilities keyed "0".."k-1" (with legend), noul exposing
    only the scalar P(true).
    """
    t = qdef["type"]
    if t == "choice":
        return np.array([float(ans["probabilities"][k]) for k in qdef["criteria"].keys()], float)
    if t == "score":
        legend = ans.get("legend") or {}
        n = len(legend) if legend else len(qdef.get("criteria") or [])
        return np.array([float(ans["probabilities"].get(str(i), 0.0)) for i in range(n)], float)
    # noul: options are [false, true]
    pt = float(ans["noul"])
    return np.array([round(1.0 - pt, 4), round(pt, 4)], float)


def metrics(rows: list) -> dict:
    """Same metric family as laya's research bench. rows = (gold, probs) pairs."""
    rows = [r for r in rows if r[1] is not None]
    if not rows:
        return {"n": 0}
    g = np.array([x[0] for x in rows])
    p = np.array([int(np.argmax(x[1])) for x in rows])
    c = np.array([float(np.max(x[1])) for x in rows])
    corr = (p == g).astype(float)

    f1s = []
    for cl in sorted(set(g.tolist()) | set(p.tolist())):
        tp = int(((p == cl) & (g == cl)).sum())
        fp = int(((p == cl) & (g != cl)).sum())
        fn = int(((p != cl) & (g == cl)).sum())
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    macro_f1 = float(np.mean(f1s))

    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        sel = (c > lo) & (c <= hi)
        if sel.any():
            e += sel.mean() * abs(c[sel].mean() - corr[sel].mean())

    brier = float(np.mean([
        ((np.asarray(x[1]) - np.eye(len(x[1]))[x[0]]) ** 2).sum() for x in rows
    ]))
    nll = float(np.mean([-math.log(max(float(x[1][x[0]]), 1e-12)) for x in rows]))
    return {
        "n": len(rows),
        "accuracy": round(float(corr.mean()), 4),
        "macro_f1": round(macro_f1, 4),
        "ece": round(float(e), 4),
        "brier": round(brier, 4),
        "nll": round(nll, 4),
        "mean_confidence": round(float(c.mean()), 4),
        "acc_at_50_coverage": round(float(corr[np.argsort(-c)[: max(1, len(c) // 2)]].mean()), 4),
    }


def score_agent(agent, cases: list):
    """Run every case through agent.system_one; collect (gold, probs, qtype, workflow)."""
    rows = []
    detail = []
    for ci, case in enumerate(cases):
        state = case["state"]
        questions = case["questions"]
        gold = case["gold"]
        wf = case.get("workflow", "unlabeled")
        out = agent.system_one(state, questions)
        answers = out["answers"]
        for qid, qdef in questions.items():
            g = gold.get(qid)
            if g is None:
                continue
            if qid not in answers:
                detail.append({"case": case.get("id"), "qid": qid, "error": "missing answer"})
                rows.append((None, None, qdef["type"], wf))
                continue
            probs = probs_vector(qdef, answers[qid])
            gi = gold_index(qdef, g)
            rows.append((gi, probs, qdef["type"], wf))
            detail.append(
                {
                    "case": case.get("id"),
                    "qid": qid,
                    "type": qdef["type"],
                    "gold": gi,
                    "predicted": int(np.argmax(probs)),
                    "probs": [round(float(x), 4) for x in probs],
                }
            )
        if (ci + 1) % 50 == 0:
            print(f"   ...{ci + 1}/{len(cases)} cases", flush=True)
    return rows, detail


def main():
    ap = argparse.ArgumentParser(description="narde quality benchmark")
    ap.add_argument("--model", default=None, help="HF repo id or local checkpoint dir")
    ap.add_argument("--subfolder", default=None, help="subfolder inside a bundle repo")
    ap.add_argument("--dataset", default=None, help="path to JSONL dataset")
    ap.add_argument("--canary", action="store_true",
                    help="use the built-in offline canary dataset (bench/canary.jsonl)")
    ap.add_argument("--tiny", action="store_true",
                    help="offline mode: seeded tiny BERT (pipeline check only)")
    ap.add_argument("--out", default=os.path.join(common.RESULTS_DIR, "quality.json"))
    a = ap.parse_args()

    if a.tiny:
        agent, _ = common.build_tiny_agent()
        meta = common.env_meta("tiny", common.TINY_MODEL_TAG)
        print(f"[quality] TINY offline mode: {common.TINY_MODEL_TAG} "
              f"(numbers are a pipeline sanity check, not a quality claim)", flush=True)
    else:
        import narde

        src, subfolder = a.model or "convaiinnovations/laya", a.subfolder
        if src.startswith("hf:") and subfolder is None:
            m = src[3:].split(":", 1)
            src, subfolder = m[0], (m[1] if len(m) > 1 else None)
        agent = narde.load(src, subfolder=subfolder)
        agent.model.eval()
        meta = common.env_meta("real", src)

    if a.dataset:
        cases = load_jsonl(a.dataset)
        ds_name = os.path.abspath(a.dataset)
    elif a.canary or a.tiny:
        ds = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canary.jsonl")
        cases = load_jsonl(ds)
        ds_name = "canary (built-in)"
    else:
        raise SystemExit("provide --dataset PATH or --canary")

    print(f"[quality] {len(cases)} cases from {ds_name}", flush=True)
    rows, detail = score_agent(agent, cases)

    overall = metrics([(g, p) for g, p, _qt, _wf in rows])
    by_qt = {}
    by_wf = {}
    for g, p, qt, wf in rows:
        by_qt.setdefault(qt, []).append((g, p))
        by_wf.setdefault(wf, []).append((g, p))

    res = {
        "meta": meta,
        "dataset": ds_name,
        "overall": overall,
        "by_question_type": {k: metrics(v) for k, v in sorted(by_qt.items())},
        "by_workflow": {k: metrics(v) for k, v in sorted(by_wf.items())},
        "per_question": detail,
    }
    out_path = common.write_report(os.path.basename(a.out), res)

    print(f"\n=== quality: {ds_name} ===", flush=True)
    o = overall
    print(f"   n={o.get('n', 0)}  acc={o.get('accuracy')}  f1={o.get('macro_f1')}  "
          f"ECE={o.get('ece')}  brier={o.get('brier')}  conf={o.get('mean_confidence')}", flush=True)
    for k, v in res["by_question_type"].items():
        print(f"   [{k}] n={v.get('n')} acc={v.get('accuracy')} ece={v.get('ece')}", flush=True)
    for k, v in res["by_workflow"].items():
        print(f"   [{k:<12}] n={v.get('n')} acc={v.get('accuracy')}", flush=True)
    print(f"\nwrote {out_path}", flush=True)
    del agent
    gc.collect()


if __name__ == "__main__":
    main()
