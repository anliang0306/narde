"""Standalone runner for the narde<->laya parity suite (no pytest required).

Usage:  python tests/parity/run.py
Runs every test_* function in the tests/parity package and reports pass/fail.
When pytest IS available, `pytest -q` is the preferred gate; this runner exists
so the parity check works in constrained/offline environments too.
"""
import importlib
import sys
import traceback

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MODULES = [
    "test_prompts",
    "test_model",
    "test_router",
    "test_calib",
    "test_presets_email",
    "test_shortlist",
    "test_agent",
]


def main() -> int:
    passed, failed = 0, 0
    failures = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            failed += 1
            failures.append(f"{name} (import error)")
            traceback.print_exc()
            continue
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {name}::{attr}")
            except Exception:
                failed += 1
                failures.append(f"{name}::{attr}")
                print(f"FAIL {name}::{attr}")
                traceback.print_exc()
    print(f"\n[parity] {passed} passed, {failed} failed")
    if failures:
        print("failing: " + ", ".join(failures))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
