"""Packaging smoke test.

Must pass on a bare venv (pydantic only, no torch) — this is the
"did I break packaging?" canary from AGENTS.md. Heavy model tests live in
later milestones and are gated on the [engine] extra.

Runnable two ways, mirroring tests/parity/:

    pytest -q tests/test_smoke.py     # preferred
    python tests/test_smoke.py        # standalone, no pytest required

Keeping it fixture-free is deliberate: the whole point of this gate is that it
works on a bare interpreter, and `python tests/test_smoke.py` used to exit 0
without executing anything at all (pytest-style functions, no __main__).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import narde
from narde.settings import Settings, get_settings


@contextmanager
def _env(**pairs):
    """Temporarily set env vars, restoring the previous state on exit.

    Used instead of pytest's `monkeypatch` fixture so this module also runs
    as a plain script (see the __main__ block at the bottom).
    """
    saved = {k: os.environ.get(k) for k in pairs}
    os.environ.update({k: str(v) for k, v in pairs.items()})
    try:
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def test_version_is_a_string():
    assert isinstance(narde.__version__, str)
    assert narde.__version__


def test_settings_defaults_match_reference():
    s = Settings(_env_file=None)
    assert s.device == "auto"
    assert s.preload is True
    assert s.max_loaded == 1
    assert s.default_model == "english"
    assert s.max_len == 1024
    assert s.head_max_len == 256
    assert s.option_max_tokens == 48
    assert s.temp_min == 0.5
    assert s.temp_max == 5.0
    assert s.gate_confidence == 0.85
    assert s.batch_size == 8
    assert s.batch_timeout_ms == 5
    assert s.shortlist_k == 20
    assert s.high_cardinality_threshold == 20
    assert s.log_routing_reasons is True
    assert s.seed is None


def test_settings_env_override():
    with _env(NARDE_DEVICE="cpu", NARDE_MAX_LEN="512"):
        s = Settings(_env_file=None)
        assert s.device == "cpu"
        assert s.max_len == 512


def test_get_settings_is_cached():
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()


if __name__ == "__main__":
    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()

    print(f"\n[smoke] {len(tests) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("failing: " + ", ".join(failed))
    raise SystemExit(1 if failed else 0)
