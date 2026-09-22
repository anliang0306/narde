"""Packaging smoke test.

Must pass on a bare venv (pydantic only, no torch) — this is the
"did I break packaging?" canary from AGENTS.md. Heavy model tests live in
later milestones and are gated on the [engine] extra.
"""

from __future__ import annotations

import narde
from narde.settings import Settings, get_settings


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


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("NARDE_DEVICE", "cpu")
    monkeypatch.setenv("NARDE_MAX_LEN", "512")
    s = Settings(_env_file=None)
    assert s.device == "cpu"
    assert s.max_len == 512


def test_get_settings_is_cached():
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()
