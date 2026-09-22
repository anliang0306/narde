"""narde — a non-autoregressive, System-1 decision engine (Laya-compatible).

Answers typed decisions (``choice`` / ``score`` / ``noul``) over any state in a
single forward pass. See ``docs/architecture-notes.md`` for the design.

Heavy imports (torch, transformers) are deferred behind ``__getattr__`` so that
``import narde`` stays lightweight; only Settings is eagerly available.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .settings import Settings, get_settings

__version__ = "0.1.0"

if TYPE_CHECKING:
    from .agent import Agent, RLAgent, load
    from .router import Router, RouteDecision, DEFAULT_MODELS
    from .shortlist import shortlist_choice, predict_shortlist, embed_fn_from_agent
    from .lang import detect_language, detect_script, is_english
    from .email import clean_email_body, email_state
    from .presets import email_questions, guard_questions, moderation_questions, router_questions, triage_questions
    from .model import proper_reward, td_lambda_targets
    from .calib import ece_score, confidence_from_probs
    from .prompts import render_options, QTYPES, QTYPE_NAMES

# Lazy import map: public name -> (submodule, attribute)
_LAZY = {
    "Agent": (".agent", "Agent"),
    "RLAgent": (".agent", "RLAgent"),
    "load": (".agent", "load"),
    "Router": (".router", "Router"),
    "RouteDecision": (".router", "RouteDecision"),
    "DEFAULT_MODELS": (".router", "DEFAULT_MODELS"),
    "shortlist_choice": (".shortlist", "shortlist_choice"),
    "predict_shortlist": (".shortlist", "predict_shortlist"),
    "embed_fn_from_agent": (".shortlist", "embed_fn_from_agent"),
    "detect_language": (".lang", "analyse"),
    "detect_script": (".lang", "detect_script"),
    "is_english": (".lang", "is_english"),
    "clean_email_body": (".email", "clean_email_body"),
    "email_state": (".email", "email_state"),
    "email_questions": (".presets", "email_questions"),
    "guard_questions": (".presets", "guard_questions"),
    "moderation_questions": (".presets", "moderation_questions"),
    "router_questions": (".presets", "router_questions"),
    "triage_questions": (".presets", "triage_questions"),
    "proper_reward": (".model", "proper_reward"),
    "td_lambda_targets": (".model", "td_lambda_targets"),
    "ece_score": (".calib", "ece_score"),
    "confidence_from_probs": (".calib", "confidence_from_probs"),
    "render_options": (".prompts", "render_options"),
    "QTYPES": (".prompts", "QTYPES"),
    "QTYPE_NAMES": (".prompts", "QTYPE_NAMES"),
}

__all__ = list(_LAZY.keys()) + ["Settings", "get_settings", "__version__", "detect_language"]


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        module = importlib.import_module(mod_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
