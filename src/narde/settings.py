"""Typed, env-overridable settings for narde.

Every field can be overridden by an environment variable with the ``NARDE_``
prefix (e.g. ``NARDE_DEVICE=cpu``, ``NARDE_GATE_CONFIDENCE=0.9``) or a local
``.env`` file. This module imports only pydantic, so ``import narde`` stays
cheap on machines that have not installed the ``engine`` extra.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the narde engine.

    Defaults mirror the Laya reference checkpoints so that, when the matching
    weights are supplied, narde reproduces Laya's behaviour. See
    ``docs/architecture-notes.md`` for the source of each default.
    """

    model_config = SettingsConfigDict(env_prefix="NARDE_", env_file=".env", extra="ignore")

    # --- device / lifecycle ---------------------------------------------------
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    preload: bool = True  # keep resident checkpoints in memory
    max_loaded: int = 1  # LRU checkpoint cache size (router)
    default_model: str = "english"

    # --- token budgets (must match the checkpoint under use) -------------------
    # laya english: 512 ctx / head 192; laya multilingual + typed: 1024 / 256
    max_len: int = 1024
    head_max_len: int = 256
    option_max_tokens: int = 48

    # --- calibration / gating --------------------------------------------------
    temp_min: float = 0.5  # temperature clamp floor (below sharpens -> distrust)
    temp_max: float = 5.0  # temperature clamp ceiling
    gate_confidence: float = 0.85  # below this -> flag for human review

    # --- batching ---------------------------------------------------------------
    batch_size: int = 8
    batch_timeout_ms: int = 5
    max_inflight: int = 8

    # --- high-cardinality shortlisting ------------------------------------------
    shortlist_k: int = 20
    high_cardinality_threshold: int = 20  # choice questions with > this many options

    # --- misc -------------------------------------------------------------------
    log_routing_reasons: bool = True
    seed: int | None = None  # None = non-deterministic; set for reproducible runs


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()


__all__ = ["Settings", "get_settings"]
