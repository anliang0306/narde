"""DecisionModel: bidirectional transformer encoder + typed decision head, plus the
RLCD reward / target helpers.

Adapted from laya/common.py (Apache-2.0, Convai Innovations) — see NOTICE.md.
Architecture and forward semantics are pinned to laya by tests/parity/.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from narde.prompts import QTYPES


class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            # A single-option question has exactly one marker, so p.topk(2, ...)
            # has nothing to select for the second slot and raises. The answer
            # is still well-defined: softmax over one logit is 1.0 regardless of
            # its value, so pad the missing second entry with 0.0 - that gives
            # the act head top1 - top2 == 1.0, the same "fully decided" signal
            # it would see for any other unambiguous top-1-vs-rest gap.
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_model(cfg: Dict, encoder_dir: Optional[str] = None) -> DecisionModel:
    """Reconstruct the DecisionModel for a laya checkpoint.

    Encoders are built uninitialised (or from a local config dir); the Agent then loads the
    bundled model.safetensors (encoder.* + head weights) with strict=True.
    """
    from transformers import AutoConfig, AutoModel

    if encoder_dir and os.path.exists(encoder_dir):
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def amp_dtype(name: Optional[str]) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16
