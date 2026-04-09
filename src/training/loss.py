# src/training/loss.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class LossConfig:
    pos_weight: Optional[float] = None  # for imbalance, e.g. (#neg/#pos)


def build_bce_with_logits_loss(cfg: LossConfig) -> nn.Module:
    if cfg.pos_weight is None:
        return nn.BCEWithLogitsLoss()
    pw = torch.tensor([cfg.pos_weight], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw)
