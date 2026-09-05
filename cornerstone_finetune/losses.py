"""Losses for the multi-label sigmoid head.

Every term here exists because of a measured failure mode:

* **``pos_weight`` on the BCE.** At 0.03-0.3% prevalence, unweighted BCE+Dice
  converged to predicting background everywhere for three of four channels
  over a full 300-epoch run -- train loss fell the whole time. The
  all-background solution is a very strong local minimum at this imbalance
  (``training_diary.md`` Run 0).
* **clDice on the tubular channels.** finetuning.md Sec. 3 calls a
  centreline-aware term "the highest-leverage single choice in this document":
  an overlap loss under-weights thin distal branches by construction, and
  plain Dice converges happily to a solution that captures the proximal trunk
  and drops every distal branch.
* **``channel_mask``.** Nine of the fifteen source classes come from meshes
  some subjects do not have. Those channels are *unannotated* there, not
  empty; counting them as background teaches the network to erase them
  (finetuning.md Sec. 1: "mask out; do not let it count as background").
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _broadcast_mask(mask: Optional[torch.Tensor], logits: torch.Tensor) -> torch.Tensor:
    """Normalise whatever the batch carried into a dense ``(N,C)`` float mask."""
    n, c = logits.shape[0], logits.shape[1]
    if mask is None:
        return torch.ones((n, c), device=logits.device, dtype=logits.dtype)
    m = torch.as_tensor(mask, device=logits.device, dtype=logits.dtype)
    m = m.reshape(-1, c) if m.numel() % c == 0 and m.numel() >= c else m
    if m.shape[0] == 1 and n > 1:
        m = m.expand(n, c)
    elif m.shape[0] != n and n % m.shape[0] == 0:
        # Collation gives one mask per *subject*; a subject contributes
        # ``samples_per_volume`` patches, so repeat rather than truncate.
        m = m.repeat_interleave(n // m.shape[0], dim=0)
    if m.shape != (n, c):
        return torch.ones((n, c), device=logits.device, dtype=logits.dtype)
    return m


def masked_dice_loss(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                     smooth: float = 1.0) -> torch.Tensor:
    dims = tuple(range(2, probs.dim()))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return 1.0 - (dice * mask).sum() / mask.sum().clamp(min=1.0)


def masked_tversky_loss(probs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                        alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> torch.Tensor:
    """Tversky index, channel-masked. ``alpha`` penalises false positives,
    ``beta`` penalises false negatives -- Dice is the special case alpha=beta=0.5.

    finetuning.md Sec. 5: plain Dice converges to the proximal trunk and drops
    distal branches; beta > alpha trades precision for recall on exactly the
    thin structures where that under-segmentation shows up.
    """
    dims = tuple(range(2, probs.dim()))
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - probs) * targets).sum(dim=dims)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - (tversky * mask).sum() / mask.sum().clamp(min=1.0)


class DiceBCELoss(nn.Module):
    """Per-channel soft Dice (or Tversky) + ``pos_weight``-ed BCE, both channel-masked."""

    def __init__(
        self,
        pos_weight: Optional[list[float]] = None,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        use_tversky: bool = False,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.use_tversky = use_tversky
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits, targets, channel_mask=None) -> torch.Tensor:
        mask = _broadcast_mask(channel_mask, logits)
        probs = torch.sigmoid(logits)
        if self.use_tversky:
            dice = masked_tversky_loss(probs, targets, mask,
                                       alpha=self.tversky_alpha, beta=self.tversky_beta)
        else:
            dice = masked_dice_loss(probs, targets, mask)

        pw = None
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.dtype).view(1, -1, *([1] * (logits.dim() - 2)))
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets.float(),
            pos_weight=None if pw is None else pw.float(),
            reduction="none",
        )
        mask_b = mask.view(*mask.shape, *([1] * (logits.dim() - 2))).float()
        n_elem = mask_b.sum() * int(torch.tensor(logits.shape[2:]).prod())
        bce = (bce * mask_b).sum() / n_elem.clamp(min=1.0)
        return self.lambda_dice * dice + self.lambda_bce * bce


def _soft_erode(x): return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)
def _soft_dilate(x): return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
def _soft_open(x): return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """Differentiable morphological skeletonisation (Shit et al., clDice, CVPR 2021)."""
    skel = F.relu(x - _soft_open(x))
    cur = x
    for _ in range(iterations):
        cur = _soft_erode(cur)
        delta = F.relu(cur - _soft_open(cur))
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """Centreline overlap: weights topology instead of volume."""

    def __init__(self, iterations: int = 10, smooth: float = 1.0) -> None:
        super().__init__()
        self.iterations = iterations
        self.smooth = smooth

    def forward(self, logits, targets, channel_mask=None) -> torch.Tensor:
        mask = _broadcast_mask(channel_mask, logits)
        probs = torch.sigmoid(logits.float())
        targets = targets.float()
        skel_pred = soft_skeletonize(probs, self.iterations)
        skel_true = soft_skeletonize(targets, self.iterations)
        dims = tuple(range(2, probs.dim()))
        prec = ((skel_pred * targets).sum(dims) + self.smooth) / (skel_pred.sum(dims) + self.smooth)
        sens = ((skel_true * probs).sum(dims) + self.smooth) / (skel_true.sum(dims) + self.smooth)
        cl = 2.0 * prec * sens / (prec + sens)
        return 1.0 - (cl * mask).sum() / mask.sum().clamp(min=1.0)


class MultiLabelLoss(nn.Module):
    """``DiceBCELoss`` over every channel + clDice over the tubular ones.

    ``lambda_cldice`` defaults below 1: at weight 1.0 the clDice term (~0.5)
    dominated Dice+BCE (~1.0 combined) early in training and the run stalled
    (``training_diary.md``, Run 3 -> Run 4 fixes).
    """

    def __init__(
        self,
        cldice_channels: Optional[list[int]] = None,
        pos_weight: Optional[list[float]] = None,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
        lambda_cldice: float = 0.5,
        cldice_iterations: int = 10,
        use_tversky: bool = False,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ) -> None:
        super().__init__()
        self.dice_bce = DiceBCELoss(pos_weight=pos_weight, lambda_dice=lambda_dice, lambda_bce=lambda_bce,
                                    use_tversky=use_tversky, tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)
        self.cldice = SoftClDiceLoss(iterations=cldice_iterations)
        self.cldice_channels = list(cldice_channels or [])
        self.lambda_cldice = lambda_cldice

    def forward(self, logits, targets, channel_mask=None) -> torch.Tensor:
        loss = self.dice_bce(logits, targets, channel_mask=channel_mask)
        if self.cldice_channels and self.lambda_cldice > 0:
            idx = self.cldice_channels
            mask = _broadcast_mask(channel_mask, logits)[:, idx]
            loss = loss + self.lambda_cldice * self.cldice(
                logits[:, idx], targets[:, idx], channel_mask=mask
            )
        return loss
