"""
Asymmetric Loss (ASL) for multi-label classification.

Implements ASL as recommended by Ridnik et al. (ICCV 2021):
    γ+ = 0   (no focusing on positives — keep full gradient)
    γ- = 4   (strong focusing on negatives)
    clip = 0.05  (hard threshold for very easy negatives)

Supports configurable loss reduction:
    'mean'     — average over B × C (safe default)
    'sum_mean' — sum over classes, mean over batch (matches reference scaling)

Also includes QueryDiversityLoss for optional query orthogonality regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Parameters
    ----------
    gamma_pos : float
        Focusing parameter for positive samples. Default 0 (recommended).
    gamma_neg : float
        Focusing parameter for negative samples. Default 4 (recommended).
    clip : float
        Probability margin for hard thresholding negatives. Default 0.05.
    reduction : str
        Loss reduction mode: 'mean' or 'sum_mean'. Default 'mean'.
    eps : float
        Numerical stability constant.

    References
    ----------
    Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
    Optimal fixed config: γ+=0, γ-=4, m=0.05 → 86.6% mAP on MS-COCO.
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = "mean",
        class_weights: torch.Tensor = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        self.eps = eps

        if reduction not in ("mean", "sum_mean"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. Use 'mean' or 'sum_mean'."
            )

        # Per-class weights for class-balanced ASL
        # Shape: (C,) — will broadcast over batch dimension
        # FIX: always register as a buffer, even when None, so that any
        # later reassignment (e.g. `criterion.class_weights = weights`)
        # goes through nn.Module's buffer-aware __setattr__ and gets moved
        # to the right device automatically by .to(device)/model.to(device).
        # Previously, when class_weights=None at construction time, the
        # attribute was set via plain object.__setattr__ instead of
        # register_buffer, so a later direct assignment of a CPU tensor
        # (as done by the training script's class-balanced weighting step)
        # was never picked up by .to(device) — causing a
        # "cuda:0 and cpu" RuntimeError the moment training started.
        self.register_buffer("class_weights", class_weights)

    def set_class_weights(self, class_weights: torch.Tensor):
        """Safely (re)assign per-class weights after construction.

        FIX: prefer this over `criterion.class_weights = tensor` directly.
        Buffers already registered in __init__ are picked up correctly by
        plain attribute assignment too (nn.Module routes it through
        self._buffers), but this helper also snaps the incoming tensor to
        whatever device the module's existing buffer/parameters are on,
        as a second safety net against CPU/GPU mismatches.
        """
        device = next(self.buffers(), torch.tensor(0.0)).device
        self.class_weights = class_weights.to(device) if class_weights is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ASL loss.

        Parameters
        ----------
        logits  : (B, C) raw logits
        targets : (B, C) binary labels (0 or 1), may be soft (from CutMix/MixUp)

        Returns
        -------
        loss : scalar tensor
        """
        # Sigmoid probabilities
        p = torch.sigmoid(logits)

        # Positive loss: L+ = -(1-p)^γ+ · log(p)
        pos_loss = -targets * torch.log(p.clamp(min=self.eps))
        if self.gamma_pos > 0:
            pos_loss = pos_loss * ((1.0 - p) ** self.gamma_pos)

        # Negative loss with probability shifting
        # p_m = max(p - clip, 0)  (hard threshold easy negatives)
        p_neg = (p - self.clip).clamp(min=0.0)

        # L- = -(p_m)^γ- · log(1 - p_m)
        neg_loss = -(1.0 - targets) * torch.log((1.0 - p_neg).clamp(min=self.eps))
        neg_loss = neg_loss * (p_neg ** self.gamma_neg)

        loss = pos_loss + neg_loss

        # Apply per-class weights (effective number of samples weighting)
        # FIX: belt-and-suspenders device guard. Even with class_weights
        # registered as a buffer, move it to logits.device explicitly here
        # so this loss is robust to any external code path that swaps in a
        # CPU tensor after construction (e.g. `criterion.class_weights = w`
        # done outside this class, or DataParallel replica quirks).
        if self.class_weights is not None:
            loss = loss * self.class_weights.to(loss.device).unsqueeze(0)  # (1, C) * (B, C)

        if self.reduction == "sum_mean":
            return loss.sum(dim=1).mean(dim=0)
        else:
            return loss.mean()


def compute_effective_number_weights(
    class_counts: list,
    beta: float = 0.999,
    num_classes: int = 7,
) -> torch.Tensor:
    """Compute per-class weights using Effective Number of Samples.

    From Cui et al., "Class-Balanced Loss Based on Effective Number of
    Samples", CVPR 2019.

    E_n = (1 - β^n) / (1 - β)
    weight_c = 1 / E_n_c
    Weights are normalized so that they sum to num_classes
    (i.e., average weight = 1.0).

    Parameters
    ----------
    class_counts : list of int
        Number of positive samples per class in the training set.
    beta : float
        Hyperparameter controlling re-weighting strength.
        0 → uniform weights; 0.999 → moderate; 0.9999 → aggressive.
    num_classes : int
        Number of classes.

    Returns
    -------
    weights : (num_classes,) float tensor
    """
    if beta <= 0.0:
        return torch.ones(num_classes)

    effective_num = []
    for n in class_counts:
        en = (1.0 - beta ** n) / (1.0 - beta)
        effective_num.append(en)

    weights = [1.0 / max(en, 1e-8) for en in effective_num]

    # Normalize so weights sum to num_classes (average weight = 1.0)
    total = sum(weights)
    weights = [w * num_classes / total for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


class QueryDiversityLoss(nn.Module):
    """Encourage orthogonality between label query outputs.

    Penalizes high cosine similarity between decoded query representations,
    encouraging each query to learn distinct, non-redundant features.

    Parameters
    ----------
    margin : float
        Target maximum absolute cosine similarity. Default 0.1.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, decoded_queries: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss.

        Parameters
        ----------
        decoded_queries : (B, num_labels, d_model)
            Decoded query representations from the Q2L decoder.

        Returns
        -------
        loss : scalar tensor
        """
        # Normalize queries along feature dimension
        q_norm = F.normalize(decoded_queries, dim=-1)  # (B, L, D)
        # Cosine similarity matrix between all pairs of queries
        sim = torch.bmm(q_norm, q_norm.transpose(1, 2))  # (B, L, L)
        # Zero out diagonal (self-similarity = 1, not penalized)
        mask = ~torch.eye(
            sim.size(1), dtype=torch.bool, device=sim.device
        ).unsqueeze(0)
        sim = sim * mask
        # Penalize absolute similarities above margin
        loss = torch.clamp(sim.abs() - self.margin, min=0.0).mean()
        return loss
