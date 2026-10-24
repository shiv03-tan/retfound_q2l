"""
Evaluation metrics for multi-label lesion classification.

Reports:
    - Per-class F1, precision, recall
    - Macro F1 (primary metric)
    - Per-class accuracy
    - Overall sample accuracy
    - Exact match ratio
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

from .config import LESION_NAMES, LESION_DISPLAY, NUM_CLASSES


def compute_metrics(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict:
    """Compute comprehensive metrics for multi-label classification.

    Parameters
    ----------
    all_probs  : (N, 7) float array of predicted probabilities
    all_labels : (N, 7) int array of ground-truth binary labels
    thresholds : dict mapping lesion name → float threshold.
                 If None, uses 0.5 for all classes.

    Returns
    -------
    metrics : dict with per-class and aggregate metrics
    """
    assert all_probs.shape == all_labels.shape
    N, C = all_probs.shape
    assert C == NUM_CLASSES

    # Build threshold array
    if thresholds is None:
        t = np.full(C, 0.5)
    else:
        t = np.array([thresholds.get(name, 0.5) for name in LESION_NAMES])

    # Binarize predictions
    all_preds = (all_probs >= t[np.newaxis, :]).astype(int)

    metrics = {}

    # Per-class metrics
    per_class_f1 = []
    for j in range(C):
        name = LESION_NAMES[j]
        display = LESION_DISPLAY[j]
        y_true = all_labels[:, j]
        y_pred = all_preds[:, j]
        y_prob = all_probs[:, j]

        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        # AUC (only if both classes present)
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = float("nan")

        metrics[f"{name}_f1"] = round(f1, 4)
        metrics[f"{name}_prec"] = round(prec, 4)
        metrics[f"{name}_rec"] = round(rec, 4)
        metrics[f"{name}_acc"] = round(acc, 4)
        metrics[f"{name}_auc"] = round(auc, 4) if not np.isnan(auc) else None
        metrics[f"{name}_threshold"] = round(float(t[j]), 4)
        metrics[f"{name}_pos_count"] = int(y_true.sum())
        per_class_f1.append(f1)

    # Aggregate metrics
    metrics["macro_f1"] = round(float(np.mean(per_class_f1)), 4)
    metrics["macro_prec"] = round(
        precision_score(all_labels, all_preds, average="macro", zero_division=0), 4
    )
    metrics["macro_rec"] = round(
        recall_score(all_labels, all_preds, average="macro", zero_division=0), 4
    )

    # Element-wise accuracy (flatten all labels)
    metrics["element_accuracy"] = round(
        accuracy_score(all_labels.flatten(), all_preds.flatten()), 4
    )

    # Exact match (all 7 labels correct for a sample)
    exact_match = (all_preds == all_labels).all(axis=1).mean()
    metrics["exact_match"] = round(float(exact_match), 4)

    metrics["num_samples"] = N

    return metrics


def format_metrics_line(metrics: Dict, prefix: str = "") -> str:
    """Format metrics as a single-line log string.

    Example: "F1=0.7845 | MA=0.89 HE=0.82 IH=0.81 VB=0.61 NV=0.68 VH=0.64 RD=0.44"
    """
    parts = [f"{prefix}F1={metrics['macro_f1']:.4f}"]

    short_names = ["MA", "HE", "IH", "VB", "NV", "VH", "RD"]
    for sn, name in zip(short_names, LESION_NAMES):
        f1 = metrics.get(f"{name}_f1", 0.0)
        parts.append(f"{sn}={f1:.2f}")

    return " | ".join([parts[0], " ".join(parts[1:])])


def format_comparison_table(
    metrics_05: Dict, metrics_opt: Dict, thresholds: Dict
) -> str:
    """Format a comparison table: fixed 0.5 vs optimized thresholds."""
    lines = []
    lines.append(f"{'Lesion':<10} {'Thresh':>6} {'F1@0.5':>7} {'F1@opt':>7} {'Diff':>7}")
    lines.append("-" * 45)

    for name, display in zip(LESION_NAMES, LESION_DISPLAY):
        t = thresholds.get(name, 0.5)
        f1_05 = metrics_05.get(f"{name}_f1", 0)
        f1_opt = metrics_opt.get(f"{name}_f1", 0)
        delta = f1_opt - f1_05
        lines.append(
            f"{display:<10} {t:>6.2f} {f1_05:>7.4f} {f1_opt:>7.4f} {delta:>+7.4f}"
        )

    lines.append("-" * 45)
    f1_05_macro = metrics_05.get("macro_f1", 0)
    f1_opt_macro = metrics_opt.get("macro_f1", 0)
    delta_macro = f1_opt_macro - f1_05_macro
    lines.append(
        f"{'Macro':<10} {'':>6} {f1_05_macro:>7.4f} {f1_opt_macro:>7.4f} {delta_macro:>+7.4f}"
    )

    return "\n".join(lines)
