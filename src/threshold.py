"""
Per-class threshold optimization on validation set (Mod 3).

After training, searches optimal binary thresholds independently for each
of the 7 lesion classes by maximizing per-class F1 on the validation set.

Strategy:
    - Grid search over [0.05, 0.10, ..., 0.95] (19 candidates)
    - Independent per-class optimization (no joint search)
    - Saves thresholds.json for use during evaluation
    - Reports metrics at both 0.5 and optimized thresholds
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, LESION_NAMES, NUM_CLASSES
from .metrics import compute_metrics, format_comparison_table


def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on loader, collect probabilities and labels.

    Returns
    -------
    all_probs  : (N, 7) float array
    all_labels : (N, 7) int array
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images, labels, mod_ids = batch
            images = images.to(device)
            mod_ids = torch.tensor(
                [mod_ids] if isinstance(mod_ids, int) else mod_ids,
                dtype=torch.long, device=device,
            ) if not isinstance(mod_ids, torch.Tensor) else mod_ids.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images, mod_ids)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)
    return all_probs, all_labels


def optimize_thresholds(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    search_range: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Find optimal threshold per class to maximize per-class F1.

    Parameters
    ----------
    all_probs  : (N, 7) predicted probabilities
    all_labels : (N, 7) ground-truth binary labels
    search_range : list of thresholds to try

    Returns
    -------
    thresholds : dict mapping lesion name → optimal threshold
    """
    if search_range is None:
        search_range = [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.85, 0.90, 0.95,
        ]

    from sklearn.metrics import f1_score as sk_f1

    thresholds = {}
    N, C = all_probs.shape

    for j in range(C):
        name = LESION_NAMES[j]
        y_true = all_labels[:, j]

        best_f1 = -1.0
        best_t = 0.5

        for t in search_range:
            y_pred = (all_probs[:, j] >= t).astype(int)
            f1 = sk_f1(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        thresholds[name] = round(best_t, 4)

    return thresholds


def run_threshold_optimization(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    logger=None,
) -> Tuple[Dict[str, float], Dict, Dict]:
    """Full threshold optimization pipeline.

    1. Collect predictions on validation set
    2. Search optimal thresholds
    3. Compute metrics at 0.5 and optimized thresholds
    4. Save thresholds.json

    Returns
    -------
    thresholds   : dict mapping lesion name → optimal threshold
    metrics_05   : metrics dict at fixed 0.5
    metrics_opt  : metrics dict at optimized thresholds
    """
    log = logger.info if logger else print

    log("Collecting validation predictions...")
    all_probs, all_labels = _collect_predictions(
        model, val_loader, device, use_amp=cfg.use_amp
    )

    log("Searching optimal thresholds per class...")
    thresholds = optimize_thresholds(
        all_probs, all_labels, cfg.threshold_search_range
    )

    # Metrics at fixed 0.5
    metrics_05 = compute_metrics(all_probs, all_labels, thresholds=None)

    # Metrics at optimized thresholds
    metrics_opt = compute_metrics(all_probs, all_labels, thresholds=thresholds)

    # Print comparison
    comparison = format_comparison_table(metrics_05, metrics_opt, thresholds)
    log("\n=== Threshold Optimization Results (Validation Set) ===")
    log("\n" + comparison)

    # Save thresholds
    out_path = cfg.output_dir / "thresholds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log(f"Thresholds saved to {out_path}")

    return thresholds, metrics_05, metrics_opt
