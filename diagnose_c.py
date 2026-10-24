"""
Diagnostic script for Experiment C (joint CFP+UWF, naive merge).

Purpose
-------
train.py fits ONE set of per-lesion thresholds on the pooled (CFP+UWF)
validation set, then applies that same threshold to both the CFP and
UWF test sets separately. That means C's weak CFP VB/IRMA score
(F1 = 0.1389, recall = 0.1163 at the pooled-validation threshold) is a
confound: it could reflect a genuinely weak CFP-VB/IRMA representation,
or it could just be a threshold that was never tuned for CFP on its own
(UWF has far more images and likely dominates the pooled validation
set when the two get combined).

This script separates those two explanations by fitting a SECOND set of
thresholds per modality (CFP-only validation set -> CFP thresholds,
UWF-only validation set -> UWF thresholds) and comparing:

    1. Pooled-threshold test metrics (byte-for-byte what train.py
       already computed and saved in results.json) -- re-derived here
       from the same checkpoint, as a sanity check.
    2. Per-modality-threshold test metrics -- the new comparison.

If CFP VB/IRMA F1 is still low under its OWN CFP-specific threshold,
that's real evidence of a weak CFP representation for that lesion. If
it recovers substantially, the pooled threshold was largely the
problem, not the representation.

It does not retrain anything -- it only needs the exp_c_joint
best_model.pth checkpoint, which is NOT included in the diagnostic
code bundle. Point --checkpoint at wherever you saved it (Kaggle
output, or wherever you downloaded outputs/exp_c_joint/best_model.pth
to).

Usage
-----
    python diagnose_c.py \\
        --checkpoint outputs/exp_c_joint/best_model.pth \\
        --config outputs/exp_c_joint/config.json \\
        --project-root .

Requires a GPU-capable machine with the MMRDR dataset available at the
paths in config.json (or pass --cfp-dir / --uwf-dir to override them),
and the RETFound pretrained backbone reachable the same way train.py
already reaches it. This is a short, checkpoint-only pass over the
validation and test sets -- minutes, not hours, no training involved.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import Config, LESION_NAMES
from src.dataset import (
    LesionDataset, get_val_transform, load_splits, IMAGE_SIZE,
)
from src.model import Q2LLesionModel
from src.threshold import _collect_predictions, optimize_thresholds
from src.metrics import compute_metrics, format_comparison_table
from src.utils import load_checkpoint


def build_per_modality_loaders(cfg: Config):
    """Like dataset.build_dataloaders, but keeps val/test split per
    modality instead of pooling CFP+UWF validation together. Returns
    two dicts: val_loaders and test_loaders, each {"cfp": ..., "uwf": ...}.
    """
    from torch.utils.data import DataLoader

    project = Path(cfg.project_root)
    val_transform = get_val_transform(IMAGE_SIZE)

    val_loaders, test_loaders = {}, {}
    for modality in cfg.modalities:
        if modality == "cfp":
            data_dir = project / cfg.cfp_dir
            csv_name = cfg.cfp_csv
        else:
            data_dir = project / cfg.uwf_dir
            csv_name = cfg.uwf_csv

        _, val_df, test_df = load_splits(
            data_dir, csv_name, cfg.val_split, cfg.seed,
            use_cv=cfg.use_cv, n_folds=cfg.n_folds, fold=cfg.fold,
        )
        val_ds = LesionDataset(val_df, data_dir, modality, val_transform)
        test_ds = LesionDataset(test_df, data_dir, modality, val_transform)

        val_loaders[modality] = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
        test_loaders[modality] = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )
    return val_loaders, test_loaders


def main():
    parser = argparse.ArgumentParser(description="Diagnose Experiment C's per-modality weakness")
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to exp_c_joint's best_model.pth")
    parser.add_argument("--config", type=str, default="outputs/exp_c_joint/config.json",
                         help="Path to exp_c_joint's saved config.json")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--cfp-dir", type=str, default=None, help="Override cfg.cfp_dir")
    parser.add_argument("--uwf-dir", type=str, default=None, help="Override cfg.uwf_dir")
    parser.add_argument("--out", type=str, default="outputs/exp_c_joint/diagnostic_per_modality.json")
    parser.add_argument("--backbone-weights", type=str, default=None,
                         help="Path to RETFound .pth (same flag as train.py). Optional here: "
                              "load_checkpoint() below overwrites every weight with the trained "
                              "exp_c_joint checkpoint right after construction, so the backbone's "
                              "own pretrained init doesn't matter for this script. Only pass this "
                              "if you'd rather not rely on that.")
    args = parser.parse_args()

    cfg = Config.load(Path(args.config))
    cfg.project_root = args.project_root
    if args.cfp_dir:
        cfg.cfp_dir = args.cfp_dir
    if args.uwf_dir:
        cfg.uwf_dir = args.uwf_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device found -- this will be slow but will still run.")

    print(f"Loading model for adaptive_query_mode={cfg.adaptive_query_mode!r}, "
          f"use_modality_conditioning={cfg.use_modality_conditioning} ...")
    if args.backbone_weights is None:
        # Skip RETFound's own pretrained-weight loading: this script only
        # ever runs a trained checkpoint forward, and load_checkpoint()
        # below replaces every parameter (backbone included) with
        # best_model.pth's saved state_dict right after construction. So
        # the backbone would otherwise need a local RETFound .pth just to
        # avoid RETFoundBackbone's offline-download guard, for a value
        # that gets thrown away a few lines later. cfg.backbone_pretrained
        # = False takes that guard's other branch (random init + a
        # harmless RuntimeWarning) instead of raising.
        cfg.backbone_pretrained = False
    model = Q2LLesionModel(cfg, weights_path=args.backbone_weights).to(device)
    ckpt = load_checkpoint(Path(args.checkpoint), model, device=device)
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("Using EMA weights (matches what train.py used for its saved results.json).")
    model.eval()

    print("Building per-modality validation and test loaders "
          "(kept separate, unlike train.py's pooled validation set)...")
    val_loaders, test_loaders = build_per_modality_loaders(cfg)

    results = {}

    # ---- 1. Reproduce the pooled-threshold thresholds train.py used ----
    # (CombinedLesionDataset pools val_loaders["cfp"] + val_loaders["uwf"];
    #  we approximate that here by concatenating predictions from both,
    #  which is what train.py's val_loader effectively does.)
    print("\n=== Reproducing train.py's pooled-validation thresholds ===")
    pooled_probs, pooled_labels = [], []
    for modality in cfg.modalities:
        p, l = _collect_predictions(model, val_loaders[modality], device, cfg.use_amp)
        pooled_probs.append(p)
        pooled_labels.append(l)
    pooled_probs = np.concatenate(pooled_probs, axis=0)
    pooled_labels = np.concatenate(pooled_labels, axis=0)
    pooled_thresholds = optimize_thresholds(pooled_probs, pooled_labels, cfg.threshold_search_range)
    print("Pooled thresholds:", pooled_thresholds)

    # ---- 2. Fit thresholds separately per modality ----
    print("\n=== Fitting per-modality thresholds ===")
    per_modality_thresholds = {}
    for modality in cfg.modalities:
        p, l = _collect_predictions(model, val_loaders[modality], device, cfg.use_amp)
        t = optimize_thresholds(p, l, cfg.threshold_search_range)
        per_modality_thresholds[modality] = t
        print(f"{modality.upper()} thresholds:", t)

    # ---- 3. Evaluate each modality's test set under BOTH threshold sets ----
    for modality, test_loader in test_loaders.items():
        print(f"\n--- {modality.upper()} Test Set ---")
        all_probs, all_labels = _collect_predictions(model, test_loader, device, cfg.use_amp)

        metrics_pooled = compute_metrics(all_probs, all_labels, thresholds=pooled_thresholds)
        metrics_own = compute_metrics(all_probs, all_labels, thresholds=per_modality_thresholds[modality])

        print(f"  Under POOLED threshold (what train.py used):   macro_F1 = {metrics_pooled['macro_f1']:.4f}")
        print(f"  Under {modality.upper()}-OWN threshold:                    macro_F1 = {metrics_own['macro_f1']:.4f}")
        print(format_comparison_table(metrics_pooled, metrics_own, per_modality_thresholds[modality]))

        results[modality] = {
            "under_pooled_threshold": metrics_pooled,
            "under_own_threshold": metrics_own,
            "pooled_thresholds_used": pooled_thresholds,
            "own_thresholds_used": per_modality_thresholds[modality],
        }

    # ---- 4. Headline comparison for the VB/IRMA question specifically ----
    print("\n=== VB/IRMA specifically ===")
    for modality in cfg.modalities:
        pooled_f1 = results[modality]["under_pooled_threshold"]["VB_IRMA_f1"]
        own_f1 = results[modality]["under_own_threshold"]["VB_IRMA_f1"]
        own_rec = results[modality]["under_own_threshold"]["VB_IRMA_rec"]
        print(f"  {modality.upper()}: pooled-threshold F1={pooled_f1:.4f}  ->  own-threshold F1={own_f1:.4f} (recall={own_rec:.4f})")
    print(
        "\nIf CFP's own-threshold F1 is still close to the pooled-threshold\n"
        "figure, the weakness is representational (the ablation is the right\n"
        "next step). If it recovers a lot, most of what looked like a CFP\n"
        "weakness was the shared-threshold artifact, and that should be\n"
        "flagged explicitly rather than folded into the ablation's premise."
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
