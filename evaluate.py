"""
Standalone evaluation script.

Usage:
    # Single model evaluation:
    python evaluate.py --experiment c --checkpoint outputs/exp_c_joint/best_model.pth
    python evaluate.py --experiment a --checkpoint outputs/exp_a_cfp/best_model.pth --tta

    # Multi-fold ensemble evaluation (5-fold CV):
    python evaluate.py --experiment b --ensemble --n-folds 5 --tta

Loads a trained model and evaluates on test sets with both fixed (0.5) and
optimized thresholds. If thresholds.json exists in the experiment directory,
uses those; otherwise re-runs threshold optimization on the validation set.

Features:
    - Test-Time Augmentation (TTA) via flip ensembling
    - Multi-fold ensemble: averages predictions from K fold models
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config, LESION_NAMES, LESION_DISPLAY
from src.dataset import build_dataloaders
from src.model import Q2LLesionModel
from src.metrics import compute_metrics, format_comparison_table
from src.threshold import _collect_predictions, optimize_thresholds
from src.utils import set_seed, setup_logging


# ──────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA)
# ──────────────────────────────────────────────────────────────────────

def _apply_flip(images: torch.Tensor, flip_mode: str) -> torch.Tensor:
    """Apply flip augmentation to a batch.

    Parameters
    ----------
    images : (B, C, H, W) tensor
    flip_mode : str
        'none', 'hflip', 'vflip', or 'hflip+vflip'

    Returns
    -------
    flipped : (B, C, H, W) tensor
    """
    if flip_mode == "none":
        return images
    elif flip_mode == "hflip":
        return torch.flip(images, dims=[3])
    elif flip_mode == "vflip":
        return torch.flip(images, dims=[2])
    elif flip_mode == "hflip+vflip":
        return torch.flip(images, dims=[2, 3])
    else:
        raise ValueError(f"Unknown flip mode: {flip_mode}")


@torch.no_grad()
def collect_predictions_tta(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    tta_flips: list,
) -> tuple:
    """Collect predictions with test-time augmentation.

    For each input, runs the model on multiple flipped versions and
    averages the sigmoid probabilities.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device
    use_amp : bool
    tta_flips : list of str
        E.g., ['none', 'hflip', 'vflip', 'hflip+vflip']

    Returns
    -------
    all_probs : (N, C) numpy array
    all_labels : (N, C) numpy array
    """
    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        images, labels, mod_ids = batch
        images = images.to(device, non_blocking=True)

        if isinstance(mod_ids, torch.Tensor):
            mod_ids = mod_ids.to(device, non_blocking=True)
        else:
            mod_ids = torch.full(
                (images.size(0),), mod_ids, dtype=torch.long, device=device
            )

        # Accumulate probabilities across flip augmentations
        batch_probs = torch.zeros(
            images.size(0), 7, device=device, dtype=torch.float32
        )

        for flip_mode in tta_flips:
            flipped = _apply_flip(images, flip_mode)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(flipped, mod_ids)
            batch_probs += torch.sigmoid(logits)

        # Average across augmentations
        batch_probs /= len(tta_flips)

        all_probs.append(batch_probs.cpu().numpy())
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)

    return all_probs, all_labels


# ──────────────────────────────────────────────────────────────────────
# Multi-fold ensemble
# ──────────────────────────────────────────────────────────────────────

def _load_fold_models(
    cfg, n_folds: int, device, backbone_weights=None, use_ema=True,
):
    """Load all fold models for ensemble evaluation.

    Expects checkpoints at:
        outputs/<experiment>/fold_0/best_model.pth
        outputs/<experiment>/fold_1/best_model.pth
        ...

    Parameters
    ----------
    cfg : Config
        Base config (fold=-1 or fold=0, doesn't matter).
    n_folds : int
    device : torch.device
    backbone_weights : str or None
    use_ema : bool

    Returns
    -------
    models : list of nn.Module
        Loaded and eval-mode models.
    """
    base_dir = Path(cfg.project_root) / cfg.output_root / cfg.experiment
    models = []

    for fold_idx in range(n_folds):
        fold_dir = base_dir / f"fold_{fold_idx}"
        ckpt_path = fold_dir / "best_model.pth"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Fold {fold_idx} checkpoint not found: {ckpt_path}"
            )

        model = Q2LLesionModel(cfg, weights_path=backbone_weights)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if use_ema and "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])

        model = model.to(device)
        model.eval()
        models.append(model)

    return models


@torch.no_grad()
def collect_predictions_ensemble(
    models: list,
    loader,
    device: torch.device,
    use_amp: bool,
    tta_flips: list = None,
) -> tuple:
    """Collect predictions from an ensemble of models.

    Averages sigmoid probabilities across all models (and TTA flips if provided).

    Parameters
    ----------
    models : list of nn.Module
    loader : DataLoader
    device : torch.device
    use_amp : bool
    tta_flips : list of str or None

    Returns
    -------
    all_probs : (N, C) numpy array
    all_labels : (N, C) numpy array
    """
    if tta_flips is None:
        tta_flips = ["none"]

    for m in models:
        m.eval()

    all_probs = []
    all_labels = []

    n_augments = len(models) * len(tta_flips)

    for batch in loader:
        images, labels, mod_ids = batch
        images = images.to(device, non_blocking=True)

        if isinstance(mod_ids, torch.Tensor):
            mod_ids = mod_ids.to(device, non_blocking=True)
        else:
            mod_ids = torch.full(
                (images.size(0),), mod_ids, dtype=torch.long, device=device
            )

        batch_probs = torch.zeros(
            images.size(0), 7, device=device, dtype=torch.float32
        )

        for model in models:
            for flip_mode in tta_flips:
                flipped = _apply_flip(images, flip_mode)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(flipped, mod_ids)
                batch_probs += torch.sigmoid(logits)

        batch_probs /= n_augments

        all_probs.append(batch_probs.cpu().numpy())
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)

    return all_probs, all_labels


def main():
    parser = argparse.ArgumentParser(description="Q2L Evaluation")
    parser.add_argument(
        "--experiment", "-e", type=str, required=True,
        choices=["a", "b", "c", "d"],
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to model checkpoint (.pth). Required for single-model eval.",
    )
    parser.add_argument(
        "--project-root", type=str, default=".",
    )
    parser.add_argument(
        "--backbone-weights", type=str, default=None,
        help="Path to RETFound (ViT-Large/16) weights (Kaggle offline)",
    )
    parser.add_argument(
        "--use-ema", action="store_true", default=True,
        help="Use EMA weights if available in checkpoint",
    )
    parser.add_argument(
        "--tta", action="store_true", default=False,
        help="Enable test-time augmentation (4-flip ensemble)",
    )
    parser.add_argument(
        "--tta-flips", type=str, nargs="+",
        default=["none", "hflip", "vflip", "hflip+vflip"],
        help="TTA flip modes",
    )
    # Multi-fold ensemble arguments
    parser.add_argument(
        "--ensemble", action="store_true", default=False,
        help="Enable multi-fold ensemble evaluation",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of folds for ensemble (default: 5)",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.ensemble and args.checkpoint is None:
        parser.error("--checkpoint is required for single-model evaluation")

    cfg = get_config(args.experiment, args.project_root)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For ensemble mode, use base experiment dir for logging
    if args.ensemble:
        log_dir = Path(cfg.project_root) / cfg.output_root / cfg.experiment
    else:
        log_dir = cfg.output_dir

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir, name=f"eval_{cfg.experiment}")

    logger.info("=" * 70)
    logger.info(f"EVALUATION — {cfg.experiment}")
    if args.ensemble:
        logger.info(f"Mode: {args.n_folds}-fold ensemble")
    else:
        logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"TTA: {'enabled' if args.tta else 'disabled'}")
    if args.tta:
        logger.info(f"TTA flips: {args.tta_flips}")
    logger.info("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────
    _, val_loader, test_loaders = build_dataloaders(cfg)

    # ── Model loading ────────────────────────────────────────────────
    if args.ensemble:
        # Load all fold models
        logger.info(f"Loading {args.n_folds} fold models...")
        models = _load_fold_models(
            cfg, args.n_folds, device,
            backbone_weights=args.backbone_weights,
            use_ema=args.use_ema,
        )
        logger.info(f"  Loaded {len(models)} models for ensemble")

        # Choose prediction function
        tta_flips = args.tta_flips if args.tta else ["none"]
        def predict_fn(model_unused, loader, device, use_amp):
            return collect_predictions_ensemble(
                models, loader, device, use_amp, tta_flips
            )
    else:
        # Single model
        model = Q2LLesionModel(cfg, weights_path=args.backbone_weights)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if args.use_ema and "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"])
            logger.info("Loaded EMA weights")
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded model weights")

        model = model.to(device)
        model.eval()

        logger.info(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
        logger.info(f"Best val F1: {ckpt.get('best_val_f1', '?')}")

        if args.tta:
            def predict_fn(model, loader, device, use_amp):
                return collect_predictions_tta(
                    model, loader, device, use_amp, args.tta_flips
                )
        else:
            predict_fn = _collect_predictions

    # ── Thresholds ───────────────────────────────────────────────────
    thresh_path = log_dir / "thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            thresholds = json.load(f)
        logger.info(f"Loaded thresholds from {thresh_path}")
    else:
        logger.info("No thresholds.json found — optimizing on validation set...")
        if args.ensemble:
            val_probs, val_labels = predict_fn(None, val_loader, device, cfg.use_amp)
        else:
            val_probs, val_labels = predict_fn(
                model, val_loader, device, cfg.use_amp
            )
        thresholds = optimize_thresholds(val_probs, val_labels)
        with open(thresh_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        logger.info(f"Thresholds saved to {thresh_path}")

    logger.info("Optimized thresholds:")
    for name in LESION_NAMES:
        logger.info(f"  {name}: {thresholds.get(name, 0.5):.2f}")

    # ── Test evaluation ──────────────────────────────────────────────
    for modality, test_loader in test_loaders.items():
        logger.info("")
        logger.info(f"{'=' * 50}")
        logger.info(f"TEST: {modality.upper()} ({len(test_loader.dataset)} samples)")
        logger.info(f"{'=' * 50}")

        if args.ensemble:
            all_probs, all_labels = predict_fn(
                None, test_loader, device, cfg.use_amp
            )
        else:
            all_probs, all_labels = predict_fn(
                model, test_loader, device, cfg.use_amp
            )

        metrics_05 = compute_metrics(all_probs, all_labels, thresholds=None)
        metrics_opt = compute_metrics(all_probs, all_labels, thresholds=thresholds)

        table = format_comparison_table(metrics_05, metrics_opt, thresholds)
        logger.info("\n" + table)
        logger.info(f"\nExact match @0.5:      {metrics_05['exact_match']:.4f}")
        logger.info(f"Exact match @optimized: {metrics_opt['exact_match']:.4f}")

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    main()
