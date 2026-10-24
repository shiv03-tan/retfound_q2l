"""
Main training script for Q2L lesion detection.

Usage:
    python train.py --experiment a        # CFP-only baseline
    python train.py --experiment b        # UWF-only baseline
    python train.py --experiment c        # Joint naive merge
    python train.py --experiment d        # Joint + modality conditioning

Architecture:
    RETFound ViT-Large/16 (~303.3M) + Q2L Decoder (d=1024, 1enc+2dec, 4 heads)
    + ASL (γ+=0, γ-=4, clip=0.05)
    + Differential LR (backbone 1e-5, decoder 1e-4)
    + Per-class threshold optimization
    + Modality-balanced sampling (Exp C/D)

Optimization roadmap features:
    Stage 1: Bug fixes (ASL defaults, warmup scheduler, inference-only ckpt)
    Stage 2: CutMix/MixUp, gradient accumulation, per-class classifiers
    Stage 3: Query diversity loss, EMA warmup, configurable decoder capacity
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

# ── Single-GPU enforcement (Phase 4) ─────────────────────────────────────
# Must happen BEFORE `import torch` triggers any CUDA runtime init. This
# Kaggle notebook's accelerator can present 2x T4 GPUs; DataParallel used
# to spread the model across both, which the RAM-growth investigation
# identified as the primary driver of the unbounded host-RSS growth that
# killed the kernel. Restricting the process's CUDA visibility to
# physical GPU 0 here is a stronger guarantee than just "don't call
# nn.DataParallel" — it makes it structurally impossible for anything
# (this script, timm, a future contributor) to allocate on a second
# device by accident, since torch never even sees it exists.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, get_config, LESION_NAMES, LESION_DISPLAY, NUM_CLASSES
from src.dataset import (
    build_dataloaders, ModalityBalancedSampler,
    apply_cutmix, apply_mixup,
    count_class_positives_fast,
)
from src.model import Q2LLesionModel
from src.losses import AsymmetricLoss, QueryDiversityLoss, compute_effective_number_weights
from src.metrics import compute_metrics, format_metrics_line
from src.threshold import (
    run_threshold_optimization,
    _collect_predictions,
    optimize_thresholds,
)
from src.utils import (
    set_seed,
    worker_init_fn,
    setup_logging,
    save_checkpoint,
    load_checkpoint,
    verify_checkpoint,
    append_epoch_history,
    ModelEMA,
    Timer,
    process_rss_gb,
    child_process_count,
    children_rss_gb,
    system_ram_gb,
    gpu_stats_all,
    PeakTracker,
)


# ──────────────────────────────────────────────────────────────────────
# Cosine scheduler with warmup
# ──────────────────────────────────────────────────────────────────────

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup.

    During warmup (epoch < warmup_epochs): LR linearly ramps from 0 to base_lr.
    After warmup: cosine decay to 0.

    Bug fix (Stage 1): The scheduler is now stepped BEFORE the first training
    epoch begins (via last_epoch=-1 convention), so epoch 0 of training
    correctly uses the first warmup LR rather than skipping it.
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup: epoch 0 → alpha=0, epoch warmup-1 → alpha=(warmup-1)/warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [base_lr * alpha for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────
# OOM diagnostics
# ──────────────────────────────────────────────────────────────────────

def _log_oom_and_raise(batch_idx: int, images: torch.Tensor, cfg, err: Exception):
    """Write full OOM diagnostics to train.log (on disk, flushed) before
    raising a clean exception.

    This exists because the original failure mode was a hard kernel death
    with *no* Python traceback anywhere — the process was killed before
    anything could be written. By explicitly catching the OOM here and
    forcing a flush to the FileHandler-backed logger, at minimum the
    on-disk train.log will contain the exact batch index and memory state
    at the moment of failure, even if the process is killed a moment
    later while unwinding.
    """
    logger = logging.getLogger(cfg.experiment)
    msg_lines = [
        "",
        "=" * 70,
        "CUDA OUT OF MEMORY",
        "=" * 70,
        f"  Batch index      : {batch_idx}",
        f"  Batch shape      : {tuple(images.shape)}",
        f"  Batch size (cfg) : {cfg.batch_size}",
        f"  Grad accum steps : {cfg.grad_accum_steps}",
        f"  Effective batch  : {cfg.effective_batch_size}",
        f"  Freeze blocks    : {cfg.freeze_blocks}",
        f"  Gradient ckpt    : {getattr(cfg, 'gradient_checkpointing', 'N/A')}",
    ]
    if torch.cuda.is_available():
        msg_lines += [
            f"  Allocated        : {torch.cuda.memory_allocated()/1e9:.2f} GB",
            f"  Reserved         : {torch.cuda.memory_reserved()/1e9:.2f} GB",
            f"  Peak allocated   : {torch.cuda.max_memory_allocated()/1e9:.2f} GB",
        ]
    msg_lines += [str(err), "=" * 70, ""]

    for line in msg_lines:
        logger.error(line)
    # Belt-and-suspenders: force-flush every handler explicitly, in case
    # anything upstream (subprocess pipe, container, driver) is about to
    # tear the process down.
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    # Free whatever we can before propagating.
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    raise RuntimeError(
        f"CUDA OOM at batch {batch_idx} (see train.log for full diagnostics). "
        f"Reduce batch_size/grad_accum, enable gradient_checkpointing, or "
        f"re-freeze early backbone stages."
    ) from err


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    cfg: Config,
    ema: ModelEMA = None,
    diversity_criterion: nn.Module = None,
    global_step: int = 0,
    epoch: int = 0,
    best_val_f1: float = 0.0,
    peaks: "PeakTracker" = None,
    logger: logging.Logger = None,
    scheduler=None,
) -> "tuple[float, int]":
    """Train for one epoch with gradient accumulation and augmentation.

    Returns (avg_loss, updated_global_step).

    Runs for the full epoch unconditionally — no self-imposed time budget
    or early stop. If Kaggle hard-kills the process mid-epoch, whatever
    the most recent mid-epoch checkpoint captured (see
    `checkpoint_every_batches`) is what the next run resumes from — that
    checkpoint cadence is what makes resuming safe, not any attempt by
    this process to predict or beat Kaggle's own timeout.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    accum_steps = cfg.grad_accum_steps
    use_diversity = (
        diversity_criterion is not None and cfg.query_diversity_weight > 0
    )
    need_queries = use_diversity
    logger = logger or logging.getLogger(cfg.experiment)
    peaks = peaks or PeakTracker()

    # Zero gradients at the start
    optimizer.zero_grad(set_to_none=True)

    # FIX (crash forensics): the Kaggle kernel-death investigation showed
    # that a CUDA OOM inside this loop can kill the process hard enough
    # that no Python traceback is ever flushed anywhere — the notebook
    # just shows "Kernel died" 45 minutes later with zero epoch logs.
    # Logging peak memory to train.log (a real file on disk, not just the
    # notebook's stdout/stderr pipe) every few batches means that even if
    # the process is killed with no traceback, the last few lines of
    # train.log tell you exactly which batch and how much memory was in
    # use right before the crash.
    mem_log_every = max(1, len(loader) // 5)

    raw_model = model  # single-GPU only (Phase 4) — no DataParallel wrapper to unwrap

    def _save_mid_epoch_checkpoint(batch_idx: int):
        """Mid-epoch checkpoint (Mod 8 / Part 3). Restores weights,
        optimizer, scheduler, scaler, and EMA state exactly on resume;
        the epoch's data iteration itself restarts from batch 0 (see
        save_checkpoint()'s docstring for why — DataLoader/sampler state
        is not resumable without much higher-risk custom sampler work).
        """
        save_checkpoint(
            raw_model, optimizer, scheduler, scaler, epoch, best_val_f1,
            cfg.output_dir / "mid_epoch.pth",
            ema_model=ema.ema_model if ema is not None else None,
            inference_only=False,
            global_step=global_step,
            batch_idx=batch_idx,
        )
        logger.info(f"    [checkpoint] mid-epoch save at batch {batch_idx} "
                    f"(global_step={global_step})")

    for batch_idx, batch in enumerate(loader):
        images, labels, mod_ids = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if isinstance(mod_ids, torch.Tensor):
            mod_ids = mod_ids.to(device, non_blocking=True)
        else:
            # Single modality: mod_ids is an int
            mod_ids = torch.full(
                (images.size(0),), mod_ids, dtype=torch.long, device=device
            )

        # ── Batch-level augmentation (CutMix / MixUp) ────────────────
        if cfg.use_cutmix or cfg.use_mixup:
            if np.random.random() < cfg.augmix_prob:
                # Randomly choose CutMix or MixUp (or whichever is enabled)
                aug_choices = []
                if cfg.use_cutmix:
                    aug_choices.append("cutmix")
                if cfg.use_mixup:
                    aug_choices.append("mixup")
                chosen = aug_choices[np.random.randint(len(aug_choices))]

                if chosen == "cutmix":
                    images, labels = apply_cutmix(
                        images, labels, alpha=cfg.cutmix_alpha
                    )
                else:
                    images, labels = apply_mixup(
                        images, labels, alpha=cfg.mixup_alpha
                    )

        # ── Forward + backward pass (OOM-safe) ─────────────────────────
        # FIX: a bare CUDA OOM here previously propagated as an
        # unrecoverable process kill with no Python traceback (see
        # kernel-death investigation). We now catch it explicitly, dump
        # full memory diagnostics + the exact batch index to train.log
        # (flushed immediately, on disk), free what we can, and raise a
        # clean, descriptive exception instead of letting the process
        # die silently.
        try:
            with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
                if need_queries:
                    logits, decoded_queries = model(
                        images, mod_ids, return_queries=True
                    )
                else:
                    logits = model(images, mod_ids)

                loss = criterion(logits, labels)

                # Query diversity regularization
                if use_diversity:
                    div_loss = diversity_criterion(decoded_queries)
                    loss = loss + cfg.query_diversity_weight * div_loss

                # Scale loss for gradient accumulation
                loss = loss / accum_steps

            # ── Backward pass ────────────────────────────────────────
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError as oom_err:  # torch>=2.0
            _log_oom_and_raise(batch_idx, images, cfg, oom_err)
        except RuntimeError as rt_err:
            if "out of memory" in str(rt_err).lower():
                _log_oom_and_raise(batch_idx, images, cfg, rt_err)
            raise

        if batch_idx % mem_log_every == 0:
            gpu_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            gpu_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            gpu_peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            ram_used, ram_total = system_ram_gb()
            self_rss = process_rss_gb()
            kids_rss = children_rss_gb()
            n_children = child_process_count()
            gpu_all = gpu_stats_all()  # ALL GPUs, closing the GPU1/DataParallel blind spot

            peaks.update("gpu_alloc_gb", gpu_alloc)
            peaks.update("gpu_reserved_gb", gpu_reserved)
            peaks.update("ram_used_gb", ram_used)
            peaks.update("self_rss_gb", self_rss)
            peaks.update("children_rss_gb", kids_rss)

            gpu_all_str = " ".join(
                f"gpu{idx}={used/1024:.2f}/{total/1024:.2f}GB({util}%)"
                for idx, used, total, util in gpu_all
            ) or "gpu_all=unavailable"

            ram_str = f"{ram_used:.2f}/{ram_total:.2f}GB" if ram_used is not None else "unavailable"
            self_rss_str = f"{self_rss:.2f}GB" if self_rss is not None else "unavailable"
            kids_rss_str = f"{kids_rss:.2f}GB" if kids_rss is not None else "unavailable"

            logger.info(
                f"    [mem] batch {batch_idx}/{len(loader)} | "
                f"alloc={gpu_alloc:.2f}GB reserved={gpu_reserved:.2f}GB peak={gpu_peak:.2f}GB | "
                f"RAM={ram_str}"
            )
            logger.info(
                f"           self_rss={self_rss_str} children_rss={kids_rss_str} "
                f"children={n_children} | {gpu_all_str}"
            )

            # ── Periodic explicit GC (Phase 5 hardening) ───────────────
            # Belt-and-suspenders against reference-cycle buildup (e.g.
            # autograd graphs from checkpointed segments, or CutMix/MixUp
            # temporaries) that Python's generational GC might otherwise
            # leave uncollected for a while under steady allocation
            # pressure. This is deliberately infrequent (same cadence as
            # the mem-log block, ~5x/epoch) since gc.collect() has a real
            # CPU cost — it is a safety net, not a substitute for not
            # leaking in the first place.
            gc.collect()

        # ── Small per-batch CPU-side cleanup ──────────────────────────
        # Drop the local reference to decoded_queries as soon as we no
        # longer need it. It's only produced when query-diversity loss is
        # enabled (need_queries=True); left as a lingering local it would
        # keep that batch's full (B, num_labels, d_model) autograd graph
        # alive one iteration longer than necessary.
        if need_queries:
            decoded_queries = None

        # ── Optimizer step (every accum_steps) ───────────────────────
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # EMA update (once per optimizer step, not per micro-batch)
            if ema is not None:
                ema.update(model)

            # ── Mid-epoch checkpoint (Mod 8) — periodic safety net so an
            # abrupt Kaggle-imposed kill (which can happen at any point,
            # with no warning) never loses more than
            # `checkpoint_every_batches` optimizer steps of work. This is
            # unconditional — there is no session-time check here, and
            # training does not stop itself early for any reason related
            # to elapsed wall-clock time. Kaggle's own timeout is the only
            # thing that ends a session; this checkpoint is what makes
            # the NEXT invocation of train.py resume seamlessly from it.
            if (cfg.checkpoint_every_batches > 0
                    and global_step % cfg.checkpoint_every_batches == 0):
                _save_mid_epoch_checkpoint(batch_idx)

        total_loss += loss.item() * accum_steps  # undo the 1/accum scaling
        n_batches += 1

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: Config,
    criterion: nn.Module = None,
) -> dict:
    """Run validation, return metrics dict (at threshold 0.5).

    If `criterion` is provided, also computes and includes 'val_loss'
    (Phase 6 requirement — previously not computed at all in validation).
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images, labels, mod_ids = batch
        images = images.to(device, non_blocking=True)

        if isinstance(mod_ids, torch.Tensor):
            mod_ids = mod_ids.to(device, non_blocking=True)
        else:
            mod_ids = torch.full(
                (images.size(0),), mod_ids, dtype=torch.long, device=device
            )

        with torch.amp.autocast(device_type="cuda", enabled=cfg.use_amp):
            logits = model(images, mod_ids)
            if criterion is not None:
                labels_dev = labels.to(device, non_blocking=True)
                loss = criterion(logits, labels_dev)
                total_loss += loss.item()
                n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)

    metrics = compute_metrics(all_probs, all_labels, thresholds=None)
    if criterion is not None:
        metrics["val_loss"] = total_loss / max(n_batches, 1)
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q2L Lesion Detection Training")
    parser.add_argument(
        "--experiment", "-e", type=str, required=True,
        choices=["a", "b", "c", "d", "j0", "j1", "j3", "j4", "j5"],
        help="Experiment: a=CFP, b=UWF, c=Joint, d=Joint+CMQC, "
             "j0=shared, j1=static, j3=gated_modality, j4=gated_content, j5=gated_full",
    )
    parser.add_argument(
        "--project-root", type=str, default=".",
        help="Path to Q2L_Try directory",
    )
    parser.add_argument(
        "--backbone-weights", type=str, default=None,
        help="Path to RETFound (ViT-Large/16) weights (for Kaggle offline)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="CV fold index (0..n_folds-1). Enables CV mode. Overrides cfg.fold.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=None,
        help="Number of CV folds. Overrides cfg.n_folds.",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Re-run even if results.json already indicates this experiment is complete.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override config batch_size (e.g. 16 for fresh experiments on larger GPU)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override config num_workers for DataLoader",
    )
    parser.add_argument(
        "--grad-accum-steps", type=int, default=None,
        help="Override config grad_accum_steps (effective_batch = batch_size * this)",
    )
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────
    cfg = get_config(args.experiment, args.project_root)

    # Apply CLI overrides for cross-validation
    if args.fold is not None:
        cfg.use_cv = True
        cfg.fold = args.fold
    if args.n_folds is not None:
        cfg.n_folds = args.n_folds

    # Apply CLI overrides for training hyperparameters
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.grad_accum_steps is not None:
        cfg.grad_accum_steps = args.grad_accum_steps

    # ── Part 4: restart-safe at the train.py level, not just the launcher.
    # The launcher already checks this before ever invoking train.py, but
    # checking again here means the experiment is correctly skipped even
    # if train.py is invoked directly/manually, defense in depth.
    results_path = cfg.output_dir / "results.json"
    best_path = cfg.output_dir / "best_model.pth"
    if not args.force and results_path.exists() and best_path.exists():
        print(f"✓ {cfg.experiment} already complete "
              f"({results_path} and {best_path} both exist). "
              f"Pass --force to re-run. Exiting 0.")
        return

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output directory ─────────────────────────────────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(cfg.output_dir, name=cfg.experiment)
    timer = Timer()

    logger.info("=" * 70)
    logger.info(f"Experiment: {cfg.experiment}")
    logger.info(f"Description: {cfg.description}")
    logger.info(f"Modalities: {cfg.modalities}")
    logger.info(f"Backbone: RETFound ViT-Large/16 (d={cfg.d_model}, LOCKED)")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {cfg.output_dir}")
    if cfg.use_cv:
        logger.info(f"Cross-validation: {cfg.n_folds}-fold, fold {cfg.fold}")
    logger.info("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────
    logger.info("Building dataloaders...")
    train_loader, val_loader, test_loaders = build_dataloaders(cfg)

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches:   {len(val_loader)}")
    for mod, tl in test_loaders.items():
        logger.info(f"  Test ({mod}):    {len(tl.dataset)} samples")

    # ── Model ────────────────────────────────────────────────────────
    logger.info("Building model...")
    logger.info(f"  --backbone-weights received: {args.backbone_weights!r}")
    if args.backbone_weights is None:
        logger.warning(
            "  ⚠ No --backbone-weights path was passed to this process — "
            "RETFoundBackbone will build with random weights unless "
            "cfg.backbone_pretrained is also False (which raises instead). "
            "This should never happen for a real training run; verify the "
            "notebook found RETFound_mae_meh.pth and passed it through."
        )
    model = Q2LLesionModel(cfg, weights_path=args.backbone_weights)
    model = model.to(device)

    # RETFoundBackbone._load_retfound_weights() already printed a full
    # load report to stdout (checkpoint path/filename, matched/missing/
    # unexpected key counts, explicit success confirmation) and raises if
    # any backbone parameter would silently be randomly initialized — this
    # just also puts the key facts into train.log so they're not only in
    # the notebook cell's transient stdout.
    load_report = getattr(model.backbone, "load_report", None)
    if load_report is not None:
        logger.info(
            f"  RETFound checkpoint: {load_report['checkpoint_filename']} "
            f"({load_report['checkpoint_path']}) — "
            f"{load_report['matched']}/{load_report['target_total_keys']} "
            f"backbone params matched, "
            f"{len(load_report['missing_keys'])} missing, "
            f"{len(load_report['unexpected_unrecognized'])} unrecognized "
            f"unexpected keys — success={load_report['success']}"
        )
    else:
        logger.warning(
            "  ⚠ model.backbone.load_report is None — no pretrained "
            "checkpoint was loaded (backbone is randomly initialized)."
        )

    summary = model.param_summary()
    logger.info(f"  Backbone: {summary['backbone_total']:,} total, "
                f"{summary['backbone_trainable']:,} trainable")
    logger.info(f"  Decoder:  {summary['decoder_total']:,} total, "
                f"{summary['decoder_trainable']:,} trainable")
    logger.info(f"  Total:    {summary['total']:,} params, "
                f"{summary['trainable']:,} trainable")
    ckpt_status = getattr(model.backbone, "gradient_checkpointing_enabled", False)
    logger.info(f"  Gradient checkpointing: {'ENABLED' if ckpt_status else 'disabled'} "
                f"(requested={getattr(cfg, 'gradient_checkpointing', True)})")
    if cfg.freeze_blocks == [] and not ckpt_status:
        logger.warning(
            "  ⚠ Backbone is fully unfrozen AND gradient checkpointing is "
            "NOT active. RETFound is a ~303M-param ViT-L — on <=16GB GPUs "
            "this combination will almost certainly CUDA OOM (far worse "
            "than the equivalent ConvNeXt-V2 case this warning was "
            "originally written for). Consider freeze_blocks=list(range(0,18)) "
            "or a smaller batch_size."
        )

    # ── Loss ─────────────────────────────────────────────────────────
    # Compute class-balanced weights from training set
    class_weights = None
    if cfg.class_balance_enabled and cfg.class_balance_beta > 0:
        # Get training datasets from the loader
        train_ds = train_loader.dataset
        if hasattr(train_ds, 'datasets'):
            train_ds_list = train_ds.datasets  # CombinedLesionDataset
        else:
            train_ds_list = [train_ds]

        class_counts = count_class_positives_fast(train_ds_list, NUM_CLASSES)
        class_weights = compute_effective_number_weights(
            class_counts, beta=cfg.class_balance_beta, num_classes=NUM_CLASSES
        )

        logger.info(f"  Class-balanced weighting (β={cfg.class_balance_beta}):")
        for name, cnt, w in zip(LESION_NAMES, class_counts, class_weights.tolist()):
            logger.info(f"    {name:8s}: {cnt:5d} positives → weight {w:.3f}")

    criterion = AsymmetricLoss(
        gamma_pos=cfg.asl_gamma_pos,
        gamma_neg=cfg.asl_gamma_neg,
        clip=cfg.asl_clip,
        reduction=cfg.loss_reduction,
        class_weights=class_weights,
    )
    # FIX: AsymmetricLoss is an nn.Module holding class_weights as a buffer.
    # Only `model` was ever moved to `device` — criterion was left on CPU,
    # which is the actual cause of the "cuda:0 and cpu" crash during the
    # first training batch. losses.py now has a forward()-level device
    # guard as a safety net, but moving the module itself is the correct
    # fix and avoids a repeated CPU→GPU copy on every batch.
    criterion = criterion.to(device)
    logger.info(f"  ASL: γ+={cfg.asl_gamma_pos}, γ-={cfg.asl_gamma_neg}, "
                f"clip={cfg.asl_clip}, reduction={cfg.loss_reduction}")

    # Optional: query diversity loss
    diversity_criterion = None
    if cfg.query_diversity_weight > 0:
        diversity_criterion = QueryDiversityLoss(margin=0.1)
        logger.info(f"  Query diversity weight: {cfg.query_diversity_weight}")

    # ── Optimizer (Mod 2: differential LR) ───────────────────────────
    param_groups = model.get_param_groups()
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    logger.info(f"  LR backbone: {cfg.lr_backbone}")
    logger.info(f"  LR decoder:  {cfg.lr_decoder}")
    logger.info(f"  Weight decay: {cfg.weight_decay}")
    logger.info(f"  Grad accumulation: {cfg.grad_accum_steps} steps "
                f"(effective batch={cfg.effective_batch_size})")
    logger.info(f"  Per-class classifiers: {cfg.per_class_classifiers}")

    # ── Augmentation logging ─────────────────────────────────────────
    if cfg.use_cutmix or cfg.use_mixup:
        aug_str = []
        if cfg.use_cutmix:
            aug_str.append(f"CutMix(α={cfg.cutmix_alpha})")
        if cfg.use_mixup:
            aug_str.append(f"MixUp(α={cfg.mixup_alpha})")
        logger.info(f"  Augmentation: {' + '.join(aug_str)} @ p={cfg.augmix_prob}")
    else:
        logger.info("  Augmentation: none (CutMix/MixUp disabled)")

    # ── Scheduler ────────────────────────────────────────────────────
    scheduler = CosineWarmupScheduler(optimizer, cfg.warmup_epochs, cfg.epochs)

    # ── AMP scaler ───────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    # ── EMA ──────────────────────────────────────────────────────────
    ema = ModelEMA(model, decay=cfg.ema_decay)
    if cfg.ema_warmup_epochs > 0:
        logger.info(f"  EMA warmup: validating on raw model for first "
                    f"{cfg.ema_warmup_epochs} epochs")

    # ── Single GPU only (Phase 4 — DataParallel removed) ──────────────
    # Root-cause investigation confirmed nn.DataParallel as the primary
    # driver of the unbounded host-RSS growth that killed the kernel
    # (self_rss climbed ~2.8GB -> ~24.8GB over one epoch while GPU memory
    # stayed flat — a host-memory leak, not a GPU OOM). DataParallel is no
    # longer used anywhere in this project. CUDA_VISIBLE_DEVICES=0 (set at
    # the top of this file, before torch was imported) already makes any
    # second GPU invisible to this process; this is just an informational
    # log line confirming that.
    visible_gpus = torch.cuda.device_count()
    logger.info(f"  CUDA devices visible to this process: {visible_gpus} "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}) "
                f"— training on {device} only, no DataParallel/DDP.")

    logger.info(f"  No self-imposed session timeout — training runs until "
                f"Kaggle itself ends the session, or until cfg.epochs / "
                f"early stopping is reached, whichever comes first. "
                f"Resuming after a Kaggle-imposed stop is handled entirely "
                f"by checkpoint auto-detection below, not by any wall-clock "
                f"budget this process tracks itself.")
    logger.info(f"  Mid-epoch checkpoint interval: "
                f"{cfg.checkpoint_every_batches} optimizer steps "
                f"({'disabled' if cfg.checkpoint_every_batches <= 0 else 'enabled'}) "
                f"— this, not any timeout prediction, is what makes an "
                f"abrupt Kaggle kill resumable.")

    # ── Resume (Mod 8: robust, verified, mid-epoch-aware) ─────────────
    start_epoch = 1
    best_val_f1 = 0.0
    global_step = 0
    thresholds_computed = False

    # Self-contained checkpoint selection: don't just trust whatever path
    # the launcher happened to pass via --resume (it only ever looks for
    # last_model.pth). Independently compare EVERY candidate checkpoint
    # in this experiment's own output directory — last_model.pth (full
    # epoch) and mid_epoch.pth (Mod 8 mid-epoch save) — and resume from
    # whichever represents more completed work, by (epoch, batch_idx)
    # with batch_idx=-1 treated as "later than any mid-epoch save" since
    # a full-epoch checkpoint always supersedes a mid-epoch one from the
    # same epoch. An explicit --resume path is honored as an override
    # ONLY if it verifies successfully; otherwise we fall back to this
    # auto-detection instead of crashing.
    def _checkpoint_rank(path: Path):
        if not path.exists() or not verify_checkpoint(path):
            return None
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            epoch_n = ck.get("epoch", 0)
            batch_n = ck.get("batch_idx")
            rank = (epoch_n, -1 if batch_n is None else batch_n)
            return rank
        except Exception:
            return None

    candidates = [
        cfg.output_dir / "last_model.pth",
        cfg.output_dir / "mid_epoch.pth",
    ]
    if args.resume:
        candidates.insert(0, Path(args.resume))

    resume_path = None
    best_rank = None
    for cand in candidates:
        rank = _checkpoint_rank(cand)
        if rank is not None and (best_rank is None or rank > best_rank):
            best_rank = rank
            resume_path = cand

    if args.resume and resume_path is None:
        logger.warning(f"  ⚠ --resume path {args.resume} failed verification "
                        f"and no valid checkpoint found in {cfg.output_dir} "
                        f"— starting fresh instead of crashing.")

    if resume_path:
        logger.info(f"Resuming from {resume_path}")
        resume_model = model  # single-GPU only (Phase 4) — no DataParallel wrapper to unwrap
        ckpt = load_checkpoint(
            Path(resume_path), resume_model, optimizer, scheduler, scaler, device
        )
        is_mid_epoch = ckpt.get("batch_idx") is not None
        # BUG FIX: a mid-epoch checkpoint's "epoch" field records the epoch
        # that was IN PROGRESS (not completed) when it was saved. Resuming
        # at ckpt["epoch"] + 1 would silently skip that epoch's remaining
        # batches forever (their gradient contributions are simply lost,
        # and the LR scheduler would advance as if a full epoch happened
        # when it didn't). A full-epoch checkpoint (last_model.pth) DID
        # complete its epoch, so +1 is correct there.
        start_epoch = ckpt["epoch"] if is_mid_epoch else ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        global_step = ckpt.get("global_step", 0)
        thresholds_computed = ckpt.get("thresholds_computed", False)
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if is_mid_epoch:
            logger.info(
                f"  Resumed from a MID-epoch checkpoint (was at batch "
                f"{ckpt['batch_idx']} of epoch {ckpt['epoch']}, which did "
                f"NOT complete). Weights/optimizer/scheduler/scaler/EMA "
                f"restored exactly; epoch {start_epoch}'s data iteration "
                f"will be REDONE from batch 0 (scheduler.step() for this "
                f"epoch has not yet been called, so this is consistent — "
                f"see save_checkpoint() docstring for the full tradeoff)."
            )

        # ── Resume summary (diagnostic) ──────────────────────────────
        _sched_restored = ckpt.get("scheduler_state_dict") is not None
        _optim_restored = "optimizer_state_dict" in ckpt
        _ema_restored = "ema_state_dict" in ckpt
        _scaler_restored = ckpt.get("scaler_state_dict") is not None
        logger.info("=" * 70)
        logger.info("RESUME SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Checkpoint file       : {Path(resume_path).name}")
        logger.info(f"  Checkpoint path       : {resume_path}")
        logger.info(f"  Checkpoint type       : {'mid-epoch (partial epoch)' if is_mid_epoch else 'full-epoch'}")
        logger.info(f"  Resumed epoch         : {start_epoch}")
        if is_mid_epoch:
            logger.info(f"  Resumed batch_idx     : {ckpt.get('batch_idx')}")
        logger.info(f"  Global step           : {global_step}")
        logger.info(f"  Best val F1 so far    : {best_val_f1:.4f}")
        logger.info(f"  Scheduler last_epoch  : {scheduler.last_epoch}")
        logger.info(f"  Scheduler warmup_epochs : {scheduler.warmup_epochs}")
        logger.info(f"  Scheduler total_epochs  : {scheduler.total_epochs}")
        _cur_lrs = scheduler.get_last_lr()
        _lr_bb = _cur_lrs[0] if len(_cur_lrs) > 0 else 0
        _lr_dec = _cur_lrs[1] if len(_cur_lrs) > 1 else _cur_lrs[0]
        logger.info(f"  Current LR (backbone / decoder) : {_lr_bb:.2e} / {_lr_dec:.2e}")
        _optim_status = "✅ restored" if _optim_restored else "⚠️  NOT in checkpoint"
        _sched_status = "✅ restored" if _sched_restored else "⚠️  NOT in checkpoint (left at freshly-constructed state)"
        _ema_status = "✅ restored" if _ema_restored else "⚠️  NOT in checkpoint"
        _scaler_status = "✅ restored" if _scaler_restored else "⚠️  NOT in checkpoint"
        logger.info(f"  Model weights         : \u2705 restored (unconditional)")
        logger.info(f"  Optimizer state       : {_optim_status}")
        logger.info(f"  Scheduler state       : {_sched_status}")
        logger.info(f"  EMA state             : {_ema_status}")
        logger.info(f"  GradScaler state      : {_scaler_status}")
        logger.info("=" * 70)

    # ── Save config ──────────────────────────────────────────────────
    cfg.save()
    logger.info(f"Config saved to {cfg.output_dir / 'config.json'}")

    # ── Training loop ────────────────────────────────────────────────
    logger.info("")
    logger.info("Starting training...")
    patience_counter = 0
    peaks = PeakTracker()
    epoch_time_ema = None   # smoothed epoch time for a steadier ETA

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()

        # Update sampler epoch for balanced sampling
        if hasattr(train_loader, "sampler") and hasattr(
            train_loader.sampler, "set_epoch"
        ):
            train_loader.sampler.set_epoch(epoch)

        # ── Train ─────────────────────────────────────────────────────
        # No self-imposed timeout: this runs the full epoch unconditionally.
        # If Kaggle kills the process mid-epoch, the most recent mid-epoch
        # checkpoint (saved inside train_one_epoch) is what the next run
        # resumes from — see the resume-detection logic above.
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, cfg,
            ema, diversity_criterion,
            global_step=global_step, epoch=epoch, best_val_f1=best_val_f1,
            peaks=peaks, logger=logger, scheduler=scheduler,
        )

        # Mid-epoch checkpoint is now stale (this epoch trained to
        # completion) — remove it so a future resume doesn't mistakenly
        # prefer a now-superseded mid-epoch file over the full-epoch one.
        mid_epoch_path = cfg.output_dir / "mid_epoch.pth"
        if mid_epoch_path.exists():
            mid_epoch_path.unlink()

        # ── Validation (only every cfg.validate_every_n_epochs) ──────
        do_validate = (epoch % max(1, cfg.validate_every_n_epochs) == 0) or (epoch == cfg.epochs)
        if not do_validate:
            logger.info(f"Ep {epoch:3d}/{cfg.epochs} | loss={train_loss:.4f} | "
                        f"validation skipped this epoch (validate_every_n_epochs="
                        f"{cfg.validate_every_n_epochs})")
            scheduler.step()
            # Part 6 asks for history "after every epoch" (the printed
            # console summary is "after every VALIDATION epoch" — these
            # are different requirements). Log a lightweight train-only
            # row here so history.csv/.json never has gaps even when
            # validate_every_n_epochs > 1.
            append_epoch_history(cfg.output_dir, {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": None,
                "macro_f1": None,
                "best_macro_f1": best_val_f1,
                "checkpoint_saved": False,
                "checkpoint_path": None,
                "thresholds_computed": thresholds_computed,
                "session_remaining_hrs": None,  # no self-imposed session budget is tracked
            })
            continue

        # EMA warmup: use raw model for first N epochs, EMA after
        if cfg.ema_warmup_epochs > 0 and epoch <= cfg.ema_warmup_epochs:
            val_model = model
            val_source = "raw"
        else:
            val_model = ema.ema_model
            val_source = "ema"

        val_metrics = validate(val_model, val_loader, device, cfg, criterion=criterion)
        val_f1 = val_metrics["macro_f1"]
        val_loss = val_metrics.get("val_loss", float("nan"))

        # Step scheduler (after training, as per PyTorch convention)
        scheduler.step()

        # Get current LRs
        lrs = scheduler.get_last_lr()
        lr_bb = lrs[0] if len(lrs) > 0 else 0
        lr_dec = lrs[1] if len(lrs) > 1 else lrs[0]

        epoch_time = time.time() - epoch_start
        epoch_time_ema = epoch_time if epoch_time_ema is None else (
            0.3 * epoch_time + 0.7 * epoch_time_ema
        )
        epochs_remaining = max(0, cfg.epochs - epoch)
        eta_seconds = epoch_time_ema * epochs_remaining

        def _fmt_hms(s):
            h, rem = divmod(int(s), 3600)
            m, sec = divmod(rem, 60)
            return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

        # ── Checkpointing ────────────────────────────────────────────
        raw_model = model  # single-GPU only (Phase 4) — no DataParallel wrapper to unwrap
        saved_this_epoch = False
        saved_path = None

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(
                raw_model, optimizer, scheduler, scaler, epoch, best_val_f1,
                cfg.output_dir / "best_model.pth",
                ema_model=ema.ema_model,
                inference_only=True,  # Stage 1: smaller checkpoint
                global_step=global_step,
                thresholds_computed=thresholds_computed,
            )
            saved_this_epoch = True
            saved_path = str(cfg.output_dir / "best_model.pth")
        else:
            patience_counter += 1

        # Last model checkpoint (full, resumable, every epoch overwrite)
        save_checkpoint(
            raw_model, optimizer, scheduler, scaler, epoch, best_val_f1,
            cfg.output_dir / "last_model.pth",
            ema_model=ema.ema_model,
            inference_only=False,  # Full checkpoint for resume
            global_step=global_step,
            thresholds_computed=thresholds_computed,
        )

        # ── Diagnostics snapshot for this epoch ───────────────────────
        ram_used, ram_total = system_ram_gb()
        self_rss = process_rss_gb()
        gpu_all = gpu_stats_all()
        peaks.update("ram_used_gb", ram_used)
        peaks.update("self_rss_gb", self_rss)

        # ── Phase 6: full per-epoch summary block ─────────────────────
        short_names = ["MA", "HE", "IH", "VB/IRMA", "NV", "VH", "RD"]
        per_lesion = "  ".join(
            f"{sn}={val_metrics.get(f'{name}_f1', 0):.3f}"
            for sn, name in zip(short_names, LESION_NAMES)
        )
        val_tag = f" [{val_source}]" if cfg.ema_warmup_epochs > 0 else ""

        logger.info(
            f"Epoch {epoch}/{cfg.epochs}{val_tag}  |  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )
        logger.info(
            f"  Macro F1: {val_f1:.4f}  (best: {best_val_f1:.4f} @ epoch "
            f"{epoch if saved_this_epoch else '—'})"
        )
        logger.info(f"  {per_lesion}")
        logger.info(f"  LR: backbone={lr_bb:.1e}  decoder={lr_dec:.1e}")
        logger.info(
            f"  Epoch time: {epoch_time:.0f}s  |  Elapsed: {timer.elapsed_str()}  |  "
            f"ETA: {_fmt_hms(eta_seconds)}"
        )
        logger.info(
            f"  Checkpoint: {'saved -> ' + saved_path if saved_this_epoch else 'not saved (val F1 below best)'}"
        )
        logger.info(
            "  Thresholds: not yet optimized (runs after training completes)"
            if not thresholds_computed else
            "  Thresholds: already optimized in a prior session"
        )
        ram_str = f"{ram_used:.2f}/{ram_total:.2f}GB" if ram_used is not None else "unavailable"
        self_rss_str = f"{self_rss:.2f}GB" if self_rss is not None else "unavailable"
        gpu_all_str = " ".join(
            f"gpu{idx}={used/1024:.2f}/{total/1024:.2f}GB({util}%)"
            for idx, used, total, util in gpu_all
        ) or "unavailable"
        logger.info(
            f"  Resources: RAM={ram_str}  self_rss={self_rss_str}  "
            f"GPU=[{gpu_all_str}]  peak_ram={peaks.get('ram_used_gb'):.2f}GB  "
            f"peak_gpu={peaks.get('gpu_alloc_gb'):.2f}GB"
        )

        resume_state = (
            f"resumed@epoch{start_epoch}" if start_epoch > 1 else "fresh start"
        )
        logger.info(
            f"  Checkpoint interval: every {cfg.checkpoint_every_batches} steps  |  "
            f"Resume state: {resume_state}"
        )

        # ── Persist to CSV/JSON history (Part 6: survives interruption) ──
        append_epoch_history(cfg.output_dir, {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss == val_loss else None,  # NaN-safe
            "macro_f1": val_f1,
            "MA_f1": val_metrics.get("MA_f1"),
            "HE_f1": val_metrics.get("HE_f1"),
            "IH_f1": val_metrics.get("IH_f1"),
            "VB_IRMA_f1": val_metrics.get("VB_IRMA_f1"),
            "NV_f1": val_metrics.get("NV_f1"),
            "VH_f1": val_metrics.get("VH_f1"),
            "RD_f1": val_metrics.get("RD_f1"),
            "lr_backbone": lr_bb,
            "lr_decoder": lr_dec,
            "epoch_time_s": round(epoch_time, 1),
            "elapsed_s": round(timer.elapsed(), 1),
            "eta_s": round(eta_seconds, 1),
            "best_macro_f1": best_val_f1,
            "checkpoint_saved": saved_this_epoch,
            "checkpoint_path": saved_path,
            "thresholds_computed": thresholds_computed,
            "session_remaining_hrs": None,  # no self-imposed session budget is tracked
        })

        # Early stopping
        if patience_counter >= cfg.patience:
            logger.info(f"  Early stopping at epoch {epoch} (patience={cfg.patience})")
            break

    logger.info("")
    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")
    logger.info(f"Total training time: {timer.elapsed_str()}")

    # ── Load best model for evaluation ───────────────────────────────
    logger.info("")
    logger.info("Loading best checkpoint for threshold optimization...")
    raw_model = model  # single-GPU only (Phase 4) — no DataParallel wrapper to unwrap
    ckpt = load_checkpoint(
        cfg.output_dir / "best_model.pth", raw_model, device=device
    )
    # Use EMA weights if available
    if "ema_state_dict" in ckpt:
        raw_model.load_state_dict(ckpt["ema_state_dict"])
        logger.info("  Using EMA weights for evaluation")

    raw_model.eval()

    # ── Threshold optimization (Mod 3) ───────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("THRESHOLD OPTIMIZATION")
    logger.info("=" * 70)

    thresholds, val_metrics_05, val_metrics_opt = run_threshold_optimization(
        raw_model, val_loader, cfg, device, logger
    )
    thresholds_computed = True

    # Re-save best_model.pth's metadata with thresholds_computed=True so a
    # resume after this point (e.g. session ended during test evaluation
    # below) correctly reports threshold-optimization status without
    # needing to just infer it from thresholds.json's mere existence.
    save_checkpoint(
        raw_model, None, None, None, ckpt.get("epoch", 0), best_val_f1,
        cfg.output_dir / "best_model.pth",
        ema_model=raw_model,  # Save the same best weights used for thresholds.
        inference_only=True,
        global_step=global_step,
        thresholds_computed=thresholds_computed,
    )

    # ── Test evaluation ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)

    all_results = {}

    for modality, test_loader in test_loaders.items():
        logger.info(f"\n--- {modality.upper()} Test Set ---")

        all_probs, all_labels = _collect_predictions(
            raw_model, test_loader, device, cfg.use_amp
        )

        # Metrics at 0.5
        metrics_05 = compute_metrics(all_probs, all_labels, thresholds=None)

        # Metrics at optimized thresholds
        metrics_opt = compute_metrics(all_probs, all_labels, thresholds=thresholds)

        logger.info(f"  At 0.5:      macro_F1 = {metrics_05['macro_f1']:.4f}")
        logger.info(f"  At optimized: macro_F1 = {metrics_opt['macro_f1']:.4f}")
        logger.info(
            f"  Δ = {metrics_opt['macro_f1'] - metrics_05['macro_f1']:+.4f}"
        )

        # Per-class breakdown
        short_names = ["MA", "HE", "IH", "VB", "NV", "VH", "RD"]
        for sn, name in zip(short_names, LESION_NAMES):
            f05 = metrics_05.get(f"{name}_f1", 0)
            fopt = metrics_opt.get(f"{name}_f1", 0)
            t = thresholds.get(name, 0.5)
            logger.info(f"    {sn:4s}: F1@0.5={f05:.4f}  F1@opt={fopt:.4f}  t={t:.2f}")

        all_results[modality] = {
            "metrics_at_0.5": metrics_05,
            "metrics_at_optimized": metrics_opt,
            "thresholds": thresholds,
        }

    # Save final results
    results_path = cfg.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info(f"Total time: {timer.elapsed_str()}")
    logger.info("=" * 70)

    # ── Explicit resource release (orchestration hardening) ─────────────
    # Not required for correctness (process exit reclaims everything
    # anyway), but on a *normal* exit this lets DataLoader worker
    # processes shut down deterministically and promptly instead of
    # relying on GC timing during interpreter teardown, and frees GPU
    # memory before the process actually terminates so the parent
    # notebook's post-experiment `nvidia-smi` check reads a clean state
    # sooner rather than racing the OS's own teardown.
    try:
        del train_loader, val_loader, test_loaders
    except NameError:
        pass
    try:
        del model, optimizer, scheduler, scaler, ema, criterion
    except NameError:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # FIX: previously an unhandled exception (esp. CUDA OOM escalating to a
    # driver-level failure) could kill this process — and the parent Kaggle
    # kernel that launched it via os.system() — with zero traceback ever
    # reaching the notebook. We now catch everything at the top level,
    # write the full traceback to a dedicated crash file on disk (separate
    # from train.log, so it exists even if setup_logging() itself never
    # ran), flush stdout/stderr explicitly, and exit with a non-zero code
    # so the notebook launcher can detect and report the failure instead
    # of hanging silently.
    try:
        main()
    except BaseException:
        crash_path = Path(__file__).resolve().parent / "outputs" / "CRASH.log"
        try:
            crash_path.parent.mkdir(parents=True, exist_ok=True)
            with open(crash_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"CRASH at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n")
                traceback.print_exc(file=f)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
