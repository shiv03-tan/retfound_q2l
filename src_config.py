"""
Central configuration for Q2L lesion detection pipeline.

All hyperparameters, paths, and experiment-specific settings live here.
Backbone: RETFound (ViT-Large/16, MAE-pretrained, ~303.3M params,
1024-dim patch tokens). Migrated from ConvNeXt-V2-Base — see backbone.py
docstring for the compatibility analysis and adapter design. d_model is
unchanged at 1024, so nothing downstream of the backbone (Q2LDecoder,
losses, metrics, threshold optimization) needed to change.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import json


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

SEED = 42

LESION_NAMES = ["MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD"]
LESION_DISPLAY = ["MA", "HE", "IH", "VB/IRMA", "NV", "VH", "RD"]
NUM_CLASSES = 7

IMAGE_SIZE = 512

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


@dataclass
class Config:
    """Complete experiment configuration.

    Backbone is locked to RETFound-MEH (ViT-Large/16, MAE-pretrained):
        - Encoder output (post-adapter): (B, 1024, 32, 32)
        - d_model = 1024 (no projection needed)
    """

    # ── Experiment identity ──────────────────────────────────────────
    experiment: str = "exp_c_joint"          # a, b, c, d
    description: str = ""

    # ── Paths (relative to project root) ─────────────────────────────
    project_root: str = "."
    # Windows default preserved for your local runs; on the pod, set the
    # MMRDR_CFP_DIR / MMRDR_UWF_DIR environment variables (Create Pods ->
    # Environment Variables) to the PVC-mounted dataset paths instead of
    # editing this file per-environment.
    cfp_dir: str = field(
        default_factory=lambda: os.environ.get("MMRDR_CFP_DIR", "C:/Datasets/MMRDR/MMRDR-CFP")
    )
    uwf_dir: str = field(
        default_factory=lambda: os.environ.get("MMRDR_UWF_DIR", "C:/Datasets/MMRDR/MMRDR-UWF")
    )
    cfp_csv: str = "FP.csv"
    uwf_csv: str = "UWF_clean.csv"
    output_root: str = "outputs"

    # ── Dataset ──────────────────────────────────────────────────────
    modalities: List[str] = field(default_factory=lambda: ["cfp", "uwf"])
    val_split: float = 0.125                 # 12.5% of train for validation
    num_workers: int = 4
    pin_memory: bool = True

    # ── Backbone (LOCKED: RETFound, ViT-Large/16, MAE-pretrained) ────
    # Migrated from ConvNeXt-V2-Base. d_model stays 1024 — see
    # backbone.py's module docstring for the full compatibility
    # analysis (token sequence -> spatial-grid adapter).
    backbone_name: str = "retfound_vit_large_patch16"
    backbone_timm_name: str = "vit_large_patch16_224"
    # Kept as an explicit, separate field (not hardcoded in backbone.py) so
    # switching model family/size later — e.g. if a genuinely retina-
    # pretrained ViT-Base checkpoint becomes available — is a one-line
    # config change. As of this writing, no official "RETFound-Base"
    # exists (the official RETFound MAE and DINOv2 families are both
    # ViT-Large-only, ~300M params each — verified against the current
    # rmaphoh/RETFound release); do not set this to a Base-sized model
    # name without first confirming you have compatible, retina-pretrained
    # weights to go with it (see backbone.py module docstring / migration
    # notes for the analysis behind this).
    backbone_pretrained: bool = True         # RETFound MAE pretraining (offline weights required — see Phase 7)
    backbone_channels: int = 1024            # ViT-L embedding dim — unchanged, no projection needed
    image_size: int = 512                    # must match dataset.py's IMAGE_SIZE; drives patch grid + pos_embed interpolation
    patch_size: int = 16                     # RETFound is a patch16 ViT — do not change without a different checkpoint family
    feature_map_size: int = 32               # image_size // patch_size = 512 // 16 = 32 (was 16 under ConvNeXt-V2's /32 stride)
    freeze_blocks: List[int] = field(default_factory=lambda: [])  # Kaggle patch: fully unfrozen backbone
    # RENAMED from freeze_stages (ConvNeXt-V2's notion of frozen conv
    # stages doesn't exist in a ViT). freeze_blocks holds 0-indexed
    # transformer block indices to freeze, out of 24 total for ViT-Large.
    # Default freezes the first 18 blocks (Stage 1 of the staged
    # fine-tuning recipe — see Phase 4/9); patch_embed/cls_token/
    # pos_embed are frozen alongside block 0 automatically by
    # RETFoundBackbone._freeze_blocks(). NOT backward compatible with old
    # config.json files that still have a "freeze_stages" key —
    # Config.load() already ignores unknown keys, so old files load fine
    # but silently fall back to this new default.
    use_layer_decay: bool = False            # Phase 9: optional layer-wise LR decay (RETFound's own recipe uses 0.65). Off by default to keep the existing 2-group differential-LR behavior unchanged unless explicitly opted into.
    layer_decay: float = 0.65
    gradient_checkpointing: bool = True      # Mod 5, ported: activation checkpointing.
    # ViT-L/16 at 1024 spatial tokens (512x512 input) has a substantially
    # larger self-attention activation footprint per unfrozen block than
    # ConvNeXt-V2's convolutional stages did, and the target hardware per
    # the Kaggle launcher notebook is a 16GB T4 (not the 102GB card the
    # early ConvNeXt-V2 baseline notebook used). Unlike the ConvNeXt-V2
    # wrapper, timm's plain VisionTransformer implements
    # set_grad_checkpointing() directly — no features_only-wrapper
    # indirection — so this is now a single reliable call. Recommended ON
    # for both Stage 1 (frozen backbone, cheap either way) and especially
    # Stage 2 (partially unfrozen ViT-L, where it is required to fit on a
    # T4 — see Phase 4/9).

    # ── Single-GPU training only (DataParallel removed) ──────────────────
    # A prior version gated `nn.DataParallel` behind a `use_data_parallel`
    # flag (default True). The root-cause investigation into the
    # "tried to allocate more memory than is available" kernel death
    # CONFIRMED DataParallel as the primary driver: self-process RSS grew
    # from ~2.8GB to ~24.8GB over a single epoch while GPU memory stayed
    # completely flat (~6.7GB/~4.8GB on the two GPUs, well under budget).
    # That signature — steady host-RAM growth with flat GPU memory — is a
    # host-side leak, not a GPU OOM, and DataParallel's per-forward-call
    # `replicate()` (which re-copies all 134M parameters as new CPU/GPU
    # tensor objects on every single forward pass) combined with gradient
    # checkpointing's autograd-graph retention is the confirmed mechanism.
    # DataParallel and DistributedDataParallel are no longer used anywhere
    # in this project. train.py additionally pins CUDA_VISIBLE_DEVICES=0
    # before importing torch, so only physical GPU 0 is ever visible to
    # the training process, regardless of how many GPUs Kaggle attaches.
    # The `use_data_parallel` field has been intentionally removed (not
    # just set to False) so no code path can silently re-enable it.

    # ── Kaggle session handling — intentionally NOT self-managed ─────────
    # A prior version of this project tracked its own wall-clock budget
    # (session_limit_hrs / session_buffer_hrs) and voluntarily stopped
    # training early to exit cleanly before Kaggle's own timeout. That
    # subsystem has been REMOVED by explicit project decision: training
    # now runs unconditionally until Kaggle itself ends the session (or
    # until cfg.epochs / early stopping is reached). Surviving an abrupt,
    # unplanned kill at any point relies entirely on the mid-epoch +
    # full-epoch checkpointing below (`checkpoint_every_batches`,
    # `last_model.pth`) and train.py's automatic resume-on-relaunch logic
    # — not on this process predicting or racing the timeout itself.

    # ── Checkpoint frequency (Mod 8) ─────────────────────────────────────
    # Full-epoch checkpointing (last_model.pth, every epoch) already existed
    # and is kept. checkpoint_every_batches adds MID-epoch checkpointing so
    # a kill mid-epoch loses at most one interval's worth of optimizer
    # steps, not the whole epoch. 0 disables mid-epoch checkpointing.
    checkpoint_every_batches: int = 200

    # ── Validation frequency (Mod 9) ─────────────────────────────────────
    # Kept at 1 (validate every epoch) by default: epoch time is already
    # dominated by training, not validation (~4-8 min of a 50-120 min
    # epoch), and validating every epoch is what makes best-checkpoint
    # tracking and resume granularity precise. Exposed as a config knob
    # per the request, not changed by default.
    validate_every_n_epochs: int = 1

    # ── Q2L Decoder ──────────────────────────────────────────────────
    d_model: int = 1024                      # Mod 1: matches backbone output
    n_heads: int = 4                         # Q2L paper default
    n_encoder_layers: int = 1                # Q2L paper default
    n_decoder_layers: int = 2                # Q2L paper default
    dim_feedforward: int = 4096              # 4 × d_model
    dropout: float = 0.1
    num_labels: int = NUM_CLASSES

    # ── Modality conditioning (Exp D only, legacy CMQC) ──────────────
    use_modality_conditioning: bool = False
    num_modalities: int = 2

    # ── Adaptive lesion-query gating (J0/J1/J3/J4/J5 pilot) ──────────
    # "none" (default, byte-identical to pre-existing A/B/C/D behavior),
    # or one of: "shared" (J0), "static" (J1), "gated_modality" (J3),
    # "gated_content" (J4), "gated_full" (J5). Mutually exclusive with
    # use_modality_conditioning -- Q2LDecoder raises if both are set.
    adaptive_query_mode: str = "none"

    # ── Training ─────────────────────────────────────────────────────
    epochs: int = 75
    batch_size: int = 8  # Kaggle patch: fits T4 16GB VRAM with gradient checkpointing
    lr_backbone: float = 1e-5               # Mod 2: differential LR
    lr_decoder: float = 1e-4                 # Mod 2: differential LR
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    patience: int = 15                       # early stopping
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    ema_decay: float = 0.999  # Kaggle patch: 0.9997 too slow for small dataset

    # ── ASL loss ─────────────────────────────────────────────────────
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 6.0  # Kaggle patch: stronger negative focusing for rare classes
    asl_clip: float = 0.10  # Kaggle patch: higher clip for rare class imbalance

    # ── Augmentation (Stage 2) ───────────────────────────────────────
    use_cutmix: bool = True                  # Enable CutMix augmentation
    use_mixup: bool = True                   # Enable MixUp augmentation
    cutmix_alpha: float = 1.0                # Beta distribution alpha for CutMix
    mixup_alpha: float = 0.8                 # Beta distribution alpha for MixUp
    augmix_prob: float = 0.5                 # Probability of applying CutMix/MixUp per batch

    # ── Gradient accumulation (Stage 2) ──────────────────────────────
    grad_accum_steps: int = 2                # Accumulate N steps (effective_batch = batch_size * N)

    # ── Classifier architecture (Stage 2) ────────────────────────────
    per_class_classifiers: bool = True       # True: independent head per class; False: shared nn.Linear(d,1)

    # ── Loss configuration (Stage 3 — experimental) ──────────────────
    loss_reduction: str = "mean"             # 'mean' (safe) or 'sum_mean' (sum over classes, mean over batch)
    query_diversity_weight: float = 0.0      # Weight for query diversity regularization (0 = disabled)

    # ── Class-balanced weighting ─────────────────────────────────────
    class_balance_enabled: bool = True       # Enable effective-number class weighting on ASL
    class_balance_beta: float = 0.999        # β for effective number: 0→uniform, 0.999→moderate, 0.9999→aggressive

    # ── Cross-validation ─────────────────────────────────────────────
    use_cv: bool = False                     # Enable k-fold cross-validation
    n_folds: int = 5                         # Number of folds
    fold: int = -1                           # Current fold index (-1 = single split, 0..n_folds-1 = specific fold)

    # ── EMA configuration (Stage 3 — experimental) ───────────────────
    ema_warmup_epochs: int = 0               # Validate on raw model for first N epochs (0 = always use EMA)

    # ── Test-time augmentation (Stage 3 — experimental) ──────────────
    use_tta: bool = False                    # Enable TTA during evaluation
    tta_flips: List[str] = field(
        default_factory=lambda: ["none", "hflip", "vflip", "hflip+vflip"]
    )

    # ── Decoder capacity override (Stage 3 — experimental) ───────────
    decoder_d_model_override: int = 0        # If > 0, use as decoder hidden dim (adds projection)
    decoder_dim_ff_override: int = 0         # If > 0, override dim_feedforward in decoder
    decoder_add_positional_encoding: bool = True
    # RETFound migration ablation gate (see q2l_decoder.py's Q2LDecoder
    # docstring / __init__ note for the full reasoning). True preserves
    # exact prior behavior (load-bearing for ConvNeXt-V2, which has no
    # explicit position signal of its own). Set False to test whether
    # RETFound's own learned positional embeddings make Q2L's added fixed
    # 2D sinusoidal PE redundant-or-harmful for this task — run both and
    # compare val Macro F1 before picking a default for real experiments.

    # ── Threshold optimization (Mod 3) ───────────────────────────────
    threshold_search_range: List[float] = field(
        default_factory=lambda: [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.85, 0.90, 0.95,
        ]
    )

    # ── Modality-balanced sampling (Mod 4) ───────────────────────────
    balanced_sampling: bool = True           # True for Exp C/D

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = SEED

    # ── Derived ──────────────────────────────────────────────────────
    @property
    def output_dir(self) -> Path:
        base = Path(self.project_root) / self.output_root / self.experiment
        if self.use_cv and self.fold >= 0:
            return base / f"fold_{self.fold}"
        return base

    @property
    def n_spatial_tokens(self) -> int:
        return self.feature_map_size ** 2     # 32×32 = 1024 (was 16×16=256 under ConvNeXt-V2)

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size including gradient accumulation."""
        return self.batch_size * self.grad_accum_steps

    def save(self, path: Optional[Path] = None):
        p = path or (self.output_dir / "config.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        # dataclass → dict, convert non-serialisable types
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        with open(p, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────
# Experiment-specific config factories
# ──────────────────────────────────────────────────────────────────────

def get_config(experiment: str, project_root: str = ".") -> Config:
    """Return a Config for one of the four planned experiments."""

    if experiment == "a":
        return Config(
            experiment="exp_a_cfp",
            description="CFP-only baseline (Q2L + ASL)",
            modalities=["cfp"],
            balanced_sampling=False,
            use_modality_conditioning=False,
            project_root=project_root,
        )
    elif experiment == "b":
        return Config(
            experiment="exp_b_uwf",
            description="UWF-only baseline (Q2L + ASL)",
            modalities=["uwf"],
            balanced_sampling=False,
            use_modality_conditioning=False,
            project_root=project_root,
        )
    elif experiment == "c":
        return Config(
            experiment="exp_c_joint",
            description="Joint CFP+UWF naive merge (Q2L + ASL)",
            modalities=["cfp", "uwf"],
            balanced_sampling=True,
            use_modality_conditioning=False,
            project_root=project_root,
        )
    elif experiment == "d":
        return Config(
            experiment="exp_d_cmqc",
            description="Joint CFP+UWF with modality conditioning (Q2L + ASL + CMQC)",
            modalities=["cfp", "uwf"],
            balanced_sampling=True,
            use_modality_conditioning=True,
            project_root=project_root,
        )
    elif experiment in ("j0", "j1", "j3", "j4", "j5"):
        # Adaptive lesion-query gating pilot ladder. All run joint CFP+UWF
        # training under the SAME balanced-sampling protocol as C/D, so J0
        # (shared queries, no residual) is the matched adapter-pipeline
        # baseline for J1/J3/J4/J5 -- NOT a re-use of the plain Exp C run,
        # since C goes through the legacy label_queries path, not the
        # adapter path, and the two should not be assumed interchangeable
        # without checking (same architecture family, different code path).
        mode_map = {
            "j0": "shared", "j1": "static", "j3": "gated_modality",
            "j4": "gated_content", "j5": "gated_full",
        }
        return Config(
            experiment=f"exp_{experiment}",
            description=f"Adaptive lesion-query gating: {mode_map[experiment]} ({experiment.upper()})",
            modalities=["cfp", "uwf"],
            balanced_sampling=True,
            use_modality_conditioning=False,
            adaptive_query_mode=mode_map[experiment],
            project_root=project_root,
        )
    else:
        raise ValueError(
            f"Unknown experiment: {experiment!r}. Choose from a/b/c/d/j0/j1/j3/j4/j5."
        )
