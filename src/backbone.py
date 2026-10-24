"""
RETFound (ViT-Large/16, MAE-pretrained) backbone wrapper.

Replaces the ConvNeXt-V2-Base backbone while preserving the exact output
contract the rest of the pipeline already depends on:

    forward(x) -> (B, 1024, H/16, W/16)      # a spatial feature grid

Why this shape, given RETFound is a plain ViT
------------------------------------------------
RETFound (Zhou et al., Nature 2023) is a ViT-Large/16 encoder trained with
a Masked-Autoencoder objective on retinal images. It has no convolutional
downsampling and no native (B, C, H, W) feature map — internally it
produces a token sequence:

    (B, 1 + N, 1024)   =   [CLS] token  +  N patch tokens

where N = (H / 16) * (W / 16) for a patch size of 16. For the project's
locked 512x512 input, N = 32 * 32 = 1024 patch tokens, each 1024-dim.

The adapter implemented below (drop CLS, reshape (B, N, C) -> (B, C, H, W))
turns that sequence back into the spatial-grid contract that q2l_decoder.py,
model.py's differential-LR grouping, and utils.py's checkpointing already
assume. Net effect: ZERO changes required in model.py's forward pass or in
q2l_decoder.py — Q2LDecoder still receives a (B, 1024, Hf, Wf) tensor, still
flattens it itself, and still adds its own 2D sinusoidal positional
encoding on top (redundant with RETFound's own learned positional
embeddings, but harmless — this mirrors how the original Q2L paper adds a
fixed PE on top of whatever CNN backbone feature map it is given, regardless
of what the backbone already encodes internally).

LOCKED specification (post-migration)
--------------------------------------
    Model:        RETFound (timm vit_large_patch16, MAE-pretrained)
    Patch size:   16
    Params:       ~303.3M (encoder only; the MAE reconstruction decoder in
                  the official checkpoint is pretraining-only and is never
                  loaded)
    d_model:      1024  (UNCHANGED from ConvNeXt-V2 — no projection needed,
                  Mod 1 / decoder_d_model_override stay irrelevant)
    512x512 input -> Hf = Wf = 32  ->  1024 spatial tokens (+1 CLS, dropped)

Compatibility note (pos_embed interpolation)
---------------------------------------------
The public RETFound checkpoints were pretrained at 224x224 (14x14=196
patches + CLS). Loading them at 512x512 (32x32=1024 patches) requires
interpolating the learned position embedding from a 14x14 grid to a 32x32
grid. This is the same bicubic-interpolation technique used by
DeiT/MAE/RETFound's own fine-tuning scripts (`interpolate_pos_embed`); it
is re-implemented here from scratch so this project has no dependency on
the external RETFound_MAE repository, only on `timm`.
"""

import math
from pathlib import Path
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None  # handled at runtime


class RETFoundBackbone(nn.Module):
    """RETFound (ViT-Large/16) feature extractor with a CNN-shaped output.

    Parameters
    ----------
    image_size : int
        Input resolution (assumed square). Default 512 (project spec).
    patch_size : int
        ViT patch size. RETFound is patch16 — do not change unless you are
        loading a different checkpoint family.
    pretrained : bool
        If True and weights_path is None, attempts timm's default
        pretrained loading (requires internet — will fail on Kaggle with
        internet off; use weights_path instead).
    freeze_blocks : list[int]
        Indices (0-indexed) of transformer blocks to freeze, out of 24
        total for ViT-Large. Also freezes patch_embed / cls_token /
        pos_embed whenever freeze_blocks is non-empty (there is no
        "stem" separate from the patch embedding in a ViT, unlike
        ConvNeXt, so patch_embed is tied to block-0 freezing).
        Default: freeze the first 18 of 24 blocks (Stage-1 fine-tuning —
        see Phase 4 recommendation).
    weights_path : str or None
        Path to a local RETFound `.pth` checkpoint (Kaggle offline). The
        official checkpoint format is `{'model': state_dict}`.
    gradient_checkpointing : bool
        Enable activation checkpointing. Unlike the ConvNeXt-V2 wrapper,
        timm's `VisionTransformer` implements `set_grad_checkpointing`
        directly (no `features_only=True` wrapper indirection), so this
        is now a single reliable call instead of a best-effort fallback
        chain. Strongly recommended: ViT-L/16 at 1024 tokens has a much
        larger activation footprint per unfrozen block than ConvNeXt-V2's
        conv stages, and the target hardware for this project (per the
        Kaggle launcher notebook) is a 16GB T4.
    """

    OUT_CHANNELS = 1024  # ViT-Large embedding dim — unchanged from ConvNeXt-V2 Stage-4

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        timm_model_name: str = "vit_large_patch16_224",
        pretrained: bool = True,
        freeze_blocks: Optional[List[int]] = None,
        weights_path: Optional[str] = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()

        if freeze_blocks is None:
            freeze_blocks = list(range(0, 18))  # freeze first 18 of 24 blocks

        if timm is None:
            raise ImportError(
                "timm is required for RETFound (ViT-Large/16). Install: pip install timm"
            )
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # Hf = Wf
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size "
                f"({patch_size}). Note: some ViT families here use patch14 "
                f"(e.g. DINOv2-based models) rather than patch16 — if you "
                f"switch timm_model_name to one of those, image_size must "
                f"change accordingly (e.g. 518 = 14*37), it is NOT a drop-in "
                f"swap at 512."
            )

        # `img_size=` at construction sizes patch_embed and pos_embed for
        # our target resolution directly (no post-hoc surgery needed for
        # patch_embed; pos_embed is loaded via _load_retfound_weights'
        # interpolation step below).
        self.backbone = timm.create_model(
            timm_model_name,
            img_size=image_size,
            pretrained=False,   # never let timm try to download on Kaggle
            num_classes=0,      # no classification head
            global_pool="",     # keep full token sequence, no built-in pooling
        )
        # Embedding dim is read off the constructed model rather than
        # hardcoded, so swapping timm_model_name (e.g. to a ViT-Base
        # variant, should a suitable retinal-pretrained checkpoint ever
        # become available) doesn't silently mismatch OUT_CHANNELS.
        self.OUT_CHANNELS = self.backbone.embed_dim
        self.load_report = None  # populated by _load_retfound_weights(); stays
                                  # None only if no pretrained weights were
                                  # requested at all (see else branch below)

        if weights_path is not None:
            self._load_retfound_weights(weights_path)
        elif pretrained:
            raise RuntimeError(
                "pretrained=True with weights_path=None would require an "
                "internet download via timm, which is unavailable on "
                "Kaggle (isInternetEnabled=False). Pass weights_path "
                "pointing at a local RETFound .pth checkpoint instead."
            )
        else:
            import warnings
            warnings.warn(
                "RETFoundBackbone built with pretrained=False and no "
                "weights_path — the ViT-L encoder is RANDOMLY INITIALIZED. "
                "RETFound's entire value is its pretrained weights; this "
                "should only happen deliberately (e.g. an architecture "
                "smoke test), never for a real training run.",
                RuntimeWarning,
            )

        self._freeze_blocks(freeze_blocks)

        self.gradient_checkpointing_enabled = False
        if gradient_checkpointing:
            self.gradient_checkpointing_enabled = self._enable_gradient_checkpointing()

    # ──────────────────────────────────────────────────────────────
    # Weight loading + position-embedding interpolation
    # ──────────────────────────────────────────────────────────────

    # Prefixes that a raw MAE *pretraining* checkpoint routinely contains
    # in addition to the encoder — the MAE reconstruction decoder and mask
    # token. These are NEVER expected to match anything in a
    # features_only=False, num_classes=0 timm ViT encoder (that's by
    # design — we only ever want the encoder), so seeing these listed as
    # "unexpected" checkpoint keys is normal and does NOT indicate a
    # mismatch. Anything unexpected that does NOT start with one of these
    # prefixes is unrecognized and IS flagged prominently, since that
    # would indicate a genuine architecture/version mismatch.
    _KNOWN_MAE_PRETRAIN_ONLY_PREFIXES = (
        "decoder_", "mask_token",
    )

    def _load_retfound_weights(self, weights_path: str):
        weights_path = str(weights_path)
        try:
            raw = torch.load(weights_path, map_location="cpu", weights_only=True)
        except Exception as e:
            # The official RETFound reference loading code uses plain
            # torch.load() with no weights_only argument at all, which
            # means the real checkpoint may bundle non-tensor pickled
            # objects (e.g. an argparse.Namespace of training args, or
            # optimizer state) that weights_only=True's restricted
            # unpickler will refuse. Falling back to weights_only=False
            # is safe ONLY because this path is a checkpoint YOU chose to
            # download from an explicit, trusted, named source (the
            # gated official HF repo) — never do this for an
            # arbitrary/untrusted file.
            import warnings
            warnings.warn(
                f"torch.load(weights_only=True) failed on {weights_path!r} "
                f"({type(e).__name__}: {e}); retrying with weights_only=False. "
                f"This is expected if the checkpoint bundles non-tensor "
                f"training metadata (args/optimizer/etc.) alongside the "
                f"model weights — only safe for a trusted, known source.",
                RuntimeWarning,
            )
            raw = torch.load(weights_path, map_location="cpu", weights_only=False)

        if not isinstance(raw, dict) or ("model" not in raw and "state_dict" not in raw and not all(
            hasattr(v, "shape") for v in raw.values()
        )):
            raise RuntimeError(
                f"'{weights_path}' does not look like a RETFound state-dict "
                f"checkpoint (expected a dict with a 'model' or 'state_dict' "
                f"key, or a raw tensor state dict). Got type "
                f"{type(raw).__name__} with top-level keys "
                f"{list(raw.keys())[:10] if isinstance(raw, dict) else 'N/A'}. "
                f"If this is the HuggingFace-Transformers-format export "
                f"(config.json + model.safetensors, e.g. the 'iszt/...' "
                f"community fork), that format is NOT compatible with this "
                f"loader — you need the official raw .pth from the gated "
                f"YukunZhou/RETFound_mae_meh repository instead."
            )

        state_dict = raw.get("model", raw.get("state_dict", raw))
        checkpoint_keys_original = set(state_dict.keys())
        target_sd = self.backbone.state_dict()

        # Drop any classifier head present in the checkpoint (fine-tuned
        # exports sometimes include one; MAE-pretrain-only exports don't).
        dropped_head_keys = []
        for k in ("head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"):
            if k in state_dict and (
                k not in target_sd or state_dict[k].shape != target_sd[k].shape
            ):
                state_dict.pop(k)
                dropped_head_keys.append(k)

        # Interpolate pos_embed if the pretrained grid doesn't match ours.
        pos_embed_interpolated = False
        if "pos_embed" in state_dict and state_dict["pos_embed"].shape != target_sd["pos_embed"].shape:
            state_dict["pos_embed"] = self._interpolate_pos_embed(
                state_dict["pos_embed"], target_sd["pos_embed"]
            )
            pos_embed_interpolated = True

        # Only keep keys that exist in our model with matching shapes.
        filtered = {
            k: v for k, v in state_dict.items()
            if k in target_sd and v.shape == target_sd[k].shape
        }

        load_result = self.backbone.load_state_dict(filtered, strict=False)
        # PyTorch's own IncompatibleKeys is the ground truth for what did
        # and didn't end up loaded — trust it over a hand-rolled set diff.
        missing_keys = sorted(load_result.missing_keys)
        # unexpected_keys from load_state_dict will always be empty here
        # by construction (we only ever pass keys already confirmed to be
        # in target_sd) — the checkpoint-side "keys we didn't use" instead
        # come from comparing against the ORIGINAL checkpoint state dict,
        # before our own pre-filtering:
        unexpected_keys = sorted(checkpoint_keys_original - set(filtered.keys()) - set(dropped_head_keys))
        unexpected_known = sorted(
            k for k in unexpected_keys
            if k.startswith(self._KNOWN_MAE_PRETRAIN_ONLY_PREFIXES)
        )
        unexpected_unrecognized = sorted(set(unexpected_keys) - set(unexpected_known))

        matched_count = len(filtered)
        target_count = len(target_sd)
        checkpoint_count = len(checkpoint_keys_original)

        self.load_report = {
            "checkpoint_path": weights_path,
            "checkpoint_filename": Path(weights_path).name,
            "checkpoint_total_keys": checkpoint_count,
            "target_total_keys": target_count,
            "matched": matched_count,
            "missing_keys": missing_keys,
            "unexpected_known_mae_pretrain_only": unexpected_known,
            "unexpected_unrecognized": unexpected_unrecognized,
            "dropped_head_keys": dropped_head_keys,
            "pos_embed_interpolated": pos_embed_interpolated,
            "success": len(missing_keys) == 0 and len(unexpected_unrecognized) == 0,
        }

        # ── Unmissable, printed (not just logged/warned) confirmation ────
        # print(), not just warnings.warn(), because Kaggle notebook output
        # and warning dedup/suppression behavior make warnings easy to miss
        # scrolling through a long cell's stdout.
        print("=" * 70)
        print("RETFound-MEH checkpoint load report")
        print("=" * 70)
        print(f"  Checkpoint path      : {weights_path}")
        print(f"  Checkpoint filename  : {self.load_report['checkpoint_filename']}")
        print(f"  Checkpoint keys      : {checkpoint_count}")
        print(f"  Target model keys    : {target_count}")
        print(f"  Matched (loaded)     : {matched_count}/{target_count}"
              + (" (pos_embed interpolated 14x14->32x32)" if pos_embed_interpolated else ""))
        print(f"  Missing (NOT loaded) : {len(missing_keys)}"
              + (f"  -> {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}" if missing_keys else "  (none — every backbone parameter received pretrained weights)"))
        print(f"  Unexpected, expected (MAE decoder/mask_token, safely ignored): {len(unexpected_known)}")
        print(f"  Unexpected, UNRECOGNIZED (potential real mismatch)          : {len(unexpected_unrecognized)}"
              + (f"  -> {unexpected_unrecognized[:10]}{'...' if len(unexpected_unrecognized) > 10 else ''}" if unexpected_unrecognized else ""))
        if dropped_head_keys:
            print(f"  Dropped classifier head keys (expected, harmless): {dropped_head_keys}")

        if self.load_report["success"]:
            print(f"  ✅ RETFound-MEH checkpoint loaded successfully — "
                  f"{matched_count}/{target_count} backbone parameters "
                  f"initialized from pretrained weights. Model is NOT "
                  f"training from random initialization.")
        print("=" * 70)

        # ── Fail closed on a genuine architecture mismatch ───────────────
        # missing_keys non-empty here means part of the ViT-L encoder would
        # silently start from random initialization — exactly the failure
        # mode this verification pass exists to prevent. Given
        # num_classes=0 (no head), missing_keys should always be empty for
        # a correctly-matched RETFound-MEH checkpoint; a non-empty list
        # means the wrong file, wrong timm_model_name, or a genuinely
        # incompatible checkpoint format was supplied. Raise rather than
        # warn-and-continue.
        if missing_keys:
            raise RuntimeError(
                f"RETFound checkpoint load INCOMPLETE: {len(missing_keys)} "
                f"backbone parameters have no matching pretrained weight and "
                f"would silently train from random initialization: "
                f"{missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}. "
                f"This means the checkpoint at {weights_path!r} does not "
                f"fully match the expected 'vit_large_patch16_224' "
                f"architecture — verify this is really the official "
                f"RETFound-MEH .pth (not the HF-Transformers-format fork, "
                f"not a different RETFound variant/size) before proceeding."
            )
        if unexpected_unrecognized:
            import warnings
            warnings.warn(
                f"RETFound checkpoint has {len(unexpected_unrecognized)} "
                f"unrecognized unused keys that are not the known MAE "
                f"decoder/mask_token pattern — double-check this is the "
                f"expected checkpoint: {unexpected_unrecognized[:10]}",
                RuntimeWarning,
            )

    @staticmethod
    def _interpolate_pos_embed(
        pretrained_pe: torch.Tensor, target_pe: torch.Tensor
    ) -> torch.Tensor:
        """Bicubically interpolate a (1, 1+N_old, C) pos_embed to (1, 1+N_new, C).

        Standard MAE/DeiT/RETFound fine-tuning technique for changing input
        resolution: the CLS-token position embedding is kept as-is; the
        patch-token embeddings are reshaped into their original square
        grid, bicubically resized to the new grid, and flattened back.
        """
        num_extra_tokens = 1  # CLS token
        old_n = pretrained_pe.shape[1] - num_extra_tokens
        new_n = target_pe.shape[1] - num_extra_tokens
        if old_n == new_n:
            return pretrained_pe

        old_side = int(math.sqrt(old_n))
        new_side = int(math.sqrt(new_n))
        assert old_side * old_side == old_n, "pretrained pos_embed is not a square grid"
        assert new_side * new_side == new_n, "target pos_embed is not a square grid"

        embed_dim = pretrained_pe.shape[-1]
        cls_pe = pretrained_pe[:, :num_extra_tokens]
        patch_pe = pretrained_pe[:, num_extra_tokens:]

        patch_pe = patch_pe.reshape(1, old_side, old_side, embed_dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(new_side, new_side), mode="bicubic", align_corners=False
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_side * new_side, embed_dim)

        return torch.cat([cls_pe, patch_pe], dim=1)

    # ──────────────────────────────────────────────────────────────
    # Freezing / gradient checkpointing
    # ──────────────────────────────────────────────────────────────

    def _freeze_blocks(self, block_indices: List[int]):
        """Freeze patch_embed/cls_token/pos_embed + the given transformer blocks.

        A ViT has no separate "stem" the way ConvNeXt does — patch_embed
        IS the first thing that touches pixels, so it is frozen together
        with block indices whenever any freezing is requested at all
        (freezing blocks but leaving patch_embed trainable is rarely
        useful and not a pattern RETFound's own fine-tuning recipe uses).
        """
        if not block_indices:
            return

        for name in ("cls_token", "pos_embed"):
            p = getattr(self.backbone, name, None)
            if isinstance(p, nn.Parameter):
                p.requires_grad = False
        for name, param in self.backbone.patch_embed.named_parameters():
            param.requires_grad = False

        for idx in block_indices:
            for param in self.backbone.blocks[idx].parameters():
                param.requires_grad = False

    def set_freeze_blocks(self, block_indices: List[int]):
        """Re-apply freezing after construction (Stage-1 -> Stage-2 progressive
        unfreezing). Unfreezes everything first, then re-freezes exactly
        the requested set, so this is safe to call repeatedly with a
        shrinking list across training stages.
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._freeze_blocks(block_indices)

    def _enable_gradient_checkpointing(self) -> bool:
        """timm's VisionTransformer implements set_grad_checkpointing directly
        (no features_only wrapper indirection like the ConvNeXt-V2 case),
        so this is a single reliable call rather than a best-effort chain.
        """
        set_ckpt = getattr(self.backbone, "set_grad_checkpointing", None)
        if callable(set_ckpt):
            try:
                set_ckpt(enable=True)
                return True
            except TypeError:
                set_ckpt(True)
                return True

        import warnings
        warnings.warn(
            "Gradient checkpointing was requested but this timm version's "
            "VisionTransformer does not expose set_grad_checkpointing(). "
            "On a 16GB T4 with more than a few unfrozen blocks this will "
            "likely OOM — upgrade timm or re-freeze more blocks.",
            RuntimeWarning,
        )
        return False

    # ──────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract RETFound patch-token features, reshaped to a spatial grid.

        Parameters
        ----------
        x : (B, 3, H, W)   H, W must equal self.image_size

        Returns
        -------
        features : (B, 1024, H/16, W/16)   — SAME contract as ConvNeXtV2Backbone
        """
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"RETFoundBackbone was built for {self.image_size}x{self.image_size} "
                f"input (fixed learned position embeddings after interpolation); "
                f"got {H}x{W}. Rebuild the backbone at the new resolution instead "
                f"of feeding it a different size at runtime."
            )

        tokens = self.backbone.forward_features(x)  # (B, 1 + N, 1024)
        patch_tokens = tokens[:, 1:, :]              # drop CLS -> (B, N, 1024)

        Hf = Wf = self.grid_size
        features = patch_tokens.transpose(1, 2).reshape(B, self.OUT_CHANNELS, Hf, Wf)
        return features

    def backbone_params(self) -> Iterator[nn.Parameter]:
        """All trainable backbone parameters (for differential LR)."""
        return (p for p in self.backbone.parameters() if p.requires_grad)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def named_blocks(self):
        """Yield (index, block_module) for layer-wise LR decay (Phase 9)."""
        for i, block in enumerate(self.backbone.blocks):
            yield i, block
