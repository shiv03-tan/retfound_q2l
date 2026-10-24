"""
Full Q2L lesion detection model: backbone + decoder.

Composes ConvNeXt-V2-Base backbone with Q2L Transformer decoder.
Provides get_param_groups() for differential LR (Mod 2).
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .backbone import RETFoundBackbone
from .q2l_decoder import Q2LDecoder
from .config import Config


class Q2LLesionModel(nn.Module):
    """RETFound (ViT-Large/16) + Q2L Decoder for 7-class multi-label lesion detection.

    Migrated from ConvNeXt-V2-Base — see backbone.py's module docstring
    for the compatibility analysis (RETFound's token sequence is adapted
    back into a (B, 1024, Hf, Wf) spatial grid, so Q2LDecoder below is
    byte-for-byte unchanged from the ConvNeXt-V2 version).

    Architecture
    ------------
    Input: (B, 3, 512, 512) image + optional modality_id (B,)
    RETFound ViT-L/16 → (B, 1024, 32, 32)   [~304.1M params]
    Q2L Decoder → (B, 7) logits              [~46.2M params — decoder
                                               param count depends only on
                                               d_model/n_heads/n_layers,
                                               not on feature_map_size, so
                                               it is unchanged in nature
                                               from the ConvNeXt-V2 build;
                                               the ~33.6M figure in the old
                                               docstring undercounted it]
    Total: ~350.4M params (measured; see Phase 10 validation)

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    weights_path : str or None
        Path to local backbone weights for offline loading.
    """

    def __init__(self, cfg: Config, weights_path: Optional[str] = None):
        super().__init__()
        self.cfg = cfg

        # Backbone: RETFound (ViT-Large/16, MAE-pretrained)
        self.backbone = RETFoundBackbone(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            pretrained=cfg.backbone_pretrained,
            freeze_blocks=cfg.freeze_blocks,
            weights_path=weights_path,
            gradient_checkpointing=getattr(cfg, "gradient_checkpointing", True),
        )

        # Decoder: Q2L Transformer
        self.decoder = Q2LDecoder(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_encoder_layers=cfg.n_encoder_layers,
            n_decoder_layers=cfg.n_decoder_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            num_labels=cfg.num_labels,
            feature_map_size=cfg.feature_map_size,
            use_modality_conditioning=cfg.use_modality_conditioning,
            num_modalities=cfg.num_modalities,
            per_class_classifiers=cfg.per_class_classifiers,
            d_model_override=cfg.decoder_d_model_override,
            dim_feedforward_override=cfg.decoder_dim_ff_override,
            add_positional_encoding=cfg.decoder_add_positional_encoding,
            adaptive_query_mode=getattr(cfg, "adaptive_query_mode", "none"),
        )

    def forward(
        self,
        images: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
        return_queries: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        images : (B, 3, H, W)
        modality_ids : (B,) int tensor, optional
        return_queries : bool
            If True, return (logits, decoded_queries) for diversity loss.

        Returns
        -------
        logits : (B, 7) or (logits, decoded_queries) if return_queries=True
        """
        features = self.backbone(images)          # (B, 1024, 16, 16)
        return self.decoder(features, modality_ids, return_queries=return_queries)

    def get_param_groups(self) -> List[Dict]:
        """Return parameter groups for differential LR (Mod 2).

        Default (cfg.use_layer_decay=False, UNCHANGED behavior from the
        ConvNeXt-V2 version):
            Group 1: backbone params (lr_backbone = 1e-5)
            Group 2: decoder + classifier params (lr_decoder = 1e-4)

        Optional (cfg.use_layer_decay=True, Phase 9 recommendation):
            One group per ViT block, LR = lr_backbone * layer_decay^(depth
            from output), following RETFound's own fine-tuning recipe
            (layer_decay=0.65 in the reference implementation) plus a
            separate group for patch_embed/cls_token/pos_embed (decayed
            one step further than block 0) and the decoder group at
            lr_decoder as before. This is opt-in and off by default so
            existing experiment configs (a/b/c/d) keep training exactly
            as they did with ConvNeXt-V2 unless explicitly changed.
        """
        decoder_params = list(self.decoder.parameters())

        if not getattr(self.cfg, "use_layer_decay", False):
            backbone_params = list(self.backbone.backbone_params())
            return [
                {"params": backbone_params, "lr": self.cfg.lr_backbone, "name": "backbone"},
                {"params": decoder_params, "lr": self.cfg.lr_decoder, "name": "decoder"},
            ]

        # ── Layer-wise LR decay across ViT-L's 24 blocks ─────────────
        layer_decay = getattr(self.cfg, "layer_decay", 0.65)
        n_blocks = len(self.backbone.backbone.blocks)
        groups: List[Dict] = []

        embed_params = [
            p for p in (
                list(self.backbone.backbone.patch_embed.parameters())
                + [self.backbone.backbone.cls_token, self.backbone.backbone.pos_embed]
            ) if p.requires_grad
        ]
        if embed_params:
            groups.append({
                "params": embed_params,
                "lr": self.cfg.lr_backbone * (layer_decay ** (n_blocks + 1)),
                "name": "backbone_embed",
            })

        for idx, block in self.backbone.named_blocks():
            block_params = [p for p in block.parameters() if p.requires_grad]
            if not block_params:
                continue
            # Deeper (later) blocks get less decay — block n_blocks-1 uses
            # layer_decay^1, block 0 uses layer_decay^n_blocks.
            depth_from_output = n_blocks - idx
            groups.append({
                "params": block_params,
                "lr": self.cfg.lr_backbone * (layer_decay ** depth_from_output),
                "name": f"backbone_block{idx}",
            })

        groups.append({"params": decoder_params, "lr": self.cfg.lr_decoder, "name": "decoder"})
        return groups

    def param_summary(self) -> Dict[str, int]:
        """Return parameter count summary."""
        bb_total = self.backbone.num_total_params()
        bb_train = self.backbone.num_trainable_params()
        dec_total = sum(p.numel() for p in self.decoder.parameters())
        dec_train = sum(
            p.numel() for p in self.decoder.parameters() if p.requires_grad
        )
        return {
            "backbone_total": bb_total,
            "backbone_trainable": bb_train,
            "decoder_total": dec_total,
            "decoder_trainable": dec_train,
            "total": bb_total + dec_total,
            "trainable": bb_train + dec_train,
        }
