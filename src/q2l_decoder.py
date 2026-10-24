"""
Q2L Transformer Decoder for multi-label classification.

Architecture follows the Q2L paper (Liu et al., 2021):
    - 1 Transformer encoder layer  (global context on spatial KV tokens)
    - 2 Transformer decoder layers (label queries cross-attend to spatial features)
    - 4 attention heads
    - d_model = 1024 (matches RETFound ViT-Large/16 token dim; also matched
      ConvNeXt-V2-Base's Stage-4 output before the backbone migration —
      no change was needed here for that migration, by design)

Label queries: 7 learnable embeddings, one per lesion class.
Positional encoding: 2D sinusoidal (following DETR).
Optional modality conditioning: additive offset on queries (Exp D).

Stage 2 additions:
    - Per-class classifier heads (independent nn.Linear per label query)
    - Configurable decoder capacity (d_model override with projection)
    - Return decoded queries for diversity regularization
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .lesion_query_adapter import LesionQueryAdapter, MODES as ADAPTER_MODES


# ──────────────────────────────────────────────────────────────────────
# 2D Sinusoidal Positional Encoding (following DETR)
# ──────────────────────────────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """Fixed 2D sine-cosine positional encoding for spatial feature maps.

    For a feature map of size (H, W), generates a (H*W, d_model) encoding
    where the first d_model//2 dimensions encode the y-position and the
    last d_model//2 dimensions encode the x-position.
    """

    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64,
                 temperature: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"

        self.d_model = d_model
        pe = self._build_encoding(d_model, max_h, max_w, temperature)
        self.register_buffer("pe", pe)  # (max_h, max_w, d_model)

    @staticmethod
    def _build_encoding(
        d_model: int, max_h: int, max_w: int, temperature: float
    ) -> torch.Tensor:
        half_d = d_model // 2
        dim_t = torch.arange(half_d, dtype=torch.float32)
        # Each pair of dimensions uses different frequency
        dim_t = temperature ** (2 * (dim_t // 2) / half_d)

        # Create position grids
        pos_h = torch.arange(max_h, dtype=torch.float32).unsqueeze(1)  # (H, 1)
        pos_w = torch.arange(max_w, dtype=torch.float32).unsqueeze(1)  # (W, 1)

        # Encode positions
        pe_h = torch.zeros(max_h, half_d)
        pe_h[:, 0::2] = torch.sin(pos_h / dim_t[0::2])
        pe_h[:, 1::2] = torch.cos(pos_h / dim_t[1::2])

        pe_w = torch.zeros(max_w, half_d)
        pe_w[:, 0::2] = torch.sin(pos_w / dim_t[0::2])
        pe_w[:, 1::2] = torch.cos(pos_w / dim_t[1::2])

        # Combine: (H, W, d_model) = [pe_h repeated over W, pe_w repeated over H]
        pe_h = pe_h.unsqueeze(1).expand(-1, max_w, -1)  # (H, W, half_d)
        pe_w = pe_w.unsqueeze(0).expand(max_h, -1, -1)  # (H, W, half_d)

        pe = torch.cat([pe_h, pe_w], dim=-1)  # (H, W, d_model)
        return pe

    def forward(self, h: int, w: int) -> torch.Tensor:
        """Return positional encoding for feature map of size (h, w).

        Returns
        -------
        pe : (h*w, d_model)
        """
        return self.pe[:h, :w].reshape(h * w, self.d_model)


# ──────────────────────────────────────────────────────────────────────
# Q2L Decoder
# ──────────────────────────────────────────────────────────────────────

class Q2LDecoder(nn.Module):
    """Query-to-Label Transformer Decoder.

    Parameters
    ----------
    d_model : int
        Input dimension from backbone. Default 1024 (RETFound ViT-L/16;
        unchanged from the earlier ConvNeXt-V2-Base backbone).
    n_heads : int
        Number of attention heads. Default 4 (Q2L paper).
    n_encoder_layers : int
        Number of encoder layers for global context. Default 1.
    n_decoder_layers : int
        Number of decoder layers. Default 2 (Q2L paper).
    dim_feedforward : int
        FFN intermediate dimension. Default 4096 (4 × d_model).
    dropout : float
        Dropout rate. Default 0.1.
    num_labels : int
        Number of label classes. Default 7.
    feature_map_size : int
        Spatial size of backbone feature map (assumed square). Default 16.
    use_modality_conditioning : bool
        If True, add modality offset to queries (Exp D). Default False.
    num_modalities : int
        Number of modality types. Default 2 (CFP, UWF).
    per_class_classifiers : bool
        If True, use independent classifier head per label. Default True.
    d_model_override : int
        If > 0 and != d_model, use this as internal decoder dimension
        with a learned projection from d_model. Default 0 (disabled).
    dim_feedforward_override : int
        If > 0, override dim_feedforward. Default 0 (use dim_feedforward).
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 4,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 2,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        num_labels: int = 7,
        feature_map_size: int = 16,
        use_modality_conditioning: bool = False,
        num_modalities: int = 2,
        per_class_classifiers: bool = True,
        d_model_override: int = 0,
        dim_feedforward_override: int = 0,
        add_positional_encoding: bool = True,
        adaptive_query_mode: str = "none",
    ):
        super().__init__()

        self.input_d_model = d_model  # backbone output dimension
        self.num_labels = num_labels
        self.feature_map_size = feature_map_size
        self.use_modality_conditioning = use_modality_conditioning
        self.per_class_classifiers = per_class_classifiers
        self.add_positional_encoding = add_positional_encoding
        self.adaptive_query_mode = adaptive_query_mode

        if adaptive_query_mode != "none" and adaptive_query_mode not in ADAPTER_MODES:
            raise ValueError(
                f"adaptive_query_mode must be 'none' or one of {ADAPTER_MODES}, "
                f"got {adaptive_query_mode!r}"
            )
        if adaptive_query_mode != "none" and use_modality_conditioning:
            raise ValueError(
                "adaptive_query_mode and use_modality_conditioning (legacy CMQC) "
                "are mutually exclusive -- applying both would add the old "
                "uniform modality offset AND the new per-lesion residual to the "
                "same queries. Set use_modality_conditioning=False when using "
                "adaptive_query_mode, or adaptive_query_mode='none' to run "
                "legacy CMQC (Exp D) as before."
            )
        # RETFound migration note (see backbone.py / config.py docstrings):
        # RETFound's own ViT already has learned, content-entangled
        # positional embeddings baked into every token by the time they
        # reach this decoder — unlike ConvNeXt-V2's feature map, which
        # carried no explicit position signal at all, so Q2L's fixed 2D
        # sinusoidal PE below was originally load-bearing. Whether adding
        # a second, independent PE on top of RETFound's is helpful,
        # neutral, or actively harmful was flagged in the architectural
        # review as genuinely undetermined without an empirical test —
        # hence this is a config-gated ablation (cfg.decoder_add_positional_encoding),
        # not a silent removal. Default True preserves exact prior
        # behavior for ConvNeXt-V2-backed experiments/configs.

        # Determine effective internal dimension
        if d_model_override > 0 and d_model_override != d_model:
            self.internal_d = d_model_override
            # Projection from backbone dim → internal dim
            self.input_proj = nn.Sequential(
                nn.Linear(d_model, d_model_override),
                nn.LayerNorm(d_model_override),
            )
        else:
            self.internal_d = d_model
            self.input_proj = None

        # Effective feedforward dimension
        eff_dim_ff = dim_feedforward_override if dim_feedforward_override > 0 else dim_feedforward

        # Positional encoding for spatial features (uses internal dim)
        self.pos_encoder = PositionalEncoding2D(self.internal_d, max_h=64, max_w=64)

        if adaptive_query_mode != "none":
            # Single source of truth for the shared queries lives inside the
            # adapter (its q_shared parameter) -- deliberately NOT also
            # creating self.label_queries here, since that would leave an
            # unused duplicate (1024, 1024) parameter block sitting in the
            # model that receives no gradient and would silently bloat
            # checkpoints / param counts without doing anything.
            self.label_queries = None
            self.modality_offset = None
            self.query_adapter = LesionQueryAdapter(
                num_lesions=num_labels,
                num_modalities=num_modalities,
                d_model=self.internal_d,
                mode=adaptive_query_mode,
            )
        else:
            # Legacy path (Exp A/B/C, and Exp D via use_modality_conditioning
            # below) -- byte-for-byte unchanged from the original.
            self.label_queries = nn.Parameter(
                torch.randn(num_labels, self.internal_d) * 0.02  # small init
            )
            self.query_adapter = None

            # Optional: modality conditioning (Exp D)
            if use_modality_conditioning:
                self.modality_offset = nn.Embedding(num_modalities, self.internal_d)
                # Initialize near zero so it starts as a small perturbation
                nn.init.normal_(self.modality_offset.weight, mean=0.0, std=0.01)
            else:
                self.modality_offset = None

        # Transformer encoder (global context fusion on spatial tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.internal_d,
            nhead=n_heads,
            dim_feedforward=eff_dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )

        # Transformer decoder (label queries attend to spatial features)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.internal_d,
            nhead=n_heads,
            dim_feedforward=eff_dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_decoder_layers
        )

        # Final layer norm
        self.output_norm = nn.LayerNorm(self.internal_d)

        # Classifier head(s)
        if per_class_classifiers:
            # Independent classifier per label — allows per-class decision boundaries
            self.classifiers = nn.ModuleList([
                nn.Linear(self.internal_d, 1) for _ in range(num_labels)
            ])
        else:
            # Shared classifier — all labels use same projection
            self.classifier = nn.Linear(self.internal_d, 1)

    def forward(
        self,
        spatial_features: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
        return_queries: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        spatial_features : (B, C, H, W)
            Backbone Stage-4 output. C must equal input d_model.
        modality_ids : (B,) int tensor, optional
            0 = CFP, 1 = UWF. Required if use_modality_conditioning=True.
        return_queries : bool
            If True, return (logits, decoded_queries) for diversity loss.

        Returns
        -------
        logits : (B, num_labels)
            Raw logits for each class.
        decoded_queries : (B, num_labels, internal_d), optional
            Only returned when return_queries=True.
        """
        B, C, H, W = spatial_features.shape
        assert C == self.input_d_model, (
            f"Feature channels ({C}) must match input d_model ({self.input_d_model})"
        )

        # (B, C, H, W) → (B, H*W, C) = (B, 256, 1024)
        kv = spatial_features.flatten(2).transpose(1, 2)

        # Optional: project to internal dimension
        if self.input_proj is not None:
            kv = self.input_proj(kv)  # (B, H*W, internal_d)

        # Adapter content pooling happens on kv BEFORE the fixed 2D
        # positional encoding is added -- the gate should condition on image
        # content, not on the arbitrary sinusoidal position vectors, which
        # carry no content information and would only add noise to g_k(x,m).
        adapter_content = kv if self.query_adapter is not None else None

        # Add 2D positional encoding (RETFound ablation gate — see __init__ note)
        if self.add_positional_encoding:
            pos = self.pos_encoder(H, W).to(kv.device)  # (H*W, internal_d)
            kv = kv + pos.unsqueeze(0)

        # Encoder: global context fusion on spatial tokens
        memory = self.encoder(kv)  # (B, 256, internal_d)

        if self.query_adapter is not None:
            if modality_ids is None:
                raise ValueError(
                    "adaptive_query_mode is set but modality_ids was None -- "
                    "every adapter mode (including 'shared'/J0) requires a "
                    "modality_ids tensor, even when a given mode doesn't use "
                    "it internally, so J0/J1/... stay drop-in compatible."
                )
            adapter_out = self.query_adapter(modality_ids, patch_tokens=adapter_content)
            queries = adapter_out["queries"]  # (B, 7, internal_d)
        else:
            # Build label queries: (B, 7, internal_d)
            queries = self.label_queries.unsqueeze(0).expand(B, -1, -1)

            # Optional modality conditioning (additive offset) -- legacy CMQC
            if self.use_modality_conditioning and modality_ids is not None:
                mod_offset = self.modality_offset(modality_ids)  # (B, internal_d)
                mod_offset = mod_offset.unsqueeze(1)  # (B, 1, internal_d)
                queries = queries + mod_offset  # broadcast to all 7 queries

        # Decoder: label queries cross-attend to spatial memory
        decoded = self.decoder(queries, memory)  # (B, 7, internal_d)

        # Normalize
        decoded = self.output_norm(decoded)  # (B, 7, internal_d)

        # Classify
        if self.per_class_classifiers:
            logits = torch.cat([
                self.classifiers[i](decoded[:, i:i+1, :])
                for i in range(self.num_labels)
            ], dim=1).squeeze(-1)  # (B, num_labels)
        else:
            logits = self.classifier(decoded).squeeze(-1)  # (B, 7)

        if return_queries:
            return logits, decoded
        return logits
