"""
Standalone lesion-query adaptation module for the RETFound + Query2Label
gated modality-adaptation pilot.

Implements q_k(x, m) = q_k_shared + g_k(x, m) * delta_q_{k,m}

Supported modes (see MODES below):
  "shared"          J0 - shared queries only, no residual, no gate
  "static"          J1 - static lesion-specific modality residual, no gate
  "gated_modality"  J3 - per-lesion gate conditioned on modality identity only
  "gated_content"   J4 - per-lesion gate conditioned on image content only
  "gated_full"      J5 - per-lesion gate conditioned on image content AND modality

Each mode with a gate builds a gate MLP that ONLY consumes the branches it is
allowed to see (the unused branch's Linear layer is not instantiated at all,
so no information can leak through a shared bias term).

Design choices, as specified:
  - z(x) is LayerNorm(mean-pooled RETFound patch tokens).
  - delta_q is initialized small-but-nonzero (not exactly zero) so the gate
    receives a nonzero upstream gradient (dq_k/dg_k = delta_q_{k,m}) from the
    first backward pass.
  - Gate output-layer weights are initialized small and its bias near 0, so
    gates start near sigmoid(0) = 0.5 regardless of the (small, symmetric)
    upstream activations -- this is a *design choice*, not a guarantee that
    initial gate logits will land exactly at 0.5 for every lesion; the smoke
    test below checks the actual resulting distribution rather than assuming it.
  - Lesion embeddings are small, independent random inits (breaks symmetry
    without pre-assigning differential adaptation strength to any lesion).

Passing the smoke tests below establishes implementation correctness only:
correct shapes, correct modality indexing, finite/nonzero gradients where
expected, a non-degenerate initial gate distribution, and exact recovery of
shared-query behavior when adaptation is disabled. It says nothing about
whether the mechanism is useful or novel -- that is a separate, later
question to be settled by the J0/J1/J5 pilot and the literature comparison.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("shared", "static", "gated_modality", "gated_content", "gated_full")


class LesionQueryAdapter(nn.Module):
    def __init__(
        self,
        num_lesions: int = 7,
        num_modalities: int = 2,
        d_model: int = 1024,
        mode: str = "gated_full",
        emb_dim: int = 64,
        gate_hidden_dim: int = 128,
        residual_init_scale: float = 0.002,
        query_init_scale: float = 0.02,
        emb_init_scale: float = 0.02,
        gate_out_init_scale: float = 0.02,
        gate_bias_init: float = 0.0,
    ):
        # NOTE on gate_out_init_scale: at 1e-3 (tried first), every gate
        # measured ~0.5000 with std ~8e-6 across a batch of 64 -- not exactly
        # zero, so this is not numerical underflow in the literal sense
        # (values did not hit float precision limits). What happened is
        # ordinary signal attenuation through a composition of small-variance
        # linear maps: emb_init_scale (0.02) -> gate_mlp_in (default init) ->
        # GELU -> gate_mlp_out weights (1e-3). Each stage shrinks the signal
        # further, and the product was small enough that the resulting gate
        # variation had no practical effect (a gate that is always ~0.5 up to
        # 1e-5 behaves, for any realistic training dynamics, as a constant).
        # 0.02 was chosen empirically (see the scale sweep in the accompanying
        # notes) to keep gates centered near 0.5 while producing measurable,
        # non-saturated variation. There is no principled "correct" scale here
        # -- 0.02 is a design choice with a stated goal (visible sensitivity
        # without saturation), and the smoke test's threshold is calibrated to
        # that goal, not derived from a correctness requirement. Retune both
        # together if the goal changes (e.g. if you want gates closer to
        # deterministic 0.5 at init for a more conservative start).
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

        self.mode = mode
        self.num_lesions = num_lesions
        self.num_modalities = num_modalities
        self.d_model = d_model

        self.use_content = mode in ("gated_content", "gated_full")
        self.use_modality_in_gate = mode in ("gated_modality", "gated_full")
        self.is_gated = mode.startswith("gated")
        self.has_residual = mode != "shared"

        # --- shared queries: (K, D) ---
        self.q_shared = nn.Parameter(torch.empty(num_lesions, d_model))
        nn.init.normal_(self.q_shared, mean=0.0, std=query_init_scale)

        # --- modality-specific residuals: (K, M, D) ---
        # Deliberately NOT zero-initialized: dq_k/dg_k = delta_q_{k,m}, so a
        # zero init would silence the gate's classification-loss gradient
        # via this pathway until the residual itself moved off zero.
        if self.has_residual:
            self.delta_q = nn.Parameter(torch.empty(num_lesions, num_modalities, d_model))
            nn.init.normal_(self.delta_q, mean=0.0, std=residual_init_scale)
            # residual_init_scale (0.002) is deliberately ~10x smaller than
            # query_init_scale (0.02), not matched to it. For an iid N(0, s^2)
            # vector in D dims, expected norm ~= s * sqrt(D); with s equal for
            # both, ||delta_q|| ~= ||q_shared||, so in static mode (full
            # residual, no gate) the perturbation would start at ~100% of the
            # shared-query norm, and ~50% in gated mode at gate~0.5 -- i.e. not
            # a small residual at all, effectively a second, independent query
            # at init. At (0.02, 0.002) the ratio is ~10% (static) / ~5%
            # (gated, gate~0.5). See effective_init_ratio() below to check
            # this against your actual d_model/init scales rather than
            # trusting these numbers blindly.
        else:
            self.register_parameter("delta_q", None)

        # --- lesion identity embedding (always present when gated: the gate
        # is per-lesion, so it must be able to tell lesions apart) ---
        if self.is_gated:
            self.lesion_emb = nn.Embedding(num_lesions, emb_dim)
            nn.init.normal_(self.lesion_emb.weight, mean=0.0, std=emb_init_scale)

            gate_in_dim = emb_dim
            if self.use_content:
                self.content_proj = nn.Linear(d_model, emb_dim, bias=False)
                gate_in_dim += emb_dim
            else:
                self.content_proj = None

            if self.use_modality_in_gate:
                self.modality_emb = nn.Embedding(num_modalities, emb_dim)
                nn.init.normal_(self.modality_emb.weight, mean=0.0, std=emb_init_scale)
                gate_in_dim += emb_dim
            else:
                self.modality_emb = None

            self.gate_mlp_in = nn.Linear(gate_in_dim, gate_hidden_dim)
            self.gate_mlp_out = nn.Linear(gate_hidden_dim, 1)
            # Small output weights + near-zero bias -> gates start close to
            # 0.5 and largely insensitive to (small, random) upstream inputs
            # at init, so both the gate and the residual receive comparable
            # early gradient signal.
            nn.init.normal_(self.gate_mlp_out.weight, mean=0.0, std=gate_out_init_scale)
            nn.init.constant_(self.gate_mlp_out.bias, gate_bias_init)
        else:
            self.lesion_emb = None
            self.content_proj = None
            self.modality_emb = None
            self.gate_mlp_in = None
            self.gate_mlp_out = None

        self.content_norm = nn.LayerNorm(d_model) if self.use_content else None

    def pool_content(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (B, N, D) -> z(x): (B, D), LayerNorm(mean over N)."""
        if patch_tokens.dim() != 3 or patch_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"patch_tokens must be (B, N, {self.d_model}), got {tuple(patch_tokens.shape)}"
            )
        pooled = patch_tokens.mean(dim=1)
        return self.content_norm(pooled)

    @torch.no_grad()
    def effective_init_ratio(self, gate_value: float = 0.5) -> float:
        """
        Diagnostic, not a correctness check by itself: ratio of the expected
        effective-residual norm (gate_value * delta_q) to the expected
        q_shared norm at CURRENT parameter values. Call this after
        construction (or after loading real weights) to sanity-check that
        the initial perturbation size matches what you intend, rather than
        assuming the default init scales are right for your d_model/K/M.
        Returns NaN for mode == "shared" (no residual exists).
        """
        if not self.has_residual:
            return float("nan")
        q_norm = self.q_shared.norm(dim=-1).mean().item()
        delta_norm = self.delta_q.norm(dim=-1).mean().item()
        g = 1.0 if self.mode == "static" else gate_value
        return (g * delta_norm) / q_norm

    def _validate_forward_inputs(self, modality_idx: torch.Tensor, force_gate: Optional[float]):
        if modality_idx.dim() != 1:
            raise ValueError(f"modality_idx must be 1-D (B,), got shape {tuple(modality_idx.shape)}")
        if modality_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"modality_idx must be an integer tensor, got dtype {modality_idx.dtype}")
        if modality_idx.numel() > 0:
            lo, hi = int(modality_idx.min()), int(modality_idx.max())
            if lo < 0 or hi >= self.num_modalities:
                raise ValueError(
                    f"modality_idx values must be in [0, {self.num_modalities}), "
                    f"got range [{lo}, {hi}]. A negative index will silently "
                    f"select the wrong residual via Python's negative-indexing "
                    f"semantics rather than raising -- this check exists "
                    f"specifically to catch that case before it happens."
                )
        if force_gate is not None and not (0.0 <= force_gate <= 1.0):
            raise ValueError(f"force_gate must be in [0, 1], got {force_gate}")

    def forward(
        self,
        modality_idx: torch.Tensor,
        patch_tokens: Optional[torch.Tensor] = None,
        pooled_content: Optional[torch.Tensor] = None,
        force_gate: Optional[float] = None,
    ):
        """
        modality_idx: (B,) long tensor of modality indices in [0, num_modalities)
        patch_tokens: (B, N, D) raw RETFound patch tokens (used if pooled_content is None)
        pooled_content: (B, D) precomputed z(x); takes precedence over patch_tokens
        force_gate: if set (e.g. 0.0), overrides the computed gate everywhere -
                    used by the smoke test to check exact shared-query recovery.

        Returns a dict:
          queries:       (B, K, D) adapted queries q_k(x, m)
          gates:         (B, K) in [0,1], or None for "shared"/"static"
          delta_selected:(B, K, D) the modality-indexed residual before gating,
                         or None for "shared"
          effective_residual_norm: (B, K) L2 norm of gate * delta_selected,
                         or (K,) static norm for "static", or None for "shared"
        """
        self._validate_forward_inputs(modality_idx, force_gate)

        B = modality_idx.shape[0]
        K, D = self.num_lesions, self.d_model
        device = modality_idx.device

        q_shared = self.q_shared.unsqueeze(0).expand(B, K, D)

        if self.mode == "shared":
            return {
                "queries": q_shared,
                "gates": None,
                "delta_selected": None,
                "effective_residual_norm": None,
            }

        # index the modality-specific residual: delta_q (K, M, D), modality_idx (B,)
        # -> (K, B, D) -> (B, K, D)
        delta_selected = self.delta_q[:, modality_idx, :].permute(1, 0, 2)

        if self.mode == "static":
            queries = q_shared + delta_selected
            static_norm = delta_selected.norm(dim=-1)  # (B, K) - constant across images per (lesion, modality)
            return {
                "queries": queries,
                "gates": None,
                "delta_selected": delta_selected,
                "effective_residual_norm": static_norm,
            }

        # --- gated modes ---
        lesion_ids = torch.arange(K, device=device)
        lesion_feat = self.lesion_emb(lesion_ids).unsqueeze(0).expand(B, K, -1)  # (B,K,E)

        parts = [lesion_feat]

        if self.use_content:
            if pooled_content is None:
                if patch_tokens is None:
                    raise ValueError("gated_content/gated_full modes require patch_tokens or pooled_content")
                pooled_content = self.pool_content(patch_tokens)
            content_feat = self.content_proj(pooled_content).unsqueeze(1).expand(B, K, -1)  # (B,K,E)
            parts.append(content_feat)

        if self.use_modality_in_gate:
            modality_feat = self.modality_emb(modality_idx).unsqueeze(1).expand(B, K, -1)  # (B,K,E)
            parts.append(modality_feat)

        gate_in = torch.cat(parts, dim=-1)  # (B, K, gate_in_dim)
        h = F.gelu(self.gate_mlp_in(gate_in))
        logits = self.gate_mlp_out(h).squeeze(-1)  # (B, K)
        gates = torch.sigmoid(logits)

        if force_gate is not None:
            gates = torch.full_like(gates, force_gate)

        queries = q_shared + gates.unsqueeze(-1) * delta_selected
        effective_residual = gates.unsqueeze(-1) * delta_selected
        effective_norm = effective_residual.norm(dim=-1)  # (B, K)

        return {
            "queries": queries,
            "gates": gates,
            "delta_selected": delta_selected,
            "effective_residual_norm": effective_norm,
        }
