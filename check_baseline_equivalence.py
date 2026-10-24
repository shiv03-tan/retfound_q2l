"""
Baseline-logit equivalence check: does adaptive_query_mode="shared" (J0)
produce EXACTLY the same logits as adaptive_query_mode="none" (the legacy
path used by Experiments A/B/C), given the same weights and the same input?

This is the one item that's been sitting on the "outstanding" list across
every version of the doc and deck so far. It needs no GPU, no dataset, and
no RETFound weights -- it only exercises Q2LDecoder directly with random
spatial features, so it runs anywhere torch is installed.

What it does:
  1. Build two Q2LDecoders, identical configs except adaptive_query_mode.
  2. Copy every shared submodule's weights from the legacy decoder onto the
     J0 decoder (pos_encoder, encoder, decoder, output_norm, classifiers,
     input_proj if present).
  3. Copy legacy label_queries -> J0's query_adapter.q_shared.
  4. Run both on the SAME random spatial_features + modality_ids, in eval
     mode (so dropout doesn't introduce nondeterministic noise).
  5. Assert the logits match to float precision.

Passing this does not by itself prove J1/J3/J4/J5 are implemented
correctly -- it only proves the J0 code path, with the residual mechanism
switched off, is mathematically identical to the code every existing A/B/C/D
result was produced with. That is exactly the guarantee needed before
trusting J0 as a fair "shared queries" arm of the ablation.
"""

import torch
from src.q2l_decoder import Q2LDecoder

torch.manual_seed(0)

COMMON_KW = dict(
    d_model=1024, n_heads=4, n_encoder_layers=1, n_decoder_layers=2,
    dim_feedforward=4096, dropout=0.0,  # dropout=0 so eval-mode equivalence isn't masked by different RNG paths
    num_labels=7, feature_map_size=8,   # small spatial grid -- just needs to be self-consistent, not 32x32
    use_modality_conditioning=False, num_modalities=2,
    per_class_classifiers=True, d_model_override=0, dim_feedforward_override=0,
    add_positional_encoding=True,
)

legacy = Q2LDecoder(adaptive_query_mode="none", **COMMON_KW)
j0 = Q2LDecoder(adaptive_query_mode="shared", **COMMON_KW)

legacy.eval()
j0.eval()

# --- Copy every shared submodule so the only difference is the query path ---
j0.pos_encoder.load_state_dict(legacy.pos_encoder.state_dict())
j0.encoder.load_state_dict(legacy.encoder.state_dict())
j0.decoder.load_state_dict(legacy.decoder.state_dict())
j0.output_norm.load_state_dict(legacy.output_norm.state_dict())
if legacy.per_class_classifiers:
    for i in range(len(legacy.classifiers)):
        j0.classifiers[i].load_state_dict(legacy.classifiers[i].state_dict())
else:
    j0.classifier.load_state_dict(legacy.classifier.state_dict())
if legacy.input_proj is not None:
    j0.input_proj.load_state_dict(legacy.input_proj.state_dict())

# --- Copy the shared queries themselves ---
with torch.no_grad():
    j0.query_adapter.q_shared.copy_(legacy.label_queries)

# --- Same random input for both ---
B, H, W = 4, 8, 8
spatial_features = torch.randn(B, 1024, H, W)
modality_ids = torch.randint(0, 2, (B,))

with torch.no_grad():
    logits_legacy = legacy(spatial_features, modality_ids)
    logits_j0 = j0(spatial_features, modality_ids)

max_abs_diff = (logits_legacy - logits_j0).abs().max().item()
allclose = torch.allclose(logits_legacy, logits_j0, atol=1e-6, rtol=1e-5)

print(f"logits_legacy: {logits_legacy}")
print(f"logits_j0:     {logits_j0}")
print(f"max |diff|:    {max_abs_diff:.3e}")
print(f"allclose (atol=1e-6, rtol=1e-5): {allclose}")

if allclose:
    print("\nPASS: adaptive_query_mode='shared' (J0) is numerically equivalent "
          "to the legacy path given matched weights. The J0 code path is safe "
          "to use as the ablation's shared-query baseline.")
else:
    print("\nFAIL: J0 and the legacy path diverge even with matched weights and "
          "dropout disabled. Do not trust J0 as a baseline until this is fixed "
          "-- the bug is somewhere in query_adapter's 'shared' branch or in how "
          "its output feeds the decoder, not in training itself.")
    raise SystemExit(1)
