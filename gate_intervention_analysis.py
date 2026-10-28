"""
Gate mechanistic analysis + test-time interventions for the J5 (gated_full) checkpoint.

Produces the evidence a Q1 reviewer asks for (per the ablation guidance you've
collected): does the per-lesion, image-and-modality-conditioned gate g_k(x,m)
actually do something, or is it mathematically absorbable into a static residual?

Reports, per lesion k and modality m:
  1. Gate value distribution (mean, std, min/max) across all test images
  2. Pairwise cosine correlation of gate values across lesions (are lesions'
     gates independent, or do they move together like the CMQC offsets did?)
  3. Effective adaptation magnitude: g_k(x,m) * ||delta_q_{k,m}||
  4. Test-time interventions, each re-run through evaluation to get macro-F1:
       - mean-replace: g_k(x,m) -> mean_x[g_k(x,m)]  (per lesion, per modality)
       - shuffle: g_k(x,m) permuted across images within a modality
       - force g=0  (-> J0-equivalent behaviour)
       - force g=1  (-> static-residual-equivalent behaviour, ~J1)

Decision rule (per your plan): if mean-replace or shuffle do NOT measurably
drop macro-F1 relative to the real J5 gate, the image-conditioning is not
contributing meaningfully in this checkpoint, and that's a result to report
honestly, not to paper over.

===============================================================================
THREE THINGS YOU MUST FILL IN BELOW (search for "ADAPT:") before this runs:
  1. Import of your model class + checkpoint-loading function
  2. The attribute path to the gate-producing submodule (the one that outputs
     g_k(x,m) in forward()) so this script can hook it
  3. Your existing val/test dataloader construction (reuse whatever
     evaluate_and_save_predictions.py already uses, for consistency)
===============================================================================
"""

import argparse
import json
import copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

LESIONS = ["MA", "HE", "IH", "VB/IRMA", "NV", "VH", "RD"]
MODALITIES = ["CFP", "UWF"]


# -----------------------------------------------------------------------------
# ADAPT: 1. Model + checkpoint loading
# -----------------------------------------------------------------------------
def load_model_and_dataloader(args):
    """
    Replace this with your actual imports / construction, e.g.:

        from retfound_q2l.models import RETFoundQ2L
        from retfound_q2l.data import build_dataloader
        from retfound_q2l.config import load_config

        cfg = load_config(args.config)          # configs/j5.yaml
        model = RETFoundQ2L(cfg)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["ema_state_dict"])   # you use EMA for eval
        model.eval().to(args.device)

        test_loader = build_dataloader(cfg, split="test")  # same split used
                                                             # by evaluate_and_save_predictions.py

        return model, test_loader, cfg
    """
    raise NotImplementedError(
        "ADAPT: wire up your model class, checkpoint loader, and test dataloader here. "
        "Reuse whatever evaluate_and_save_predictions.py already does so the test "
        "split/order matches the numbers already in the paper."
    )


# -----------------------------------------------------------------------------
# ADAPT: 2. Gate submodule hook
# -----------------------------------------------------------------------------
def get_gate_module(model):
    """
    Return the nn.Module whose forward() outputs g_k(x, m), shape
    (batch, num_lesions) in (0, 1), i.e. the MLP+sigmoid block in Fig. 1b.

    Likely candidates based on your Method section's naming:
        model.decoder.query_adapter.gate_mlp
        model.query_adapter.gate
        model.adapter.gate_head
    Print model to find the real attribute path:
        print(model)
    """
    raise NotImplementedError("ADAPT: return the gate-producing nn.Module here.")


def get_delta_q(model):
    """
    Return the learned residual tensor Delta q_{k,m}, shape
    (num_modalities, num_lesions, d_model) or (num_lesions, num_modalities, d_model)
    depending on how you registered it (likely an nn.Parameter on the adapter
    module, e.g. model.query_adapter.delta_q).
    """
    raise NotImplementedError("ADAPT: return the Delta q_{k,m} parameter tensor here.")


# -----------------------------------------------------------------------------
# Gate capture via forward hook (no model surgery needed for read-only analysis)
# -----------------------------------------------------------------------------
class GateCapture:
    """Hooks the gate module to record every g_k(x,m) it produces."""

    def __init__(self, gate_module):
        self.values = []        # list of (batch, num_lesions) tensors
        self.modalities = []    # list of modality labels aligned to batches
        self._handle = gate_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.values.append(output.detach().cpu())

    def remove(self):
        self._handle.remove()

    def as_array(self):
        return torch.cat(self.values, dim=0).numpy()  # (N, num_lesions)


# -----------------------------------------------------------------------------
# Intervention via monkey-patched forward (override gate at inference time)
# -----------------------------------------------------------------------------
class GateOverride:
    """
    Temporarily replaces a gate module's forward() to apply an intervention.
    Use as a context manager so the original forward is always restored,
    even if evaluation raises.
    """

    def __init__(self, gate_module, mode, fixed_value=None, mean_lookup=None,
                 shuffle_index=None):
        self.module = gate_module
        self.mode = mode                  # "mean_replace" | "shuffle" | "force"
        self.fixed_value = fixed_value     # 0.0 or 1.0 for "force"
        self.mean_lookup = mean_lookup     # (num_lesions,) tensor for "mean_replace"
        self.shuffle_index = shuffle_index  # precomputed permutation for "shuffle"
        self._orig_forward = gate_module.forward
        self._call_count = 0

    def __enter__(self):
        parent = self

        def patched_forward(*args, **kwargs):
            out = parent._orig_forward(*args, **kwargs)
            if parent.mode == "force":
                return torch.full_like(out, parent.fixed_value)
            elif parent.mode == "mean_replace":
                return parent.mean_lookup.to(out.device).expand_as(out)
            elif parent.mode == "shuffle":
                idx = parent.shuffle_index[
                    parent._call_count: parent._call_count + out.shape[0]
                ]
                parent._call_count += out.shape[0]
                return out[idx % out.shape[0]]
            return out

        self.module.forward = patched_forward
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.module.forward = self._orig_forward


# -----------------------------------------------------------------------------
# Evaluation loop (ADAPT 3: swap in your real macro-F1 / thresholding logic;
# this assumes you already have per-class validation-selected thresholds
# saved from evaluate_and_save_predictions.py, as `thresholds.json`)
# -----------------------------------------------------------------------------
def run_eval(model, test_loader, thresholds, device):
    """
    Minimal macro-F1 pass. Replace the metric computation with your existing
    utility if you have one (keeps this consistent with what's in the paper).
    """
    from sklearn.metrics import f1_score

    all_logits, all_labels, all_modalities = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            modality = batch["modality"]           # list/tensor of "CFP"/"UWF"
            labels = batch["labels"]                # (batch, num_lesions)

            logits = model(images, modality)        # ADAPT: match your model's call signature
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels)
            all_modalities.extend(
                modality if isinstance(modality, list) else modality.tolist()
            )

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))

    per_modality_f1 = {}
    for m in MODALITIES:
        mask = np.array([mm == m for mm in all_modalities])
        if mask.sum() == 0:
            continue
        th = np.array([thresholds[m][lesion] for lesion in LESIONS])
        preds = (probs[mask] >= th[None, :]).astype(int)
        f1s = [
            f1_score(labels[mask][:, k], preds[:, k], zero_division=0)
            for k in range(len(LESIONS))
        ]
        per_modality_f1[m] = {"macro": float(np.mean(f1s)),
                               "per_lesion": dict(zip(LESIONS, f1s))}
    return per_modality_f1


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/j5.yaml")
    parser.add_argument("--checkpoint", required=True, help="path to J5 EMA checkpoint")
    parser.add_argument("--thresholds", required=True,
                         help="JSON: {'CFP': {lesion: th, ...}, 'UWF': {...}} "
                              "from your J5 validation-threshold selection")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="gate_analysis_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, test_loader, cfg = load_model_and_dataloader(args)
    gate_module = get_gate_module(model)
    delta_q = get_delta_q(model)  # (num_lesions, num_modalities, d_model) — confirm your axis order

    with open(args.thresholds) as f:
        thresholds = json.load(f)

    results = {}

    # --- 1-3: capture real gate values, no intervention ---------------------
    capture = GateCapture(gate_module)
    baseline_metrics = run_eval(model, test_loader, thresholds, device)
    capture.remove()
    gate_values = capture.as_array()  # (N, num_lesions)

    results["baseline_macro_f1"] = baseline_metrics

    gate_stats = {}
    for k, lesion in enumerate(LESIONS):
        g = gate_values[:, k]
        gate_stats[lesion] = {
            "mean": float(g.mean()), "std": float(g.std()),
            "min": float(g.min()), "max": float(g.max()),
        }
    results["gate_distribution"] = gate_stats

    # pairwise lesion correlation (this is the CMQC-style collapse check,
    # applied to the gate instead of the raw offset)
    corr = np.corrcoef(gate_values.T)  # (num_lesions, num_lesions)
    results["gate_pairwise_correlation"] = corr.tolist()

    # effective adaptation magnitude g_k(x,m) * ||Delta q_{k,m}||
    # ADAPT: confirm delta_q axis order matches (lesion, modality, dim)
    delta_norms = delta_q.norm(dim=-1).detach().cpu().numpy()  # (num_lesions, num_modalities)
    eff_magnitude = {}
    for k, lesion in enumerate(LESIONS):
        eff_magnitude[lesion] = {
            m: float(gate_stats[lesion]["mean"] * delta_norms[k, mi])
            for mi, m in enumerate(MODALITIES)
        }
    results["effective_adaptation_magnitude"] = eff_magnitude

    # --- 4: interventions -----------------------------------------------------
    mean_per_lesion = torch.tensor(gate_values.mean(axis=0), dtype=torch.float32)
    rng = np.random.default_rng(42)
    shuffle_index = torch.tensor(rng.permutation(gate_values.shape[0]))

    interventions = {
        "force_g0": GateOverride(gate_module, "force", fixed_value=0.0),
        "force_g1": GateOverride(gate_module, "force", fixed_value=1.0),
        "mean_replace": GateOverride(gate_module, "mean_replace",
                                      mean_lookup=mean_per_lesion),
        "shuffle": GateOverride(gate_module, "shuffle",
                                 shuffle_index=shuffle_index),
    }

    results["interventions"] = {}
    for name, override in interventions.items():
        with override:
            metrics = run_eval(model, test_loader, thresholds, device)
        results["interventions"][name] = metrics
        print(f"[{name}] " + json.dumps(metrics, indent=2))

    # --- decision summary -----------------------------------------------------
    real_f1 = {m: baseline_metrics[m]["macro"] for m in baseline_metrics}
    summary = {}
    for m in real_f1:
        drop_mean_replace = real_f1[m] - results["interventions"]["mean_replace"][m]["macro"]
        drop_shuffle = real_f1[m] - results["interventions"]["shuffle"][m]["macro"]
        summary[m] = {
            "real_macro_f1": real_f1[m],
            "drop_if_mean_replaced": drop_mean_replace,
            "drop_if_shuffled": drop_shuffle,
            "gate_appears_meaningful": bool(drop_mean_replace > 0.005 or drop_shuffle > 0.005),
        }
    results["decision_summary"] = summary

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
