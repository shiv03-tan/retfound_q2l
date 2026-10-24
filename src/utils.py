"""
Utility functions: reproducibility, logging, checkpointing, EMA.
"""

import copy
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """Seed each DataLoader worker for reproducibility."""
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────

def setup_logging(exp_dir: Path, name: str = "train") -> logging.Logger:
    """Set up logger that writes to both console and exp_dir/train.log."""
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # FIX: without this, messages also propagate to the root logger. If
    # anything else in the process (timm, matplotlib, etc.) has already
    # attached a handler to root, every logger.info() call gets printed
    # an extra time — this is why the Kaggle log showed each setup block
    # (config, dataloaders, model summary) repeated 2-3x in a row.
    logger.propagate = False

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(exp_dir / "train.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_f1: float,
    path: Path,
    ema_model: Optional[nn.Module] = None,
    config_dict: Optional[Dict] = None,
    inference_only: bool = False,
    global_step: int = 0,
    batch_idx: Optional[int] = None,
    thresholds_computed: bool = False,
    save_rng_state: bool = True,
):
    """Save training checkpoint.

    Parameters
    ----------
    inference_only : bool
        If True, save only model weights (and EMA weights), omitting
        optimizer/scheduler/scaler state. Reduces checkpoint size from
        ~2.17 GB to ~0.54 GB. Used for best_model.pth.
    global_step : int
        Cumulative optimizer.step() count across the whole run (Mod 8).
        Not used to resume optimizer/scheduler position (that's fully
        captured by their own state_dicts) — recorded so downstream
        tooling (ETA, throughput diagnostics) has a stable counter that
        survives resume without recomputation.
    batch_idx : int, optional
        Position within the epoch at save time, for MID-epoch checkpoints
        (Mod 8, checkpoint_every_batches). None for full-epoch checkpoints.
        NOTE on resume semantics: PyTorch's DataLoader/sampler iteration
        state is not itself resumable, so a mid-epoch checkpoint's weights
        and optimizer/scheduler/scaler/EMA state ARE restored exactly, but
        the epoch's data iteration restarts from batch 0 on resume rather
        than seeking to this exact batch_idx. This is a deliberate,
        documented tradeoff: it avoids the much higher-risk work of
        building custom resumable samplers, at the cost of some batches
        within the interrupted epoch being seen twice. No gradient
        updates or optimizer/scheduler state are lost either way.
    thresholds_computed : bool
        Whether run_threshold_optimization() has already produced
        thresholds.json for this experiment (Mod 2 / Part 2 requirement).
    save_rng_state : bool
        Save torch/numpy/python RNG state for closer (not guaranteed
        bit-exact, since DataLoader workers have their own RNG streams)
        reproducibility across a resume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_f1": best_val_f1,
        "global_step": global_step,
        "batch_idx": batch_idx,
        "thresholds_computed": thresholds_computed,
    }

    if not inference_only:
        # Include full training state for resumable checkpoints
        state["optimizer_state_dict"] = optimizer.state_dict()
        state["scheduler_state_dict"] = scheduler.state_dict() if scheduler else None
        state["scaler_state_dict"] = scaler.state_dict() if scaler else None
        if save_rng_state:
            state["rng_state"] = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }

    if ema_model is not None:
        state["ema_state_dict"] = ema_model.state_dict()
    if config_dict is not None:
        state["config"] = config_dict

    # Atomic write: save to a temp path then rename, so a kill mid-`torch.save`
    # (e.g. the exact SIGKILL scenario from the prior investigation) can
    # never leave a corrupt/truncated checkpoint at the real path.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    device: torch.device = None,
    restore_rng_state: bool = True,
) -> Dict:
    """Load checkpoint, return metadata dict.

    Handles both full and inference-only checkpoints gracefully, and is
    backward-compatible with checkpoints saved before Mod 8 (missing keys
    default sensibly via .get()).
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if restore_rng_state and ckpt.get("rng_state"):
        try:
            rs = ckpt["rng_state"]
            torch.set_rng_state(rs["torch"])
            if rs.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs["cuda"])
            np.random.set_state(rs["numpy"])
            random.setstate(rs["python"])
        except Exception:
            pass  # Never fail a resume over best-effort RNG restoration.
    return ckpt


def verify_checkpoint(path: Path) -> bool:
    """Sanity-check a checkpoint before trusting it for resume (Part 2).

    Confirms the file loads, is a dict, and contains the minimum keys a
    resumable checkpoint must have. Used by the launcher/train.py before
    committing to a resume path, so a truncated/corrupt file (e.g. from a
    kill during a non-atomic save prior to this fix) falls back to
    starting fresh instead of crashing the run.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return (
            isinstance(ckpt, dict)
            and "model_state_dict" in ckpt
            and "epoch" in ckpt
        )
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Exponential Moving Average (EMA)
# ──────────────────────────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model with smoothed weights.
    Used for evaluation only — the EMA model often generalizes better.

    Parameters
    ----------
    model : nn.Module
        Source model to track.
    decay : float
        EMA decay factor. Default 0.9997 (Q2L paper).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9997):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights after each training step."""
        for ema_p, model_p in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


# ──────────────────────────────────────────────────────────────────────
# Resource diagnostics (Mod 10 — Part 5 requirement)
# ──────────────────────────────────────────────────────────────────────

def process_rss_gb(pid: Optional[int] = None) -> Optional[float]:
    """RSS (resident set size) of a process in GB, via psutil if available."""
    try:
        import psutil
        p = psutil.Process(pid or os.getpid())
        return p.memory_info().rss / 1e9
    except Exception:
        return None


def child_process_count(pid: Optional[int] = None) -> int:
    """Count of live child processes (e.g. DataLoader workers) of `pid`
    (defaults to the current process). Returns -1 if psutil unavailable."""
    try:
        import psutil
        p = psutil.Process(pid or os.getpid())
        return len(p.children(recursive=True))
    except Exception:
        return -1


def children_rss_gb(pid: Optional[int] = None) -> Optional[float]:
    """Aggregate RSS (GB) of all child processes (e.g. DataLoader
    workers) of `pid` (defaults to the current process). Kept separate
    from process_rss_gb() (this process's own RSS) so the two can be
    reported side by side (Part 5: parent RSS vs. child RSS)."""
    try:
        import psutil
        p = psutil.Process(pid or os.getpid())
        total = 0.0
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except Exception:
                continue
        return total / 1e9
    except Exception:
        return None


def system_ram_gb() -> "tuple[Optional[float], Optional[float]]":
    """(used_gb, total_gb) system-wide host RAM, via psutil if available."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.used / 1e9, vm.total / 1e9
    except Exception:
        return None, None


def gpu_stats_all() -> list:
    """Per-GPU [(index, used_mb, total_mb, util_pct), ...] via nvidia-smi.

    Queries ALL GPUs (not just index 0) so DataParallel's use of a second
    device is actually visible in diagnostics, closing the exact blind
    spot identified in the prior investigation. Returns [] on any failure
    (e.g. nvidia-smi not present) rather than raising, since this must
    never be allowed to interrupt training.
    """
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()
        stats = []
        for line in out:
            idx, used, total, util = (x.strip() for x in line.split(","))
            stats.append((int(idx), int(used), int(total), int(util)))
        return stats
    except Exception:
        return []


class PeakTracker:
    """Tracks a running maximum for a named set of metrics (Part 5: peak
    RAM, peak GPU memory). Plain dict wrapper; exists mainly for a
    consistent, obvious call site (`peaks.update('ram_gb', value)`)."""

    def __init__(self):
        self._peaks: Dict[str, float] = {}

    def update(self, key: str, value: Optional[float]):
        if value is None:
            return
        if key not in self._peaks or value > self._peaks[key]:
            self._peaks[key] = value

    def get(self, key: str, default=0.0):
        return self._peaks.get(key, default)

    def as_dict(self) -> Dict[str, float]:
        return dict(self._peaks)


# ──────────────────────────────────────────────────────────────────────
# Training history persistence (Mod 11 — Part 6 requirement)
# ──────────────────────────────────────────────────────────────────────

_HISTORY_FIELDS = [
    "epoch", "timestamp", "train_loss", "val_loss", "macro_f1",
    "MA_f1", "HE_f1", "IH_f1", "VB_IRMA_f1", "NV_f1", "VH_f1", "RD_f1",
    "lr_backbone", "lr_decoder", "epoch_time_s", "elapsed_s", "eta_s",
    "best_macro_f1", "checkpoint_saved", "checkpoint_path",
    "thresholds_computed", "session_remaining_hrs",
]


def append_epoch_history(output_dir: Path, record: Dict) -> None:
    """Append one epoch's metrics to history.csv and history.json.

    Survives notebook interruption: both files are rewritten in full
    (atomic tmp+rename, matching the checkpoint-write pattern) on every
    call, from an in-memory list re-read from the existing JSON file each
    time — so a kill mid-write can never leave a truncated/corrupt
    history file, only a slightly-stale one from the last successful
    write.

    De-duplicates by 'epoch': if this epoch's row already exists (e.g. a
    mid-epoch-checkpoint resume redoes an epoch that had already logged a
    row before the interruption), the existing row is REPLACED rather
    than duplicated, so history.csv/.json always reflect exactly one row
    per completed epoch.

    Parameters
    ----------
    output_dir : Path
        Experiment output directory (history.csv/.json live here,
        alongside best_model.pth / last_model.pth / results.json).
    record : dict
        Must include 'epoch'; any _HISTORY_FIELDS keys not present are
        filled with None so the CSV header stays stable across runs
        even if a future caller adds fields incrementally.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "history.json"
    csv_path = output_dir / "history.csv"

    # Load existing history (tolerant of a missing/corrupt file — a
    # broken history file must never block training or resume).
    history = []
    if json_path.exists():
        try:
            history = json.loads(json_path.read_text())
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []

    row = {k: record.get(k) for k in _HISTORY_FIELDS}
    row["timestamp"] = row["timestamp"] or time.strftime("%Y-%m-%d %H:%M:%S")

    # De-dupe by epoch
    history = [h for h in history if h.get("epoch") != row["epoch"]]
    history.append(row)
    history.sort(key=lambda h: (h.get("epoch") is None, h.get("epoch", 0)))

    # Atomic JSON write
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(history, indent=2))
    os.replace(tmp_json, json_path)

    # Atomic CSV write (full rewrite — history is small, at most `epochs`
    # rows, so this is cheap and simplest to keep correct/consistent
    # with the JSON file rather than maintaining two divergent
    # append-only formats).
    import csv as _csv
    tmp_csv = csv_path.with_suffix(".csv.tmp")
    with open(tmp_csv, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=_HISTORY_FIELDS)
        writer.writeheader()
        for h in history:
            writer.writerow(h)
    os.replace(tmp_csv, csv_path)


# ──────────────────────────────────────────────────────────────────────
# Timer
# ──────────────────────────────────────────────────────────────────────

class Timer:
    """Simple elapsed-time tracker."""

    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        """Return elapsed time as 'Xh Ym Zs' string."""
        s = self.elapsed()
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        if h > 0:
            return f"{h}h {m}m {sec}s"
        elif m > 0:
            return f"{m}m {sec}s"
        return f"{sec}s"
