"""
Dataset classes and modality-balanced sampler for MMRDR lesion detection.

Handles:
- Single-modality datasets (Exp A, B)
- Joint CFP+UWF datasets (Exp C, D)
- Modality-balanced sampling (Mod 4)
- Train/val/test splits (tr/ts prefix convention)
"""

import ast
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms

from .config import (
    Config, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES, SEED,
)


# ──────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────

def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_val_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────
# Batch-level augmentation: CutMix / MixUp for multi-label
# ──────────────────────────────────────────────────────────────────────

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix.

    Parameters
    ----------
    size : tuple
        (B, C, H, W) tensor shape.
    lam : float
        Lambda from Beta distribution.

    Returns
    -------
    bbx1, bby1, bbx2, bby2 : int
        Bounding box coordinates.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix(images, labels, alpha=1.0):
    """Apply CutMix augmentation to a batch (multi-label safe).

    Randomly cuts a patch from a permuted image and pastes it onto the
    original. Labels are mixed proportionally to the area ratio.

    Parameters
    ----------
    images : (B, C, H, W) tensor
    labels : (B, num_classes) tensor
    alpha : float
        Beta distribution parameter.

    Returns
    -------
    mixed_images : (B, C, H, W) tensor
    mixed_labels : (B, num_classes) tensor
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))

    # Multi-label mixing: weighted blend
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    mixed_labels = mixed_labels.clamp(0, 1)

    return images, mixed_labels


def apply_mixup(images, labels, alpha=0.8):
    """Apply MixUp augmentation to a batch (multi-label safe).

    Linearly blends images and labels from a random permutation.

    Parameters
    ----------
    images : (B, C, H, W) tensor
    labels : (B, num_classes) tensor
    alpha : float
        Beta distribution parameter.

    Returns
    -------
    mixed_images : (B, C, H, W) tensor
    mixed_labels : (B, num_classes) tensor
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    mixed_labels = mixed_labels.clamp(0, 1)

    return mixed_images, mixed_labels


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

class LesionDataset(Dataset):
    """Multi-label retinal lesion dataset.

    Returns
    -------
    image       : (3, H, W) float32 tensor
    lesion_vec  : (7,) float32 binary vector
    modality_id : int  (0 = CFP, 1 = UWF)
    """

    MODALITY_MAP = {"cfp": 0, "uwf": 1}

    def __init__(
        self,
        df: pd.DataFrame,
        img_root: Path,
        modality: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.modality_id = self.MODALITY_MAP[modality.lower()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.img_root / str(row["image"])
        image = Image.open(img_path).convert("RGB")

        # Parse lesion vector: "[0, 1, 0, 0, 1, 0, 0]" → tensor
        lesion_str = str(row["lesion"]).strip()
        lesion_list = ast.literal_eval(lesion_str)
        lesion_vec = torch.tensor(lesion_list, dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, lesion_vec, self.modality_id


# ──────────────────────────────────────────────────────────────────────
# Combined dataset (for joint training)
# ──────────────────────────────────────────────────────────────────────

class CombinedLesionDataset(Dataset):
    """Concatenates multiple LesionDatasets, preserving modality_id."""

    def __init__(self, datasets: List[LesionDataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int):
        # Find which sub-dataset this idx belongs to
        for i, cs in enumerate(self.cumulative_sizes):
            if idx < cs:
                offset = self.cumulative_sizes[i - 1] if i > 0 else 0
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range for size {len(self)}")

    def get_modality_indices(self) -> List[List[int]]:
        """Return list of index arrays, one per sub-dataset (modality)."""
        indices = []
        offset = 0
        for ds in self.datasets:
            indices.append(list(range(offset, offset + len(ds))))
            offset += len(ds)
        return indices


# ──────────────────────────────────────────────────────────────────────
# Modality-balanced sampler  (Mod 4)
# ──────────────────────────────────────────────────────────────────────

class ModalityBalancedSampler(Sampler):
    """Yields batches with ~50 % CFP and ~50 % UWF indices.

    Strategy
    --------
    - Maintain two independent, shuffled index pools.
    - Each __iter__ call interleaves chunks:
        [cfp_half_batch] + [uwf_half_batch] repeated.
    - If one pool runs out first, reshuffle and restart it so the other
      pool can finish (guarantees every sample is seen at least once
      per epoch from the larger pool).

    Edge cases
    ----------
    - Odd batch size: CFP gets ceil(B/2), UWF gets floor(B/2),
      alternating each batch.
    - One modality much larger: the smaller modality's pool is reshuffled
      mid-epoch to keep batches balanced.
    """

    def __init__(
        self,
        modality_indices: List[List[int]],
        batch_size: int,
        seed: int = SEED,
    ):
        assert len(modality_indices) == 2, "Balanced sampler needs exactly 2 modalities"
        self.pool_a = list(modality_indices[0])  # CFP
        self.pool_b = list(modality_indices[1])  # UWF
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        # Shuffle both pools
        pool_a = self.pool_a.copy()
        pool_b = self.pool_b.copy()
        rng.shuffle(pool_a)
        rng.shuffle(pool_b)

        half = self.batch_size // 2
        other_half = self.batch_size - half  # handles odd batch_size

        idx_a, idx_b = 0, 0
        indices = []

        # Total batches based on larger pool
        max_samples = max(len(pool_a), len(pool_b)) * 2
        yielded = 0

        while yielded < max_samples:
            # Get chunk from pool A (CFP)
            chunk_a = []
            for _ in range(half):
                if idx_a >= len(pool_a):
                    rng.shuffle(pool_a)
                    idx_a = 0
                chunk_a.append(pool_a[idx_a])
                idx_a += 1

            # Get chunk from pool B (UWF)
            chunk_b = []
            for _ in range(other_half):
                if idx_b >= len(pool_b):
                    rng.shuffle(pool_b)
                    idx_b = 0
                chunk_b.append(pool_b[idx_b])
                idx_b += 1

            indices.extend(chunk_a + chunk_b)
            yielded += self.batch_size

            # Swap half sizes each batch for odd batch_size fairness
            half, other_half = other_half, half

            # Stop once both pools have been fully traversed at least once
            if idx_a >= len(pool_a) and idx_b >= len(pool_b):
                break

        return iter(indices)

    def __len__(self) -> int:
        # Approximate: 2 × max(pool_a, pool_b), rounded up to batch boundary
        max_pool = max(len(self.pool_a), len(self.pool_b))
        return (max_pool * 2 // self.batch_size) * self.batch_size


# ──────────────────────────────────────────────────────────────────────
# Data loading utilities
# ──────────────────────────────────────────────────────────────────────

def _split_train_val(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train rows into train/val with fixed seed."""
    rng = np.random.RandomState(seed)
    n = len(df)
    n_val = int(n * val_ratio)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _split_train_val_kfold(
    df: pd.DataFrame, n_folds: int, fold: int, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train rows into train/val using stratified k-fold.

    Stratifies on the rarest class (last column = RD) to ensure each fold
    has at least some positive samples for the most imbalanced class.
    Falls back to random k-fold if stratification fails.

    Parameters
    ----------
    df : pd.DataFrame
        Training pool dataframe (prefix 'tr' images only).
    n_folds : int
        Total number of folds.
    fold : int
        Current fold index (0 to n_folds-1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_df, val_df : pd.DataFrame
    """
    n = len(df)
    rng = np.random.RandomState(seed)

    # Parse labels to get stratification key
    labels = np.array([ast.literal_eval(str(r)) for r in df["lesion"]])

    # Use the rarest class (RD, index 6) for stratification
    rare_class = labels[:, -1].astype(int)

    # Stratified split: separate positive and negative indices
    pos_idx = np.where(rare_class == 1)[0]
    neg_idx = np.where(rare_class == 0)[0]

    # Shuffle each group independently
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    # Split each group into k folds
    def _kfold_split(indices, k, f):
        fold_size = len(indices) // k
        remainder = len(indices) % k
        # Each fold gets fold_size items; first 'remainder' folds get +1
        start = 0
        for i in range(k):
            extra = 1 if i < remainder else 0
            end = start + fold_size + extra
            if i == f:
                return indices[start:end], np.concatenate([indices[:start], indices[end:]])
            start = end
        return np.array([], dtype=int), indices  # fallback

    val_pos, train_pos = _kfold_split(pos_idx, n_folds, fold)
    val_neg, train_neg = _kfold_split(neg_idx, n_folds, fold)

    val_indices = np.concatenate([val_pos, val_neg])
    train_indices = np.concatenate([train_pos, train_neg])

    return df.iloc[train_indices].copy(), df.iloc[val_indices].copy()


def count_class_positives(
    train_loader_or_datasets,
    num_classes: int = NUM_CLASSES,
) -> List[int]:
    """Count positive samples per class across training datasets.

    Parameters
    ----------
    train_loader_or_datasets : DataLoader or list of LesionDataset
        Training data source.
    num_classes : int
        Number of label classes.

    Returns
    -------
    counts : list of int
        Number of positive samples per class.
    """
    counts = [0] * num_classes

    # Accept either a list of datasets or a single DataLoader
    if isinstance(train_loader_or_datasets, list):
        datasets = train_loader_or_datasets
    elif hasattr(train_loader_or_datasets, "dataset"):
        ds = train_loader_or_datasets.dataset
        if hasattr(ds, "datasets"):
            datasets = ds.datasets  # CombinedLesionDataset
        else:
            datasets = [ds]
    else:
        datasets = [train_loader_or_datasets]

    for ds in datasets:
        for idx in range(len(ds)):
            _, labels, _ = ds[idx]
            for c in range(num_classes):
                counts[c] += int(labels[c].item())

    return counts


def count_class_positives_fast(
    datasets: list,
    num_classes: int = NUM_CLASSES,
) -> List[int]:
    """Count positive samples per class from dataframes (fast, no image loading).

    Parameters
    ----------
    datasets : list of LesionDataset
        Training datasets with .df attribute.
    num_classes : int
        Number of label classes.

    Returns
    -------
    counts : list of int
        Number of positive samples per class.
    """
    counts = np.zeros(num_classes, dtype=int)

    for ds in datasets:
        labels = np.array([ast.literal_eval(str(r)) for r in ds.df["lesion"]])
        counts += labels.sum(axis=0).astype(int)

    return counts.tolist()


def load_splits(
    data_dir: Path, csv_name: str, val_ratio: float = 0.125, seed: int = SEED,
    use_cv: bool = False, n_folds: int = 5, fold: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSV → split into train / val / test by filename prefix.

    Convention: filenames starting with 'tr' → train pool, 'ts' → test.

    When use_cv=True and fold >= 0, uses stratified k-fold instead of
    random split. The val_ratio parameter is ignored in CV mode.
    """
    df = pd.read_csv(data_dir / csv_name)
    basenames = df["image"].apply(lambda x: x.split("/")[-1])
    train_mask = basenames.str.startswith("tr").values
    test_mask = basenames.str.startswith("ts").values

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if use_cv and fold >= 0:
        train_df, val_df = _split_train_val_kfold(train_df, n_folds, fold, seed)
    else:
        train_df, val_df = _split_train_val(train_df, val_ratio, seed)

    return train_df, val_df, test_df


def build_dataloaders(cfg: Config):
    """Build train / val / test DataLoaders respecting all modifications.

    Returns
    -------
    train_loader, val_loader, test_loaders : dict
        test_loaders is a dict {"cfp": DataLoader, "uwf": DataLoader}
        with separate test loaders per modality for fair evaluation.
    """
    project = Path(cfg.project_root)

    train_transform = get_train_transform(IMAGE_SIZE)
    val_transform = get_val_transform(IMAGE_SIZE)

    def _worker_kwargs():
        """Shared DataLoader worker-lifecycle settings (Mod 4 — hardening).

        persistent_workers=False (explicit, matches the prior implicit
        default): workers are recreated each epoch/iterator rather than
        kept alive, so nothing survives between epochs to leak.
        prefetch_factor=1 (down from PyTorch's default of 2, only valid
        when num_workers>0): each worker holds at most one pre-loaded,
        pinned batch in flight instead of two, shrinking the amount of
        host RAM / pinned memory that could be orphaned if a worker's
        parent process is ever SIGKILL'd before it can clean up after
        itself (SIGKILL does not cascade to a process's own
        multiprocessing children).
        """
        kw = {"persistent_workers": False}
        if cfg.num_workers > 0:
            kw["prefetch_factor"] = 1
        return kw

    train_datasets = []
    val_datasets = []
    test_loaders = {}

    for modality in cfg.modalities:
        if modality == "cfp":
            data_dir = project / cfg.cfp_dir
            csv_name = cfg.cfp_csv
        else:
            data_dir = project / cfg.uwf_dir
            csv_name = cfg.uwf_csv

        train_df, val_df, test_df = load_splits(
            data_dir, csv_name, cfg.val_split, cfg.seed,
            use_cv=cfg.use_cv, n_folds=cfg.n_folds, fold=cfg.fold,
        )

        train_ds = LesionDataset(train_df, data_dir, modality, train_transform)
        val_ds = LesionDataset(val_df, data_dir, modality, val_transform)
        test_ds = LesionDataset(test_df, data_dir, modality, val_transform)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

        # Separate test loader per modality (always)
        test_loaders[modality] = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            **_worker_kwargs(),
        )

    # ── Combine for joint training ───────────────────────────────────
    if len(train_datasets) == 1:
        # Single modality (Exp A or B)
        train_loader = DataLoader(
            train_datasets[0],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
            **_worker_kwargs(),
        )
        val_loader = DataLoader(
            val_datasets[0],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            **_worker_kwargs(),
        )
    else:
        # Joint training (Exp C or D)
        combined_train = CombinedLesionDataset(train_datasets)
        combined_val = CombinedLesionDataset(val_datasets)

        if cfg.balanced_sampling:
            modality_indices = combined_train.get_modality_indices()
            train_sampler = ModalityBalancedSampler(
                modality_indices, cfg.batch_size, cfg.seed
            )
            train_loader = DataLoader(
                combined_train,
                batch_size=cfg.batch_size,
                sampler=train_sampler,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                drop_last=True,
                **_worker_kwargs(),
            )
        else:
            train_loader = DataLoader(
                combined_train,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                drop_last=True,
                **_worker_kwargs(),
            )

        val_loader = DataLoader(
            combined_val,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            **_worker_kwargs(),
        )

    return train_loader, val_loader, test_loaders

