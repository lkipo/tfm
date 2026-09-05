"""LightningDataModule over the cached cohort written by ``prepare_data.py``.

Two decisions here are load-bearing and both come from the failed runs logged
in ``training_diary.md``:

* **Crop centres are drawn per class, not from the foreground union.** A
  portal pedicle is ~0.3% of the liver and ~0.03% of the volume; sampling
  patch centres from the union of all channels draws a liver-only centre
  30-100x more often than a vessel one, and a 300-epoch run under that scheme
  left val Dice at ~0 for three of four channels. ``RandCropByLabelClassesd``
  with explicit ratios fixes the sampling distribution directly.
* **Augmentation never mirrors.** finetuning.md Sec. 4: the classes are
  laterality-specific and the Couinaud pipeline recovers its anatomical frame
  from them, so a flip in any axis teaches the wrong side and corrupts every
  downstream decision. Rotations <=15 deg, scaling, elastic, and intensity
  transforms only -- no ``RandFlipd``, no ``RandRotate90d``.

The label stays a compact integer map all the way through augmentation and is
expanded to C sigmoid channels only after cropping (``BuildChannelsd``): the
expansion is ~C x 86 MB on a whole cropped case and ~C x 3.5 MB on a patch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import lightning.pytorch as pl
import numpy as np
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import (
    CastToTyped,
    ClassesToIndicesd,
    Compose,
    LoadImaged,
    MapTransform,
    Rand3DElasticd,
    RandAdjustContrastd,
    RandAffined,
    RandCropByLabelClassesd,
    RandGaussianNoised,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

from labels import Channel, channel_availability, channel_names, get_channels


# --------------------------------------------------------------------------
# target construction
# --------------------------------------------------------------------------
class BuildChannelsd(MapTransform):
    """Integer label map ``(1,*S)`` -> multi-label float target ``(C,*S)``.

    Sigmoid, not softmax. The channels genuinely overlap by construction --
    ``vessels`` is a subset of ``liver``, ``portal_tree`` is a subset of
    ``vessels`` -- and even at the raw-id level finetuning.md Sec. 1 measures
    up to 20-29% pairwise overlap at the portal bifurcation and the caval
    confluence, the two places the pipeline is most sensitive. A softmax head
    forces a decision the annotation never made, exactly there.
    """

    def __init__(self, keys, channels: list[Channel]) -> None:
        super().__init__(keys)
        self.channels = channels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = np.asarray(d[key])
            flat = label[0] if label.ndim == 4 else label
            out = np.zeros((len(self.channels), *flat.shape), dtype=np.float32)
            for i, ch in enumerate(self.channels):
                acc = np.zeros(flat.shape, dtype=bool)
                for src in ch.sources:
                    acc |= flat == src
                out[i] = acc
            d[key] = out
        return d


class SampleMapd(MapTransform):
    """Integer label map -> the categorical map crop centres are drawn from.

    Channels are painted in ascending ``sample_priority`` so the rarest,
    thinnest structure wins every overlap: bucket ``i+1`` is "channel i and
    nothing more specific than it". Bucket 0 is background. Without this,
    ``liver`` -- which is a superset of most other channels -- would own every
    voxel and the per-class ratios would be meaningless.
    """

    def __init__(self, keys, channels: list[Channel], out_key: str = "sample_label") -> None:
        super().__init__(keys)
        self.channels = channels
        self.out_key = out_key
        self.order = sorted(range(len(channels)), key=lambda i: channels[i].sample_priority)

    def __call__(self, data):
        d = dict(data)
        (key,) = self.keys
        label = np.asarray(d[key])
        flat = label[0] if label.ndim == 4 else label
        cat = np.zeros(flat.shape, dtype=np.uint8)
        for i in self.order:
            acc = np.zeros(flat.shape, dtype=bool)
            for src in self.channels[i].sources:
                acc |= flat == src
            cat[acc] = i + 1
        d[self.out_key] = cat[None]
        return d


def crop_ratios(channels: list[Channel], background: float, thin_share: float,
                liver_share: float = 0.0) -> list[float]:
    """Ratio per sampling bucket: background first, then one per channel.

    ``thin_share`` of the foreground budget goes to the tubular channels split
    evenly, the remainder to the bulky ones. Everything is renormalised, so the
    numbers below are shares of crop centres, not of voxels -- which is the
    whole point: they are deliberately unrelated to prevalence.

    ``liver_share``, if set, carves out a dedicated crop-centre share for the
    ``liver`` channel instead of splitting the bulk budget evenly with
    ``aorta``/``gallbladder``. core6 Run 1 plateaued at liver Dice 0.82 while
    still nominally "bulk" -- diluted 1/3 against two smaller organs it should
    dominate.
    """
    thin = [i for i, c in enumerate(channels) if c.thin]
    liver_idx = next((i for i, c in enumerate(channels) if c.name == "liver"), None) \
        if liver_share > 0 else None
    bulk = [i for i, c in enumerate(channels) if not c.thin and i != liver_idx]
    fg = max(0.0, 1.0 - background)
    reserved = liver_share if liver_idx is not None else 0.0
    remaining = max(0.0, fg - reserved)
    per_thin = (remaining * thin_share / len(thin)) if thin else 0.0
    per_bulk = (remaining * (1.0 - thin_share if thin else 1.0) / len(bulk)) if bulk else 0.0
    ratios = [background]
    for i, c in enumerate(channels):
        if i == liver_idx:
            ratios.append(liver_share)
        elif c.thin:
            ratios.append(per_thin)
        else:
            ratios.append(per_bulk)
    total = sum(ratios)
    return [r / total for r in ratios]


# --------------------------------------------------------------------------
# datamodule
# --------------------------------------------------------------------------
class CornerstoneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        prepared_root: str = "./data/cornerstone_prepared",
        channel_preset: str = "core6",
        roi_size: Sequence[int] = (96, 96, 96),
        intensity_range: Sequence[float] = (-175.0, 250.0),
        batch_size: int = 1,
        samples_per_volume: int = 4,
        num_workers: int = 4,
        seed: int = 42,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        test_frac: float = 0.15,
        n_folds: int = 0,
        fold: int = 0,
        exclude_modality: Optional[list[str]] = None,
        require_channels: Optional[list[str]] = None,
        group_by: str = "case",
        crop_background_ratio: float = 0.05,
        crop_thin_share: float = 0.7,
        crop_liver_share: float = 0.0,
        max_samples_per_class: int = 20000,
        elastic_prob: float = 0.15,
        affine_prob: float = 0.3,
        cache_rate_val: float = 1.0,
        cache_rate_train: float = 1.0,
    ) -> None:
        super().__init__()
        if n_folds == 0 and abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
            raise ValueError("train/val/test fractions must sum to 1.0")
        if group_by not in {"case", "suffix"}:
            raise ValueError("group_by must be 'case' or 'suffix'")
        self.save_hyperparameters()
        self.channels = get_channels(channel_preset)
        self.channel_names = channel_names(self.channels)
        self.train_ds = self.val_ds = self.test_ds = None
        self._records: list[dict] | None = None

    # ------------------------------------------------------------------
    # cohort
    # ------------------------------------------------------------------
    def _manifest(self) -> dict:
        path = Path(self.hparams.prepared_root) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run cornerstone_finetune/prepare_data.py first."
            )
        return json.loads(path.read_text())

    def records(self) -> list[dict]:
        """One dict per usable subject: paths, per-channel availability mask,
        and the group key the split must not straddle."""
        if self._records is not None:
            return self._records
        manifest = self._manifest()
        root = Path(self.hparams.prepared_root)
        exclude = set(self.hparams.exclude_modality or [])
        require = set(self.hparams.require_channels or [])
        unknown = require - set(self.channel_names)
        if unknown:
            raise ValueError(f"require_channels names no such channel: {sorted(unknown)}")

        records, skipped = [], []
        from labels import LABEL_IDS

        for meta in manifest["cases"]:
            case = meta["case"]
            if meta.get("modality") in exclude:
                skipped.append((case, f"modality {meta['modality']}"))
                continue
            counts = {LABEL_IDS[name]: n for name, n in meta["label_counts"].items()}
            mask = channel_availability(counts, self.channels)
            missing = {n for n, m in zip(self.channel_names, mask) if not m}
            if require & missing:
                skipped.append((case, f"missing required {sorted(require & missing)}"))
                continue
            records.append(
                {
                    "case": case,
                    "image": str(root / case / "image.nii.gz"),
                    "label": str(root / case / "label.nii.gz"),
                    "channel_mask": np.asarray(mask, dtype=np.float32),
                    "group": case.rsplit("_", 1)[-1] if self.hparams.group_by == "suffix" else case,
                }
            )
        if not records:
            raise RuntimeError("no usable cases after filtering")
        self._records = records
        self._skipped = skipped
        return records

    def splits(self) -> tuple[list, list, list]:
        records = self.records()
        groups = sorted({r["group"] for r in records})
        rng = np.random.default_rng(self.hparams.seed)
        order = rng.permutation(len(groups))
        groups = [groups[i] for i in order]

        if self.hparams.n_folds and self.hparams.n_folds > 1:
            # Grouped k-fold. finetuning.md Sec. 4: at n=38 a single held-out
            # split is too noisy to compare runs by -- report variance across
            # folds instead. Fold k is validation, fold k+1 is test.
            k = self.hparams.n_folds
            folds = [groups[i::k] for i in range(k)]
            f = self.hparams.fold % k
            val_groups = set(folds[f])
            test_groups = set(folds[(f + 1) % k])
            train_groups = set(groups) - val_groups - test_groups
        else:
            n = len(groups)
            n_train = int(round(self.hparams.train_frac * n))
            n_val = int(round(self.hparams.val_frac * n))
            train_groups = set(groups[:n_train])
            val_groups = set(groups[n_train : n_train + n_val])
            test_groups = set(groups[n_train + n_val :])

        pick = lambda gs: [r for r in records if r["group"] in gs]  # noqa: E731
        return pick(train_groups), pick(val_groups), pick(test_groups)

    def summary(self) -> str:
        train, val, test = self.splits()
        lines = [
            f"[data] {len(self.records())} usable subjects "
            f"({len(train)} train / {len(val)} val / {len(test)} test), "
            f"preset={self.hparams.channel_preset} ({len(self.channels)} channels)"
        ]
        for name, split in (("train", train), ("val", val), ("test", test)):
            lines.append(f"  {name:5s}: {', '.join(r['case'] for r in split) or '-'}")
        for skip_case, why in getattr(self, "_skipped", []):
            lines.append(f"  skipped {skip_case}: {why}")
        avail = np.stack([r["channel_mask"] for r in train]) if train else np.zeros((0, len(self.channels)))
        lines.append("  channel availability in train split:")
        for i, name in enumerate(self.channel_names):
            n = int(avail[:, i].sum()) if len(avail) else 0
            lines.append(f"    {name:24s} {n:2d}/{len(train)}")
        lines.append(f"  crop ratios (bg + per channel): "
                     f"{[round(r, 3) for r in self.sampling_ratios()]}")
        return "\n".join(lines)

    def sampling_ratios(self) -> list[float]:
        return crop_ratios(self.channels, self.hparams.crop_background_ratio,
                           self.hparams.crop_thin_share, self.hparams.crop_liver_share)

    def class_prevalence(self) -> list[float]:
        """Mean per-channel foreground fraction over the *training* subjects,
        used for the head's bias prior and for ``pos_weight``. Computed from
        the manifest's exact voxel counts, so it costs nothing."""
        from labels import LABEL_IDS, channel_counts

        manifest = self._manifest()
        by_case = {m["case"]: m for m in manifest["cases"]}
        train, _, _ = self.splits()
        out = np.zeros(len(self.channels))
        weight = np.zeros(len(self.channels))
        for r in train:
            meta = by_case[r["case"]]
            counts = {LABEL_IDS[name]: n for name, n in meta["label_counts"].items()}
            frac = np.asarray(channel_counts(counts, self.channels), dtype=float) / meta["n_voxels"]
            m = r["channel_mask"]
            out += frac * m
            weight += m
        prevalence = np.divide(out, np.maximum(weight, 1e-6))
        return [float(max(p, 1e-6)) for p in prevalence]

    # ------------------------------------------------------------------
    # transforms
    # ------------------------------------------------------------------
    def _deterministic(self) -> list:
        lo, hi = self.hparams.intensity_range
        return [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=True),
            CastToTyped(keys=["image"], dtype=np.float32),
            ScaleIntensityRanged(keys=["image"], a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True),
            # Pad before the sampling map so the indices below index the same
            # array the crop will index.
            SpatialPadd(keys=["image", "label"], spatial_size=tuple(self.hparams.roi_size)),
        ]

    def _train_transforms(self) -> Compose:
        n_buckets = len(self.channels) + 1
        return Compose(
            [
                *self._deterministic(),
                SampleMapd(keys=["label"], channels=self.channels),
                ClassesToIndicesd(
                    keys=["sample_label"],
                    num_classes=n_buckets,
                    max_samples_per_class=self.hparams.max_samples_per_class,
                ),
                # --- everything below is random and therefore never cached ---
                RandCropByLabelClassesd(
                    keys=["image", "label"],
                    label_key="sample_label",
                    indices_key="sample_label_cls_indices",
                    spatial_size=tuple(self.hparams.roi_size),
                    ratios=self.sampling_ratios(),
                    num_classes=n_buckets,
                    num_samples=self.hparams.samples_per_volume,
                    warn=False,
                ),
                RandAffined(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=self.hparams.affine_prob,
                    rotate_range=(0.26, 0.26, 0.26),  # ~15 deg, the cap in finetuning.md Sec. 4
                    scale_range=(0.1, 0.1, 0.1),
                    padding_mode="border",
                ),
                Rand3DElasticd(
                    keys=["image", "label"],
                    mode=("bilinear", "nearest"),
                    prob=self.hparams.elastic_prob,
                    sigma_range=(5, 7),
                    magnitude_range=(50, 150),
                    padding_mode="border",
                ),
                RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
                RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                BuildChannelsd(keys=["label"], channels=self.channels),
                ToTensord(keys=["image", "label"]),
            ]
        )

    def _eval_transforms(self) -> Compose:
        return Compose(
            [
                *self._deterministic(),
                BuildChannelsd(keys=["label"], channels=self.channels),
                ToTensord(keys=["image", "label"]),
            ]
        )

    # ------------------------------------------------------------------
    # lightning hooks
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        train, val, test = self.splits()
        if stage in (None, "fit"):
            self.train_ds = CacheDataset(
                data=train,
                transform=self._train_transforms(),
                cache_rate=self.hparams.cache_rate_train,
                num_workers=self.hparams.num_workers,
                copy_cache=False,
            )
            self.val_ds = CacheDataset(
                data=val,
                transform=self._eval_transforms(),
                cache_rate=self.hparams.cache_rate_val,
                num_workers=self.hparams.num_workers,
                copy_cache=False,
            )
        if stage in (None, "validate") and self.val_ds is None:
            self.val_ds = CacheDataset(
                data=val, transform=self._eval_transforms(),
                cache_rate=self.hparams.cache_rate_val, num_workers=self.hparams.num_workers,
                copy_cache=False,
            )
        if stage in (None, "test", "predict"):
            self.test_ds = CacheDataset(
                data=test,
                transform=self._eval_transforms(),
                cache_rate=self.hparams.cache_rate_val,
                num_workers=self.hparams.num_workers,
                copy_cache=False,
            )

    def _loader(self, ds, shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size or self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            collate_fn=list_data_collate,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False, batch_size=1)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False, batch_size=1)

    def predict_dataloader(self):
        return self._loader(self.test_ds, shuffle=False, batch_size=1)
