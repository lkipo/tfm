#!/usr/bin/env python3
"""Turn ``data/cornerstone_masks`` + ``data/0_test_nifti/imagesTs`` into a
cached, training-ready cohort under ``data/cornerstone_prepared``.

Three things happen here, all of which have to happen *once* rather than in
every epoch of every run:

1. **Reorient to RAS.** Every source volume is ``L,A,S``. Reorientation moves
   image and label together, so it does not touch laterality -- unlike a flip,
   which finetuning.md Sec. 4 forbids because the pipeline recovers its
   anatomical frame from these very labels.
2. **Resample to a common spacing** (default 1.0 mm isotropic). The cohort
   spans 0.63-0.98 mm in-plane and **0.40-1.50 mm** through-plane, so a fixed
   voxel ROI would be a 3.75x-varying *physical* receptive field across
   subjects. finetuning.md's "do not resample" applies to the 1 mm cohort in
   ``continuity/``; these are the native scans and they are not isotropic.
3. **Crop to the liver.** Source volumes are 150-500 MB of float64 each
   (13 GB for the cohort); the liver bounding box plus a margin is ~2% of
   that. Everything downstream reads the crop.

The output keeps NIfTI with a correct affine (finetuning.md Sec. 4: "write a
correct affine on export") so ``predict.py`` can map predictions back onto the
original CT grid, and so the crops can be dropped into any viewer as-is.

    python cornerstone_finetune/prepare_data.py --workers 4
    python cornerstone_finetune/prepare_data.py --case RMV2025_0001_CT_NCT_HBP_X --force
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
from scipy.ndimage import zoom

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from labels import ID_TO_NAME, NUM_LABELS  # noqa: E402

DEFAULT_IMAGES = REPO / "data" / "0_test_nifti" / "imagesTs"
DEFAULT_MASKS = REPO / "data" / "cornerstone_masks"
DEFAULT_OUT = REPO / "data" / "cornerstone_prepared"


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def to_ras(data: np.ndarray, affine: np.ndarray, order_is_nearest: bool):
    """Reorient an array (and its affine) to RAS. Pure axis permutation and
    flipping of an integer-indexed array -- no interpolation, so it is exact
    for the label map as well as the image."""
    start = io_orientation(affine)
    target = axcodes2ornt(("R", "A", "S"))
    transform = ornt_transform(start, target)
    out = apply_orientation(data, transform)
    new_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
    return out, new_affine


def bbox_of(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-axis ``any()`` reductions rather than ``argwhere``: on these
    volumes the reduction form is ~100x faster and allocates nothing."""
    lo, hi = [], []
    for axis_pair in ((1, 2), (0, 2), (0, 1)):
        nz = np.where(mask.any(axis=axis_pair))[0]
        if nz.size == 0:
            return None
        lo.append(int(nz.min()))
        hi.append(int(nz.max()) + 1)
    return np.array(lo), np.array(hi)


def resample(volume: np.ndarray, affine: np.ndarray, target_spacing, order: int):
    """Resample to ``target_spacing`` and return the volume with its updated
    affine. ``order=0`` for the label map keeps it a valid partition."""
    current = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    factors = current / np.asarray(target_spacing, dtype=float)
    if np.allclose(factors, 1.0, atol=1e-3):
        return volume, affine
    out = zoom(volume, factors, order=order, mode="nearest", grid_mode=False)
    # The zoom factor actually realised differs from the requested one by the
    # rounding of the output shape; derive the affine from what we got so the
    # crop stays registered to the patient, not to the request.
    realised = np.array(out.shape) / np.array(volume.shape)
    scale = np.diag(1.0 / realised)
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] @ scale
    # zoom() samples at voxel centres of the new grid; shift the origin by the
    # half-voxel difference so the two grids share a world origin.
    shift = 0.5 * (np.diag(scale) - 1.0)
    new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ shift
    return out, new_affine


# --------------------------------------------------------------------------
# per-case work
# --------------------------------------------------------------------------
def prepare_case(
    case: str,
    images_dir: Path,
    masks_dir: Path,
    out_dir: Path,
    spacing: tuple[float, float, float],
    margin_mm: float,
    force: bool,
) -> dict:
    case_out = out_dir / case
    meta_path = case_out / "meta.json"
    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text())

    image_path = images_dir / f"{case}.nii.gz"
    mask_path = masks_dir / f"{case}_mask.nii.gz"
    img_nii = nib.load(str(image_path))
    msk_nii = nib.load(str(mask_path))
    if img_nii.shape != msk_nii.shape:
        raise ValueError(f"{case}: image {img_nii.shape} != mask {msk_nii.shape}")
    if not np.allclose(img_nii.affine, msk_nii.affine, atol=1e-4):
        raise ValueError(f"{case}: image and mask affines differ")

    source_shape = tuple(int(s) for s in img_nii.shape)
    source_affine = img_nii.affine.copy()
    source_spacing = tuple(float(z) for z in img_nii.header.get_zooms()[:3])

    label = np.asarray(msk_nii.dataobj).astype(np.uint8)
    label, ras_affine = to_ras(label, source_affine, order_is_nearest=True)

    # Crop from the liver, not from the union of every class: the aorta runs
    # the full length of the scan, so including it in the box would drag most
    # of the thorax back in and undo the point of cropping. The aorta *labels*
    # inside the box are kept; only the part of it far from the liver is lost,
    # and that part is not what this model is for.
    liver = label == 1
    box = bbox_of(liver)
    if box is None:
        raise ValueError(f"{case}: empty classLiver, nothing to crop to")
    lo, hi = box
    ras_spacing = np.sqrt((ras_affine[:3, :3] ** 2).sum(axis=0))
    margin_vox = np.ceil(margin_mm / ras_spacing).astype(int)
    lo = np.maximum(lo - margin_vox, 0)
    hi = np.minimum(hi + margin_vox, np.array(label.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

    label = np.ascontiguousarray(label[slices])
    crop_affine = ras_affine.copy()
    crop_affine[:3, 3] = ras_affine[:3, 3] + ras_affine[:3, :3] @ lo

    # Read only the cropped window of the image. dataobj slicing on a gzipped
    # NIfTI still decompresses the stream, but it never materialises the
    # full float64 array (150-500 MB per case).
    image = np.asarray(img_nii.dataobj)
    image, _ = to_ras(image, source_affine, order_is_nearest=False)
    image = np.ascontiguousarray(image[slices]).astype(np.float32)

    label_rs, out_affine = resample(label, crop_affine, spacing, order=0)
    image_rs, _ = resample(image, crop_affine, spacing, order=1)
    if label_rs.shape != image_rs.shape:  # rounding can disagree by one voxel
        common = tuple(min(a, b) for a, b in zip(label_rs.shape, image_rs.shape))
        label_rs = label_rs[: common[0], : common[1], : common[2]]
        image_rs = image_rs[: common[0], : common[1], : common[2]]

    # CT stays in HU; int16 covers [-32768, 32767] and the scanners' out-of-FOV
    # padding (-8192 on some Siemens series here) round-trips exactly.
    image_rs = np.clip(np.rint(image_rs), -32768, 32767).astype(np.int16)
    label_rs = label_rs.astype(np.uint8)

    case_out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(image_rs, out_affine), str(case_out / "image.nii.gz"))
    nib.save(nib.Nifti1Image(label_rs, out_affine), str(case_out / "label.nii.gz"))

    counts = np.bincount(label_rs.reshape(-1), minlength=NUM_LABELS + 1)
    fg = np.asarray(image_rs, dtype=np.float32)
    meta = {
        "case": case,
        "modality": case.split("_")[2] if len(case.split("_")) > 2 else "CT",
        "shape": [int(s) for s in label_rs.shape],
        "spacing": [float(s) for s in spacing],
        "affine": out_affine.tolist(),
        "source_shape": list(source_shape),
        "source_spacing": list(source_spacing),
        "source_affine": source_affine.tolist(),
        "ras_crop_origin": [int(v) for v in lo],
        "ras_crop_end": [int(v) for v in hi],
        "n_voxels": int(label_rs.size),
        "label_counts": {ID_TO_NAME[i]: int(counts[i]) for i in range(1, NUM_LABELS + 1)},
        "hu_percentiles": {
            "p01": float(np.percentile(fg, 1)),
            "p50": float(np.percentile(fg, 50)),
            "p99": float(np.percentile(fg, 99)),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def _worker(args) -> tuple[str, dict | None, str | None]:
    case = args[0]
    try:
        return case, prepare_case(*args), None
    except Exception:
        return case, None, traceback.format_exc()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help="target voxel spacing in mm (default 1 1 1)")
    ap.add_argument("--margin-mm", type=float, default=32.0,
                    help="margin around the liver bounding box (default 32)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--case", action="append", default=None, help="prepare only these cases")
    ap.add_argument("--force", action="store_true", help="rebuild cases that are already prepared")
    args = ap.parse_args()

    cases = sorted(p.name[: -len("_mask.nii.gz")] for p in args.masks.glob("*_mask.nii.gz"))
    if args.case:
        wanted = set(args.case)
        missing = wanted - set(cases)
        if missing:
            print(f"unknown cases: {sorted(missing)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c in wanted]
    if not cases:
        print(f"no *_mask.nii.gz under {args.masks}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (c, args.images, args.masks, args.out, tuple(args.spacing), args.margin_mm, args.force)
        for c in cases
    ]

    metas: dict[str, dict] = {}
    failures: dict[str, str] = {}
    print(f"preparing {len(jobs)} cases -> {args.out} "
          f"(spacing {args.spacing}, margin {args.margin_mm} mm, {args.workers} workers)", flush=True)
    if args.workers <= 1:
        results = (_worker(j) for j in jobs)
        for case, meta, err in results:
            (metas if meta else failures).__setitem__(case, meta or err)
            print(f"  {'ok  ' if meta else 'FAIL'} {case}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, j): j[0] for j in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                case, meta, err = fut.result()
                if meta:
                    metas[case] = meta
                    print(f"  [{i}/{len(jobs)}] ok   {case} {meta['shape']}", flush=True)
                else:
                    failures[case] = err
                    print(f"  [{i}/{len(jobs)}] FAIL {case}\n{err}", flush=True)

    manifest_path = args.out / "manifest.json"
    manifest = {
        "spacing": list(args.spacing),
        "margin_mm": args.margin_mm,
        "cases": [metas[c] for c in sorted(metas)],
        "failed": sorted(failures),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest_path} ({len(metas)} cases, {len(failures)} failed)")

    if metas:
        print("\nper-class presence over prepared cases:")
        for i in range(1, NUM_LABELS + 1):
            name = ID_TO_NAME[i]
            counts = [m["label_counts"][name] for m in metas.values()]
            present = [c for c in counts if c > 0]
            med = int(np.median(present)) if present else 0
            print(f"  {i:2d} {name:36s} {len(present):2d}/{len(metas)}  median {med:>9d} voxels")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
