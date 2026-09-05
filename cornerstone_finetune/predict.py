#!/usr/bin/env python3
"""Run a trained checkpoint over subjects and write per-class masks.

Output layout is the one the Couinaud pipeline consumes -- one binary
``class*.nii.gz`` per class per subject -- so a run can be scored end to end
with ``continuity/couinaud/run_couinaud.py`` rather than only by Dice, which
finetuning.md Sec. 3 argues is the wrong target here in the first place.

    python cornerstone_finetune/predict.py \
        --checkpoint runs/cornerstone/core6/checkpoints/last.ckpt \
        --split test --out runs/cornerstone/core6/predictions

Predictions are resampled back onto the *original* CT grid and written with
that volume's own affine (finetuning.md Sec. 4: "write a correct affine on
export" -- every mask in the source dataset declares R,A,S and almost none of
them are). Pass ``--grid prepared`` to keep them on the 1 mm crop instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent

from labels import get_channels  # noqa: E402


def channel_to_filename(name: str) -> str:
    """``portal_tree`` -> ``classPortalTree`` -- the pipeline globs ``class*``."""
    return "class" + "".join(part.capitalize() for part in name.split("_"))


def restore_to_source(prob: np.ndarray, meta: dict, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Put a prediction back on the original CT voxel grid.

    Inverts prepare_data.py exactly: resample the crop to the source spacing,
    paste it into a full-size volume at the recorded crop origin, then undo the
    RAS reorientation. Thresholding happens on the resampled probabilities, not
    before, so a structure two voxels wide does not vanish into rounding.
    """
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
    from scipy.ndimage import zoom

    source_affine = np.asarray(meta["source_affine"])
    source_shape = tuple(meta["source_shape"])
    ras_ornt = ornt_transform(io_orientation(source_affine), axcodes2ornt(("R", "A", "S")))
    ras_shape = list(source_shape)
    for axis, (to_axis, _) in enumerate(ras_ornt):
        ras_shape[int(to_axis)] = source_shape[axis]

    lo = np.asarray(meta["ras_crop_origin"])
    hi = np.asarray(meta["ras_crop_end"])
    target_shape = tuple(int(h - l) for l, h in zip(lo, hi))
    factors = np.asarray(target_shape) / np.asarray(prob.shape)
    resampled = zoom(prob, factors, order=1, mode="nearest") if not np.allclose(factors, 1.0) else prob
    # zoom's output shape can round a voxel away from the request
    pad = [(0, max(0, t - s)) for t, s in zip(target_shape, resampled.shape)]
    resampled = np.pad(resampled, pad)[: target_shape[0], : target_shape[1], : target_shape[2]]

    full = np.zeros(ras_shape, dtype=np.uint8)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (resampled > threshold).astype(np.uint8)
    inverse = ornt_transform(axcodes2ornt(("R", "A", "S")), io_orientation(source_affine))
    return apply_orientation(full, inverse), source_affine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None,
                    help="defaults to resolved_config.yaml beside the checkpoint's run dir")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", choices=["source", "prepared"], default="source")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save-probs", action="store_true", help="also write float32 probability maps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import yaml
    from datamodule import CornerstoneDataModule
    from module import CornerstoneSegModule

    try:
        from monai.data import MetaTensor

        torch.serialization.add_safe_globals([MetaTensor])
    except Exception:  # pragma: no cover
        pass

    run_dir = args.checkpoint.resolve().parent.parent
    config_path = Path(args.config or (run_dir / "resolved_config.yaml"))
    if not config_path.exists():
        raise SystemExit(
            f"no config at {config_path}. train.py writes resolved_config.yaml into the run "
            f"directory; pass --config explicitly if the checkpoint was moved."
        )
    cfg = yaml.safe_load(config_path.read_text())

    data_cfg = dict(cfg.get("data", {}))
    data_cfg.setdefault("prepared_root", str(REPO / "data" / "cornerstone_prepared"))
    data_cfg.setdefault("seed", cfg.get("seed", 42))
    data_cfg["num_workers"] = 0
    dm = CornerstoneDataModule(**data_cfg)

    model = CornerstoneSegModule.load_from_checkpoint(str(args.checkpoint), map_location="cpu")
    model.eval().to(args.device)
    names = model.channel_names
    channels = get_channels(dm.hparams.channel_preset)
    print(f"[predict] {args.checkpoint}\n[predict] channels: {names}")

    train, val, test = dm.splits()
    pool = {"train": train, "val": val, "test": test, "all": train + val + test}[args.split]
    if args.case:
        wanted = set(args.case)
        pool = [r for r in pool if r["case"] in wanted]
    if not pool:
        print("no cases selected", file=sys.stderr)
        return 2

    manifest = json.loads((Path(dm.hparams.prepared_root) / "manifest.json").read_text())
    metas = {m["case"]: m for m in manifest["cases"]}
    transforms = dm._eval_transforms()
    args.out.mkdir(parents=True, exist_ok=True)

    for record in pool:
        case = record["case"]
        sample = transforms(dict(record))
        image = torch.as_tensor(np.asarray(sample["image"]))[None].to(args.device)
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16,
                                             enabled=args.device == "cuda"):
            probs = torch.sigmoid(model._infer(image).float())[0].cpu().numpy()

        case_out = args.out / case
        case_out.mkdir(parents=True, exist_ok=True)
        meta = metas[case]
        for i, name in enumerate(names):
            fname = channel_to_filename(name)
            if args.grid == "prepared":
                data = (probs[i] > args.threshold).astype(np.uint8)
                affine = np.asarray(meta["affine"])
            else:
                data, affine = restore_to_source(probs[i], meta, args.threshold)
            nib.save(nib.Nifti1Image(data, affine), str(case_out / f"{fname}.nii.gz"))
            if args.save_probs:
                nib.save(nib.Nifti1Image(probs[i].astype(np.float32), np.asarray(meta["affine"])),
                         str(case_out / f"{fname}_prob.nii.gz"))
        counts = {n: int((probs[i] > args.threshold).sum()) for i, n in enumerate(names)}
        print(f"  {case}: " + ", ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

    print(f"\nwrote {len(pool)} cases to {args.out}")
    print("score end-to-end with:  python continuity/couinaud/run_couinaud.py --all --no-figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
