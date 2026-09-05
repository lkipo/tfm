#!/usr/bin/env python
"""
Run the Couinaud pipeline on one case or on every case in the dataset.

    python run_couinaud.py                          # one random usable case
    python run_couinaud.py RMV2025_0001_CT_NCT_HBP_X
    python run_couinaud.py --all                    # every usable case + batch summary
    python run_couinaud.py --list                   # which cases can and cannot run

Figures and a JSON summary land in `out/`. `--nifti` additionally writes the label
volume next to them so it can be opened in a viewer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import nibabel as nib

from couinaud import (SEG_DESCRIPTION, SEG_IDS, SEG_NAMES, Config, load_case,
                      missing_labels, segment_liver)
from checks import run_checks, _dice
import viz

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data" / "0_test_nifti"
OUT = HERE / "out"


def usable_cases(root: Path = DATA_ROOT) -> list[Path]:
    return [c for c in sorted(root.glob("RMV*")) if c.is_dir() and not missing_labels(c)]


def coverage_bias(summaries: list[dict]) -> dict:
    """
    Quantify the nearest-seed assignment's dependence on how completely each
    pedicle was traced.

    If the method were unbiased, how many skeleton points a pedicle happens to have
    would say nothing about how much parenchyma it wins. Comparing the two right
    pedicles inside each case controls for liver size and for the frame: the only
    thing that varies is which of the two was labelled further into the periphery.
    A strong correlation is direct evidence that label completeness, not anatomy,
    is driving part of the result.
    """
    import re
    ratios, volumes, names = [], [], []
    for s in summaries:
        note = next((n for n in s["notes"] if n.startswith("pedicle skeletons:")), None)
        if note is None:
            continue
        counts = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", note)}
        ant, post = counts.get("right_ant", 0), counts.get("right_post", 0)
        if ant < 5 or post < 5:
            continue
        pct = s["volume_pct"]
        anterior = pct["5"] + pct["8"]
        posterior = pct["6"] + pct["7"]
        if anterior <= 0 or posterior <= 0:
            continue
        ratios.append(post / ant)
        volumes.append(posterior / anterior)
        names.append(s["case"])

    if len(ratios) < 5:
        return {"n": len(ratios), "pearson_log": None}
    r = float(np.corrcoef(np.log(ratios), np.log(volumes))[0, 1])
    return {
        "n": len(ratios),
        "pearson_log": r,
        "seed_ratio_median": float(np.median(ratios)),
        "volume_ratio_median": float(np.median(volumes)),
        # Leelaudomlipi means: (VI + VII) / (V + VIII)
        "volume_ratio_reference": (8.4 + 14.9) / (10.6 + 17.6),
        "cases": names,
    }


def _jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def process(case_dir: Path, cfg: Config, out_dir: Path, figures: bool = True,
            write_nifti: bool = False, verbose: bool = True) -> dict:
    """
    Run both assignment modes on one case.

    Both are always computed: the Voronoi labelling is the only one that can be
    cross-validated against the hepatic vein planes without circularity, and the
    agreement between the two modes is itself a useful stability signal.
    """
    t0 = time.time()
    case = load_case(case_dir)
    if verbose:
        print(f"[{case.name}] shape={case.shape} zooms={np.round(case.zooms, 2).tolist()}")
        print(f"  header axcodes {nib.aff2axcodes(case.affine)} (nominal); "
              f"recovered frame {case.frame.describe()}")
        for w in case.frame.warnings:
            print(f"  ! {w}")

    reference = segment_liver(case, replace(cfg, mode="voronoi"), verbose=verbose)
    if cfg.mode == "voronoi":
        res, other = reference, segment_liver(case, replace(cfg, mode="hybrid"),
                                              verbose=False)
    else:
        res, other = segment_liver(case, cfg, verbose=verbose), reference

    result = run_checks(case, res, reference=reference)
    both = (res.labels > 0)
    agreement = float(np.count_nonzero((res.labels == other.labels) & both) /
                      max(np.count_nonzero(both), 1))
    per_segment_dice = {str(s): _dice(res.labels == s, other.labels == s) for s in SEG_IDS}
    result["metrics"]["mode_agreement"] = agreement

    pct, vols = res.volume_pct(), res.volumes_ml()
    other_pct = other.volume_pct()
    if verbose:
        print()
        print(f"  {'seg':<5}{'name':<26}{'mL':>9}{'%':>7}     {'other mode %':>13}")
        for s in SEG_IDS:
            print(f"  {SEG_NAMES[s]:<5}{SEG_DESCRIPTION[s]:<26}{vols[s]:>9.1f}"
                  f"{pct[s]:>7.1f}{other_pct[s]:>18.1f}")
        print(f"\n  the two modes agree on {agreement:.1%} of liver voxels")
        print()
        for name, ok, detail in result["rows"]:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
        print(f"\n  {result['passed']}/{result['total']} checks passed "
              f"in {time.time() - t0:.1f}s")

    if figures:
        case_out = out_dir / case.name
        case_out.mkdir(parents=True, exist_ok=True)
        viz.figure_frame(case, case_out / "1_orientation.png")
        viz.figure_geometry(case, res, case_out / "2_geometry.png")
        viz.figure_seeds_and_territories(case, res, case_out / "3_seeds_territories.png")
        viz.figure_slices(case, res, case_out / "4_segments.png")
        viz.figure_volumes(case, res, result, case_out / "5_volumes.png")
        vor, hyb = (res, other) if res.mode == "voronoi" else (other, res)
        viz.figure_modes(case, vor, hyb, case_out / "6_modes.png")
        if verbose:
            print(f"  figures -> {case_out}")
        if write_nifti:
            nib.save(nib.Nifti1Image(res.labels, case.affine),
                     str(case_out / f"{case.name}_couinaud.nii.gz"))

    return {
        "case": case.name,
        "mode": res.mode,
        "frame": case.frame.describe(),
        "frame_confidence": case.frame.confidence,
        "frame_warnings": case.frame.warnings,
        "planes": sorted(res.planes),
        "volume_ml": {str(s): vols[s] for s in SEG_IDS},
        "volume_pct": {str(s): pct[s] for s in SEG_IDS},
        "volume_pct_other_mode": {str(s): other_pct[s] for s in SEG_IDS},
        "mode_agreement": agreement,
        "mode_dice_per_segment": per_segment_dice,
        "checks": result,
        "notes": res.notes,
        "seconds": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", help="case folder name under data/0_test_nifti")
    ap.add_argument("--all", action="store_true", help="run every usable case")
    ap.add_argument("--list", action="store_true", help="report which cases can run")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--nifti", action="store_true", help="also write the label volume")
    ap.add_argument("--smooth", type=int, default=0, help="majority-smoothing passes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assign", choices=("euclidean", "geodesic"), default="euclidean",
                    help="metric the seed labels travel under (default: euclidean, "
                         "the straight-line nearest-seed rule)")
    ap.add_argument("--flow-alpha", type=float, default=0.0,
                    help="Murray weighting on the pedicle root radius, w ~ r^(3a); "
                         "needs --assign geodesic")
    ap.add_argument("--vein-barrier", type=float, default=0.0,
                    help="extra cost of crossing a hepatic vein; needs --assign geodesic")
    ap.add_argument("--vein-barrier-mm", type=float, default=12.0,
                    help="distance over which the vein cost ramps up")
    ap.add_argument("--holdout", choices=("", "mhv", "rhv", "lhv"), default="",
                    help="keep this vein out of the barrier so its plane stays "
                         "independent evidence for the cross-validation")
    args = ap.parse_args()

    if not DATA_ROOT.exists():
        print(f"data root {DATA_ROOT} not found", file=sys.stderr)
        return 2

    if args.list:
        all_cases = sorted(c for c in DATA_ROOT.glob("RMV*") if c.is_dir())
        ok = usable_cases()
        print(f"{len(ok)}/{len(all_cases)} cases have every label the pipeline needs\n")
        for c in all_cases:
            miss = missing_labels(c)
            if miss:
                print(f"  skip {c.name}: missing "
                      + ", ".join(m.replace('class', '') for m in miss))
        print()
        for c in ok:
            print(f"  run  {c.name}")
        return 0

    cfg = Config(smooth_iterations=args.smooth, rng_seed=args.seed,
                 assign=args.assign, flow_alpha=args.flow_alpha,
                 vein_barrier=args.vein_barrier, vein_barrier_mm=args.vein_barrier_mm,
                 vein_barrier_holdout=args.holdout)
    OUT.mkdir(exist_ok=True)

    if args.all:
        cases = usable_cases()
        print(f"running {len(cases)} cases\n")
        summaries, failures = [], []
        for i, c in enumerate(cases, 1):
            print(f"--- [{i}/{len(cases)}] {c.name} " + "-" * 30)
            try:
                summaries.append(process(c, cfg, OUT, figures=not args.no_figures,
                                         write_nifti=args.nifti, verbose=True))
            except Exception as exc:                        # noqa: BLE001
                failures.append((c.name, repr(exc)))
                print(f"  !! failed: {exc}")
                traceback.print_exc()
            print()

        if summaries:
            bias = coverage_bias(summaries)
            viz.figure_batch(summaries, OUT / "batch_summary.png", bias)
            payload = {"cases": summaries, "failures": failures, "coverage_bias": bias}
            (OUT / "batch_summary.json").write_text(
                json.dumps(payload, indent=2, default=_jsonable))
            passed = sum(s["checks"]["passed"] for s in summaries)
            total = sum(s["checks"]["total"] for s in summaries)
            print("=" * 62)
            print(f"{len(summaries)} cases succeeded, {len(failures)} failed")
            print(f"{passed}/{total} checks passed overall")
            if bias.get("pearson_log") is not None:
                print(f"coverage bias: the two right pedicles' skeleton-size ratio and "
                      f"their volume ratio correlate at r={bias['pearson_log']:.2f} "
                      f"(log-log, n={bias['n']})")
                print(f"  median volume ratio (VI+VII)/(V+VIII) = "
                      f"{bias['volume_ratio_median']:.2f} against a reference of "
                      f"{bias['volume_ratio_reference']:.2f}")
            print(f"summary -> {OUT / 'batch_summary.png'}")
        return 0 if not failures else 1

    if args.case:
        case_dir = DATA_ROOT / args.case
        if not case_dir.exists():
            print(f"no such case: {case_dir}", file=sys.stderr)
            return 2
        miss = missing_labels(case_dir)
        if miss:
            print(f"warning: {args.case} is missing "
                  + ", ".join(m.replace('class', '') for m in miss))
    else:
        case_dir = random.Random(args.seed).choice(usable_cases())
        print(f"picked {case_dir.name} at random\n")

    summary = process(case_dir, cfg, OUT, figures=not args.no_figures,
                      write_nifti=args.nifti)
    (OUT / f"{summary['case']}.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable))
    return 0 if summary["checks"]["passed"] == summary["checks"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
