#!/usr/bin/env python
"""
Inventory a DICOM archive: one row per series, so you can decide which series each
mask case was drawn on.

The archive is a DICOM File-Set -- hashed filenames, no extensions, a DICOMDIR at
the root of each case. Nothing in a filename tells you what a file is, so every
file has to be opened. That is the whole reason this script exists.

    python dicom_inventory.py /mnt/disco4t/removirt_test_repo -o series.csv
    python dicom_inventory.py /mnt/disco4t/removirt_test_repo --case RMV2025_0001_CT_NCT_HBP_X -v

Why series selection matters here. The case folder names encode modality and
contrast phase -- `RMV2025_0001_CT_NCT_HBP_X` is CT, non-contrast plus
hepatobiliary phase; `RMV2025_0011_CT_UTV_VEP_S` is venous phase; `RMV2025_0002_MR_
NCT_ARP_N` is MR, arterial. A study holds several series, and the portal pedicles
are only conspicuous in the portal-venous phase. Picking the wrong series gives you
an image in which the structure you are training to find is invisible, and the
network will happily learn to hallucinate it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    sys.exit("pydicom is required:  pip install pydicom")

# Tags worth reading. Kept short: reading every tag of every file over a slow mount
# is the difference between a minute and an hour.
FIELDS = ["SeriesInstanceUID", "SeriesNumber", "SeriesDescription", "Modality",
          "StudyInstanceUID", "StudyDate", "ProtocolName", "ContrastBolusAgent",
          "AcquisitionTime", "ImageOrientationPatient", "ImagePositionPatient",
          "PixelSpacing", "SliceThickness", "Rows", "Columns", "ConvolutionKernel",
          "PatientID"]


def read_header(path: Path):
    """Header only. `stop_before_pixels` is what makes scanning 900 files bearable."""
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except (InvalidDicomError, OSError, AttributeError):
        return None


def scan_case(case_dir: Path, verbose: bool = False) -> list[dict]:
    series: dict[str, dict] = {}
    positions: dict[str, list] = defaultdict(list)
    n_files = n_read = 0

    for path in sorted(case_dir.rglob("*")):
        if not path.is_file() or path.name.upper() == "DICOMDIR":
            continue
        n_files += 1
        ds = read_header(path)
        if ds is None or not hasattr(ds, "SeriesInstanceUID"):
            continue
        n_read += 1
        uid = str(ds.SeriesInstanceUID)
        if uid not in series:
            series[uid] = {f: getattr(ds, f, "") for f in FIELDS}
            series[uid]["case"] = case_dir.name
            series[uid]["n_slices"] = 0
            series[uid]["example_file"] = str(path.relative_to(case_dir))
        series[uid]["n_slices"] += 1
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            positions[uid].append(float(ipp[2]))

    rows = []
    for uid, s in series.items():
        z = sorted(positions[uid])
        # Derive the real slice spacing from the positions rather than trusting
        # SliceThickness, which is the reconstructed thickness and can differ from
        # the actual increment when slices overlap.
        if len(z) > 1:
            gaps = [round(b - a, 3) for a, b in zip(z, z[1:])]
            s["z_spacing"] = min(gaps, key=lambda g: abs(g - (z[-1] - z[0]) / (len(z) - 1)))
            s["z_extent_mm"] = round(z[-1] - z[0], 1)
            s["z_irregular"] = len(set(gaps)) > 1
        else:
            s["z_spacing"] = s["z_extent_mm"] = ""
            s["z_irregular"] = ""
        ps = s.get("PixelSpacing") or ["", ""]
        s["xy_spacing"] = f"{float(ps[0]):.3f}" if ps[0] != "" else ""
        s["ImageOrientationPatient"] = ("axial" if _is_axial(s["ImageOrientationPatient"])
                                        else str(s["ImageOrientationPatient"]))
        s.pop("ImagePositionPatient", None)
        rows.append(s)
    rows.sort(key=lambda r: (str(r.get("SeriesNumber") or ""), r["SeriesInstanceUID"]))
    if verbose:
        print(f"  {case_dir.name}: {n_read}/{n_files} files parsed, "
              f"{len(rows)} series", file=sys.stderr)
    return rows


def _is_axial(iop) -> bool:
    try:
        v = [float(x) for x in iop]
    except (TypeError, ValueError):
        return False
    return len(v) == 6 and abs(abs(v[0]) - 1) < 1e-3 and abs(abs(v[4]) - 1) < 1e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory holding the RMV* case folders")
    ap.add_argument("-o", "--out", type=Path, default=Path("series_inventory.csv"))
    ap.add_argument("--case", help="only this case")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cases = ([args.root / args.case] if args.case
             else sorted(p for p in args.root.glob("RMV*") if p.is_dir()))
    if not cases:
        return print(f"no RMV* case folders under {args.root}", file=sys.stderr) or 2

    all_rows = []
    for c in cases:
        if not c.is_dir():
            print(f"  ! {c} is not a directory", file=sys.stderr)
            continue
        all_rows += scan_case(c, args.verbose)

    cols = (["case", "SeriesNumber", "Modality", "SeriesDescription", "n_slices",
             "xy_spacing", "z_spacing", "z_extent_mm", "z_irregular", "Rows", "Columns",
             "ImageOrientationPatient", "ContrastBolusAgent", "AcquisitionTime",
             "ProtocolName", "ConvolutionKernel", "StudyDate", "PatientID",
             "SeriesInstanceUID", "StudyInstanceUID", "example_file"])
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"{len(all_rows)} series across {len(cases)} case(s) -> {args.out}\n")
    print(f"{'case':<30}{'#':>3} {'mod':<4}{'slices':>7}{'xy':>7}{'dz':>7}  description")
    for r in all_rows:
        print(f"{r['case'][:29]:<30}{str(r.get('SeriesNumber') or '?'):>3} "
              f"{str(r['Modality']):<4}{r['n_slices']:>7}{str(r['xy_spacing']):>7}"
              f"{str(r['z_spacing']):>7}  {str(r['SeriesDescription'])[:40]}")

    thin = [r for r in all_rows if r["n_slices"] < 20]
    if thin:
        print(f"\n{len(thin)} series have <20 slices -- scouts, dose reports or "
              f"localisers. Not training data.")
    irreg = [r for r in all_rows if r.get("z_irregular") is True]
    if irreg:
        print(f"{len(irreg)} series have irregular slice spacing -- check these "
              f"before converting; a naive stack will be geometrically wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
