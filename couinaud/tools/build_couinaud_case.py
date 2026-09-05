#!/usr/bin/env python
"""
Reconstruct correctly-aligned per-class label masks for one Couinaud case,
on the same grid as the existing `data/0_test_nifti/imagesTs/<case>.nii.gz`
CT volume -- no registration against a third-party model's output.

Background: `continuity/data/0_test_nifti/<case>/class*.nii.gz` carries a
broken affine (identity rotation, see `continuity/couinaud/FINETUNING.md`
Sec. 0 / `anatomy_frame.py`). This script instead voxelizes the ORIGINAL
annotation meshes (`data/0_test_glb/<case>/class*.glb`, glTF binary, no
scene transform) directly onto `imagesTs`'s own grid, using ONLY `imagesTs`'s
own affine as the geometric source of truth end to end -- no DICOM read at
all. (An earlier version of this script cross-checked/derived geometry from
the case's raw DICOM `ImagePositionPatient`/`Rows`/`PixelSpacing`; on cases
with an ambiguous multi-series study or a non-standard file layout that
silently picked a *different* series than whatever `imagesTs` was actually
built from, producing believable-looking but wrong placement. `imagesTs` is
self-consistent by construction, so reading geometry only from its own
affine removes that whole failure mode.)

The GLB vertex -> real (LPS) space mapping (reverse-engineered and verified
against hard anatomical facts independent of any label -- IVC right of
aorta, portal vein anterior to IVC, gallbladder anterior to aorta,
right-anterior pedicle anterior to right-posterior pedicle; also verified
that the per-class meshes nest inside the classLiver mesh's own extent,
which they should not do if the transform were wrong):

    X_LPS = -T_R + a1                             # T_R = imagesTs.affine[R-dominant axis, 3]
    Y_LPS = -T_A - a2                              # T_A = imagesTs.affine[A-dominant axis, 3]; row axis stored flipped
    Z_LPS = a0 + z_shift                           # see calibration below

(`T_R`/`T_A` reproduce an independently-DICOM-derived `IPP_X` and
`IPP_Y+(Rows-1)*rowSpacing` to <1e-3mm on every case checked -- imagesTs's
own affine already carries exactly this information, so there is nothing
left for a separate DICOM read to add.)

`a0` is *not* directly the source slice's absolute Z (verified: placing it
directly lands classLiver on pelvis-level CT slices). What is reliable is
the mesh's own internal Z-profile shape -- voxel count per cross-section
rises sharply from one tip to a single widest cross-section, then tapers
more slowly to the other tip, the shape of a real liver's craniocaudal
profile. `z_shift` is calibrated per case by matching that mesh-internal
peak against the widest liver-plausible cross-section actually visible in
the CT (largest right-upper-quadrant soft-tissue connected component,
searched across the whole slice range). This anchors the region that
matters most for this project -- the porta hepatis / pedicle-bearing
mid-liver -- even though the mesh's own far superior/inferior extent can
still disagree with the image by some margin (flagged, not hidden, by the
per-case QA outputs this script writes).

Usage:
    python build_couinaud_case.py --list
    python build_couinaud_case.py RMV2026_0022_CT_UTV_VEP_N -v
    python build_couinaud_case.py --all -o /path/to/out_dir
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required: pip install nibabel")
try:
    import trimesh
except ImportError:
    sys.exit("trimesh is required: pip install trimesh shapely scikit-image scipy")
from skimage.draw import polygon as sk_polygon
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[3]
GLB_ROOT = REPO_ROOT / "data" / "0_test_glb"
IMAGES_TS = REPO_ROOT / "data" / "0_test_nifti" / "imagesTs"

ALL_CLASSES = [
    "classLiver", "classVenaPorta", "classIVC",
    "classPediculoPortalIzquierdo", "classPediculoPortalDerecho",
    "classPediculoPortalAnteriorDerecho", "classPediculoPortalPosteriorDerecho",
    "classVenaHepaticaMedia", "classVenaHepaticaDerecha", "classVenaHepaticaIzquierda",
    "classAorta", "classGallbladder",
    "classArteriaHepatica", "classBloodVessels", "classTumour",
]

# Gating checks: X/Y placement between well-separated, large structures --
# these have wide margins and have proven robust across every case tried.
# A failure here means the case's placement is actually wrong.
ANATOMICAL_CHECKS = [
    ("classIVC", "classAorta", 0, "lt", "IVC right of aorta"),
    ("classVenaPorta", "classIVC", 1, "lt", "portal vein anterior to IVC"),
    ("classGallbladder", "classAorta", 1, "lt", "gallbladder anterior to aorta"),
    ("classPediculoPortalAnteriorDerecho", "classPediculoPortalPosteriorDerecho", 1, "lt",
     "right-anterior pedicle anterior to right-posterior pedicle"),
]

# Diagnostic only, not gating: hepatic veins and portal pedicles sit close
# together in the S-I direction (tens of mm), inside the margin of the
# peak-anchored Z-calibration's own known residual error (~20mm, see
# calibrate_z_shift's docstring/module notes) -- this check is informative
# about calibration quality but not reliable enough to exclude a case on.
DIAGNOSTIC_CHECKS = [
    ("classVenaHepaticaMedia", "classPediculoPortalIzquierdo", 2, "gt",
     "hepatic vein superior to portal pedicle"),
]


# ---------------------------------------------------------------------------
# Geometry -- read entirely from imagesTs's own affine. Earlier versions of
# this script re-scanned the case's raw DICOM for ImagePositionPatient/Rows/
# PixelSpacing, which silently picks a *different* series than whatever
# imagesTs was actually built from on some cases (ambiguous multi-series
# studies, non-standard folder layouts) -- producing believable-looking but
# wrong geometry. imagesTs is the single source of truth end to end instead:
# for whichever array axis is dominantly R (resp. A), the mesh's raw a1
# (resp. a2) coordinate maps to real space via that axis's own affine
# translation term alone -- no separate IPP/Rows/PixelSpacing needed. See
# `mesh_to_grid_index` for the derivation.
# ---------------------------------------------------------------------------

def load_target_grid(case_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    """Returns (volume, affine, inv_affine, slice_axis, geo) for imagesTs."""
    img = nib.load(str(IMAGES_TS / f"{case_name}.nii.gz"))
    vol = np.asarray(img.dataobj)
    affine = img.affine
    inv_affine = np.linalg.inv(affine)
    r_axis = int(np.argmax(np.abs(affine[0, :3])))
    a_axis = int(np.argmax(np.abs(affine[1, :3])))
    slice_axis = int(np.argmax(np.abs(affine[2, :3])))
    assert len({r_axis, a_axis, slice_axis}) == 3, f"{case_name}: affine is not axis-aligned"
    geo = {"t_r": float(affine[0, 3]), "t_a": float(affine[1, 3])}
    return vol, affine, inv_affine, slice_axis, geo


# ---------------------------------------------------------------------------
# GLB mesh -> real LPS/RAS mm, then -> imagesTs array index space
# ---------------------------------------------------------------------------

def load_mesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(list(scene.geometry.values()))
    return scene


def mesh_to_grid_index(mesh: trimesh.Trimesh, geo: dict, inv_affine: np.ndarray,
                        z_shift: float) -> trimesh.Trimesh:
    """a1 is column_index*colSpacing (direct); a2 is (Rows-1-row_index)*rowSpacing
    (flipped -- see module docstring). In terms of imagesTs's own affine:
    X_LPS = -T_R + a1, Y_LPS = -T_A - a2, where T_R/T_A are imagesTs's own
    affine translation on its R-dominant/A-dominant axes -- verified exactly
    against an independent DICOM read (T_R/T_A reproduce IPP_X and
    IPP_Y+(Rows-1)*rowSpacing to 1e-3mm)."""
    a0, a1, a2 = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    x_lps = -geo["t_r"] + a1
    y_lps = -geo["t_a"] - a2
    z_lps = a0 + z_shift
    x_ras, y_ras, z_ras = -x_lps, -y_lps, z_lps
    pts = np.stack([x_ras, y_ras, z_ras, np.ones_like(x_ras)], axis=1)
    idx = (inv_affine @ pts.T).T[:, :3]
    out = mesh.copy()
    out.vertices = idx
    return out


def voxelize_mask(mesh_idx: trimesh.Trimesh, shape: tuple[int, int, int], slice_axis: int) -> np.ndarray:
    """Per-slice contour rasterization along `slice_axis`."""
    other = [a for a in range(3) if a != slice_axis]
    sl_coords = mesh_idx.vertices[:, slice_axis]
    n = shape[slice_axis]
    kmin = max(0, int(np.floor(sl_coords.min())))
    kmax = min(n - 1, int(np.ceil(sl_coords.max())))
    out = np.zeros(shape, dtype=bool)
    if kmax < kmin:
        return out
    normal = np.zeros(3); normal[slice_axis] = 1.0
    origin = np.zeros(3)
    heights = np.arange(kmin, kmax + 1, dtype=float)
    sections = mesh_idx.section_multiplane(plane_origin=origin, plane_normal=normal, heights=heights)
    out_shape_2d = (shape[other[0]], shape[other[1]])
    for offset, sec in enumerate(sections):
        if sec is None:
            continue
        k = kmin + offset
        for poly in sec.polygons_full:
            xs, ys = poly.exterior.coords.xy  # xs -> other[0], ys -> other[1]
            rr, cc = sk_polygon(np.asarray(xs), np.asarray(ys), shape=out_shape_2d)
            idx3 = [None, None, None]
            idx3[other[0]] = rr; idx3[other[1]] = cc; idx3[slice_axis] = k
            out[tuple(idx3)] = True
            for interior in poly.interiors:
                xs2, ys2 = interior.coords.xy
                rr2, cc2 = sk_polygon(np.asarray(xs2), np.asarray(ys2), shape=out_shape_2d)
                idx3b = [None, None, None]
                idx3b[other[0]] = rr2; idx3b[other[1]] = cc2; idx3b[slice_axis] = k
                out[tuple(idx3b)] = False
    return out


# ---------------------------------------------------------------------------
# Z calibration: anchor the mesh's own widest cross-section on the liver's
# widest cross-section actually visible in the CT.
# ---------------------------------------------------------------------------

def _mesh_area_profile(mesh: trimesh.Trimesh, n_samples: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional area vs. raw a0 height, computed directly from the
    mesh's own geometry -- no dependency on any target grid or its bounds,
    so this works even when a0's raw range currently falls entirely outside
    imagesTs's valid index range (only the *shift* fixes that)."""
    a0 = mesh.vertices[:, 0]
    heights = np.linspace(a0.min() + 1e-3, a0.max() - 1e-3, n_samples)
    areas = np.zeros(n_samples)
    for i, h in enumerate(heights):
        sec = mesh.section(plane_origin=[h, 0, 0], plane_normal=[1, 0, 0])
        if sec is None:
            continue
        try:
            path2d, _ = sec.to_2D()
            areas[i] = float(path2d.area)
        except Exception:
            continue
    return heights, areas


def _footprint_from_mesh(mesh: trimesh.Trimesh, geo: dict, inv_affine: np.ndarray,
                          other: list[int], shape: tuple[int, int, int], pad: int = 40) -> tuple[slice, slice]:
    """(other-axis-0, other-axis-1) index bounds of `mesh`'s footprint,
    computed directly from its (a1,a2) extent via the X/Y transform -- valid
    regardless of the (still unknown) Z shift, since the affine is
    axis-aligned and the Z component doesn't couple into the other two rows."""
    a1, a2 = mesh.vertices[:, 1], mesh.vertices[:, 2]
    x_lps = -geo["t_r"] + a1
    y_lps = -geo["t_a"] - a2
    pts = np.stack([-x_lps, -y_lps, np.zeros_like(x_lps), np.ones_like(x_lps)], axis=1)
    idx = (inv_affine @ pts.T).T[:, :3]
    lo0, hi0 = max(0, int(idx[:, other[0]].min()) - pad), min(shape[other[0]], int(idx[:, other[0]].max()) + pad)
    lo1, hi1 = max(0, int(idx[:, other[1]].min()) - pad), min(shape[other[1]], int(idx[:, other[1]].max()) + pad)
    return slice(lo0, hi0), slice(lo1, hi1)


def calibrate_z_shift(liver_mesh: trimesh.Trimesh, geo: dict, vol: np.ndarray,
                       affine: np.ndarray, inv_affine: np.ndarray, slice_axis: int) -> tuple[float, dict]:
    other = [a for a in range(3) if a != slice_axis]

    heights, areas = _mesh_area_profile(liver_mesh)
    if areas.max() == 0:
        return 0.0, {"ok": False, "reason": "degenerate liver mesh (no cross-sectional area)"}
    a0_peak = float(heights[int(np.argmax(areas))])

    sl0, sl1 = _footprint_from_mesh(liver_mesh, geo, inv_affine, other, vol.shape)

    window = (vol > -20) & (vol < 120)
    n = vol.shape[slice_axis]
    # Restrict the search to the middle of the volume: this cohort's scans
    # run from thigh/pelvis through chest/neck, and a large right-upper-
    # quadrant soft-tissue blob can otherwise be spuriously matched against
    # thigh/pelvic muscle near the extremes.
    k_lo, k_hi = int(0.25 * n), int(0.85 * n)
    true_sig = np.zeros(n)
    for k in range(k_lo, k_hi):
        idx = [slice(None), slice(None), slice(None)]
        idx[slice_axis] = k
        idx[other[0]] = sl0
        idx[other[1]] = sl1
        sl = window[tuple(idx)]
        if sl.sum() < 100:
            continue
        lbl, ncomp = ndimage.label(sl)
        if ncomp == 0:
            continue
        sizes = ndimage.sum(sl, lbl, range(1, ncomp + 1))
        true_sig[k] = sizes.max()
    if true_sig.max() == 0:
        return 0.0, {"ok": False, "reason": "no plausible liver cross-section found in image"}
    k_true_peak = int(np.argmax(true_sig))

    # Z_LPS at k_true_peak, straight from imagesTs's own (axis-aligned)
    # affine -- Z_RAS == Z_LPS, sign is unaffected by the LPS/RAS flip.
    z_lps_true_peak = float(affine[2, slice_axis] * k_true_peak + affine[2, 3])
    z_shift_mm = z_lps_true_peak - a0_peak

    info = {"ok": True, "a0_peak": a0_peak, "k_true_peak": k_true_peak,
            "z_lps_true_peak": z_lps_true_peak, "z_shift_mm": z_shift_mm,
            "true_peak_area": float(true_sig[k_true_peak])}
    return z_shift_mm, info


# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

def centroid_lps(mesh: trimesh.Trimesh, geo: dict, z_shift: float) -> np.ndarray:
    """[X_LPS, Y_LPS, Z_LPS]: +X=Left, +Y=Posterior, +Z=Superior."""
    a0, a1, a2 = mesh.vertices[:, 0].mean(), mesh.vertices[:, 1].mean(), mesh.vertices[:, 2].mean()
    x_lps = -geo["t_r"] + a1
    y_lps = -geo["t_a"] - a2
    z_lps = a0 + z_shift
    return np.array([x_lps, y_lps, z_lps])


def run_checks(meshes: dict[str, trimesh.Trimesh], geo: dict, z_shift: float, checks: list) -> list[dict]:
    results = []
    for cls_a, cls_b, axis, op, label in checks:
        if cls_a not in meshes or cls_b not in meshes:
            continue
        va = centroid_lps(meshes[cls_a], geo, z_shift)[axis]
        vb = centroid_lps(meshes[cls_b], geo, z_shift)[axis]
        ok = (va < vb) if op == "lt" else (va > vb)
        results.append({"check": label, "a": cls_a, "b": cls_b, "va": float(va), "vb": float(vb), "passed": bool(ok)})
    return results


def largest_component_fraction(mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return 0.0
    lbl, n = ndimage.label(mask)
    if n <= 1:
        return 1.0
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    return float(sizes.max() / mask.sum())


def save_qa_overlay(vol: np.ndarray, masks: dict[str, np.ndarray], slice_axis: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    liver = masks.get("classLiver")
    if liver is None or liver.sum() == 0:
        return
    idx = np.argwhere(liver)
    centre = [int(idx[:, a].mean()) for a in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    window = (-175, 250)
    for ax, fixed_axis in zip(axes, range(3)):
        sl = [slice(None)] * 3
        sl[fixed_axis] = centre[fixed_axis]
        img2d = np.clip(vol[tuple(sl)], *window)
        ax.imshow(img2d, cmap="gray", vmin=window[0], vmax=window[1])
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(masks), 2)))
        for color, (cls, m) in zip(colors, masks.items()):
            m2d = m[tuple(sl)]
            if m2d.any():
                ax.contour(m2d, levels=[0.5], colors=[color], linewidths=0.8)
        ax.set_title(f"axis{fixed_axis} fixed @ {centre[fixed_axis]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build_case(case_name: str, out_dir: Path, verbose: bool = False) -> dict:
    glb_dir = GLB_ROOT / case_name
    t0 = time.time()

    vol, affine, inv_affine, slice_axis, geo = load_target_grid(case_name)
    if verbose:
        print(f"  imagesTs volume {vol.shape}, slice_axis={slice_axis}")

    liver_path = glb_dir / "classLiver.glb"
    if not liver_path.exists():
        raise RuntimeError("no classLiver.glb -- cannot calibrate")
    liver_mesh = load_mesh(liver_path)
    z_shift, calib_info = calibrate_z_shift(liver_mesh, geo, vol, affine, inv_affine, slice_axis)
    if verbose:
        print(f"  z-calibration: {calib_info}")

    meshes: dict[str, trimesh.Trimesh] = {}
    masks: dict[str, np.ndarray] = {}
    for cls in ALL_CLASSES:
        p = glb_dir / f"{cls}.glb"
        if not p.exists():
            continue
        mesh = load_mesh(p)
        meshes[cls] = mesh
        mesh_idx = mesh_to_grid_index(mesh, geo, inv_affine, z_shift)
        masks[cls] = voxelize_mask(mesh_idx, vol.shape, slice_axis)
        if verbose:
            frac = largest_component_fraction(masks[cls])
            print(f"    {cls:38s} n_voxels={masks[cls].sum():8d}  largest_cc_frac={frac:.2f}")

    checks = run_checks(meshes, geo, z_shift, ANATOMICAL_CHECKS)
    diagnostics = run_checks(meshes, geo, z_shift, DIAGNOSTIC_CHECKS)
    n_pass = sum(c["passed"] for c in checks)
    passed = n_pass == len(checks) and len(checks) > 0 and calib_info.get("ok", False)

    case_out = out_dir / case_name
    case_out.mkdir(parents=True, exist_ok=True)
    for cls, m in masks.items():
        nib.save(nib.Nifti1Image(m.astype(np.uint8), affine), case_out / f"{cls}.nii.gz")
    save_qa_overlay(vol, masks, slice_axis, case_out / "qa_overlay.png")

    report = {
        "case": case_name,
        "classes_found": sorted(masks.keys()),
        "z_calibration": calib_info,
        "anatomical_checks": checks, "checks_passed": passed,
        "diagnostic_checks": diagnostics,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (case_out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="?", help="single case name")
    ap.add_argument("--all", action="store_true", help="process every RMV* case with a GLB dir")
    ap.add_argument("--list", action="store_true", help="list available cases and exit")
    ap.add_argument("-o", "--out", type=Path, default=REPO_ROOT / "data" / "couinaud_nifti")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cases = sorted(p.name for p in GLB_ROOT.glob("RMV*")
                   if p.is_dir() and (IMAGES_TS / f"{p.name}.nii.gz").exists())
    if args.list:
        for c in cases:
            print(c)
        return 0

    if args.all:
        targets = cases
    elif args.case:
        targets = [args.case]
    else:
        ap.error("pass a case name, --all, or --list")
        return 2

    reports = []
    for c in targets:
        print(f"[{c}]")
        try:
            r = build_case(c, args.out, verbose=args.verbose)
        except Exception as e:
            print(f"  FAILED: {e}")
            reports.append({"case": c, "error": str(e)})
            continue
        status = "OK" if r["checks_passed"] else "CHECKS FAILED"
        print(f"  {status}  ({r['elapsed_s']}s, {len(r['classes_found'])} classes, "
              f"{sum(c['passed'] for c in r['anatomical_checks'])}/{len(r['anatomical_checks'])} checks, "
              f"z_shift={r['z_calibration'].get('z_shift_mm', 'n/a')})")
        reports.append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "build_report.json").write_text(json.dumps(reports, indent=2))
    n_ok = sum(1 for r in reports if r.get("checks_passed"))
    print(f"\n{n_ok}/{len(reports)} cases passed all checks -> {args.out}/build_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
