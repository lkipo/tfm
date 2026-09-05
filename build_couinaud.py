#!/usr/bin/env python3
"""
Build Couinaud training data: cornerstone 6 classes + GLB sector splits.

Usage:
    python build_couinaud.py              # build all 38 cases
    python build_couinaud.py 0            # build case 0 only (debug)
"""
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import nibabel as nib
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt

ROOT = Path("data")
IMAGES = ROOT / "0_test_nifti/imagesTs"
ANN_DIR = ROOT / "0_test_nifti/cornerstone_annotations"
GLB_ROOT = ROOT / "0_test_glb"
OUT_DIR = ROOT / "couinaud_hybrid"

LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
GAP_SPLIT = 10
MIN_CLUSTER = 0.05
MAX_FILL_GAP = 5


def list_cases():
    return sorted([p.name.replace(".nii.gz", "") for p in IMAGES.glob("*.nii.gz")])


def read_contours(case):
    d = json.loads((ANN_DIR / f"{case}.json").read_text())
    uid2cls = {u: c for c, us in d["classes"].items() for u in us}
    out = defaultdict(list)
    for e in d["state"]:
        cls = uid2cls.get(e["annotationUID"])
        if cls:
            out[cls].append(np.asarray(e["data"]["contour"]["polyline"], float))
    return out


def _fill_slice(polys, shape):
    acc = np.zeros(shape, bool)
    for p in polys:
        img = Image.new("1", (shape[1], shape[0]), 0)
        ImageDraw.Draw(img).polygon([(y, x) for x, y in p], fill=1)
        acc ^= np.asarray(img, bool)
    return acc


def _reject_outliers(vol, per_z):
    zs = np.array(sorted(per_z))
    if len(zs) < 2:
        return zs, []
    clusters = np.split(zs, np.where(np.diff(zs) > GAP_SPLIT)[0] + 1)
    sizes = [int(vol[:, :, cl].sum()) for cl in clusters]
    total = sum(sizes) or 1
    keep, dropped = [], []
    for cl, n in zip(clusters, sizes):
        if n / total >= MIN_CLUSTER:
            keep.append(cl)
        else:
            vol[:, :, cl] = False
            dropped.append((int(cl[0]), int(cl[-1]), n, round(100 * n / total, 2)))
    return (np.concatenate(keep) if keep else np.array([], int)), dropped


def _interpolate(vol, zs, max_gap=MAX_FILL_GAP):
    filled = 0
    for z0, z1 in zip(zs[:-1], zs[1:]):
        if z1 - z0 <= 1 or z1 - z0 - 1 > max_gap:
            continue
        d0 = distance_transform_edt(~vol[:, :, z0]) - distance_transform_edt(vol[:, :, z0])
        d1 = distance_transform_edt(~vol[:, :, z1]) - distance_transform_edt(vol[:, :, z1])
        for z in range(z0 + 1, z1):
            a = (z - z0) / (z1 - z0)
            vol[:, :, z] = ((1 - a) * d0 + a * d1) < 0
            filled += 1
    return filled


def rasterize_contours(case):
    """Cornerstone contours -> 6 binary masks."""
    im = nib.load(IMAGES / f"{case}.nii.gz")
    inv, shape = np.linalg.inv(im.affine), im.shape

    masks, report = {}, {}
    for cls, polys in read_contours(case).items():
        per_z = defaultdict(list)
        for pl in polys:
            v = nib.affines.apply_affine(inv, pl * LPS_TO_RAS)
            z = int(round(v[:, 2].mean()))
            if 0 <= z < shape[2]:
                per_z[z].append(v[:, :2])

        vol = np.zeros(shape, bool)
        for z, ps in per_z.items():
            vol[:, :, z] = _fill_slice(ps, shape[:2])

        zs, dropped = _reject_outliers(vol, per_z)
        n_fill = _interpolate(vol, list(zs)) if len(zs) else 0

        masks[cls] = vol
        report[cls] = {"traced": len(zs), "interpolated": n_fill,
                       "dropped": dropped, "voxels": int(vol.sum())}

    return masks, im.affine


def load_mesh(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = trimesh.load(str(path))
    return scene.to_geometry() if hasattr(scene, "to_geometry") else scene


def mesh_to_mask(mesh, affine, shape):
    """Ray-cast mesh containment on CT grid."""
    # All voxel centers
    coords_ijk = np.stack(np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij'
    ), axis=-1).reshape(-1, 3).astype(np.float64)

    # To world coords
    world_pts = nib.affines.apply_affine(affine, coords_ijk)

    # Ray-cast containment
    print(f"      ray-casting {len(world_pts):,} voxels...", end=" ", flush=True)
    inside = mesh.contains(world_pts)
    print(f"done", flush=True)

    mask = inside.reshape(shape)
    mask = binary_dilation(mask, iterations=1)
    return mask


def build_couinaud_case(case):
    """Build hybrid Couinaud for one case."""
    print(f"\n{case}")

    # Cornerstone
    print(f"  Loading cornerstone contours...", end=" ", flush=True)
    masks_cs, affine = rasterize_contours(case)
    print(f"done ({len(masks_cs)} classes)")

    if "classLiver" not in masks_cs:
        print(f"  ERROR: no liver contour")
        return None

    liver = masks_cs["classLiver"]
    im = nib.load(IMAGES / f"{case}.nii.gz")
    shape = im.shape

    result = dict(masks_cs)

    # Sectors
    sectors = {
        "classPediculoPortalIzquierdo": "CouinaudLeft",
        "classPediculoPortalAnteriorDerecho": "CouinaudRightAnterior",
        "classPediculoPortalPosteriorDerecho": "CouinaudRightPosterior",
    }

    for glb_name, out_name in sectors.items():
        glb_path = GLB_ROOT / case / "full_model" / f"{glb_name}.glb"

        if not glb_path.exists():
            print(f"  {glb_name}: not found")
            continue

        try:
            print(f"  Loading {glb_name}...", end=" ", flush=True)
            mesh = load_mesh(glb_path)
            print(f"done ({len(mesh.vertices):,} verts)")

            sector_mask = mesh_to_mask(mesh, affine, shape)
            sector_mask = sector_mask & liver

            result[f"class{out_name}"] = sector_mask

            n_vox = int(sector_mask.sum())
            pct = 100 * n_vox / max(int(liver.sum()), 1)
            print(f"      → {out_name:25s} {n_vox:9,d} vox ({pct:5.1f}% of liver)")

        except Exception as e:
            print(f"  {glb_name}: ERROR {str(e)[:60]}")

    # Write
    print(f"  Writing {len(result)} classes...", end=" ", flush=True)
    case_dir = OUT_DIR / case
    case_dir.mkdir(parents=True, exist_ok=True)

    for cls, vol in result.items():
        nib.save(nib.Nifti1Image(vol.astype(np.uint8), affine),
                 case_dir / f"{cls}.nii.gz")

    print(f"done")
    return len(result)


def build_couinaud_case_wrapper(case):
    """Wrapper for multiprocessing (no verbose output)."""
    try:
        n = build_couinaud_case(case)
        return (case, n, None)
    except Exception as e:
        return (case, None, str(e))


def main():
    from multiprocessing import Pool
    import os

    cases = list_cases()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse args
    if len(sys.argv) > 1:
        if sys.argv[1] == "--serial":
            serial = True
            ncores = 1
        else:
            serial = False
            case_idx = int(sys.argv[1])
            cases = [cases[case_idx]]
            ncores = 1
    else:
        serial = False
        ncores = os.cpu_count() or 4

    if len(cases) == 1:
        serial = True
        print(f"Building case 0: {cases[0]}\n")
    else:
        print(f"Building Couinaud hybrid for {len(cases)} cases")
        print(f"Output: {OUT_DIR}")
        print(f"Workers: {ncores}\n")

    # Build
    if serial or ncores == 1:
        ok = 0
        for i, case in enumerate(cases, 1):
            try:
                n = build_couinaud_case(case)
                if n:
                    ok += 1
            except Exception as e:
                print(f"  FATAL: {e}")
    else:
        with Pool(ncores) as pool:
            results = pool.map(build_couinaud_case_wrapper, cases)

        ok = 0
        for case, n, error in results:
            if error:
                print(f"{case}: ERROR {error}")
            elif n:
                ok += 1

    print(f"\n{'='*60}")
    print(f"Built {ok}/{len(cases)} cases")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
