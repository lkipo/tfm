#!/usr/bin/env python3
"""
Build Couinaud training data: cornerstone 6 classes + GLB mesh sectors.

Strategy: Keep the 6 accurate cornerstone annotations, then add sector splits
by rasterizing PediculoPortal meshes and intersecting with the liver.
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

ROOT = Path("data")
IMAGES = ROOT / "0_test_nifti/imagesTs"
ANN_DIR = ROOT / "0_test_nifti/cornerstone_annotations"
GLB_ROOT = ROOT / "0_test_glb"
OUT_DIR = ROOT / "couinaud_hybrid"

LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])


def list_cases():
    return sorted([p.stem for p in IMAGES.glob("*.nii.gz")])


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


def _reject_outliers(vol, per_z, gap_split=10, min_cluster=0.05):
    zs = np.array(sorted(per_z))
    if len(zs) < 2:
        return zs, []
    clusters = np.split(zs, np.where(np.diff(zs) > gap_split)[0] + 1)
    sizes = [int(vol[:, :, cl].sum()) for cl in clusters]
    total = sum(sizes) or 1
    keep, dropped = [], []
    for cl, n in zip(clusters, sizes):
        if n / total >= min_cluster:
            keep.append(cl)
        else:
            vol[:, :, cl] = False
            dropped.append((int(cl[0]), int(cl[-1]), n, round(100 * n / total, 2)))
    return (np.concatenate(keep) if keep else np.array([], int)), dropped


def _interpolate(vol, zs, max_gap=5):
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
    """Load and rasterize cornerstone contours."""
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


def mesh_to_mask_sdf(mesh, affine, shape, dilate=1):
    """Rasterize mesh to binary mask via signed distance + ray-casting."""
    # Voxel center coordinates
    coords_ijk = np.stack(np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing='ij'
    ), axis=-1).reshape(-1, 3).astype(np.float32)

    # World coordinates
    world_pts = nib.affines.apply_affine(affine, coords_ijk)

    # Closest distance to mesh surface
    try:
        distances, _ = trimesh.proximity.closest_point(mesh, world_pts)
    except Exception:
        # Fallback: use a grid-based approach
        return np.zeros(shape, bool)

    # Determine sign via ray-casting
    hit = mesh.contains(world_pts)
    distances = np.where(hit, -distances, distances)

    # Mask: inside mesh (distance < 0)
    mask = (distances < 0).reshape(shape)

    # Dilate to fill in boundary voxels
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)

    return mask


def build_couinaud_case(case, out_dir=OUT_DIR, verbose=True):
    """Build Couinaud training mask: 6 cornerstone + 3 sector splits."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cornerstone
    masks_cs, affine = rasterize_contours(case)
    liver = masks_cs["classLiver"]

    # Start result
    result = dict(masks_cs)
    im = nib.load(IMAGES / f"{case}.nii.gz")
    shape = im.shape

    # Add sectors
    sectors = {
        "PediculoPortalIzquierdo": "CouinaudLeft",
        "PediculoPortalAnteriorDerecho": "CouinaudRightAnterior",
        "PediculoPortalPosteriorDerecho": "CouinaudRightPosterior",
    }

    if verbose:
        print(f"{case}")
        print(f"  Cornerstone: {len(masks_cs)} classes")

    for glb_name, out_name in sectors.items():
        glb_path = GLB_ROOT / case / "full_model" / f"class{glb_name}.glb"

        if not glb_path.exists():
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scene = trimesh.load(str(glb_path))
            mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene

            # Rasterize and intersect with liver
            sector_mask = mesh_to_mask_sdf(mesh, affine, shape, dilate=1)
            sector_mask = sector_mask & liver

            result[f"class{out_name}"] = sector_mask

            n_vox = int(sector_mask.sum())
            pct = 100 * n_vox / max(int(liver.sum()), 1)
            if verbose:
                print(f"  + {out_name:25s}  {n_vox:9d} vox ({pct:5.1f}% of liver)")

        except Exception as e:
            if verbose:
                print(f"  ✗ {glb_name:25s}  {str(e)[:50]}")

    # Write outputs
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    for cls, vol in result.items():
        nib.save(nib.Nifti1Image(vol.astype(np.uint8), affine),
                 case_dir / f"{cls}.nii.gz")

    return result


def build_all_couinaud(out_dir=OUT_DIR):
    """Build Couinaud data for all 38 cases."""
    cases = list_cases()
    print(f"Building Couinaud hybrid masks for {len(cases)} cases")
    print(f"Output: {out_dir}\n")

    summary = {}
    for i, case in enumerate(cases, 1):
        try:
            result = build_couinaud_case(case, out_dir=out_dir, verbose=True)
            summary[case] = {"classes": len(result), "status": "ok"}
            print()
        except Exception as e:
            print(f"  ERROR: {str(e)[:60]}\n")
            summary[case] = {"status": f"error: {str(e)[:40]}"}

    # Save summary
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBuilt {sum(1 for s in summary.values() if s['status'] == 'ok')}/{len(cases)} cases")
    print(f"Summary saved to {out_dir / 'build_summary.json'}")


if __name__ == "__main__":
    build_all_couinaud()
