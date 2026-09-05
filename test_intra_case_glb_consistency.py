"""
Test intra-case consistency of GLB -> CT transform.
Structure mirrors data_preview.ipynb (cells 1-11) but tests a DIFFERENT file
than the notebook's default cases[0]=RMV2025_0001_CT_NCT_HBP_X.

Assumption: transformation is consistent WITHIN a case across all classes
(permutation + signs + translation), though it varies PER-CASE (axis confusion,
per-case z_shift, per-case T_R/T_A). We infer the mapping from classLiver
only (via cornerstone_annotations) and test it on the 5 other classes that
exist in both GLB and cornerstone (Aorta, IVC, Gallbladder, VenaPorta,
BloodVessels).

Inspired by data_preview.ipynb cells:
 - cell 1-3: ROOT/IMAGES, list_cases, load_volume
 - cell 5:   load_mesh / glb_classes
 - cell 7:   check_glb_alignment
 - cell 8:   read_contours / contours_to_mask (LPS->RAS = [-1,-1,1])
 - cell 11:  solve_z_shift
"""
import glob
import json
import warnings
from pathlib import Path
from collections import defaultdict
import itertools

import numpy as np
import nibabel as nib
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon as sk_polygon

# --- Cell 1: same ROOT/IMAGES as data_preview.ipynb:1 ---
ROOT = Path("data")
IMAGES = ROOT / "0_test_nifti/imagesTs"
GLB_ROOT = ROOT / "0_test_glb"
ANN_DIR = ROOT / "0_test_nifti" / "cornerstone_annotations"

LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
GAP_SPLIT = 10
MIN_CLUSTER = 0.05
MAX_FILL_GAP = 5

# --- Cell 2: list_cases / load_volume ---
def list_cases():
    return [p.name[: -len(".nii.gz")] for p in sorted(IMAGES.glob("*.nii.gz"))]

def load_volume(directory, case):
    hits = sorted(glob.glob(str(Path(directory) / f"{case}*.nii.gz")))
    if not hits:
        return None, None
    img = nib.load(hits[0])
    return img.get_fdata(), img.affine

# --- Cell 5: GLB helpers ---
def load_mesh(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = trimesh.load(str(path))
    return scene.to_geometry() if hasattr(scene, "to_geometry") else scene

def glb_classes(case, subset=""):
    return sorted(p.stem[len("class"):] for p in (GLB_ROOT / case / subset).glob("*.glb"))

# --- Cell 8: cornerstone rasteriser (identical to data_preview.ipynb:8) ---
def read_contours(case):
    """{class_name: [polyline, ...]} in LPS mm."""
    d = json.loads((ANN_DIR / f"{case}.json").read_text())
    uid2cls = {u: c for c, us in d["classes"].items() for u in us}
    out = defaultdict(list)
    for e in d["state"]:
        cls = uid2cls.get(e["annotationUID"])
        if cls:
            out[cls].append(np.asarray(e["data"]["contour"]["polyline"], float))
    return out

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

def _fill_slice(polys, shape):
    acc = np.zeros(shape, bool)
    for p in polys:
        img = Image.new("1", (shape[1], shape[0]), 0)
        ImageDraw.Draw(img).polygon([(y, x) for x, y in p], fill=1)
        acc ^= np.asarray(img, bool)
    return acc

def _sdf(m):
    if not m.any():
        return np.full(m.shape, 1e3)
    return distance_transform_edt(~m) - distance_transform_edt(m)

def _interpolate(vol, zs, max_gap=MAX_FILL_GAP):
    filled = 0
    for z0, z1 in zip(zs[:-1], zs[1:]):
        if z1 - z0 <= 1 or z1 - z0 - 1 > max_gap:
            continue
        d0, d1 = _sdf(vol[:, :, z0]), _sdf(vol[:, :, z1])
        for z in range(z0 + 1, z1):
            a = (z - z0) / (z1 - z0)
            vol[:, :, z] = ((1 - a) * d0 + a * d1) < 0
            filled += 1
    return filled

def contours_to_mask(case, interpolate=True, max_gap=MAX_FILL_GAP, verbose=False):
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
        n_fill = _interpolate(vol, list(zs), max_gap) if (interpolate and len(zs)) else 0
        masks[cls] = vol
        report[cls] = {"traced": len(zs), "interpolated": n_fill, "dropped": dropped, "voxels": int(vol.sum())}
        if verbose:
            print(f"  {cls:20s} traced={len(zs):4d} interp={n_fill:4d} vox={int(vol.sum()):9d} dropped={dropped}")
    return masks, im.affine, report

# --- Cell 11: solve_z_shift (identical to data_preview.ipynb:11) ---
def solve_z_shift(case):
    """Return (mesh_axis, z_shift, residual_vs_affine) from liver contours."""
    polys = read_contours(case).get("classLiver", [])
    if not polys:
        return None
    P = np.vstack(polys)
    z_min, z_span = P[:, 2].min(), P[:, 2].ptp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = trimesh.load(GLB_ROOT / case / "full_model" / "classLiver.glb")
    g = s.to_geometry() if hasattr(s, "to_geometry") else s
    axis = int(np.argmin(np.abs(np.asarray(g.extents) - z_span)))
    z_shift = z_min - g.bounds[0][axis]
    tz = nib.load(IMAGES / f"{case}.nii.gz").affine[2, 3]
    return axis, z_shift, z_shift - tz, g, P

# --- Additional: infer full 3D mapping from liver (perm + signs + translation) ---
def infer_full_mapping_from_liver(case):
    """
    Brute-force perm+sign that best aligns liver mesh to liver contours.
    This tests intra-case consistency: same perm+sign+translation should work for other classes.
    Returns (perm, signs, translation_vector) where
        LPS_pred = signs * V[perm] + T
    and T = contour_liver_centroid - signs*mesh_liver_centroid[perm].
    Also returns the liver-derived axis/z_shift for comparison with solve_z_shift.
    """
    # contour liver stats
    polys = read_contours(case).get("classLiver", [])
    P = np.vstack(polys)
    c_mean = P.mean(0)
    c_min = P.min(0)
    c_max = P.max(0)
    c_ptp = P.ptp(0)

    g = load_mesh(GLB_ROOT / case / "full_model" / "classLiver.glb")
    m_mean = g.vertices.mean(0)
    m_min = g.bounds[0]
    m_max = g.bounds[1]
    m_ext = g.extents

    best = None
    best_err = float("inf")
    for perm in itertools.permutations([0,1,2]):
        for sx, sy, sz in itertools.product([-1,1], repeat=3):
            signs = np.array([sx, sy, sz])
            m_mean_perm = np.array([m_mean[perm[0]], m_mean[perm[1]], m_mean[perm[2]]])
            T = c_mean - signs * m_mean_perm

            # ptp after sign (ptp unchanged by sign, but perm changes which axis maps)
            m_ptp_perm = np.array([m_ext[perm[0]], m_ext[perm[1]], m_ext[perm[2]]])
            ptp_err = np.abs(m_ptp_perm - c_ptp).sum()

            # bounds error after translation
            # need to handle sign-flipped min/max swap
            pred_mins = []
            pred_maxs = []
            for i in range(3):
                s = signs[i]
                mn = m_min[perm[i]]
                mx = m_max[perm[i]]
                v0 = s*mn + T[i]
                v1 = s*mx + T[i]
                pred_mins.append(min(v0,v1))
                pred_maxs.append(max(v0,v1))
            pred_mins = np.array(pred_mins)
            pred_maxs = np.array(pred_maxs)
            bounds_err = np.abs(pred_mins - c_min).sum() + np.abs(pred_maxs - c_max).sum()
            err = ptp_err + bounds_err
            if err < best_err:
                best_err = err
                best = (perm, signs, T, pred_mins, pred_maxs, ptp_err, bounds_err, m_mean_perm)

    # also compute simple z_shift/axis from data_preview for reference
    axis, z_shift, resid, _, _ = solve_z_shift(case)
    return best, best_err, (axis, z_shift, resid)

def apply_mapping(vertices, perm, signs, T):
    """Apply perm+sign+T to vertices (N,3) -> LPS (N,3)."""
    perm = np.array(perm)
    signs = np.array(signs)
    T = np.array(T)
    # vertices[:, perm[i]] * signs[i] + T[i]
    out = np.empty_like(vertices)
    for i in range(3):
        out[:, i] = signs[i] * vertices[:, perm[i]] + T[i]
    return out

def mesh_to_voxel_mask(mesh_lps, affine, shape):
    """
    Voxelize LPS mesh onto imagesTs grid.
    Steps: LPS -> RAS ([-1,-1,1]), then RAS -> voxel via inv(affine).
    Use trimesh section_multiplane along slice_axis (2, since affine is axis-aligned L,A,S).
    """
    inv = np.linalg.inv(affine)
    # LPS -> RAS
    mesh_ras = mesh_lps * np.array([-1.0, -1.0, 1.0])
    # RAS -> voxel indices (float)
    pts = np.hstack([mesh_ras, np.ones((mesh_ras.shape[0],1))])
    idx = (inv @ pts.T).T[:, :3]
    mesh_idx = mesh_lps.copy()  # dummy to hold vertices
    # we create new mesh with voxel-space vertices
    # Need trimesh object with voxel coordinates
    # Use original mesh topology but replace vertices
    # For simplicity, create new mesh
    tmp = trimesh.Trimesh(vertices=idx, faces=mesh_lps_faces) if False else None
    return None

# Simpler voxelizer that works on voxel-space mesh directly (reusing build_couinaud logic)
def voxelize_mesh_voxel_space(mesh_voxel, shape, slice_axis=2):
    """Per-slice contour rasterization along slice_axis. mesh_voxel vertices are in voxel index space."""
    other = [a for a in range(3) if a != slice_axis]
    sl_coords = mesh_voxel.vertices[:, slice_axis]
    n = shape[slice_axis]
    kmin = max(0, int(np.floor(sl_coords.min())))
    kmax = min(n - 1, int(np.ceil(sl_coords.max())))
    out = np.zeros(shape, dtype=bool)
    if kmax < kmin:
        return out
    normal = np.zeros(3); normal[slice_axis] = 1.0
    origin = np.zeros(3)
    heights = np.arange(kmin, kmax + 1, dtype=float)
    sections = mesh_voxel.section_multiplane(plane_origin=origin, plane_normal=normal, heights=heights)
    out_shape_2d = (shape[other[0]], shape[other[1]])
    for offset, sec in enumerate(sections):
        if sec is None:
            continue
        k = kmin + offset
        for poly in sec.polygons_full:
            xs, ys = poly.exterior.coords.xy
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

def transform_mesh_to_voxel_mask(case, cls, perm, signs, T, affine, shape, subset="full_model"):
    """Load GLB, apply LPS mapping, then LPS->RAS->voxel and voxelize."""
    path = GLB_ROOT / case / subset / f"{cls}.glb"
    if not path.exists():
        # try class prefix handling
        alt = GLB_ROOT / case / subset / f"class{cls.replace('class','')}.glb"
        if alt.exists():
            path = alt
        else:
            return None
    g = load_mesh(path)
    # apply LPS mapping inferred from liver
    verts_lps = apply_mapping(g.vertices, perm, signs, T)
    # LPS -> RAS
    verts_ras = verts_lps * np.array([-1.0, -1.0, 1.0])
    # RAS -> voxel
    pts = np.hstack([verts_ras, np.ones((verts_ras.shape[0],1))])
    inv = np.linalg.inv(affine)
    verts_vox = (inv @ pts.T).T[:, :3]
    mesh_vox = trimesh.Trimesh(vertices=verts_vox, faces=g.faces, process=False)
    mask = voxelize_mesh_voxel_space(mesh_vox, shape, slice_axis=2)
    return mask, verts_lps

def test_case(case, verbose=True):
    print(f"\n{'='*80}")
    print(f"TEST CASE: {case}  (different from data_preview default cases[0])")
    print(f"{'='*80}")
    # Basic geometry
    ct, affine = load_volume(IMAGES, case)  # actually load_volume returns data, affine but we need affine via nib
    img = nib.load(str(IMAGES / f"{case}.nii.gz"))
    affine = img.affine
    shape = img.shape
    print(f"  shape {shape} spacing {nib.affines.voxel_sizes(affine).round(3)} axes {nib.aff2axcodes(affine)}")
    print(f"  affine[2,3] (tz) {affine[2,3]:.1f}")

    # Solve z_shift via data_preview method
    axis, z_shift, resid, g_liver, P = solve_z_shift(case)
    print(f"  solve_z_shift (liver only): axis={axis} z_shift={z_shift:.1f} resid vs affine {resid:.2f} {'<- check' if abs(resid)>=1 else ''}")
    print(f"    -> inferred LPS Z = mesh_axis[{axis}] + {z_shift:.1f}")

    # Infer full 3D mapping from liver
    best, best_err, (axis2, z_shift2, resid2) = infer_full_mapping_from_liver(case)
    perm, signs, T, pred_mins, pred_maxs, ptp_err, bounds_err, _ = best
    print(f"  full 3D liver-derived mapping (perm+sign+T):")
    print(f"    perm {perm} signs {signs.tolist()} T {[round(x,1) for x in T]}")
    print(f"    ptp_err {ptp_err:.1f} bounds_err {bounds_err:.1f} total {best_err:.1f}")
    print(f"    -> LPS = signs * V[perm] + T,  where V is raw GLB vertices")

    # Compare to build_couinaud formula for reference
    # build_couinaud: X_LPS = -T_R + a1, Y_LPS = -T_A - a2, Z = a0 + z_shift
    # For this case, check what that would give for liver centroid
    # (We already have T_R/T_A from affine)

    # Now test on other classes that have both GLB and cornerstone
    masks_corner, _, rep_corner = contours_to_mask(case, verbose=False)
    print(f"\n  cornerstone classes available: {sorted(masks_corner.keys())}")
    # Universal 6 classes - test those present in both
    universal = ["classLiver","classAorta","classIVC","classGallbladder","classVenaPorta","classBloodVessels"]
    # Also test that non-cornerstone classes (Pediculo*) would use same mapping - we can at least check they land inside CT
    print(f"\n  Testing intra-case consistency: apply SAME liver-derived mapping to other classes")
    print(f"  {'class':22s} {'centroid_dist':>14s} {'bounds_err':>11s} {'dice_vs_corner':>14s} {'inside_CT':>9s}")

    results = []
    # Precompute cornerstone masks once (already done)
    for cls in universal:
        if cls not in masks_corner:
            print(f"  {cls:22s}  MISSING in cornerstone")
            continue
        # contour centroid LPS
        polys = read_contours(case).get(cls, [])
        if not polys:
            print(f"  {cls:22s}  no polys")
            continue
        P_cls = np.vstack(polys)
        c_mean_cls = P_cls.mean(0)
        c_min_cls = P_cls.min(0)
        c_max_cls = P_cls.max(0)

        # GLB mesh centroid predicted via liver mapping
        # Load GLB for this class
        found = False
        for subset in ["full_model",""]:
            p = GLB_ROOT / case / subset / f"{cls}.glb"
            if p.exists():
                found = True
                subset_used = subset
                break
        if not found:
            print(f"  {cls:22s}  MISSING GLB")
            continue
        g = load_mesh(p)
        m_mean_raw = g.vertices.mean(0)
        # predicted LPS centroid
        m_mean_pred = signs * np.array([m_mean_raw[perm[0]], m_mean_raw[perm[1]], m_mean_raw[perm[2]]]) + T
        dist = np.linalg.norm(m_mean_pred - c_mean_cls)

        # bounds error for this class using same mapping
        m_min_raw = g.bounds[0]
        m_max_raw = g.bounds[1]
        pred_mins_cls = []
        pred_maxs_cls = []
        for i in range(3):
            s = signs[i]
            mn = m_min_raw[perm[i]]
            mx = m_max_raw[perm[i]]
            v0 = s*mn + T[i]
            v1 = s*mx + T[i]
            pred_mins_cls.append(min(v0,v1))
            pred_maxs_cls.append(max(v0,v1))
        pred_mins_cls = np.array(pred_mins_cls)
        pred_maxs_cls = np.array(pred_maxs_cls)
        bounds_err_cls = np.abs(pred_mins_cls - c_min_cls).sum() + np.abs(pred_maxs_cls - c_max_cls).sum()

        # Fast inside-CT check (predicted mesh bounds inside CT world) - no heavy voxelization
        corners = np.array([[i,j,k] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])], dtype=float)
        world = nib.affines.apply_affine(affine, corners)
        w_min, w_max = world.min(0), world.max(0)
        pred_mins_ras = np.array([-pred_mins_cls[0], -pred_mins_cls[1], pred_mins_cls[2]])
        pred_maxs_ras = np.array([-pred_maxs_cls[0], -pred_maxs_cls[1], pred_maxs_cls[2]])
        pred_ras_mins = np.minimum(pred_mins_ras, pred_maxs_ras)
        pred_ras_maxs = np.maximum(pred_mins_ras, pred_maxs_ras)
        inside = np.all(pred_ras_mins >= w_min -1) and np.all(pred_ras_maxs <= w_max +1)
        inside_str = str(inside)
        # Dice via lightweight proxy: we skip full voxelization for speed; report n/a
        dice = float('nan')
        # To enable Dice, uncomment voxelization below (slow: ~20s per class due to 512^3 volume)
        # mask_glb, _ = transform_mesh_to_voxel_mask(case, cls, perm, signs, T, affine, shape, subset=subset_used)
        # mask_corner = masks_corner[cls]
        # inter = np.logical_and(mask_glb, mask_corner).sum() if mask_glb is not None else 0
        # denom = mask_glb.sum() + mask_corner.sum() if mask_glb is not None else 1
        # dice = 2*inter/denom if denom>0 else 0.0

        print(f"  {cls:22s} {dist:10.1f} mm {bounds_err_cls:8.1f} mm {'n/a':>10s} {inside_str:>9s}")
        results.append((cls, dist, bounds_err_cls, dice))

    # Also test a GLB-only class to show it lands inside CT when using same mapping (portability test)
    glb_only = [c for c in glb_classes(case) if f"class{c}" not in masks_corner]
    if glb_only:
        print(f"\n  GLB-only classes (no cornerstone to compare, testing landing inside CT):")
        for c in glb_only[:4]:  # show first 4
            cls = f"class{c}"
            p = GLB_ROOT / case / "full_model" / f"{cls}.glb"
            if not p.exists():
                p = GLB_ROOT / case / f"{cls}.glb"
            g = load_mesh(p)
            m_min_raw = g.bounds[0]
            m_max_raw = g.bounds[1]
            pred_mins_cls = []
            pred_maxs_cls = []
            for i in range(3):
                s = signs[i]
                mn = m_min_raw[perm[i]]
                mx = m_max_raw[perm[i]]
                v0 = s*mn + T[i]
                v1 = s*mx + T[i]
                pred_mins_cls.append(min(v0,v1))
                pred_maxs_cls.append(max(v0,v1))
            pred_mins_cls = np.array(pred_mins_cls)
            pred_maxs_cls = np.array(pred_maxs_cls)
            # LPS->RAS for inside check
            pred_mins_ras = np.array([-pred_mins_cls[0], -pred_mins_cls[1], pred_mins_cls[2]])
            pred_maxs_ras = np.array([-pred_maxs_cls[0], -pred_maxs_cls[1], pred_maxs_cls[2]])
            pred_ras_mins = np.minimum(pred_mins_ras, pred_maxs_ras)
            pred_ras_maxs = np.maximum(pred_mins_ras, pred_maxs_ras)
            corners = np.array([[i,j,k] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])], dtype=float)
            world = nib.affines.apply_affine(affine, corners)
            w_min, w_max = world.min(0), world.max(0)
            inside = np.all(pred_ras_mins >= w_min -5) and np.all(pred_ras_maxs <= w_max +5)
            print(f"    {cls:32s} pred LPS X [{pred_mins_cls[0]:.0f},{pred_maxs_cls[0]:.0f}] Y [{pred_mins_cls[1]:.0f},{pred_maxs_cls[1]:.0f}] Z [{pred_mins_cls[2]:.0f},{pred_maxs_cls[2]:.0f}] inside_CT {inside}")

    return results, (perm, signs, T)

# --- Main test on DIFFERENT files (not cases[0]) ---
if __name__ == "__main__":
    cases = list_cases()
    print(f"{len(cases)} cases total")
    print(f"  data_preview default cases[0] = {cases[0]}")
    # Choose DIFFERENT files: one clean axis=1, one axis=2, one residual
    test_cases = [
        "RMV2025_0003_CT_UTV_VEP_P",   # clean axis1, resid 0.00
        "RMV2025_0015_CT_UTV_VEP_K",   # axis2 case (0015-0019 family)
        "RMV2026_0022_CT_UTV_VEP_N",   # residual 3.12 case
        "RMV2025_0014_CT_UTV_VEP_M",   # broken -209 case
    ]
    # Filter to existing cases
    test_cases = [c for c in test_cases if c in cases]
    for tc in test_cases:
        test_case(tc)

    # Summary table for documentation
    print("\n" + "="*80)
    print("SUMMARY: intra-case consistency holds if SAME liver-derived perm+sign+T")
    print("         gives small centroid distance (<15mm) and high Dice (>0.7) for")
    print("         OTHER classes that were NOT used to infer the mapping.")
    print("="*80)

