#!/usr/bin/env python3
"""
Mesh-to-contour registration: test if GLB mesh coordinate transform is consistent.

Uses ICP to find the rigid transform that best aligns mesh vertices with contour
point clouds, then validates consistency across multiple cases.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
import warnings

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

# Reuse config/functions from data_preview.ipynb
ROOT = Path("data")
IMAGES = ROOT / "0_test_nifti/imagesTs"
GLB_ROOT = ROOT / "0_test_glb"
ANN_DIR = ROOT / "0_test_nifti/cornerstone_annotations"

LPS_TO_RAS = np.array([-1.0, -1.0, 1.0])
UNIVERSAL_CLASSES = ["Liver", "VenaPorta", "IVC", "Aorta", "Gallbladder", "BloodVessels"]


def list_cases():
    """Case IDs that have a base image, in sorted order."""
    return sorted([p.stem for p in IMAGES.glob("*.nii.gz")])


def load_mesh(path):
    """GLB -> single concatenated mesh."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = trimesh.load(str(path))
    return scene.to_geometry() if hasattr(scene, "to_geometry") else scene


def read_contours(case):
    """{class_name: [polyline, ...]} in LPS mm."""
    from collections import defaultdict
    d = json.loads((ANN_DIR / f"{case}.json").read_text())
    uid2cls = {u: c for c, us in d["classes"].items() for u in us}
    out = defaultdict(list)
    for e in d["state"]:
        cls = uid2cls.get(e["annotationUID"])
        if cls:
            out[cls].append(np.asarray(e["data"]["contour"]["polyline"], float))
    return out


def get_mesh_surface_points(case, cls="Liver", subset="full_model", sample_count=10000):
    """Sample points from mesh surface."""
    mesh = load_mesh(GLB_ROOT / case / subset / f"class{cls}.glb")
    # Sample uniformly from surface
    pts, _ = trimesh.sample.sample_surface(mesh, sample_count)
    return pts


def get_contour_surface_points(case, cls="classLiver"):
    """Extract points from rasterized contours as surface voxels.

    Returns points in voxel coordinates (before affine transform).
    """
    polys = read_contours(case).get(cls, [])
    if not polys:
        return np.array([]).reshape(0, 3)

    # Load CT grid
    im = nib.load(IMAGES / f"{case}.nii.gz")
    inv, shape = np.linalg.inv(im.affine), im.shape

    # Rasterize contours to mask
    from collections import defaultdict
    from PIL import Image, ImageDraw

    vol = np.zeros(shape, bool)
    for pl in polys:
        # Transform from LPS to voxel coords
        v = nib.affines.apply_affine(inv, pl * LPS_TO_RAS)
        z = int(round(v[:, 2].mean()))
        if 0 <= z < shape[2]:
            # Fill slice
            p_xy = v[:, :2]
            img = Image.new("1", (shape[1], shape[0]), 0)
            ImageDraw.Draw(img).polygon([(y, x) for x, y in p_xy], fill=1)
            vol[:, :, z] ^= np.asarray(img, bool)

    # Extract surface voxels (on the boundary of the mask)
    from scipy import ndimage
    if not vol.any():
        return np.array([]).reshape(0, 3)

    # Find boundary: erode and XOR to get shell
    eroded = ndimage.binary_erosion(vol)
    boundary = vol & ~eroded

    # Get coordinates
    pts_voxel = np.argwhere(boundary).astype(float)  # (i, j, k) in voxel coords

    # Transform to world coords (RAS)
    pts_world = nib.affines.apply_affine(im.affine, pts_voxel)

    return pts_world


def icp_registration(src, dst, max_iterations=100, threshold=1e-6):
    """
    Rigid ICP: find R, t such that dst ≈ R @ src.T + t

    Returns (R, t, residual) where:
    - R: 3x3 rotation matrix
    - t: 3-vector translation
    - residual: mean squared error after registration
    """
    if HAS_O3D:
        return _icp_open3d(src, dst, max_iterations, threshold)
    else:
        return _icp_manual(src, dst, max_iterations, threshold)


def _icp_open3d(src, dst, max_iterations=100, threshold=1e-6):
    """ICP using open3d."""
    src_cloud = o3d.geometry.PointCloud()
    src_cloud.points = o3d.utility.Vector3dVector(src)

    dst_cloud = o3d.geometry.PointCloud()
    dst_cloud.points = o3d.utility.Vector3dVector(dst)

    result = o3d.pipelines.registration.registration_icp(
        src_cloud, dst_cloud,
        max_correspondence_distance=1000.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    T = result.transformation
    R = T[:3, :3]
    t = T[:3, 3]

    # Residual: mean squared error
    src_transformed = (R @ src.T + t[:, None]).T
    residual = np.mean(np.sum((src_transformed - dst) ** 2, axis=1)) ** 0.5

    return R, t, residual


def _icp_manual(src, dst, max_iterations=100, threshold=1e-6):
    """Simple ICP using numpy (no open3d)."""
    from scipy.spatial import cKDTree

    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    # Initialize
    R = np.eye(3)
    t = np.zeros(3)
    prev_error = np.inf

    for iteration in range(max_iterations):
        # Find nearest neighbors
        src_transformed = (R @ src.T + t[:, None]).T
        tree = cKDTree(dst)
        distances, indices = tree.query(src_transformed, k=1)

        # Current error
        error = np.mean(distances ** 2) ** 0.5

        # Check convergence
        if abs(prev_error - error) < threshold:
            break
        prev_error = error

        # Compute optimal rotation and translation
        src_centered = src_transformed - src_transformed.mean(axis=0)
        dst_centered = dst[indices] - dst[indices].mean(axis=0)

        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R_new = Vt.T @ U.T

        # Ensure proper rotation (det(R) == 1)
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T

        t_new = dst[indices].mean(axis=0) - (R_new @ src_transformed.mean(axis=0))

        R = R_new @ R
        t = R_new @ t + t_new

    src_transformed = (R @ src.T + t[:, None]).T
    residual = np.mean(np.sum((src_transformed - dst) ** 2, axis=1)) ** 0.5

    return R, t, residual


def test_mesh_consistency(cases_to_test=None, sample_count=5000):
    """
    Test if mesh→contour transform is consistent across cases.

    For each case, register the 6 universal mesh classes against
    rasterized contours, and check if the transformation matrix is stable.
    """
    if cases_to_test is None:
        cases_to_test = list_cases()[:5]  # Start with first 5

    all_cases = list_cases()
    cases_to_test = [c for c in cases_to_test if c in all_cases]

    results = {}

    print(f"Testing mesh registration on {len(cases_to_test)} cases")
    print(f"{'case':35s} {'class':20s} {'n_mesh':>7s} {'n_cont':>7s} {'rmse':>10s} {'status'}")
    print("-" * 95)

    for case in cases_to_test:
        case_result = {}

        for cls_short in UNIVERSAL_CLASSES:
            cls_mesh = f"class{cls_short}"
            cls_glb = cls_short

            try:
                # Get point clouds
                mesh_pts = get_mesh_surface_points(case, cls_glb, sample_count=sample_count)
                cont_pts = get_contour_surface_points(case, cls_mesh)

                if len(cont_pts) < 10:
                    status = f"sparse contours ({len(cont_pts)})"
                    case_result[cls_short] = {"status": status}
                    print(f"{case:35s} {cls_short:20s} {len(mesh_pts):7d} {len(cont_pts):7d} {'—':>10s} {status}")
                    continue

                # Run ICP
                R, t, rmse = icp_registration(mesh_pts, cont_pts)

                case_result[cls_short] = {
                    "R": R.tolist(),
                    "t": t.tolist(),
                    "rmse": float(rmse),
                    "n_mesh": len(mesh_pts),
                    "n_contour": len(cont_pts),
                    "status": "ok" if rmse < 50 else "high-error"
                }

                status = "ok" if rmse < 50 else "⚠️ high RMSE"
                print(f"{case:35s} {cls_short:20s} {len(mesh_pts):7d} {len(cont_pts):7d} {rmse:10.2f} {status}")

            except Exception as e:
                status = f"error: {str(e)[:40]}"
                case_result[cls_short] = {"status": status}
                print(f"{case:35s} {cls_short:20s} {'—':>7s} {'—':>7s} {'—':>10s} {status}")

        results[case] = case_result

    # Analyze consistency
    print("\n" + "=" * 95)
    print("Consistency Analysis")
    print("=" * 95)

    for cls in UNIVERSAL_CLASSES:
        rmses = [r[cls].get("rmse") for r in results.values() if cls in r and "rmse" in r[cls]]
        if rmses:
            rmses = np.array(rmses)
            print(f"{cls:20s}: mean RMSE {rmses.mean():.2f} mm, "
                  f"std {rmses.std():.2f}, range [{rmses.min():.2f}, {rmses.max():.2f}]")

    # Check if rotation matrices are similar (should be close to identity if meshes are already in CT space)
    print("\nRotation matrix analysis (should be close to identity if meshes are in CT space):")
    for cls in UNIVERSAL_CLASSES:
        Rs = [np.array(r[cls]["R"]) for r in results.values()
              if cls in r and "R" in r[cls]]
        if Rs:
            R_mean = np.mean(Rs, axis=0)
            R_deviation = np.mean([np.linalg.norm(R - R_mean) for R in Rs])
            frob_from_id = np.mean([np.linalg.norm(R - np.eye(3)) for R in Rs])
            print(f"{cls:20s}: avg deviation from mean {R_deviation:.3f}, "
                  f"avg deviation from identity {frob_from_id:.3f}")

    return results


if __name__ == "__main__":
    print(f"PyTorch available: {HAS_O3D}\n")

    # Test first 5 cases
    results = test_mesh_consistency(sample_count=5000)

    # Save results
    out_path = Path("mesh_registration_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
