#!/usr/bin/env python3
"""Derive the Couinaud pedicle masks from the predicted portal-tree *union*.

The four ``classPediculoPortal*`` masks the Couinaud pipeline consumes cannot be
trained directly: their labels in the cornerstone masks are remnants (the
``classBloodVessels`` contour overwrite leaves 21-1450 voxels, and no model
learned them). What is learnable is the portal tree *union* (``portal_tree``,
test Dice ~0.62) and the vena porta *trunk* (``vena_porta``, ~0.61). This module
derives the pedicle masks from those two at inference time:

1. skeletonize the portal-tree union and the vena-porta trunk (Lee thinning on
   the bounding box);
2. remove the trunk region from the tree skeleton (skeleton points within a
   trunk radius of the porta skeleton) -- what remains is the pedicle branches,
   which become topologically separated components;
3. split left vs right by the recovered anatomical R-L axis, and the right
   component at its first branchpoint into anterior vs posterior (the A axis);
4. assign every portal-tree voxel outside the trunk to the nearest pedicle
   skeleton (Voronoi), exactly like the pipeline's own territory assignment.

In practice step 2 never leaves anything: ``vena_porta`` is not a short stub
here (the ``classBloodVessels``-style contour overwrite documented in
``labels.py`` means it already extends through most of the tree), so its
skeleton is nearly as long as the whole union's and every tree point falls
within a trunk radius of it. ``derive_pedicle_masks`` (above) always returns
``{}`` on this data and ``_plane_split`` (below) is what actually runs. It
cuts ``portal_tree & ~vena_porta & liver`` -- not the trunk-dominated full
union -- with two planes through axes recovered by
``continuity/couinaud/anatomy_frame.estimate_frame`` (the same anatomy-fit
frame the Couinaud stage itself trusts; the NIfTI affine cannot be, per that
module's warning, and neither could an ad hoc IVC-aorta vector -- measured at
cosine similarity 0.03 with the true left/right split on case 0003). Each
plane sits at the Otsu threshold of the relevant axis (falling back to the
median if Otsu would starve a sector), not a fixed anatomical point: the true
sector boundary is off-centre, and a fixed point is fragile when trunk
removal reshapes the pedicle-only point cloud per case.

The liver gate and the largest-component filtering on ``portal_tree`` /
``vena_porta`` matter only for *predictions*, not GT: the cornerstone labels
are a hard partition (one class per voxel), so excluding the trunk already
implied intrahepatic-only there. Predictions have no such guarantee -- a
couinaud5 prediction measured 5-13% stray voxels per channel across up to 90
disconnected components, and its portal-tree union included real but
extrahepatic portal trunk tissue that, ungated, pulled the derived "left"
sector to sit almost entirely outside the predicted liver.

    python cornerstone_finetune/derive_pedicles.py \
        --case-dir runs/pipeline_tests/predictions/RMV2025_0003_CT_UTV_VEP_P

Writes classPediculoPortal{Izquierdo,AnteriorDerecho,PosteriorDerecho}.nii.gz
into the case dir. Hepatic veins and gallbladder are intentionally absent:
the Voronoi division degrades gracefully without them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO / "continuity", _REPO / "continuity" / "couinaud"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from anatomy_frame import estimate_frame  # noqa: E402

TRUNK_RADIUS_MM = 6.0
MIN_COMP_PTS = 10


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    return lbl == (np.argmax(sizes) + 1)


def _skeleton_points(mask: np.ndarray) -> np.ndarray:
    """Lee thinning on the bounding box -> full-grid voxel coordinates."""
    sl = ndimage.find_objects(mask)[0]
    skel = skeletonize(mask[sl].astype(np.uint8))
    return np.argwhere(skel).astype(float) + np.array([s.start for s in sl])


def _points_to_vol(pts: np.ndarray, shape: tuple) -> np.ndarray:
    vol = np.zeros(shape, bool)
    idx = pts.astype(int)
    ok = ((idx >= 0) & (idx < np.array(shape))).all(1)
    vol[tuple(idx[ok].T)] = True
    return vol


def _branchpoints(pts: np.ndarray) -> np.ndarray:
    """Skeleton points with >=3 occupied neighbours (degree >= 3)."""
    if len(pts) == 0:
        return np.empty((0, 3))
    idx = pts.astype(int)
    vol = np.zeros(tuple(idx.max(0) + 3), bool)
    vol[tuple(idx.T)] = True
    k = np.ones((3, 3, 3), int)
    k[1, 1, 1] = 0
    deg = ndimage.convolve(vol, k, mode="constant", cval=0)
    bp = np.argwhere((deg >= 3) & vol)
    return bp.astype(float) - 1.0


def _components(pts: np.ndarray, shape: tuple) -> list[np.ndarray]:
    """Connected components of a skeleton point set, as point arrays."""
    vol = _points_to_vol(pts, shape)
    lbl, n = ndimage.label(vol)
    out = []
    for i in range(1, n + 1):
        sel = lbl[tuple(pts.astype(int).T)] == i
        out.append(pts[sel])
    return out


def derive_pedicle_masks(
    portal_tree: np.ndarray,
    vena_porta: np.ndarray,
    ivc: np.ndarray,
    aorta: np.ndarray,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
    trunk_radius_mm: float = TRUNK_RADIUS_MM,
) -> dict[str, np.ndarray]:
    """Binary masks for the left / right-anterior / right-posterior pedicles,
    on the same grid as the inputs."""
    zooms = np.asarray(zooms, float)
    to_mm = lambda p: p * zooms

    tree_pts = _skeleton_points(_largest_component(portal_tree))
    porta_pts = _skeleton_points(_largest_component(vena_porta))
    if len(tree_pts) == 0 or len(porta_pts) == 0:
        return {}

    tree_mm = to_mm(tree_pts)
    porta_mm = to_mm(porta_pts)
    tree_kd = cKDTree(tree_mm)

    # --- trunk removal: tree skeleton points near the porta skeleton ----------
    near = tree_kd.query_ball_point(porta_mm, r=trunk_radius_mm)
    trunk_mask = np.zeros(len(tree_pts), bool)
    for hits in near:
        trunk_mask[hits] = True
    branches = tree_pts[~trunk_mask]
    if len(branches) == 0:
        return {}

    comps = [c for c in _components(branches, portal_tree.shape) if len(c) >= MIN_COMP_PTS]
    if len(comps) < 2:
        return {}

    # --- the bifurcation: tree branchpoint nearest the porta skeleton ---------
    bp = _branchpoints(tree_pts)
    if len(bp) == 0:
        return {}
    bif = bp[int(np.argmin(cKDTree(to_mm(bp)).query(porta_mm)[0]))]

    # --- anatomical axes from the predicted anchors ---------------------------
    def centroid(mask: np.ndarray) -> np.ndarray:
        return np.asarray(ndimage.center_of_mass(mask))

    ivc_c = centroid(ivc) if ivc.any() else None
    ao_c = centroid(aorta) if aorta.any() else None
    if ivc_c is None or ao_c is None:
        return {}
    si_axis = int(np.argmin(zooms))
    si = np.zeros(3); si[si_axis] = 1.0
    r_ax = ivc_c - ao_c
    r_ax = r_ax - (r_ax @ si) * si
    if np.linalg.norm(r_ax) < 1e-6:
        return {}
    r_ax = r_ax / np.linalg.norm(r_ax)
    a_ax = centroid(vena_porta) - ivc_c
    a_ax = a_ax - (a_ax @ r_ax) * r_ax
    if np.linalg.norm(a_ax) < 1e-6:
        a_ax = np.cross(si, r_ax)
    a_ax = a_ax / np.linalg.norm(a_ax)
    s_ax = np.cross(r_ax, a_ax)
    a_ax = np.cross(s_ax, r_ax)

    # --- left vs right: largest component on each side of the bifurcation -----
    largest = {True: None, False: None}   # True = right (d >= 0)
    sizes = [len(c) for c in comps]
    for i in np.argsort(sizes)[::-1]:
        d = float(np.dot((to_mm(comps[i].mean(0)) - to_mm(bif)), r_ax))
        side = d >= 0
        if largest[side] is None:
            largest[side] = i
    if largest[True] is None or largest[False] is None:
        return {}
    left_pts, right_pts = comps[largest[False]], comps[largest[True]]

    # --- split the right component at its first branchpoint -------------------
    r_bp = _branchpoints(right_pts)
    if len(r_bp) == 0:
        return {}
    fork = r_bp[int(np.argmin(cKDTree(to_mm(r_bp)).query(to_mm(bif))[0]))]
    keep = np.linalg.norm(to_mm(right_pts) - to_mm(fork), axis=1) >= trunk_radius_mm
    ants, posts = [], []
    for c in _components(right_pts[keep], portal_tree.shape):
        if len(c) < MIN_COMP_PTS:
            continue
        d = float(np.dot((to_mm(c.mean(0)) - to_mm(fork)), a_ax))
        (posts if d < 0 else ants).append(c)
    if not ants or not posts:
        return {}

    # --- Voronoi assignment of the union voxels to the seed skeletons ---------
    trunk_region = portal_tree.copy()
    pv = np.argwhere(portal_tree)
    d_trunk = cKDTree(porta_mm).query(to_mm(pv))[0]
    trunk_region[tuple(pv[d_trunk < trunk_radius_mm].T)] = False
    pv = np.argwhere(trunk_region)

    seeds: dict[str, np.ndarray] = {
        "classPediculoPortalIzquierdo": left_pts,
        "classPediculoPortalAnteriorDerecho": np.concatenate(ants),
        "classPediculoPortalPosteriorDerecho": np.concatenate(posts),
    }
    ranges: dict[str, tuple[int, int]] = {}
    stacked = []
    for name, pts in seeds.items():
        a = len(stacked)
        stacked.extend(to_mm(pts))
        ranges[name] = (a, a + len(pts))
    kd = cKDTree(np.asarray(stacked))
    _, j = kd.query(to_mm(pv))

    out: dict[str, np.ndarray] = {}
    for name, (a, b) in ranges.items():
        m = np.zeros(portal_tree.shape, bool)
        m[tuple(pv[(j >= a) & (j < b)].T)] = True
        out[name] = m
    return out


def _frame_axes(
    liver: np.ndarray, vena_porta: np.ndarray, ivc: np.ndarray, aorta: np.ndarray,
    zooms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Right- and Anterior-pointing unit vectors, in voxel-index space.

    Recovered from the anatomy via ``anatomy_frame.estimate_frame`` -- the same
    frame the Couinaud stage itself trusts -- rather than the NIfTI affine
    (unreliable per that module's own warning) or an ad hoc ``ivc - aorta``
    heuristic: measured on case 0003, that heuristic's axis has cosine
    similarity 0.03 with the true left/right pedicle split (i.e. no
    correlation at all), because the vena_porta label already extends through
    most of the tree here and drags the naive centroid geometry off-axis.
    ``estimate_frame`` jointly fits R/A/S from several independent anatomical
    relations at once, which is what makes it robust to that."""
    if not (liver.any() and ivc.any() and aorta.any()):
        return None
    masks = {"classLiver": liver, "classVenaPorta": vena_porta,
              "classIVC": ivc, "classAorta": aorta}
    try:
        frame = estimate_frame(masks, zooms=zooms)
    except ValueError:
        return None
    return frame.unit("R"), frame.unit("A")


MIN_SPLIT_FRAC = 0.15


def _otsu_split(values: np.ndarray) -> np.ndarray:
    """Boolean mask, True on the "high" side. Otsu finds the natural valley in
    a skewed distribution; a plain median forces an exact 50/50 split even
    when the true boundary sits off-centre (as it does for the ant/post
    pedicles), and a fixed threshold at some anatomical point is fragile when
    that point is poorly localised.

    Falls back to the median if Otsu is degenerate (e.g. all but one value
    identical) or, more commonly, if it finds a valley so far off to one side
    that a segment would starve: liver-gating a GT case once left Otsu
    splitting 1,223 voxels into 18/1,205, which downstream lost the whole
    right-anterior sector (V and VIII) to having no seeds at all. A median
    guarantees every side keeps a fair share; Otsu is preferred only when it
    doesn't come at that cost."""
    try:
        thr = threshold_otsu(values)
        mask = values >= thr
        frac = float(mask.mean())
        if min(frac, 1.0 - frac) < MIN_SPLIT_FRAC:
            raise ValueError("Otsu split too skewed")
    except ValueError:
        thr = np.median(values)
        mask = values >= thr
    return mask


def _plane_split(
    portal_tree: np.ndarray,
    vena_porta: np.ndarray,
    ivc: np.ndarray,
    aorta: np.ndarray,
    zooms: tuple[float, float, float],
    liver: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fallback: cut the union with the recovered anatomical R-L/A-P planes.
    Predicted unions are thick blobs (skeleton length << voxel count, no
    branchpoints), so skeleton splitting fails; the Couinaud stage reads the
    pedicle masks only for the sector *identity* and performs the fine splits
    itself (graph cuts on the pedicle skeletons), so a plane cut is a
    legitimate fallback -- the pipeline's own docs put plane cuts second in
    the cascade.

    The split is computed on ``portal_tree & ~vena_porta`` -- the trunk
    (``classVenaPorta``) is not a short stub here (it swallows most of the
    tree, per the contour-overwrite behaviour ``labels.py`` documents), so
    including it drowns the actual left/right pedicle signal.

    Both inputs are first restricted to their largest connected component.
    Predicted channels carry scattered false-positive fragments far from the
    liver (measured 5-13% of voxels, up to ~90 components, on a couinaud5
    prediction) -- GT masks don't have this, so it never showed up testing
    against GT. Unfiltered, those fragments' skeleton points become seeds at
    wherever they happen to sit (one case put a "left" seed cluster 300 mm
    away in Z), which starves the segments whose seeds got scattered and lets
    whichever segment kept a coherent seed cloud claim the whole liver via
    nearest-seed Voronoi. The Couinaud stage already does the same
    largest-component filtering for its own liver mask; this mirrors it.

    ``pedicle_only`` is then also gated by ``liver``. On the cornerstone GT
    masks this is a no-op -- the hard-partition label schema already puts the
    pedicle remnants (12-15) entirely inside INTRAHEPATIC by construction, so
    excluding the trunk was equivalent. Predictions have no such guarantee:
    the network is free to predict extrahepatic portal trunk as part of the
    union (which is anatomically real -- the portal trunk *is* extrahepatic
    before the hilum), and measured directly, the derived "left" sector
    without this gate sat at R-projection 98-169 mm against a predicted liver
    spanning 142-328 mm -- almost entirely *outside* the liver -- which starved
    segments II/III/IV of any seeds inside the organ at all."""
    zooms = np.asarray(zooms, float)
    to_mm = lambda p: p * zooms
    if liver is None:
        liver = np.zeros_like(portal_tree)
    portal_tree = _largest_component(portal_tree) if portal_tree.any() else portal_tree
    vena_porta = _largest_component(vena_porta) if vena_porta.any() else vena_porta
    liver_lcc = _largest_component(liver) if liver.any() else liver
    axes = _frame_axes(liver_lcc, vena_porta, ivc, aorta, zooms)
    if axes is None:
        return {}
    r_ax, a_ax = axes

    pedicle_only = portal_tree & ~vena_porta & liver_lcc
    base = pedicle_only if int(pedicle_only.sum()) >= MIN_COMP_PTS else portal_tree
    pv = np.argwhere(base)
    if len(pv) < MIN_COMP_PTS:
        return {}

    d = to_mm(pv.astype(float)) @ r_ax
    right_mask = _otsu_split(d)
    left = base.copy(); left[tuple(pv[right_mask].T)] = False
    right = base.copy(); right[tuple(pv[~right_mask].T)] = False

    pv_r = pv[right_mask]
    if len(pv_r) < MIN_COMP_PTS:
        return {}
    a = to_mm(pv_r.astype(float)) @ a_ax
    ant_mask = _otsu_split(a)
    ant = right.copy(); ant[tuple(pv_r[~ant_mask].T)] = False
    post = right.copy(); post[tuple(pv_r[ant_mask].T)] = False
    if not ant.any() or not post.any() or not left.any():
        return {}
    return {"classPediculoPortalIzquierdo": left,
            "classPediculoPortalAnteriorDerecho": ant,
            "classPediculoPortalPosteriorDerecho": post}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", type=Path, required=True)
    args = ap.parse_args()

    def load(name: str) -> np.ndarray:
        p = args.case_dir / f"{name}.nii.gz"
        if not p.exists():
            # predict.py writes "classIvc"; GT-derived case dirs write "classIVC"
            # (labels.json's own casing) -- accept either.
            matches = [q for q in args.case_dir.glob("class*.nii.gz")
                       if q.name.lower() == f"{name.lower()}.nii.gz"]
            if not matches:
                return np.zeros((0, 0, 0), bool)
            p = matches[0]
        return np.asarray(nib.load(str(p)).dataobj).astype(bool)

    portal = load("classPortalTree")
    porta = load("classVenaPorta")
    ivc = load("classIvc")
    aorta = load("classAorta")
    liver = load("classLiver")
    if not portal.any() or not porta.any():
        print("need classPortalTree and classVenaPorta", file=sys.stderr)
        return 2
    affine = nib.load(str(args.case_dir / "classPortalTree.nii.gz")).affine
    zooms = tuple(float(z) for z in np.sqrt((affine[:3, :3] ** 2).sum(0)))
    masks = derive_pedicle_masks(portal, porta, ivc, aorta, zooms=zooms)
    if not masks:
        masks = _plane_split(portal, porta, ivc, aorta, zooms=zooms, liver=liver)
        if masks:
            print("  (skeleton split failed -> plane-cut fallback)")
    for name, m in masks.items():
        nib.save(nib.Nifti1Image(m.astype(np.uint8), affine),
                 str(args.case_dir / f"{name}.nii.gz"))
        print(f"  {name:32s} {int(m.sum()):7d} voxels")
    print(f"wrote {len(masks)} pedicle masks to {args.case_dir}")
    return 0 if masks else 1


if __name__ == "__main__":
    raise SystemExit(main())