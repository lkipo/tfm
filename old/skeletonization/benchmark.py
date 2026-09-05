"""C1 — skeletonization benchmark on real hepatic vessel masks.

Compares 3D skeletonization methods on the MSD/Zenodo modified vessel masks
(clean, liver-only). Methods:
  * skimage_skeletonize  — skimage.morphology.skeletonize (Lee 3D thinning)
  * skimage_medial_axis  — skeletonize(method='lee') vs medial-axis (2D only) -> skip 3D
  * kimimaro             — TEASAR (Sato/Zhao), produces a connected skeleton graph

DeepVesselNet centerline head is NOT included: that task was never trained
(lab diary: only the seg head exists). Noted as a gap.

Coverage / quality metrics (no external GT skeleton needed; the mask is GT):
  * n_skel_vox      skeleton size
  * components      #connected components of the skeleton (26-conn). Fewer,
                    closer to the #vessel components, is better.
  * centeredness    mean EDT (distance-to-surface) at skeleton voxels / mean EDT
                    over all vessel voxels. >1 means the skeleton sits on thicker
                    (more central) medial regions, as it should.
  * coverage        fraction of vessel voxels within their local radius of a
                    skeleton voxel (is every part of the vessel represented?).
  * recon_dice      dilate each skeleton voxel by its local EDT radius and
                    compare the union to the original mask (Dice). This is the
                    key "does skeleton+radius reconstruct the vessel" metric.
  * time_s          wall-clock.

Run (dvn env, has skimage + kimimaro + scipy):
  ~/.virtualenvs/dvn/bin/python -m skeletonization.benchmark
"""

from __future__ import annotations

import glob
import os
import time
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

MASK_DIR = "data/clinical/Task08_HepaticVessel/labelsTr"


# --------------------------------------------------------------------------
# skeletonizers -> return (N,3) int voxel coords
# --------------------------------------------------------------------------

def skel_skimage(mask):
    from skimage.morphology import skeletonize
    s = skeletonize(mask.astype(bool))
    return np.argwhere(s)


def _kimimaro(mask, scale, const):
    import kimimaro
    return kimimaro.skeletonize(
        mask.astype(np.uint8),
        teasar_params={"scale": scale, "const": const, "pdrf_scale": 100000,
                       "pdrf_exponent": 4, "soma_detection_threshold": 0},
        dust_threshold=0, progress=False, parallel=1,
    )


def skel_kimimaro(mask):
    # default TEASAR pruning
    skels = _kimimaro(mask, scale=1.5, const=300)
    return _kimi_points(skels)


def skel_kimimaro_lowprune(mask):
    # minimal pruning -> keep fine branches (fairer coverage comparison)
    return _kimi_points(_kimimaro(mask, scale=1.0, const=10))


def _kimi_points(skels):
    if not skels:
        return np.zeros((0, 3), int)
    pts = []
    for lbl, sk in skels.items():
        v = sk.vertices  # physical coords = voxel index (anisotropy=1)
        # densify the polyline: sample ~1-voxel steps along every edge, else the
        # sparse graph vertices badly under-represent the centreline.
        for a, b in sk.edges:
            pa, pb = v[a], v[b]
            L = np.linalg.norm(pb - pa)
            n = max(int(np.ceil(L)), 1)
            for t in np.linspace(0, 1, n + 1):
                pts.append(pa + t * (pb - pa))
    if not pts:
        return np.zeros((0, 3), int)
    P = np.round(np.vstack(pts)).astype(int)
    return np.unique(P, axis=0)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _components(coords, radius=np.sqrt(3) + 1e-6):
    import networkx as nx
    if len(coords) == 0:
        return 0
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    tree = cKDTree(coords)
    for i, j in tree.query_pairs(radius):
        G.add_edge(i, j)
    return nx.number_connected_components(G)


def evaluate(mask, coords):
    vessel = np.argwhere(mask > 0)
    edt = distance_transform_edt(mask > 0)
    out = {"n_skel_vox": len(coords)}
    if len(coords) == 0:
        return {**out, "components": 0, "centeredness": 0, "coverage": 0,
                "recon_dice": 0}

    # centeredness
    skel_edt = edt[coords[:, 0], coords[:, 1], coords[:, 2]]
    mean_edt_vessel = edt[vessel[:, 0], vessel[:, 1], vessel[:, 2]].mean()
    out["centeredness"] = float(skel_edt.mean() / (mean_edt_vessel + 1e-9))

    # coverage: a vessel voxel is represented if it lies within the local vessel
    # radius (EDT at the nearest skeleton voxel = centreline calibre there) of a
    # skeleton voxel. Surface voxels of a thick vessel are far from the centre
    # but still covered by that vessel's centreline.
    tree = cKDTree(coords.astype(float))
    dist, nn = tree.query(vessel.astype(float), k=1)
    skel_r = edt[coords[:, 0], coords[:, 1], coords[:, 2]]
    covered_r = skel_r[nn] + 1.0
    out["coverage"] = float((dist <= covered_r).mean())

    # reconstruction Dice: dilate each skeleton voxel by its local radius
    recon = np.zeros_like(mask, dtype=bool)
    R = np.ceil(skel_edt).astype(int)
    shape = np.array(mask.shape)
    # precompute offset balls per integer radius
    maxr = int(R.max())
    balls = {}
    for r in range(1, maxr + 1):
        rng = np.arange(-r, r + 1)
        dz, dy, dx = np.meshgrid(rng, rng, rng, indexing="ij")
        m = (dz**2 + dy**2 + dx**2) <= r * r
        balls[r] = np.stack([dz[m], dy[m], dx[m]], axis=1)
    for c, r in zip(coords, R):
        r = max(int(r), 1)
        pts = c + balls[r]
        ok = np.all((pts >= 0) & (pts < shape), axis=1)
        pts = pts[ok]
        recon[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    inter = np.logical_and(recon, mask > 0).sum()
    out["recon_dice"] = float(2 * inter / (recon.sum() + (mask > 0).sum() + 1e-9))
    out["components"] = _components(coords)
    return out


def main():
    masks = sorted(glob.glob(os.path.join(MASK_DIR, "*_mod.nii.gz")))
    methods = {"skimage": skel_skimage, "kimimaro": skel_kimimaro,
               "kimimaro_lowprune": skel_kimimaro_lowprune}
    agg = {m: [] for m in methods}
    print(f"Benchmarking {len(masks)} masks\n")
    for mp in masks:
        case = os.path.basename(mp).replace("_mod.nii.gz", "")
        # label 1 = vessel, label 2 = tumour -> skeletonise vessels only
        mask = (nib.load(mp).get_fdata() == 1).astype(np.uint8)
        for name, fn in methods.items():
            t0 = time.time()
            coords = fn(mask)
            dt = time.time() - t0
            met = evaluate(mask, coords)
            met["time_s"] = dt
            agg[name].append(met)
    keys = ["n_skel_vox", "components", "centeredness", "coverage", "recon_dice", "time_s"]
    print(f"{'method':<12} " + " ".join(f"{k:>12}" for k in keys))
    print("-" * (12 + 13 * len(keys)))
    for name in methods:
        rows = agg[name]
        means = {k: np.mean([r[k] for r in rows]) for k in keys}
        print(f"{name:<12} " + " ".join(f"{means[k]:>12.3f}" for k in keys))


if __name__ == "__main__":
    main()
