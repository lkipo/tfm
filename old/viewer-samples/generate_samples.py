"""Generate a series of viewer-ready samples for visual inspection.

Produces layered files for skeleton-viewer/multi_viewer.py covering the main
results of the reconnection work. Each demo folder holds several files (one per
tree/cluster/layer, so the viewer auto-colours them) plus a VIEW.txt with the
exact launch command.

Formats (per multi_viewer.py):
  * skeleton : JSON  [[x,y,z], ...]                     -> points
  * tree     : JSON  [{"start":[..],"end":[..],"radius":r,"Q":q}, ...] -> lines
  * volume   : binary NIfTI (shows voxels==1)

Run (dvn env):
  ~/.virtualenvs/dvn/bin/python viewer-samples/generate_samples.py
"""

from __future__ import annotations

import json
import os
import sys
import shutil
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import networkx as nx
import nibabel as nib
from skimage.morphology import skeletonize

from reconnection.graphs import (graph_from_tree_txt, reduce_to_topological,
                                 graph_from_points, load_skeleton_json, positions)
from reconnection.fragment import break_branches
from reconnection.methods.cco import reconnect_cco
from reconnection import separate as S

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "data")
ZEN = os.path.join(DATA, "clinical", "Task08_HepaticVessel")


def _ipts(arr):
    return [[int(round(c)) for c in p] for p in np.asarray(arr)]


def save_points(arr, path):
    json.dump(_ipts(arr), open(path, "w"))


def save_branches(edges, pos, path, radius=1.0):
    br = []
    for u, v in edges:
        br.append({"start": [float(x) for x in pos[u]],
                   "end": [float(x) for x in pos[v]],
                   "radius": float(radius), "Q": 1.0})
    json.dump(br, open(path, "w"))


def tree_points(sample_dir, i):
    """Dense integer skeleton points of Tree i in a sample dir."""
    pts = []
    with open(glob.glob(f"{sample_dir}/Tree{i}.txt")[0]) as fh:
        for line in fh:
            v = [float(x) for x in line.split()]
            if v:
                pts.append(np.asarray(v).reshape(-1, 3))
    P = np.vstack(pts)
    return np.unique(np.round(P).astype(int), axis=0)


def write_view(folder, cmd_files, note):
    py = "~/.virtualenvs/viewer/bin/python"
    rel = "skeleton-viewer/multi_viewer.py"
    skels = " ".join(f"--skeleton {f}" for f in cmd_files["skeleton"])
    trees = " ".join(f"--tree {f}" for f in cmd_files.get("tree", []))  # if supported
    vols = " ".join(cmd_files.get("volume", []))
    cmd = f"{py} {rel} {vols} {skels}".strip()
    with open(os.path.join(folder, "VIEW.txt"), "w") as f:
        f.write(note + "\n\n")
        f.write("From the repo root, run:\n\n" + cmd + "\n\n")
        if cmd_files.get("tree"):
            f.write("Tree (line) files — add each in the viewer UI as a 'tree' "
                    "layer, or load via add_tree():\n")
            for t in cmd_files["tree"]:
                f.write("  " + t + "\n")


# ---------------------------------------------------------------------------
# DEMO 1 — synthetic 3-tree crossing (A1 fix): interlaced vs separated
# ---------------------------------------------------------------------------

def demo_crossing():
    d = os.path.join(ROOT, "synthetic_3tree_crossing")
    os.makedirs(d, exist_ok=True)
    names = {1: "portal", 2: "hepatic", 3: "artery"}
    srcs = {"interlaced_marginp5": "/tmp/a1ctrl", "separated_margin4": "/tmp/a1_m4.0"}
    made = []
    for tag, sd in srcs.items():
        if not glob.glob(f"{sd}/Tree1.txt"):
            continue
        for i in (1, 2, 3):
            p = os.path.join(d, f"{tag}_{names[i]}.json")
            save_points(tree_points(sd, i), p)
            made.append(os.path.relpath(p, os.path.join(ROOT, "..")))
    inter = [f for f in made if "interlaced" in f]
    sep = [f for f in made if "separated" in f]
    write_view(d, {"skeleton": sep},
               "DEMO 1 — A1 crossing fix. Load the three SEPARATED trees "
               "(margin 4): portal/hepatic/artery should occupy distinct, "
               "non-crossing territories. Then compare with the three "
               "INTERLACED trees (margin 0.5) which weave through each other.\n"
               "Interlaced files:\n  " + "\n  ".join(inter))
    print(f"[demo_crossing] {len(made)} files -> {d}")


# ---------------------------------------------------------------------------
# DEMO 2 — synthetic multi-tree separation quality (GT vs geodesic clusters)
# ---------------------------------------------------------------------------

def _labelled_skel(base):
    seg = nib.load(f"{base}/seg.nii.gz").get_fdata() > 0
    vl = nib.load(f"{base}/vessel_labels.nii.gz").get_fdata().astype(int)
    pts = np.argwhere(skeletonize(seg))
    lbl = vl[pts[:, 0], pts[:, 1], pts[:, 2]]
    keep = lbl > 0
    return pts[keep], lbl[keep]


def demo_separation():
    d = os.path.join(ROOT, "synthetic_separation")
    os.makedirs(d, exist_ok=True)
    base = "/tmp/a1_m4.0"
    if not os.path.exists(f"{base}/seg.nii.gz"):
        print("[demo_separation] skipped (no seg)")
        return
    pts, lbl = _labelled_skel(base)
    gt_files, cl_files = [], []
    for t in (1, 2, 3):
        p = os.path.join(d, f"gt_tree{t}.json")
        save_points(pts[lbl == t], p); gt_files.append(os.path.relpath(p, os.path.join(ROOT, "..")))
    cl = S.root_geodesic(pts.astype(float), n_trees=3)
    carr = np.array([cl[i] for i in range(len(pts))])
    for c in range(3):
        p = os.path.join(d, f"geodesic_cluster{c}.json")
        save_points(pts[carr == c], p); cl_files.append(os.path.relpath(p, os.path.join(ROOT, "..")))
    write_view(d, {"skeleton": gt_files},
               "DEMO 2 — multi-tree separation. gt_tree{1,2,3} are the TRUE trees "
               "(3 colours). geodesic_cluster{0,1,2} are the unsupervised "
               "separation attempt — compare: clusters are size-balanced but leak "
               "across the true trees (~0.5 purity). Load the gt set and the "
               "cluster set in two passes.\nCluster files:\n  " + "\n  ".join(cl_files))
    print(f"[demo_separation] GT+clusters -> {d}")


# ---------------------------------------------------------------------------
# DEMO 3 — synthetic single-tree reconnection (fragment -> CCO reconnect)
# ---------------------------------------------------------------------------

def demo_recon_synth():
    d = os.path.join(ROOT, "synthetic_reconnection")
    os.makedirs(d, exist_ok=True)
    T = reduce_to_topological(graph_from_tree_txt("data/synthetic/sample_01/Tree1.txt"))
    F, gt_pairs, _ = break_branches(T, n_breaks=40, seed=0)
    pos = {n: F.nodes[n]["pos"] for n in F.nodes}
    # dense points of the fragmented skeleton (from branch paths)
    frag_pts = []
    for u, v, dta in F.edges(data=True):
        pp = dta.get("path_pos")
        frag_pts.append(np.asarray(pp) if pp is not None else np.vstack([pos[u], pos[v]]))
    frag_pts = np.vstack(frag_pts)
    save_points(frag_pts, os.path.join(d, "fragmented_skeleton.json"))
    proposed = reconnect_cco(F)
    save_branches(proposed, pos, os.path.join(d, "cco_added_links.json"), radius=1.5)
    save_branches(list(F.edges()), pos, os.path.join(d, "original_branches.json"), radius=0.6)
    write_view(d, {"skeleton": ["data/../viewer-samples/synthetic_reconnection/fragmented_skeleton.json"],
                   "tree": ["viewer-samples/synthetic_reconnection/original_branches.json",
                            "viewer-samples/synthetic_reconnection/cco_added_links.json"]},
               f"DEMO 3 — single-tree reconnection. fragmented_skeleton.json = the "
               f"broken input ({len(gt_pairs)} gaps). cco_added_links.json = the "
               f"{len(proposed)} bridges CCO proposed (load as a TREE layer to see "
               f"them as thick lines spanning the gaps). original_branches.json = "
               f"the fragment branches.")
    print(f"[demo_recon_synth] frag+links -> {d}")


# ---------------------------------------------------------------------------
# DEMO 4 — real Zenodo reconnection (revised skeleton -> CCO reconnect)
# ---------------------------------------------------------------------------

def demo_recon_real(cases=("001", "007")):
    base = os.path.join(ROOT, "real_reconnection")
    for c in cases:
        skp = glob.glob(f"{ZEN}/skeletons/hepaticvessel_{c}_*.json")
        if not skp:
            continue
        d = os.path.join(base, f"hepaticvessel_{c}")
        os.makedirs(d, exist_ok=True)
        skel = load_skeleton_json(skp[0])
        shutil.copy(skp[0], os.path.join(d, "revised_skeleton.json"))
        # vessel_gt mask (vessels only) as a binary volume
        mask = nib.load(f"{ZEN}/labelsTr/hepaticvessel_{c}_mod.nii.gz")
        vgt = (mask.get_fdata() == 1).astype(np.uint8)
        nib.save(nib.Nifti1Image(vgt, mask.affine, mask.header),
                 os.path.join(d, "vessel_gt.nii.gz"))
        # reconnect
        Gp = graph_from_points(skel.astype(float))
        R = reduce_to_topological(Gp)
        pos = {n: R.nodes[n]["pos"] for n in R.nodes}
        proposed = reconnect_cco(R)
        save_branches(proposed, pos, os.path.join(d, "cco_added_links.json"), radius=2.0)
        rel = f"viewer-samples/real_reconnection/hepaticvessel_{c}"
        write_view(d, {"volume": [f"{rel}/vessel_gt.nii.gz"],
                       "skeleton": [f"{rel}/revised_skeleton.json"],
                       "tree": [f"{rel}/cco_added_links.json"]},
                   f"DEMO 4 — REAL reconnection (case {c}). vessel_gt.nii.gz = the "
                   f"annotated vessel mask; revised_skeleton.json = the Zenodo "
                   f"skeleton ({len(skel)} pts, fragmented); cco_added_links.json = "
                   f"the {len(proposed)} bridges CCO added (TREE layer). Inspect "
                   f"whether the bridges follow plausible vessel continuations.")
        print(f"[demo_recon_real {c}] -> {d}")


# ---------------------------------------------------------------------------
# DEMO 5 — real skeleton fragmentation (components coloured)
# ---------------------------------------------------------------------------

def demo_fragmentation(case="001", top=10):
    d = os.path.join(ROOT, "real_fragmentation")
    os.makedirs(d, exist_ok=True)
    skp = glob.glob(f"{ZEN}/skeletons/hepaticvessel_{case}_*.json")[0]
    skel = load_skeleton_json(skp)
    G = graph_from_points(skel.astype(float))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)[:top]
    files = []
    for i, comp in enumerate(comps):
        idx = sorted(comp)
        p = os.path.join(d, f"{case}_comp{i:02d}_n{len(idx)}.json")
        save_points(skel[idx], p)
        files.append(os.path.relpath(p, os.path.join(ROOT, "..")))
    write_view(d, {"skeleton": files},
               f"DEMO 5 — REAL skeleton fragmentation (case {case}). The {top} "
               f"largest connected components of the revised skeleton, each a "
               f"colour. Shows how a real hepatic skeleton breaks into many "
               f"pieces — the motivation for reconnection.")
    print(f"[demo_fragmentation {case}] {len(files)} comps -> {d}")


def main():
    os.chdir(os.path.join(ROOT, ".."))  # run from repo root for data paths
    demo_crossing()
    demo_separation()
    demo_recon_synth()
    demo_recon_real()
    demo_fragmentation()
    # top-level README
    with open(os.path.join(ROOT, "README.md"), "w") as f:
        f.write(VIEWER_README)
    print("\nAll demos generated under viewer-samples/. See README.md.")


VIEWER_README = """# Viewer samples — visual inspection

Generated by `generate_samples.py`. Each subfolder has a `VIEW.txt` with the
exact command. Launch the viewer (needs a browser; use `--no-browser` and open
the printed URL if remote):

    ~/.virtualenvs/viewer/bin/python skeleton-viewer/multi_viewer.py <files...>

Layer types the viewer accepts:
  * skeleton points : `--skeleton file.json`   (JSON list of [x,y,z])
  * volume (mask)   : positional `file.nii.gz`  (binary; shows voxels==1)
  * tree (lines)    : add in the UI as a 'tree' layer, or `viewer.add_tree(...)`
                      (JSON list of {start,end,radius,Q})

Each file is a separate auto-coloured layer, so loading N per-tree files shows
N colours. You can also add/remove files live from the viewer's web UI.

## Demos
1. `synthetic_3tree_crossing/` — A1 fix: 3 non-crossing trees (margin 4) vs 3
   interlaced trees (margin 0.5).
2. `synthetic_separation/`     — true 3 trees vs the unsupervised separation
   attempt (balanced but ~0.5 purity).
3. `synthetic_reconnection/`   — a fragmented synthetic tree + the CCO bridges.
4. `real_reconnection/`        — Zenodo skeleton + vessel mask + CCO bridges
   (cases 001, 007).
5. `real_fragmentation/`       — a real skeleton's connected components, coloured
   (why reconnection is needed).
"""


if __name__ == "__main__":
    main()
