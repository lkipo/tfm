"""Render static 3D PNG previews of each viewer demo (headless, Agg).

A quick-look so the demos can be judged without launching the interactive viewer.
Each PNG shows the demo's layers in distinct colours from a couple of angles.
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
       "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _pts(f):
    d = json.load(open(f))
    return np.array(d, dtype=float)


def _branch_pts(f):
    d = json.load(open(f))
    segs = []
    for b in d:
        segs.append((np.array(b["start"], float), np.array(b["end"], float)))
    return segs


def scene(point_files, branch_files, title, out, elev=20, azim=-60, psize=1.5):
    fig = plt.figure(figsize=(11, 5.5))
    for k, az in enumerate((azim, azim + 90)):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for i, f in enumerate(point_files):
            p = _pts(f)
            if len(p) == 0:
                continue
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=psize, c=PAL[i % len(PAL)],
                       label=os.path.basename(f).replace(".json", ""), alpha=0.6)
        for f in (branch_files or []):
            for s, e in _branch_pts(f):
                ax.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]],
                        c="#d62728", lw=2.0)
        ax.view_init(elev=elev, azim=az)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        if k == 0:
            ax.legend(loc="upper left", fontsize=6, markerscale=3)
    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("  wrote", os.path.relpath(out, ROOT))


def main():
    D = ROOT
    # 1. crossing
    for tag in ("interlaced_marginp5", "separated_margin4"):
        pf = sorted(glob.glob(f"{D}/synthetic_3tree_crossing/{tag}_*.json"))
        if pf:
            scene(pf, None, f"3-tree {tag} (portal/hepatic/artery)",
                  f"{D}/synthetic_3tree_crossing/preview_{tag}.png")
    # 2. separation
    gt = sorted(glob.glob(f"{D}/synthetic_separation/gt_tree*.json"))
    cl = sorted(glob.glob(f"{D}/synthetic_separation/geodesic_cluster*.json"))
    if gt:
        scene(gt, None, "Separation — GROUND TRUTH (3 trees)",
              f"{D}/synthetic_separation/preview_gt.png")
    if cl:
        scene(cl, None, "Separation — geodesic clusters (balanced, ~0.5 purity)",
              f"{D}/synthetic_separation/preview_clusters.png")
    # 3. synthetic reconnection
    fr = f"{D}/synthetic_reconnection/fragmented_skeleton.json"
    li = f"{D}/synthetic_reconnection/cco_added_links.json"
    if os.path.exists(fr):
        scene([fr], [li], "Synthetic reconnection: fragments + CCO bridges (red)",
              f"{D}/synthetic_reconnection/preview.png", psize=0.6)
    # 4. real reconnection
    for c in ("001", "007"):
        base = f"{D}/real_reconnection/hepaticvessel_{c}"
        sk = f"{base}/revised_skeleton.json"; li = f"{base}/cco_added_links.json"
        if os.path.exists(sk):
            scene([sk], [li], f"Real case {c}: skeleton + CCO bridges (red)",
                  f"{base}/preview.png", psize=1.0)
    # 5. fragmentation
    fc = sorted(glob.glob(f"{D}/real_fragmentation/001_comp*.json"))
    if fc:
        scene(fc, None, "Real skeleton fragmentation (10 largest components)",
              f"{D}/real_fragmentation/preview.png", psize=1.5)


if __name__ == "__main__":
    main()
