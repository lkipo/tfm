"""
Figures for the Couinaud pipeline.

Every view is drawn in the *recovered* anatomical frame (see `anatomy_frame.py`),
so "axial", "coronal" and "sagittal" mean what they say even though the NIfTI
affines in this dataset do not.

Colour: the eight segments use the dataviz reference categorical palette. The
segment -> slot mapping was chosen by maximising the worst OKLab Delta E over the
pairs of segments that are actually *spatially adjacent* in the liver (a map only
needs neighbours to be separable), not over slot order. The best achievable worst
adjacent pair is CVD Delta E 7.2, inside the 6-8 warn band, so secondary encoding is
mandatory and non-optional here: every slice panel carries direct roman-numeral
labels at each segment's in-slice centroid, a legend is always present, and the
volume table repeats the same numbers in text.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import matplotlib.patheffects as patheffects   # noqa: E402
from matplotlib.colors import ListedColormap   # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection   # noqa: E402
from scipy import ndimage   # noqa: E402

from couinaud import (SEG_COLOR_LIGHT, SEG_DESCRIPTION, SEG_IDS, SEG_NAMES,
                      SEG_REFERENCE_PCT, Case, Result, to_mm)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e5e4e0"

SEG_CMAP = ListedColormap([SURFACE] + [SEG_COLOR_LIGHT[s] for s in SEG_IDS])

SKEL_STYLE = {
    "portal_left": ("#2a78d6", "left portal pedicle"),
    "portal_right": ("#4a3aa7", "right portal trunk"),
    "portal_right_ant": ("#eb6834", "right anterior pedicle"),
    "portal_right_post": ("#008300", "right posterior pedicle"),
    "portal_trunk": ("#8a8983", "main portal vein"),
    "hv_mhv": ("#e34948", "middle hepatic vein"),
    "hv_rhv": ("#e87ba4", "right hepatic vein"),
    "hv_lhv": ("#eda100", "left hepatic vein"),
    "ivc": ("#52514e", "IVC axis"),
}


def _style(ax, title: str = "", three_d: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, fontsize=15.8, color=INK, pad=6)
    if three_d:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor(SURFACE)
            pane.pane.set_edgecolor(GRID)
        ax.grid(False)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def _new_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return fig


def _legend(ax, entries, **kw):
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5,
                      markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in entries]
    leg = ax.legend(handles=handles, fontsize=12.2, frameon=False, labelcolor=INK_2, **kw)
    return leg


def _subsample(points, n, seed=0):
    if len(points) <= n:
        return points
    return points[np.random.default_rng(seed).choice(len(points), n, replace=False)]


def _plane_quad(plane, centre, radius):
    """Two in-plane basis vectors -> a square patch for drawing."""
    n = plane.normal
    helper = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, helper); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    origin = centre - n * float((centre - plane.origin) @ n)
    return np.array([origin + radius * (a * u + b * v)
                     for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


# --------------------------------------------------------------------------- #
# 1. orientation check
# --------------------------------------------------------------------------- #

def figure_frame(case: Case, path: Path) -> Path:
    """Mid-slices in the recovered frame, with the landmarks that fixed each axis."""
    frame = case.frame
    liver = frame.reorient(case.masks["classLiver"])
    overlay = {"classIVC": "#e34948", "classAorta": "#4a3aa7", "classVenaPorta": "#2a78d6",
               "classGallbladder": "#eda100", "classVenaHepaticaMedia": "#1baf7a"}
    centre = np.round(ndimage.center_of_mass(liver)).astype(int)

    views = [("axial (viewed from the feet)", 2, 0, 1, "patient right →", "anterior →"),
             ("coronal (viewed from the front)", 1, 0, 2, "patient right →", "superior →"),
             ("sagittal (viewed from the left)", 0, 1, 2, "anterior →", "superior →")]

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURFACE)
    for ax, (title, fixed, hx, vy, xl, yl) in zip(axs, views):
        sl = [slice(None)] * 3
        sl[fixed] = int(centre[fixed])
        img = liver[tuple(sl)]
        ax.imshow(img.T, cmap="gray_r", origin="lower", alpha=0.25, interpolation="nearest")
        for name, colour in overlay.items():
            m = case.masks.get(name)
            if m is None or not m.any():
                continue
            plane = frame.reorient(m)[tuple(sl)]
            yy, xx = np.where(plane.T)
            if len(xx):
                ax.scatter(xx, yy, s=1.2, c=colour, linewidths=0)
        _style(ax, title)
        ax.set_xlabel(xl, fontsize=12.2, color=INK_MUTED)
        ax.set_ylabel(yl, fontsize=12.2, color=INK_MUTED)
    _legend(axs[-1], [(n.replace("class", ""), c) for n, c in overlay.items()],
            loc="upper right")

    warn = ("  ·  ".join(case.frame.warnings)) if case.frame.warnings else "no warnings"
    fig.suptitle(f"{case.name} — orientation check\n"
                 f"header says {tuple('RAS')} but the anatomy says {frame.describe()}",
                 fontsize=17.5, color=INK, y=1.03)
    fig.text(0.5, -0.03, warn, ha="center", fontsize=12.2, color=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 2. construction geometry
# --------------------------------------------------------------------------- #

def figure_geometry(case: Case, res: Result, path: Path) -> Path:
    """
    Top row: the skeletons the pipeline works from, plus the bifurcation and the
    largest-radius trunk walk. Bottom row: one panel per construction, because four
    translucent planes in a single 3D scatter read as mud.
    """
    liver_pts = _subsample(to_mm(np.argwhere(res.liver), case), 7000)
    centre = liver_pts.mean(axis=0)

    r_hat, a_hat, s_hat = case.unit("R"), case.unit("A"), case.unit("S")
    basis = np.stack([r_hat, a_hat, s_hat])          # world -> (R, A, S)
    proj = lambda p: np.asarray(p) @ basis.T          # noqa: E731
    lp = proj(liver_pts)
    span = np.ptp(lp, axis=0)
    radius = 0.55 * float(span.max())

    def frame_axes(ax, elev, azim):
        ax.view_init(elev=elev, azim=azim)
        for setter, col in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
            setter(lp[:, col].min(), lp[:, col].max())
        try:
            ax.set_box_aspect(span)
        except Exception:
            pass

    def liver_cloud(ax, alpha=0.12):
        ax.scatter(lp[:, 0], lp[:, 1], lp[:, 2], s=0.7, c="#c9c8c3", alpha=alpha,
                   linewidths=0, depthshade=False)

    def draw_skeletons(ax, keys=None, size=3.5):
        for key, pts in res.skeletons.items():
            if keys is not None and key not in keys:
                continue
            colour, _ = SKEL_STYLE.get(key, ("#8a8983", key))
            q = proj(_subsample(pts, 2500))
            ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=size, c=colour, alpha=0.95,
                       linewidths=0, depthshade=False)

    def draw_plane(ax, plane, colour):
        quad = proj(_plane_quad(plane, centre, radius))
        ax.add_collection3d(Poly3DCollection([quad], facecolor=colour, alpha=0.22,
                                             edgecolor=colour, linewidths=1.4))

    views = [(12, -78, "from the front"), (12, 12, "from the left"),
             (76, -78, "from above")]

    fig = _new_fig(15, 10)
    for k, (elev, azim, name) in enumerate(views):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        liver_cloud(ax)
        draw_skeletons(ax)
        if res.trunk_path_mm is not None and len(res.trunk_path_mm) > 1:
            t = proj(res.trunk_path_mm)
            ax.add_collection3d(Line3DCollection(
                [(t[i], t[i + 1]) for i in range(len(t) - 1)],
                colors="#0b0b0b", linewidths=2.4))
        b = proj(res.bifurcation_mm[None, :])[0]
        ax.scatter(*b, s=90, c="#0b0b0b", marker="X", depthshade=False, zorder=10)
        frame_axes(ax, elev, azim)
        _style(ax, f"skeletons — {name}", three_d=True)

    panels = [
        ("mhv", "#e34948", "Cantlie plane (MHV)\nseparates left from right",
         ("hv_mhv", "ivc"), (76, -78)),
        ("rhv", "#e87ba4", "right scissura (RHV)\nseparates V+VIII from VI+VII",
         ("hv_rhv", "ivc"), (76, -78)),
        ("umbilical", "#008300", "umbilical fissure\nseparates IV from II+III",
         ("portal_left",), (76, -78)),
    ]
    for k, (key, colour, title, skel_keys, (elev, azim)) in enumerate(panels):
        ax = fig.add_subplot(2, 3, k + 4, projection="3d")
        liver_cloud(ax, alpha=0.10)
        draw_skeletons(ax, keys=skel_keys, size=5)
        plane = res.planes.get(key)
        if plane is not None:
            draw_plane(ax, plane, colour)
            note = plane.note
        else:
            note = "not available for this case"
        if key == "umbilical" and res.trunk_path_mm is not None and len(res.trunk_path_mm) > 1:
            t = proj(res.trunk_path_mm)
            ax.add_collection3d(Line3DCollection(
                [(t[i], t[i + 1]) for i in range(len(t) - 1)],
                colors="#0b0b0b", linewidths=2.4))
        frame_axes(ax, elev, azim)
        _style(ax, f"{title}\n{note}", three_d=True)

    entries = [(l, c) for k, (c, l) in SKEL_STYLE.items() if k in res.skeletons]
    entries += [("portal bifurcation / trunk walk", "#0b0b0b")]
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=5,
                               markerfacecolor=c, markeredgecolor="none", label=l)
                        for l, c in entries],
               loc="lower center", ncol=5, fontsize=13.1, frameon=False, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{case.name} — construction geometry", fontsize=19.2, color=INK)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 3. labelled seeds + 3D territories
# --------------------------------------------------------------------------- #

def figure_seeds_and_territories(case: Case, res: Result, path: Path) -> Path:
    """Left: the portal skeleton after labelling. Right: the parenchyma it produced."""
    r_hat, a_hat, s_hat = case.unit("R"), case.unit("A"), case.unit("S")
    basis = np.stack([r_hat, a_hat, s_hat])
    proj = lambda p: np.asarray(p) @ basis.T   # noqa: E731

    labels = res.labels
    # Draw the liver's surface shell rather than its solid interior: a scatter of a
    # million interior voxels renders as one opaque blob, while the shell shows the
    # segment pattern where it is actually legible.
    shell = res.liver & ~ndimage.binary_erosion(res.liver, iterations=2)
    vox = np.argwhere(shell)
    vox = vox[_subsample(np.arange(len(vox)), 40000)]
    seg = labels[tuple(vox.T)]
    vox = vox[seg > 0]
    seg = seg[seg > 0]
    pts = proj(to_mm(vox, case))
    seeds = proj(res.seeds_mm)
    context = proj(_subsample(to_mm(np.argwhere(res.liver), case), 4000))

    views = [(10, -78, "from the front"), (10, 12, "from the left"), (76, -78, "from above")]
    fig = _new_fig(15, 9.5)
    for k, (elev, azim, name) in enumerate(views):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        ax.scatter(context[:, 0], context[:, 1], context[:, 2], s=0.5, c="#d9d8d3",
                   alpha=0.13, linewidths=0, depthshade=False)
        for s in SEG_IDS:
            m = res.seed_labels == s
            if not m.any():
                continue
            ax.scatter(seeds[m, 0], seeds[m, 1], seeds[m, 2], s=14,
                       c=SEG_COLOR_LIGHT[s], linewidths=0, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        _style(ax, f"labelled portal skeleton — {name}", three_d=True)
        try:
            ax.set_box_aspect(np.ptp(pts, axis=0))
        except Exception:
            pass
        for setter, col in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), range(3)):
            setter(pts[:, col].min(), pts[:, col].max())

        ax2 = fig.add_subplot(2, 3, k + 4, projection="3d")
        # painter's order: farthest-first along the view direction so the near face wins
        depth = {0: -pts[:, 1], 1: pts[:, 0], 2: -pts[:, 2]}[k]
        order = np.argsort(depth)
        ax2.scatter(pts[order, 0], pts[order, 1], pts[order, 2], s=2.4,
                    c=[SEG_COLOR_LIGHT[s] for s in seg[order]], alpha=0.9,
                    linewidths=0, depthshade=False)
        ax2.view_init(elev=elev, azim=azim)
        _style(ax2, f"resulting territories — {name}", three_d=True)
        try:
            ax2.set_box_aspect(np.ptp(pts, axis=0))
        except Exception:
            pass
        for setter, col in zip((ax2.set_xlim, ax2.set_ylim, ax2.set_zlim), range(3)):
            setter(pts[:, col].min(), pts[:, col].max())

    pct = res.volume_pct()
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=6,
                               markerfacecolor=SEG_COLOR_LIGHT[s], markeredgecolor="none",
                               label=f"{SEG_NAMES[s]} · {SEG_DESCRIPTION[s]} · {pct[s]:.1f}%")
                        for s in SEG_IDS],
               loc="lower center", ncol=4, fontsize=13.1, frameon=False, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{case.name} — from labelled skeleton to parenchyma\n"
                 f"each liver voxel takes the segment of its nearest portal skeleton node",
                 fontsize=17.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 4. slice montage
# --------------------------------------------------------------------------- #

def figure_slices(case: Case, res: Result, path: Path, n_slices: int = 6) -> Path:
    """Axial montage plus one coronal and one sagittal, every region direct-labelled."""
    frame = case.frame
    labels = frame.reorient(res.labels)
    liver = labels > 0
    s_idx = np.argwhere(liver)[:, 2]
    lo, hi = int(np.percentile(s_idx, 6)), int(np.percentile(s_idx, 94))
    levels = np.linspace(lo, hi, n_slices).astype(int)[::-1]

    rows, cols = 2, max(n_slices // 2 + 1, 4)
    fig, axs = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows), facecolor=SURFACE)
    axs = axs.ravel()

    panels = [("axial", 2, z, "patient right →", "anterior →") for z in levels]
    ca = np.round(ndimage.center_of_mass(liver)).astype(int)
    panels.append(("coronal", 1, int(ca[1]), "patient right →", "superior →"))
    panels.append(("sagittal", 0, int(ca[0]), "anterior →", "superior →"))

    for ax, (kind, fixed, index, xl, yl) in zip(axs, panels):
        sl = [slice(None)] * 3
        sl[fixed] = index
        img = labels[tuple(sl)].T
        ax.imshow(img, cmap=SEG_CMAP, vmin=0, vmax=8, origin="lower",
                  interpolation="nearest")
        for s in SEG_IDS:
            m = img == s
            if m.sum() < 40:
                continue
            cc, n = ndimage.label(m)
            sizes = ndimage.sum_labels(m, cc, index=range(1, n + 1))
            big = int(np.argmax(sizes)) + 1
            cy, cx = ndimage.center_of_mass(cc == big)
            ax.text(cx, cy, SEG_NAMES[s], fontsize=15.8, fontweight="bold", ha="center",
                    va="center", color="#ffffff",
                    path_effects=[patheffects.withStroke(linewidth=2.4,
                                                         foreground="#00000099")])
        _style(ax, f"{kind} @ {index}")
        ax.set_xlabel(xl, fontsize=11.4, color=INK_MUTED)
        ax.set_ylabel(yl, fontsize=11.4, color=INK_MUTED)
    for ax in axs[len(panels):]:
        ax.axis("off")

    pct = res.volume_pct()
    fig.legend(handles=[Line2D([], [], marker="s", linestyle="", markersize=7,
                               markerfacecolor=SEG_COLOR_LIGHT[s], markeredgecolor="none",
                               label=f"{SEG_NAMES[s]} {SEG_DESCRIPTION[s]} ({pct[s]:.1f}%)")
                        for s in SEG_IDS],
               loc="lower center", ncol=4, fontsize=13.1, frameon=False, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(f"{case.name} — Couinaud segments", fontsize=19.2, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 5. volumes vs the literature
# --------------------------------------------------------------------------- #

def figure_volumes(case: Case, res: Result, checks: dict, path: Path) -> Path:
    """One series (this case's share of liver volume) against the published range."""
    pct = res.volume_pct()
    vols = res.volumes_ml()
    order = SEG_IDS[::-1]
    y = np.arange(len(order))

    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    for i, s in enumerate(order):
        lo, hi = SEG_REFERENCE_PCT[s]
        ax.barh(i, hi - lo, left=lo, height=0.72, color="#eceae5", zorder=1)
    for i, s in enumerate(order):
        lo, hi = SEG_REFERENCE_PCT[s]
        inside = lo <= pct[s] <= hi
        ax.barh(i, pct[s], height=0.34, zorder=3,
                color=SEG_COLOR_LIGHT[s] if inside else "#e34948")
        ax.text(pct[s] + 0.45, i, f"{pct[s]:.1f}%", va="center", fontsize=14,
                color=INK if inside else "#e34948",
                fontweight="normal" if inside else "bold")
    ax.set_yticks(y, [f"{SEG_NAMES[s]}" for s in order], fontsize=15.8, color=INK)
    ax.set_xlabel("share of total liver volume (%)", fontsize=14, color=INK_2)
    ax.set_xlim(0, max(28, max(pct.values()) * 1.18))
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.set_facecolor(SURFACE)
    ax.set_title("segment volume vs published Couinaud volumetry\n"
                 "grey band = reference range; red bar = outside it",
                 fontsize=15.8, color=INK, loc="left")

    ax_t.axis("off")
    ax_t.set_facecolor(SURFACE)
    lines = [f"{'seg':<5}{'name':<26}{'mL':>8}{'%':>7}{'ref %':>12}"]
    lines.append("-" * 58)
    for s in SEG_IDS:
        lo, hi = SEG_REFERENCE_PCT[s]
        mark = " " if lo <= pct[s] <= hi else "*"
        lines.append(f"{SEG_NAMES[s]:<5}{SEG_DESCRIPTION[s]:<26}{vols[s]:>8.0f}"
                     f"{pct[s]:>6.1f}{mark}{f'{lo:.0f}-{hi:.0f}':>12}")
    lines.append("-" * 58)
    lines.append(f"{'':<5}{'total':<26}{sum(vols.values()):>8.0f}{100.0:>7.1f}")
    lines.append("")
    lines.append("checks")
    import textwrap
    for name, ok, detail in checks["rows"]:
        head = f"  [{'PASS' if ok else 'FAIL'}] {name}"
        lines.extend(textwrap.wrap(head, 76, subsequent_indent="         "))
        lines.extend(textwrap.wrap(detail, 76, initial_indent="         ",
                                   subsequent_indent="         "))
    ax_t.text(0, 1, "\n".join(lines), fontsize=10.8, family="monospace", va="top",
              color=INK_2, transform=ax_t.transAxes)

    fig.suptitle(f"{case.name} — volumetry and sanity checks", fontsize=19.2, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 6. the two assignment modes, side by side
# --------------------------------------------------------------------------- #

def figure_modes(case: Case, voronoi: Result, hybrid: Result, path: Path) -> Path:
    """Where the pure portal territories and the vein-plane hybrid disagree."""
    frame = case.frame
    a_v, a_h = frame.reorient(voronoi.labels), frame.reorient(hybrid.labels)
    liver = a_v > 0
    s_idx = np.argwhere(liver)[:, 2]
    levels = np.linspace(int(np.percentile(s_idx, 20)),
                         int(np.percentile(s_idx, 80)), 3).astype(int)[::-1]

    fig, axs = plt.subplots(3, 3, figsize=(11.5, 11), facecolor=SURFACE)
    for col, z in enumerate(levels):
        for row, (labels, title) in enumerate(((a_v, "voronoi — portal territories"),
                                               (a_h, "hybrid — vein-plane sectors"))):
            ax = axs[row][col]
            ax.imshow(labels[:, :, z].T, cmap=SEG_CMAP, vmin=0, vmax=8, origin="lower",
                      interpolation="nearest")
            img = labels[:, :, z].T
            for s in SEG_IDS:
                m = img == s
                if m.sum() < 40:
                    continue
                cc, n = ndimage.label(m)
                sizes = ndimage.sum_labels(m, cc, index=range(1, n + 1))
                cy, cx = ndimage.center_of_mass(cc == int(np.argmax(sizes)) + 1)
                ax.text(cx, cy, SEG_NAMES[s], fontsize=14, fontweight="bold", ha="center",
                        va="center", color="#ffffff",
                        path_effects=[patheffects.withStroke(linewidth=2.2,
                                                             foreground="#00000099")])
            _style(ax, f"{title}\naxial @ {z}" if col == 0 else f"axial @ {z}")

        ax = axs[2][col]
        diff = (a_v[:, :, z] != a_h[:, :, z]) & (a_v[:, :, z] > 0)
        ax.imshow(liver[:, :, z].T, cmap="gray_r", origin="lower", alpha=0.16,
                  interpolation="nearest")
        yy, xx = np.where(diff.T)
        ax.scatter(xx, yy, s=0.6, c="#e34948", linewidths=0)
        share = diff.sum() / max((a_v[:, :, z] > 0).sum(), 1)
        _style(ax, f"disagreement — {share:.0%} of this slice" if col else
               f"where they differ\n{share:.0%} of this slice")

    per_seg = []
    for s in SEG_IDS:
        a, b = voronoi.labels == s, hybrid.labels == s
        inter = int(np.count_nonzero(a & b))
        tot = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
        per_seg.append(2 * inter / tot if tot else 0.0)
    overall = float(np.count_nonzero((voronoi.labels == hybrid.labels) & (voronoi.labels > 0))
                    / max(np.count_nonzero(voronoi.labels > 0), 1))

    fig.legend(handles=[Line2D([], [], marker="s", linestyle="", markersize=7,
                               markerfacecolor=SEG_COLOR_LIGHT[s], markeredgecolor="none",
                               label=f"{SEG_NAMES[s]}  Dice {per_seg[i]:.2f}")
                        for i, s in enumerate(SEG_IDS)],
               loc="lower center", ncol=4, fontsize=14, frameon=False, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.008), title="per-segment agreement between modes",
               title_fontsize=24.5)
    fig.suptitle(f"{case.name} — assignment modes compared\n"
                 f"they agree on {overall:.1%} of liver voxels",
                 fontsize=19.2, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 7. assignment rules compared
# --------------------------------------------------------------------------- #

def figure_assignment(case: Case, results: dict[str, Result], path: Path,
                      veins: dict[str, np.ndarray] | None = None) -> Path:
    """
    The same seeds, propagated under each assignment rule.

    The first result given is the baseline, and every other row carries its boundary
    as a thin dark contour, so what moved is visible in place rather than inferred by
    flicking between panels. Where hepatic veins are supplied they are stippled over
    the slice: the point of the barrier is that a boundary should come to rest on
    one, and that is a claim the reader can check here directly.
    """
    names = list(results)
    base = results[names[0]]
    frame = case.frame
    arrays = {n: frame.reorient(r.labels) for n, r in results.items()}
    a_base = arrays[names[0]]
    vein_arrays = {k: frame.reorient(v) for k, v in (veins or {}).items()}

    s_idx = np.argwhere(a_base > 0)[:, 2]
    levels = np.linspace(int(np.percentile(s_idx, 20)),
                         int(np.percentile(s_idx, 80)), 3).astype(int)[::-1]

    # Crop to the liver. Without this most of each panel is empty air and the
    # boundary shifts these figures exist to show are too small to read.
    shown = a_base[:, :, levels] > 0
    xs, ys = np.where(shown.any(axis=2))
    pad = 4
    xlim = (max(xs.min() - pad, 0), min(xs.max() + pad, a_base.shape[0] - 1))
    ylim = (max(ys.min() - pad, 0), min(ys.max() + pad, a_base.shape[1] - 1))
    aspect = (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1)

    fig, axs = plt.subplots(len(names), 3, figsize=(11.5, 3.9 * aspect * len(names)),
                            facecolor=SURFACE, squeeze=False)
    for row, name in enumerate(names):
        arr = arrays[name]
        changed = float(np.count_nonzero((arr != a_base) & (a_base > 0))
                        / max(np.count_nonzero(a_base > 0), 1))
        for col, z in enumerate(levels):
            ax = axs[row][col]
            img = arr[:, :, z].T
            ax.imshow(img, cmap=SEG_CMAP, vmin=0, vmax=8, origin="lower",
                      interpolation="nearest")

            # the baseline partition, as a contour, on every row but the first
            if row:
                ax.contour(a_base[:, :, z].T, levels=np.arange(0.5, 9, 1),
                           colors="#0b0b0b", linewidths=0.55, alpha=0.5)
            for key, va in vein_arrays.items():
                yy, xx = np.where(va[:, :, z].T)
                if len(xx):
                    ax.scatter(xx, yy, s=1.4, marker=".", linewidths=0,
                               c=SKEL_STYLE[f"hv_{key}"][0], alpha=0.85)

            for s in SEG_IDS:
                m = img == s
                if m.sum() < 40:
                    continue
                cc, n = ndimage.label(m)
                sizes = ndimage.sum_labels(m, cc, index=range(1, n + 1))
                cy, cx = ndimage.center_of_mass(cc == int(np.argmax(sizes)) + 1)
                ax.text(cx, cy, SEG_NAMES[s], fontsize=14, fontweight="bold", ha="center",
                        va="center", color="#ffffff",
                        path_effects=[patheffects.withStroke(linewidth=2.2,
                                                             foreground="#00000099")])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            pct = results[name].volume_pct()
            ratio = (pct[6] + pct[7]) / max(pct[5] + pct[8], 1e-9)
            if col == 0:
                moved = "baseline" if not row else f"{changed:.1%} of voxels moved"
                _style(ax, f"{name} — {moved}\n(VI+VII)/(V+VIII) = {ratio:.2f}"
                           f"\naxial @ {z}")
            else:
                _style(ax, f"axial @ {z}")

    handles = [Line2D([], [], marker="s", linestyle="", markersize=7,
                      markerfacecolor=SEG_COLOR_LIGHT[s], markeredgecolor="none",
                      label=SEG_NAMES[s]) for s in SEG_IDS]
    if len(names) > 1:
        handles.append(Line2D([], [], color="#0b0b0b", linewidth=0.9, alpha=0.6,
                              label=f"{names[0]} boundary"))
    handles += [Line2D([], [], marker=".", linestyle="", markersize=8,
                       color=SKEL_STYLE[f"hv_{k}"][0], label=SKEL_STYLE[f"hv_{k}"][1])
                for k in vein_arrays]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=14, frameon=False,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.004))
    fig.suptitle(f"{case.name} — assignment rules compared\n"
                 f"the same portal seeds, propagated four ways "
                 f"(published sector ratio 0.83)",
                 fontsize=19.2, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.945))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 8. batch summary
# --------------------------------------------------------------------------- #

def figure_batch(summaries: list[dict], path: Path,
                 bias: dict | None = None) -> Path:
    """
    Distribution of segment volume shares across every case that ran, the check
    tally, and -- when it could be computed -- the coverage-bias scatter that shows
    how much of the result is driven by label completeness rather than anatomy.
    """
    if not summaries:
        raise ValueError("nothing to plot")
    has_bias = bool(bias and bias.get("pearson_log") is not None)
    if has_bias:
        fig, (ax, ax3, ax2) = plt.subplots(
            1, 3, figsize=(18, 5.6), facecolor=SURFACE,
            gridspec_kw={"width_ratios": [1.5, 1, 1.05]})
    else:
        ax3 = None
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE,
                                      gridspec_kw={"width_ratios": [1.5, 1]})

    data = [[s["volume_pct"][str(seg)] for s in summaries] for seg in SEG_IDS]
    for i, seg in enumerate(SEG_IDS):
        lo, hi = SEG_REFERENCE_PCT[seg]
        ax.add_patch(plt.Rectangle((i + 1 - 0.42, lo), 0.84, hi - lo, color="#eceae5",
                                   zorder=1))
    bp = ax.boxplot(data, patch_artist=True, widths=0.42, zorder=3,
                    medianprops={"color": INK, "linewidth": 1.6},
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": INK_MUTED, "markeredgecolor": "none"},
                    whiskerprops={"color": INK_MUTED, "linewidth": 1},
                    capprops={"color": INK_MUTED, "linewidth": 1})
    for patch, seg in zip(bp["boxes"], SEG_IDS):
        patch.set_facecolor(SEG_COLOR_LIGHT[seg])
        patch.set_alpha(0.55)
        patch.set_edgecolor(SEG_COLOR_LIGHT[seg])
    ax.set_xticks(range(1, 9), [SEG_NAMES[s] for s in SEG_IDS], fontsize=15.8, color=INK)
    ax.set_ylabel("share of liver volume (%)", fontsize=14, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"segment volume share across {len(summaries)} cases\n"
                 f"grey band = published reference range", fontsize=15.8, color=INK, loc="left")

    if ax3 is not None:
        import re
        xs, ys = [], []
        for s_ in summaries:
            note = next((n for n in s_["notes"] if n.startswith("pedicle skeletons:")), None)
            if note is None:
                continue
            counts = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", note)}
            a_n, p_n = counts.get("right_ant", 0), counts.get("right_post", 0)
            pct = s_["volume_pct"]
            anterior, posterior = pct["5"] + pct["8"], pct["6"] + pct["7"]
            if a_n >= 5 and p_n >= 5 and anterior > 0 and posterior > 0:
                xs.append(p_n / a_n)
                ys.append(posterior / anterior)
        ax3.scatter(xs, ys, s=34, c="#2a78d6", alpha=0.7, linewidths=0)
        lim = [min(min(xs), min(ys)) * 0.8, max(max(xs), max(ys)) * 1.2]
        ax3.plot(lim, lim, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
        ax3.axhline(bias["volume_ratio_reference"], color="#e34948", linewidth=1.2,
                    linestyle=(0, (2, 2)), zorder=0)
        ax3.text(lim[0] * 1.05, bias["volume_ratio_reference"] * 1.06,
                 "reference volume ratio", fontsize=12.2, color="#e34948")
        ax3.set_xscale("log"); ax3.set_yscale("log")
        ax3.set_xlim(*lim); ax3.set_ylim(*lim)
        ax3.set_xlabel("skeleton points, posterior / anterior pedicle", fontsize=14,
                       color=INK_2)
        ax3.set_ylabel("volume claimed, (VI+VII) / (V+VIII)", fontsize=14, color=INK_2)
        ax3.set_title(f"coverage bias: r = {bias['pearson_log']:.2f} (log-log, "
                      f"n = {bias['n']})\nhow completely a pedicle was traced predicts\n"
                      f"how much parenchyma it wins",
                      fontsize=15.8, color=INK, loc="left")
        ax3.grid(color=GRID, linewidth=0.7, zorder=0)
        ax3.set_axisbelow(True)
        for spine in ("top", "right"):
            ax3.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax3.spines[spine].set_color(GRID)
        ax3.tick_params(colors=INK_MUTED, length=0, labelsize=7)
        ax3.set_facecolor(SURFACE)

    ax2.axis("off"); ax2.set_facecolor(SURFACE)
    # key by name, not index -- a case that could not run a check has fewer rows
    tally: dict[str, list[int]] = {}
    order: list[str] = []
    for s in summaries:
        for name, ok, _ in s["checks"]["rows"]:
            if name not in tally:
                tally[name] = [0, 0]
                order.append(name)
            tally[name][0] += int(bool(ok))
            tally[name][1] += 1
    lines = [f"{'check':<52}{'pass':>8}"]
    lines.append("-" * 60)
    for name in order:
        passed, avail = tally[name]
        short = name.split(" (Dice")[0].split(" (>=")[0]
        lines.append(f"{short[:51]:<52}{passed:>4}/{avail:<3}")
    lines.append("")

    def spread(key, label):
        vals = [s["checks"]["metrics"].get(key) for s in summaries]
        vals = [v for v in vals if v is not None]
        if vals:
            lines.append(f"{label:<52}{np.median(vals):>8.3f}")

    lines.append(f"{'independent cross-validation (median)':<52}")
    spread("dice_cantlie_vs_mhv", "  Dice: portal left/right vs MHV plane")
    spread("dice_right_sector_vs_rhv", "  Dice: portal ant/post vs RHV plane")
    spread("mode_agreement", "  voronoi vs hybrid voxel agreement")
    lines.append("")
    lines.append(f"{'segment':<12}{'median %':>11}{'ref %':>14}{'in range':>12}")
    lines.append("-" * 60)
    for seg, vals in zip(SEG_IDS, data):
        lo, hi = SEG_REFERENCE_PCT[seg]
        n_in = sum(1 for v in vals if lo <= v <= hi)
        lines.append(f"{SEG_NAMES[seg]:<12}{np.median(vals):>11.1f}"
                     f"{f'{lo:.0f}-{hi:.0f}':>14}{f'{n_in}/{len(vals)}':>12}")
    ax2.text(0, 1, "\n".join(lines), fontsize=13.3, family="monospace", va="top",
             color=INK_2, transform=ax2.transAxes)

    fig.suptitle("Couinaud pipeline — batch results", fontsize=19.2, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)
    return path
