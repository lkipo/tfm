#!/usr/bin/env python
"""
Tests for the Couinaud pipeline.

Two kinds:

  synthetic  the geometry primitives are fed inputs whose right answer is known by
             construction -- a plane fit given points on a known plane, a graph cut
             given a tree built with a known division, a frame estimate given masks
             placed in a known orientation. These are the real unit tests: they can
             fail, and a failure means the maths is wrong.

  real data  one case is run end to end and the structural invariants are asserted.
             This is a smoke test with teeth: it cannot verify the segmentation is
             anatomically right, only that it is self-consistent.

    python test_couinaud.py            # everything
    python test_couinaud.py --quick    # synthetic only, no data needed
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage

import growth
from anatomy_frame import estimate_frame
from couinaud import (Config, angle_deg, best_axis_cut, fit_line_direction, fit_plane,
                      fit_plane_through_line, load_case, missing_labels, segment_liver,
                      skeleton_graph, SEG_IDS)

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "data" / "0_test_nifti"

_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((name, bool(condition), detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# synthetic geometry
# --------------------------------------------------------------------------- #

def test_fit_plane() -> None:
    print("\nfit_plane")
    rng = np.random.default_rng(0)
    normal = np.array([0.3, -0.5, 0.81])
    normal /= np.linalg.norm(normal)
    basis = np.linalg.svd(normal[None, :])[2][1:]
    coeffs = rng.normal(size=(600, 2)) * 40
    pts = coeffs @ basis + np.array([10.0, -5.0, 3.0])
    pts += rng.normal(scale=0.4, size=pts.shape)          # a little thickness

    _, fitted, planarity = fit_plane(pts)
    check("recovers a known plane normal", angle_deg(fitted, normal) < 2.0,
          f"{angle_deg(fitted, normal):.2f} deg off")
    check("reports high planarity for a flat cloud", planarity > 0.95, f"{planarity:.3f}")

    blob = rng.normal(size=(600, 3)) * 20
    _, _, blob_planarity = fit_plane(blob)
    check("reports low planarity for an isotropic cloud", blob_planarity < 0.4,
          f"{blob_planarity:.3f}")


def test_fit_plane_through_line() -> None:
    print("\nfit_plane_through_line")
    rng = np.random.default_rng(1)
    line_point = np.array([5.0, 5.0, 0.0])
    line_dir = np.array([0.05, 0.02, 1.0])
    line_dir /= np.linalg.norm(line_dir)
    in_plane = np.array([1.0, 0.0, 0.0])
    in_plane -= line_dir * (in_plane @ line_dir)
    in_plane /= np.linalg.norm(in_plane)
    expected = np.cross(line_dir, in_plane)

    # points on the plane, but bunched far from the line so a free fit would drift
    t = rng.uniform(-30, 30, 400)
    u = rng.uniform(40, 90, 400)
    pts = line_point + t[:, None] * line_dir + u[:, None] * in_plane
    pts += rng.normal(scale=1.5, size=pts.shape)

    origin, normal, _ = fit_plane_through_line(pts, line_point, line_dir)
    check("recovers the plane containing a known line",
          angle_deg(normal, expected) < 3.0, f"{angle_deg(normal, expected):.2f} deg off")
    check("the fitted normal is perpendicular to the line",
          abs(float(normal @ line_dir)) < 1e-9, f"dot {float(normal @ line_dir):.2e}")
    check("the line lies in the fitted plane",
          abs(float((line_point - origin) @ normal)) < 1e-9)


def test_best_axis_cut() -> None:
    print("\nbest_axis_cut")
    rng = np.random.default_rng(2)
    # a Y: a trunk that splits into a branch going up and a branch going down
    trunk = np.stack([np.zeros(20), np.zeros(20), np.linspace(0, 20, 20)], axis=1)
    up = np.stack([np.linspace(0, 30, 40), np.zeros(40), np.linspace(20, 55, 40)], axis=1)
    down = np.stack([np.linspace(0, 30, 40), np.zeros(40), np.linspace(20, -15, 40)], axis=1)
    pts = np.vstack([trunk, up, down]) + rng.normal(scale=0.15, size=(100, 3))
    s_hat = np.array([0.0, 0.0, 1.0])

    cut = best_axis_cut(pts, s_hat, min_gap_mm=5.0)
    check("finds a cut in a Y-shaped tree", cut is not None)
    if cut is not None:
        side, gap = cut
        up_idx = slice(20, 60)
        down_idx = slice(60, 100)
        up_purity = side[up_idx].mean()
        down_purity = (~side[down_idx]).mean()
        check("the superior branch lands on the positive side", up_purity > 0.9,
              f"{up_purity:.0%} of the up branch")
        check("the inferior branch lands on the negative side", down_purity > 0.9,
              f"{down_purity:.0%} of the down branch")
        check("reports a sensible gap", gap > 10, f"{gap:.1f} mm")

    straight = np.stack([np.zeros(60), np.zeros(60), np.linspace(0, 60, 60)], axis=1)
    straight += rng.normal(scale=0.1, size=straight.shape)
    # a straight line does divide along S; what must not happen is a cut on a
    # cloud with no supero-inferior structure at all
    flat = rng.normal(scale=0.4, size=(60, 3)) * np.array([20.0, 20.0, 0.05])
    check("refuses to cut a cloud with no supero-inferior structure",
          best_axis_cut(flat, s_hat, min_gap_mm=5.0) is None)


def test_skeleton_graph() -> None:
    print("\nskeleton_graph")
    line = np.stack([np.arange(40.0), np.zeros(40), np.zeros(40)], axis=1)
    g = skeleton_graph(line)
    check("a straight chain becomes a path graph",
          g.number_of_edges() == 39 and max(dict(g.degree()).values()) == 2,
          f"{g.number_of_edges()} edges, max degree {max(dict(g.degree()).values())}")

    far = np.vstack([line, line + np.array([500.0, 0, 0])])
    g2 = skeleton_graph(far)
    import networkx as nx
    n_comp = nx.number_connected_components(g2)
    check("two distant clouds stay separate", n_comp == 2, f"{n_comp} components")


def test_fit_line_direction() -> None:
    print("\nfit_line_direction")
    rng = np.random.default_rng(3)
    d = np.array([0.4, 0.7, -0.59]); d /= np.linalg.norm(d)
    pts = np.outer(rng.uniform(-50, 50, 300), d) + rng.normal(scale=0.5, size=(300, 3))
    got = fit_line_direction(pts)
    check("recovers a known line direction", angle_deg(got, d) < 2.0,
          f"{angle_deg(got, d):.2f} deg off")


def test_frame_estimation() -> None:
    print("\nestimate_frame (synthetic anatomy)")
    # Build a phantom in a deliberately non-RAS voxel order: axis0=Left,
    # axis1=Superior, axis2=Anterior -- the convention this dataset actually uses.
    shape = (120, 120, 120)

    def blob(centre, radius=6):
        m = np.zeros(shape, dtype=bool)
        z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
        m[((z - centre[0]) ** 2 + (y - centre[1]) ** 2 + (x - centre[2]) ** 2)
          <= radius ** 2] = True
        return m

    def column(centre_lr, centre_ap, radius=5):
        m = np.zeros(shape, dtype=bool)
        z, x = np.ogrid[:shape[0], :shape[2]]
        disc = ((z - centre_lr) ** 2 + (x - centre_ap) ** 2) <= radius ** 2
        m[:, 20:100, :] = disc[:, None, :]
        return m

    # (left, superior, anterior) coordinates
    masks = {
        "classIVC": column(58, 40),                     # slightly right of midline, posterior
        "classAorta": column(68, 38),                   # to the patient's left of the IVC
        "classLiver": blob((45, 62, 60), 26),           # bulk to the patient's right
        "classGallbladder": blob((52, 40, 76), 6),      # inferior and anterior
        "classVenaPorta": blob((58, 55, 55), 6),        # anterior to the IVC
        "classVenaHepaticaDerecha": blob((38, 78, 52), 6),
        "classVenaHepaticaIzquierda": blob((78, 78, 58), 6),
        "classVenaHepaticaMedia": blob((58, 78, 60), 6),
        "classPediculoPortalIzquierdo": blob((78, 58, 62), 6),
        "classPediculoPortalAnteriorDerecho": blob((42, 58, 70), 6),
        "classPediculoPortalPosteriorDerecho": blob((42, 58, 48), 6),
    }
    frame = estimate_frame(masks, np.ones(3))
    check("recovers Right = -axis0", np.allclose(frame.unit("R"), [-1, 0, 0]),
          f"got {frame.unit('R').tolist()}")
    check("recovers Superior = +axis1", np.allclose(frame.unit("S"), [0, 1, 0]),
          f"got {frame.unit('S').tolist()}")
    check("recovers Anterior = +axis2", np.allclose(frame.unit("A"), [0, 0, 1]),
          f"got {frame.unit('A').tolist()}")
    check("the recovered triad is right-handed", frame.right_handed)
    check("reports high confidence on clean input", frame.confidence > 0.8,
          f"{frame.confidence:.2f}")

    # mirroring left/right must flip R and nothing else
    mirrored = {k: v[::-1] for k, v in masks.items()}
    mframe = estimate_frame(mirrored, np.ones(3))
    check("mirroring the volume flips Right", np.allclose(mframe.unit("R"), [1, 0, 0]),
          f"got {mframe.unit('R').tolist()}")


# --------------------------------------------------------------------------- #
# real data
# --------------------------------------------------------------------------- #

def test_geodesic_growth() -> None:
    """
    A C-shaped mask: two seeds sit close in a straight line but far apart through
    the material. This is the gallbladder fossa in miniature, and it is the whole
    reason for confining the front.
    """
    print("\ngeodesic growth")
    mask = np.ones((3, 21, 21), dtype=bool)
    mask[:, :16, 10] = False           # a septum, leaving a gap along the bottom
    zooms = np.array([1.0, 1.0, 1.0])

    a, b = (1, 18, 2), (1, 2, 15)      # A far down the left arm, B at the top right
    seeds = np.array([a, b])
    labels = np.array([2, 3], dtype=np.uint8)
    grown = growth.grow(mask, seeds, labels, zooms)
    grown = growth.fill_unreached(grown, mask)

    check("geodesic growth labels every masked voxel",
          bool(np.all(grown[mask] > 0)) and bool(np.all(grown[~mask] == 0)))

    # A point at the top of the *left* arm. B is much nearer in a straight line, but
    # the straight line goes through the septum; reaching it from B means walking all
    # the way down, round the gap and back up.
    probe = (1, 2, 8)
    d_a = float(np.linalg.norm(np.array(probe) - np.array(a)))
    d_b = float(np.linalg.norm(np.array(probe) - np.array(b)))
    check("the septum is a real test (the blocked seed is the straight-line nearer one)",
          d_b < d_a, f"straight-line: blocked seed {d_b:.1f}, reachable seed {d_a:.1f}")
    check("the front goes round the septum instead of through it", grown[probe] == 2,
          f"probe went to {grown[probe]}")

    euclid = np.zeros_like(grown)
    vox = np.argwhere(mask)
    d = np.linalg.norm(vox[:, None, :] - seeds[None, :, :], axis=2)
    euclid[tuple(vox.T)] = labels[np.argmin(d, axis=1)]
    differ = int(np.count_nonzero((grown != euclid) & mask))
    check("geodesic disagrees with straight-line nearest across a septum", differ > 0,
          f"{differ} voxels differ ({100 * differ / mask.sum():.0f}%)")

    # Both territories must be connected -- that is the structural guarantee a
    # monotone front inside a connected mask buys, and the reason to prefer it.
    for lab in (2, 3):
        cc, n = ndimage.label(grown == lab)
        check(f"geodesic territory {lab} is a single connected blob", n == 1, f"{n} components")


def test_growth_matches_voronoi_without_obstacles() -> None:
    """On a convex mask with unit cost there is no difference, and there must not be."""
    print("\ngeodesic == Voronoi on a convex mask")
    mask = np.ones((15, 15, 15), dtype=bool)
    zooms = np.array([1.0, 1.0, 1.0])
    seeds = np.array([[3, 3, 3], [11, 11, 11], [3, 11, 7]])
    labels = np.array([2, 5, 7], dtype=np.uint8)

    grown = growth.grow(mask, seeds, labels, zooms)
    vox = np.argwhere(mask)
    d = np.linalg.norm(vox[:, None, :] - seeds[None, :, :], axis=2)
    euclid = np.zeros_like(grown)
    euclid[tuple(vox.T)] = labels[np.argmin(d, axis=1)]
    agree = float(np.count_nonzero(grown == euclid) / mask.sum())
    # Not exactly 1: a 26-neighbour lattice metric slightly overestimates diagonals,
    # so voxels equidistant to two seeds can tip either way.
    check("convex mask: geodesic reproduces the Voronoi partition", agree > 0.98,
          f"{agree:.1%} of voxels agree")


def test_murray_weights() -> None:
    print("\nMurray flow weights")
    radii = {"left": 5.0, "right_ant": 4.0, "right_post": 3.0}
    check("alpha=0 disables the weighting",
          set(growth.murray_weights(radii, 0.0, 0.7).values()) == {1.0})

    w = growth.murray_weights(radii, 1.0, 10.0)
    check("weights order with radius", w["left"] > w["right_ant"] > w["right_post"])
    check("at alpha=1 the weight is linear in the radius",
          np.isclose(w["left"] / w["right_post"], 5.0 / 3.0),
          f"{w['left'] / w['right_post']:.3f} vs {5 / 3:.3f}")
    check("weights are normalised to geometric mean 1",
          np.isclose(float(np.exp(np.mean(np.log(list(w.values()))))), 1.0))

    clipped = growth.murray_weights({"a": 10.0, "b": 2.0}, 1.0, 0.7)
    check("the clip bounds a runaway radius",
          max(abs(np.log(v)) for v in clipped.values()) <= 0.7 + 1e-9)

    floor = growth.murray_weights({"a": 1.0, "b": 1.0}, 1.0, 0.7)
    check("radii at the voxel quantisation floor are refused",
          set(floor.values()) == {1.0})

    equal = growth.murray_weights({"a": 4.0, "b": 4.0}, 1.0, 0.7)
    check("equal radii give equal weights", np.isclose(equal["a"], equal["b"]))


def test_flow_weighted_growth() -> None:
    """A heavier weight must claim more of a symmetric domain, and by the right sign."""
    print("\nflow-weighted growth")
    mask = np.ones((9, 41, 9), dtype=bool)
    zooms = np.array([1.0, 1.0, 1.0])
    seeds = np.array([[4, 5, 4], [4, 35, 4]])
    labels = np.array([5, 6], dtype=np.uint8)

    even = growth.grow(mask, seeds, labels, zooms)
    share = np.count_nonzero(even == 5) / mask.sum()
    check("equal weights split a symmetric domain evenly", abs(share - 0.5) < 0.02,
          f"{share:.1%}")

    tilted = growth.grow(mask, seeds, labels, zooms, weights={5: 1.5, 6: 1.0})
    share2 = np.count_nonzero(tilted == 5) / mask.sum()
    check("the heavier weight claims more", share2 > share + 0.05,
          f"{share:.1%} -> {share2:.1%}")

    # The whole `w = r^alpha` parameterisation rests on this: a weight divides
    # distance, so it scales a territory's linear extent and volume follows its
    # *cube*. Get this wrong and w = r^3 silently delivers volume ~ r^9, which is
    # exactly the error the first cohort ablation caught.
    box = np.ones((41, 41, 41), dtype=bool)
    pair = np.array([[10, 20, 20], [30, 20, 20]])
    for k in (1.5, 2.0):
        g = growth.grow(box, pair, labels, zooms, weights={5: k, 6: 1.0})
        ratio = np.count_nonzero(g == 5) / max(np.count_nonzero(g == 6), 1)
        check(f"claimed volume scales as roughly w^3 (w={k})",
              abs(np.log(ratio) - 3 * np.log(k)) < 0.35,
              f"volume ratio {ratio:.2f} vs w^3 = {k ** 3:.2f}")


def test_vein_barrier() -> None:
    print("\nvein barrier cost")
    shape = np.ones((3, 41, 11), dtype=bool)
    vein = np.zeros(shape.shape, dtype=bool)
    vein[:, 13, :] = True
    zooms = np.array([1.0, 1.0, 1.0])

    cost = growth.vein_barrier_cost(shape, {"rhv": vein}, zooms, strength=3.0,
                                    falloff_mm=5.0)
    check("cost peaks on the vein", np.isclose(cost[1, 13, 5], 4.0), f"{cost[1, 13, 5]:.2f}")
    check("cost is unity beyond the falloff", np.isclose(cost[1, 0, 5], 1.0))
    check("cost ramps monotonically toward the vein",
          bool(np.all(np.diff(cost[1, :14, 5]) >= -1e-12)))
    check("a switched-off barrier is a flat cost",
          bool(np.all(growth.vein_barrier_cost(shape, {"rhv": vein}, zooms, 0.0, 5.0) == 1.0)))

    # The vein is deliberately *off* the natural boundary: seeds at rows 3 and 37 meet
    # at row 20 on their own, and the barrier has to drag that boundary to row 13.
    seeds = np.array([[1, 3, 5], [1, 37, 5]])
    labels = np.array([5, 6], dtype=np.uint8)

    def boundary_row(vol: np.ndarray) -> float:
        rows = np.argwhere(vol == 5)[:, 1]
        return float(rows.max())

    off = boundary_row(growth.grow(shape, seeds, labels, zooms))
    on = boundary_row(growth.grow(shape, seeds, labels, zooms, cost=cost))
    check("without a barrier the boundary sits midway between the seeds",
          abs(off - 20) <= 1, f"row {off:.0f}, seeds at 3 and 37")
    check("the barrier drags the boundary toward the vein", abs(on - 13) < abs(off - 13),
          f"row {off:.0f} -> {on:.0f}, vein at row 13")

    # It is a ramp, not a wall, so the boundary is pulled rather than pinned. The
    # property that must hold is monotonicity: more strength, more pull -- that is
    # what makes `vein_barrier` a dial the ablation can tune instead of a switch.
    strong = growth.vein_barrier_cost(shape, {"rhv": vein}, zooms, 12.0, 5.0)
    harder = boundary_row(growth.grow(shape, seeds, labels, zooms, cost=strong))
    check("a stronger barrier pulls it further", abs(harder - 13) < abs(on - 13),
          f"strength 3 -> row {on:.0f}, strength 12 -> row {harder:.0f}")


def test_growth_config_guards() -> None:
    """Flow weighting and the barrier are meaningless without a geodesic metric."""
    print("\nconfiguration guards")
    from couinaud import _assign_voxels

    class _Stub:
        shape, zooms, masks = (2, 2, 2), np.ones(3), {}

    for cfg in (Config(flow_alpha=0.5), Config(vein_barrier=2.0), Config(assign="nope")):
        try:
            _assign_voxels(_Stub(), np.ones((2, 2, 2), bool), np.zeros((1, 3)),
                           np.array([2], np.uint8), np.array(["left"], object),
                           {}, None, {}, cfg, [])
            ok = False
        except ValueError:
            ok = True
        except Exception:                                # noqa: BLE001
            ok = False
        check(f"assign={cfg.assign!r} alpha={cfg.flow_alpha} barrier={cfg.vein_barrier} "
              f"is rejected", ok)


def test_unreached_fill() -> None:
    print("\nunreachable-island repair")
    mask = np.zeros((3, 9, 9), dtype=bool)
    mask[1, 1:4, 1:4] = True
    mask[1, 6:9, 6:9] = True           # a second island, no seed of its own
    zooms = np.array([1.0, 1.0, 1.0])
    grown = growth.grow(mask, np.array([[1, 2, 2]]), np.array([4], np.uint8), zooms)
    check("the island is left unlabelled by the front", int((grown[mask] == 0).sum()) == 9)
    filled = growth.fill_unreached(grown, mask)
    check("and is repaired by the nearest labelled voxel",
          bool(np.all(filled[mask] == 4)))


def test_real_case() -> None:
    print("\nend to end on a real case")
    usable = [c for c in sorted(DATA_ROOT.glob("RMV*")) if c.is_dir() and not missing_labels(c)]
    if not usable:
        check("a usable case exists", False, f"none under {DATA_ROOT}")
        return
    case_dir = usable[0]
    print(f"  using {case_dir.name}")
    case = load_case(case_dir)
    res = segment_liver(case, Config(), verbose=False)

    labels, liver = res.labels, res.liver
    check("every liver voxel is labelled", not (liver & (labels == 0)).any(),
          f"{int(np.count_nonzero(liver & (labels == 0)))} unlabelled")
    check("nothing outside the liver is labelled", not (~liver & (labels > 0)).any(),
          f"{int(np.count_nonzero(~liver & (labels > 0)))} outside")
    present = [s for s in SEG_IDS if (labels == s).any()]
    check("all eight segments are produced", len(present) == 8,
          f"{len(present)} present")
    check("labels stay in range 0..8", int(labels.max()) <= 8 and int(labels.min()) >= 0)
    total = sum(res.volumes_ml().values())
    expected = float(np.count_nonzero(liver)) * case.voxel_volume_ml()
    check("volumes sum to the liver volume", abs(total - expected) < 1e-6,
          f"{total:.1f} vs {expected:.1f} mL")
    check("seeds and seed labels line up",
          len(res.seeds_mm) == len(res.seed_labels) == len(res.seed_source))
    check("no seed is left unlabelled", bool(np.isin(res.seed_labels, SEG_IDS).all()))

    # determinism: the same config must give the same volume, twice
    again = segment_liver(case, Config(), verbose=False)
    check("the pipeline is deterministic", bool((again.labels == labels).all()))

    # the two modes must both be valid partitions
    hybrid = segment_liver(case, Config(mode="hybrid"), verbose=False)
    check("hybrid mode also partitions the liver exactly",
          not (liver & (hybrid.labels == 0)).any() and not (~liver & (hybrid.labels > 0)).any())

    # Each of the three propagation rules must still produce an exact partition of
    # the same liver mask, and must actually change something -- an option that
    # silently does nothing is worse than no option.
    variants = {
        "geodesic": Config(assign="geodesic"),
        "flow": Config(assign="geodesic", flow_alpha=0.5),
        "vein barrier": Config(assign="geodesic", vein_barrier=2.0,
                               vein_barrier_holdout="rhv"),
        "geodesic hybrid": Config(assign="geodesic", mode="hybrid"),
    }
    previous = labels
    for name, cfg in variants.items():
        r = segment_liver(case, cfg, verbose=False)
        exact = (not (liver & (r.labels == 0)).any()
                 and not (~liver & (r.labels > 0)).any())
        check(f"{name} partitions the liver exactly", exact,
              f"{int(np.count_nonzero(liver & (r.labels == 0)))} unlabelled, "
              f"{int(np.count_nonzero(~liver & (r.labels > 0)))} outside")
        moved = int(np.count_nonzero((r.labels != previous) & liver))
        check(f"{name} changes the labelling", moved > 0,
              f"{100 * moved / int(liver.sum()):.1f}% of liver voxels vs the rule above")
        previous = r.labels

    # A monotone front inside a connected mask cannot leave a basin in two pieces.
    geo = segment_liver(case, Config(assign="geodesic"), verbose=False)
    broken = [SEG_NAMES[s] for s in SEG_IDS if s != 1 and (geo.labels == s).any()
              and ndimage.label(geo.labels == s)[1] > 1]
    check("every geodesic territory is connected", not broken,
          f"disconnected: {', '.join(broken) or 'none'} "
          f"(segment I is excluded -- it is a geometric heuristic, not a territory)")


def test_frame_on_all_cases() -> None:
    print("\nestimate_frame on every case in the dataset")
    cases = [c for c in sorted(DATA_ROOT.glob("RMV*")) if c.is_dir()]
    if not cases:
        check("dataset present", False, str(DATA_ROOT))
        return
    handed, low = 0, []
    for c in cases:
        case = load_case(c)
        if case.frame.right_handed:
            handed += 1
        if case.frame.confidence < 0.62:
            low.append(c.name)
    check("every recovered frame is right-handed", handed == len(cases),
          f"{handed}/{len(cases)}")
    check("at most a quarter of cases are low-confidence", len(low) <= len(cases) // 4,
          f"{len(low)}/{len(cases)} low: {', '.join(n[:18] for n in low) or 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="synthetic tests only")
    args = ap.parse_args()

    tests = [test_fit_plane, test_fit_plane_through_line, test_best_axis_cut,
             test_skeleton_graph, test_fit_line_direction, test_frame_estimation,
             test_geodesic_growth, test_growth_matches_voronoi_without_obstacles,
             test_murray_weights, test_flow_weighted_growth, test_vein_barrier,
             test_growth_config_guards, test_unreached_fill]
    if not args.quick:
        tests += [test_real_case, test_frame_on_all_cases]

    for t in tests:
        try:
            t()
        except Exception as exc:                        # noqa: BLE001
            check(f"{t.__name__} raised", False, repr(exc))
            traceback.print_exc()

    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{'=' * 60}\n{passed}/{len(_results)} checks passed")
    failed = [n for n, ok, _ in _results if not ok]
    if failed:
        print("failed: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
