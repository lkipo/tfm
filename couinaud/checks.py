"""
Sanity checks for a Couinaud result.

There is no segment-level ground truth in this dataset, so nothing here can prove
a segmentation correct. What these checks can do is catch a segmentation that is
definitely *wrong*, and they come in three kinds:

  structural   the labelling must partition the liver exactly, with all eight
               segments present and each one spatially coherent
  anatomical   relations that hold in every human liver -- II above III, VIII above
               V, the posterior sector behind the anterior one, the left segments
               to the patient's left. A violated ordering means an axis or a graph
               cut went the wrong way.
  independent  the portal-territory result is compared against the hepatic-vein
               planes, which come from *different labels entirely*. Portal anatomy
               predicting venous anatomy is the closest thing to external
               validation available here.

Volume shares are also compared against published Couinaud volumetry, but that is
a wide band and only flags gross failures.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from couinaud import (SEG_IDS, SEG_NAMES, SEG_REFERENCE_PCT, Case, Result, to_mm)

LEFT_LIVER = (2, 3, 4)
RIGHT_LIVER = (5, 6, 7, 8)
RIGHT_ANTERIOR = (5, 8)
RIGHT_POSTERIOR = (6, 7)

# (name, superior segment, inferior segment)
SUPERIOR_PAIRS = [("II above III", 2, 3), ("VIII above V", 8, 5), ("VII above VI", 7, 6)]
MIN_LARGEST_CC = 0.85
MIN_DICE = 0.70


def _centroid(labels: np.ndarray, seg: int, case: Case) -> np.ndarray | None:
    m = labels == seg
    if not m.any():
        return None
    return to_mm(np.array(ndimage.center_of_mass(m)), case)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    total = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    return 2.0 * inter / total if total else 0.0


def _largest_cc_fraction(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    cc, n = ndimage.label(mask)
    if n <= 1:
        return 1.0
    sizes = ndimage.sum_labels(mask, cc, index=range(1, n + 1))
    return float(sizes.max() / sizes.sum())


def run_checks(case: Case, res: Result, reference: Result | None = None) -> dict:
    """
    Returns {'rows': [(name, passed, detail)], 'metrics': {...}}.

    `reference` is the labelling used for the hepatic-vein cross-validation. It must
    be a *pure Voronoi* result: in hybrid mode the sector boundaries are taken from
    the vein planes, so comparing that labelling back to the same planes measures
    nothing (it returns Dice 1.0 by construction). Pass the Voronoi result to keep
    the check independent; it defaults to `res`, which is correct in Voronoi mode.
    """
    liver = res.liver
    reference = reference if reference is not None else res
    labels = res.labels
    rows: list[tuple[str, bool, str]] = []
    metrics: dict[str, float] = {}

    r_hat, a_hat, s_hat = case.unit("R"), case.unit("A"), case.unit("S")
    centroids = {s: _centroid(labels, s, case) for s in SEG_IDS}

    # -- structural --------------------------------------------------------
    unlabelled = int(np.count_nonzero(liver & (labels == 0)))
    outside = int(np.count_nonzero(~liver & (labels > 0)))
    rows.append(("partitions the liver exactly", unlabelled == 0 and outside == 0,
                 f"{unlabelled} liver voxels unlabelled, {outside} labels outside the liver"))

    missing = [SEG_NAMES[s] for s in SEG_IDS if not (labels == s).any()]
    rows.append(("all eight segments present", not missing,
                 "none missing" if not missing else f"missing {', '.join(missing)}"))

    cc_fracs = {s: _largest_cc_fraction(labels == s) for s in SEG_IDS}
    worst = min(cc_fracs, key=cc_fracs.get)
    metrics["min_largest_cc_fraction"] = cc_fracs[worst]
    rows.append((f"each segment is one connected blob (>={MIN_LARGEST_CC:.0%})",
                 cc_fracs[worst] >= MIN_LARGEST_CC,
                 f"worst is {SEG_NAMES[worst]} at {cc_fracs[worst]:.0%}"))

    # -- anatomical ordering ----------------------------------------------
    bad = []
    for name, sup, inf in SUPERIOR_PAIRS:
        cs, ci = centroids[sup], centroids[inf]
        if cs is None or ci is None:
            bad.append(f"{name} (missing)")
        elif float((cs - ci) @ s_hat) <= 0:
            bad.append(f"{name} ({float((cs - ci) @ s_hat):+.0f} mm)")
    rows.append(("superior segments sit above their inferior partners", not bad,
                 "II>III, VIII>V, VII>VI all hold" if not bad else "; ".join(bad)))

    def group_centroid(group):
        m = np.isin(labels, group)
        return to_mm(np.array(ndimage.center_of_mass(m)), case) if m.any() else None

    c_ant, c_post = group_centroid(RIGHT_ANTERIOR), group_centroid(RIGHT_POSTERIOR)
    gap = float((c_ant - c_post) @ a_hat) if c_ant is not None and c_post is not None else 0.0
    metrics["anterior_posterior_gap_mm"] = gap
    rows.append(("right anterior sector is anterior to the posterior sector", gap > 0,
                 f"V+VIII are {gap:+.0f} mm anterior of VI+VII"))

    c_left, c_right = group_centroid(LEFT_LIVER), group_centroid(RIGHT_LIVER)
    lr = float((c_right - c_left) @ r_hat) if c_left is not None and c_right is not None else 0.0
    metrics["left_right_gap_mm"] = lr
    rows.append(("left segments lie to the patient's left of the right ones", lr > 0,
                 f"V-VIII are {lr:+.0f} mm right of II-IV"))

    # -- independent cross-validation against the hepatic veins ------------
    # A vein that fed the assignment cannot then judge it. `reference` handles the
    # usual case by evaluating a clean labelling instead; `spent` handles the case
    # where the caller deliberately wants *this* labelling scored, and refuses the
    # comparison outright rather than reporting a Dice that is high by construction.
    ref_labels = reference.labels
    # It is the *reference* labelling that gets compared to the planes, so it is the
    # reference's evidence that can be spent -- which is the same thing as `res` when
    # no separate reference was passed.
    spent = reference.veins_used()

    def circular(key: str) -> str:
        why = ("its sector boundaries are the vein planes" if reference.mode == "hybrid"
               else f"it was part of the growth cost (barrier {reference.vein_barrier:g})")
        return f"{key.upper()} fed this assignment -- {why} -- so this is not evidence"

    left_mask = np.isin(ref_labels, LEFT_LIVER)
    if "mhv" in spent:
        rows.append((f"portal left/right agrees with the MHV plane (Dice>={MIN_DICE:.2f})",
                     False, circular("mhv")))
    elif "mhv" in res.planes:
        vox = np.argwhere(liver)
        sd = res.planes["mhv"].sd(to_mm(vox, case))
        plane_left = np.zeros_like(liver)
        plane_left[tuple(vox[sd < 0].T)] = True     # MHV normal points right
        d = _dice(left_mask, plane_left)
        metrics["dice_cantlie_vs_mhv"] = d
        rows.append((f"portal left/right agrees with the MHV plane (Dice>={MIN_DICE:.2f})",
                     d >= MIN_DICE,
                     f"Dice {d:.3f} between the portal-derived left liver and the "
                     f"MHV (Cantlie) half-space"))
    else:
        rows.append((f"portal left/right agrees with the MHV plane (Dice>={MIN_DICE:.2f})",
                     False, "no MHV plane for this case"))

    right_vox = np.argwhere(np.isin(ref_labels, RIGHT_LIVER))
    if "rhv" in spent:
        rows.append((f"portal ant/post agrees with the RHV plane (Dice>={MIN_DICE:.2f})",
                     False, circular("rhv")))
    elif "rhv" in res.planes and len(right_vox):
        sd = res.planes["rhv"].sd(to_mm(right_vox, case))
        plane_ant = np.zeros_like(liver)
        plane_ant[tuple(right_vox[sd > 0].T)] = True   # RHV normal points anterior
        d = _dice(np.isin(ref_labels, RIGHT_ANTERIOR), plane_ant)
        metrics["dice_right_sector_vs_rhv"] = d
        rows.append((f"portal ant/post agrees with the RHV plane (Dice>={MIN_DICE:.2f})",
                     d >= MIN_DICE,
                     f"Dice {d:.3f} between V+VIII and the RHV anterior half-space"))
    else:
        rows.append((f"portal ant/post agrees with the RHV plane (Dice>={MIN_DICE:.2f})",
                     False, "no RHV plane for this case"))

    left_vox = np.argwhere(np.isin(ref_labels, LEFT_LIVER))
    if "lhv" in spent:
        rows.append((f"portal IV boundary agrees with the LHV plane (Dice>={MIN_DICE:.2f})",
                     False, circular("lhv")))
    elif "lhv" in res.planes and len(left_vox):
        sd = res.planes["lhv"].sd(to_mm(left_vox, case))
        plane_medial = np.zeros_like(liver)
        plane_medial[tuple(left_vox[sd > 0].T)] = True   # LHV normal points right
        d = _dice(ref_labels == 4, plane_medial)
        metrics["dice_segment_iv_vs_lhv"] = d
        rows.append((f"portal IV boundary agrees with the LHV plane (Dice>={MIN_DICE:.2f})",
                     d >= MIN_DICE,
                     f"Dice {d:.3f} between segment IV and the LHV medial half-space"))
    else:
        rows.append((f"portal IV boundary agrees with the LHV plane (Dice>={MIN_DICE:.2f})",
                     False, "no LHV plane for this case"))

    # -- volumetry ---------------------------------------------------------
    pct = res.volume_pct()
    in_range = [s for s in SEG_IDS if SEG_REFERENCE_PCT[s][0] <= pct[s] <= SEG_REFERENCE_PCT[s][1]]
    metrics["segments_in_reference_range"] = len(in_range)
    rows.append(("segment volumes inside the published range (>=6/8)", len(in_range) >= 6,
                 f"{len(in_range)}/8 in range; outliers "
                 + ", ".join(f"{SEG_NAMES[s]}={pct[s]:.1f}%"
                             for s in SEG_IDS if s not in in_range) or "none"))

    left_pct = sum(pct[s] for s in LEFT_LIVER)
    metrics["left_liver_pct"] = left_pct
    rows.append(("left hemiliver is 25-45% of the liver", 25.0 <= left_pct <= 45.0,
                 f"II+III+IV = {left_pct:.1f}%"))

    metrics["frame_confidence"] = case.frame.confidence
    rows.append(("anatomical frame recovered confidently", case.frame.confidence >= 0.62,
                 f"confidence {case.frame.confidence:.2f}, margin {case.frame.margin:.2f}"))

    return {"rows": rows, "metrics": metrics,
            "passed": sum(1 for _, ok, _ in rows if ok), "total": len(rows)}
