"""
Competitive front growth from the labelled portal seeds.

The nearest-seed rule in `_assign_voxels` is already a watershed -- a marker-controlled
one, on a flat elevation, with a straight-line metric. This module keeps the marker
control and replaces the other two, one at a time, so each change can be measured
on its own:

  geodesic     the front travels *through the liver*. A straight line may leave the
               organ and re-enter it across the gallbladder fossa, the IVC groove or
               the interlobar notch, letting a seed claim parenchyma no branch of its
               pedicle could ever reach. Measured on this cohort, confining the front
               moves 2-15% of liver voxels.

  flow         each territory advances at a rate set by the Murray-law flow implied
               by its pedicle's root radius, instead of every seed advancing equally.
               Built to attack the coverage bias: a pedicle traced further into the
               periphery wins more parenchyma under any nearest-seed rule, but its
               *radius* does not grow because the segmentation followed it further --
               across 23 cases the seed-count ratio correlates with the produced
               volume ratio at 0.79 and with the Murray ratio at only 0.19.

               On this cohort it does not work, and the ablation says why. Raising
               alpha lowers that correlation monotonically (0.79 -> 0.65), so the
               mechanism is sound, but the volumes move *away* from published
               volumetry at every alpha, and at alpha=1 the per-case ratios still
               correlate with the unweighted ones at 0.91. The bias is positional:
               a seed already sitting near the far capsule has claimed that
               parenchyma no matter how slowly its front advances. Fixing it means
               touching the seeds, not their speed. Off by default; see README.md.

  vein barrier crossing a hepatic vein is made expensive, so a basin boundary prefers
               to settle *on* the vein. Couinaud's scissurae are not planes, and this
               is the only one of the three that can produce a curved one. It is also
               the only one that spends evidence: see `holdout` below.

The three compose. All of them reduce to the current behaviour at their defaults, so
`Config()` still produces exactly the labelling the existing results describe.

Cost model. Each label L gets its own minimum-cost-path field d_L over the liver, and
a voxel goes to whichever label minimises d_L(x) / w_L. Because the cost image is
shared, the weights are a division applied after the fact rather than eight separate
propagations per weighting -- and since argmin is invariant to a global scale on w,
only the *ratios* between pedicles matter.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

# Which pedicle feeds which segment. Segments fed by the same pedicle must carry the
# same flow weight: their shared boundary is decided by the graph cut inside one
# pedicle, where the trace-completeness bias cancels and there is nothing to correct.
SEGMENT_PEDICLE = {2: "left", 3: "left", 4: "left",
                   5: "right_ant", 8: "right_ant",
                   6: "right_post", 7: "right_post"}

# Below this the distance-transform radius estimate is at the voxel quantisation
# floor and carries no information -- 1 mm data cannot resolve a 1 mm radius.
MIN_TRUSTED_RADIUS_MM = 1.5


def murray_weights(radii: dict[str, float], alpha: float, log_clip: float,
                   notes: list[str] | None = None) -> dict[str, float]:
    """
    Turn measured pedicle root radii into propagation weights.

    Murray's law gives flow Q ~ r^3, and a territory's volume is what its flow has to
    perfuse, so the target is volume ~ r^3.

    The weight is *not* r^3. A weight enters as a divisor on distance, so it scales a
    territory's linear extent, and volume follows its cube: doubling w multiplies the
    claimed volume by roughly 8 (measured, see `test_flow_weighted_growth`). Setting
    w ~ r^3 would therefore deliver volume ~ r^9. The weight that delivers Murray is
    linear in the radius, so:

        w = r^alpha,  alpha = 1 -> volume ~ r^3, exactly Murray
                      alpha = 0 -> radii ignored entirely

    `alpha` is a shrinkage knob between those, and belongs strictly inside the
    interval on this data: cubing a radius triples its relative error -- +-0.5 mm on
    a 3 mm radius is a factor 1.6 in flow -- and 1 mm voxels resolve only 19 distinct
    radius values across 46 pedicles. `log_clip` bounds |log w| on top of that, so one
    bad radius cannot run away with the liver.
    """
    usable = {k: r for k, r in radii.items()
              if np.isfinite(r) and r >= MIN_TRUSTED_RADIUS_MM}
    if alpha <= 0 or len(usable) < 2:
        if notes is not None and alpha > 0:
            notes.append(f"  ! flow weighting off: only {len(usable)} pedicle radii "
                         f"clear the {MIN_TRUSTED_RADIUS_MM} mm quantisation floor")
        return {k: 1.0 for k in radii}

    logs = {k: alpha * np.log(r) for k, r in usable.items()}
    centre = float(np.mean(list(logs.values())))       # geometric-mean normalisation
    w = {k: float(np.exp(np.clip(v - centre, -log_clip, log_clip)))
         for k, v in logs.items()}
    for k in radii:                                     # untrusted radii stay neutral
        w.setdefault(k, 1.0)
    if notes is not None:
        notes.append("  flow weights: " + "  ".join(
            f"{k} r={radii[k]:.2f}mm w={w[k]:.2f}" for k in sorted(w)))
    return w


def vein_barrier_cost(shape_mask: np.ndarray, vein_masks: dict[str, np.ndarray],
                      zooms: np.ndarray, strength: float, falloff_mm: float,
                      notes: list[str] | None = None) -> np.ndarray:
    """
    Elevation image that makes the hepatic veins expensive to cross.

    A ramp rather than a wall: a hard barrier would be a plane by another name and
    would break wherever a vein label is fragmentary, which on this cohort it often
    is. The cost rises smoothly from 1 at `falloff_mm` away to 1+`strength` on the
    vein itself, so a boundary is *pulled* onto the vein where one exists and is
    decided by the portal seeds where one does not.
    """
    cost = np.ones(shape_mask.shape, dtype=np.float64)
    if strength <= 0 or not vein_masks:
        return cost
    union = np.zeros(shape_mask.shape, dtype=bool)
    for m in vein_masks.values():
        union |= m
    if not union.any():
        if notes is not None:
            notes.append("  ! vein barrier requested but no vein voxels in the crop")
        return cost
    # Only the liver is ever traversed, so the distance transform is computed in its
    # bounding box, padded by the falloff so a vein just outside still reaches in.
    # A full-volume EDT here costs hundreds of megabytes for a field that is unit
    # almost everywhere.
    box = ndimage.find_objects(shape_mask.astype(np.uint8))
    if box:
        pad = np.ceil(falloff_mm / np.maximum(zooms, 1e-6)).astype(int)
        sl = tuple(slice(max(s.start - p, 0), min(s.stop + p, n))
                   for s, p, n in zip(box[0], pad, shape_mask.shape))
    else:
        sl = tuple(slice(None) for _ in shape_mask.shape)
    d = ndimage.distance_transform_edt(~union[sl], sampling=tuple(zooms))
    ramp = np.clip(1.0 - d / max(falloff_mm, 1e-6), 0.0, 1.0)
    cost[sl] += strength * ramp
    if notes is not None:
        notes.append(f"  vein barrier: {int(union.sum()):,} voxels from "
                     f"{'+'.join(sorted(vein_masks))}, cost up to "
                     f"{1 + strength:.1f}x within {falloff_mm:.0f} mm")
    return cost


def grow(liver: np.ndarray, seed_vox: np.ndarray, seed_labels: np.ndarray,
         zooms: np.ndarray, weights: dict[int, float] | None = None,
         cost: np.ndarray | None = None,
         restrict: np.ndarray | None = None,
         notes: list[str] | None = None) -> np.ndarray:
    """
    Grow every label's front from its seeds and give each voxel its cheapest claimant.

    `liver`, `cost` and `restrict` are full-volume arrays; `seed_vox` are voxel indices.
    `restrict`, when given, is a boolean volume the growth may not leave -- that is how
    hybrid mode confines each sector's growth to the region its vein planes carved out.

    Returns a uint8 label volume. Voxels no front reaches (a mask island with no seed
    of its own) are left at 0 and reported; the caller repairs them.
    """
    labels_present = sorted({int(s) for s in np.unique(seed_labels)})
    out = np.zeros(liver.shape, dtype=np.uint8)
    domain = liver if restrict is None else (liver & restrict)
    if not domain.any() or not labels_present:
        return out

    # Cropping to the liver's bounding box is what makes this affordable: the front
    # never leaves the organ, so the rest of the volume is dead weight in every
    # propagation.
    sl = ndimage.find_objects(domain.astype(np.uint8))[0]
    sub = domain[sl]
    base = np.ones(sub.shape, dtype=np.float64) if cost is None else cost[sl].astype(float)
    field = np.where(sub, base, np.inf)

    origin = np.array([s.start for s in sl])
    local = seed_vox - origin
    inside = np.all((local >= 0) & (local < np.array(sub.shape)), axis=1)
    inside &= sub[tuple(np.clip(local, 0, np.array(sub.shape) - 1).T)]
    if not inside.any():
        if notes is not None:
            notes.append("  ! no seed lies inside the growth domain")
        return out

    best = np.full(sub.shape, np.inf)
    winner = np.zeros(sub.shape, dtype=np.uint8)
    for lab in labels_present:
        starts = local[inside & (seed_labels == lab)]
        if not len(starts):
            continue
        mcp = MCP_Geometric(field, sampling=tuple(float(z) for z in zooms),
                            fully_connected=True)
        d, _ = mcp.find_costs([tuple(int(v) for v in p) for p in starts])
        d = np.asarray(d) / float((weights or {}).get(lab, 1.0))
        take = d < best
        best[take] = d[take]
        winner[take] = lab

    unreached = int(np.count_nonzero(sub & (winner == 0)))
    if unreached and notes is not None:
        notes.append(f"  ! {unreached:,} liver voxels unreachable from any seed "
                     f"({100 * unreached / int(sub.sum()):.2f}%) -- filled by nearest label")
    out[sl] = winner
    return out


def fill_unreached(labels: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """
    Give voxels no front could reach the label of the nearest voxel one did.

    These are mask islands cut off from every seed -- a lobe attached only through a
    voxel-thin bridge the largest-component filter kept. Straight-line nearest is the
    right answer here precisely because no path exists.
    """
    missing = domain & (labels == 0)
    if not missing.any():
        return labels
    known = domain & (labels > 0)
    if not known.any():
        return labels

    # Cropped, because an exact EDT with indices over a whole CT volume allocates
    # three index arrays the size of the volume -- gigabytes for a decision that only
    # ever concerns a handful of stranded voxels.
    sl = ndimage.find_objects(domain.astype(np.uint8))[0]
    sub_labels, sub_missing = labels[sl], missing[sl]
    _, idx = ndimage.distance_transform_edt(~known[sl], return_indices=True)
    out = labels.copy()
    out[sl] = np.where(sub_missing, sub_labels[tuple(idx)], sub_labels)
    return out
