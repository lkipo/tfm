"""Class ids of ``data/cornerstone_masks`` and the channel sets we train on.

The masks are a single integer volume per subject (``labels.json`` next to
them), so the annotation is already a hard partition -- one class per voxel.
Two properties of that partition drive every choice in this file:

* **The coarse contour classes overwrite the fine mesh classes.** The mask
  builder resolves collisions in favour of the contour source (ids 1-6), and
  ``classBloodVessels`` (id 6) is an *intrahepatic vessel union* that covers
  most of the portal pedicles. Measured over the 18 subjects that have both:
  ``log(BloodVessels)`` vs ``log(sum of pedicles)`` correlates at **-0.55**,
  and the pedicles that survive have a median of 105-1329 voxels -- one to two
  orders of magnitude below the 4-6.5k voxels the pipeline notes expect. A
  channel defined as a single pedicle id is therefore mostly *missing*, not
  small; a channel defined as a union with the contour class that ate it
  (``portal_tree`` below) is intact.
* **Nine of the fifteen classes come from GLB meshes that some subjects do not
  have at all** (e.g. ``RMV2025_0017`` ships six meshes, none of them portal).
  Those are unannotated, not empty: they are masked out of the loss per
  subject rather than treated as background.

``CHANNEL_PRESETS`` therefore ranges from the six channels that are complete in
all 38 subjects to the raw 15-class transcription; see ``README.md`` Sec. 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Verbatim from data/cornerstone_masks/labels.json.
LABEL_IDS: dict[str, int] = {
    "classLiver": 1,
    "classVenaPorta": 2,
    "classIVC": 3,
    "classAorta": 4,
    "classGallbladder": 5,
    "classBloodVessels": 6,
    "classVenaHepaticaDerecha": 7,
    "classVenaHepaticaMedia": 8,
    "classVenaHepaticaIzquierda": 9,
    "classArteriaHepatica": 10,
    "classViaBiliar": 11,
    "classPediculoPortalDerecho": 12,
    "classPediculoPortalIzquierdo": 13,
    "classPediculoPortalAnteriorDerecho": 14,
    "classPediculoPortalPosteriorDerecho": 15,
}
ID_TO_NAME: dict[int, str] = {v: k for k, v in LABEL_IDS.items()}
NUM_LABELS = 15

# Everything that lies inside the liver capsule. ``classLiver`` in the mask is
# parenchyma *minus* whatever else was labelled there, so a usable organ mask
# is the union -- otherwise the vessels are holes in the liver channel and the
# Couinaud pipeline has nothing to partition through the hilum.
INTRAHEPATIC = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
PEDICLES = [12, 13, 14, 15]
HEPATIC_VEINS = [7, 8, 9]


@dataclass(frozen=True)
class Channel:
    """One sigmoid output channel.

    ``sources``      label ids unioned into the target.
    ``required``     if True the channel is assumed annotated in every subject,
                     so an empty target really means "absent here" and counts
                     as background. If False and *all* its sources are empty in
                     a subject, the channel is masked out of that subject's
                     loss and metrics (finetuning.md Sec. 1: "mask out; do not
                     let it count as background").
    ``thin``         tubular structure -- gets the clDice term and a larger
                     share of the foreground-sampled crops.
    """

    name: str
    sources: list[int]
    required: bool = True
    thin: bool = False
    sample_priority: int = 0  # higher wins when building the crop-sampling map
    field_note: str = field(default="", compare=False)


def _ch(name, sources, **kw) -> Channel:
    return Channel(name=name, sources=list(sources), **kw)


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
# core6: every channel is non-empty in all 38 subjects and has >=7k voxels.
# This is the set that can actually be fitted from these masks.
CORE6 = [
    _ch("liver", [1] + INTRAHEPATIC, sample_priority=1,
        field_note="organ = parenchyma + everything inside it"),
    _ch("portal_tree", [2] + PEDICLES, thin=True, sample_priority=5,
        field_note="trunk + pedicles; the union survives the contour overwrite"),
    _ch("ivc", [3], thin=True, sample_priority=3),
    _ch("aorta", [4], sample_priority=2),
    _ch("gallbladder", [5], sample_priority=2),
    _ch("vessels", INTRAHEPATIC, thin=True, sample_priority=4,
        field_note="intrahepatic vessel union; the vesselness head of finetuning.md Sec. 1"),
]

# core5: core6 minus gallbladder. The Couinaud pipeline uses the gallbladder
# only as a soft frame-recovery signal (two evidence vectors in
# ``anatomy_frame.py``); without it the division still yields 8/8 segments and
# only the frame confidence drops (finetuning.md Sec. 2). Its val Dice was the
# noisiest channel in the core6 run (0.45-0.72), so dropping it trades a
# perturbing loss term for a bit of frame evidence.
CORE5 = [c for c in CORE6 if c.name != "gallbladder"]

# core7: core6 + the hepatic-vein union, which 25/38 subjects have (median a
# few hundred voxels). Masked where absent.
CORE7 = CORE6 + [
    _ch("hepatic_veins", HEPATIC_VEINS, required=False, thin=True, sample_priority=6),
]

# full11: adds the four named pedicles as their own channels. These are the
# Couinaud sectors, and the reason the pipeline works -- but see the module
# docstring: what is left of them in these masks is a remnant. Train this only
# after core6 has converged, and read the per-channel Dice, never the mean.
FULL11 = CORE7 + [
    _ch("pediculo_izquierdo", [13], required=False, thin=True, sample_priority=9),
    _ch("pediculo_derecho", [12], required=False, thin=True, sample_priority=7),
    _ch("pediculo_ant_derecho", [14], required=False, thin=True, sample_priority=8),
    _ch("pediculo_post_derecho", [15], required=False, thin=True, sample_priority=8),
]

# couinaud8: the minimal channel set the Couinaud pipeline consumes, each class
# separate. Everything else the pipeline reads (classAorta/classIVC/classLiver
# for the frame, classVenaPorta for the bifurcation, the four pedicles for the
# sectors) is here; classGallbladder is dropped (soft frame evidence only,
# finetuning.md Sec. 2), the hepatic veins are dropped (validation-only; the
# Voronoi division degrades gracefully without them) and the union channels
# (portal_tree, vessels) are redundant once the individual classes exist.
# Pedicles are remnants in the masks (the classBloodVessels contour overwrite)
# and are masked where absent.
COUINAUD8 = [
    _ch("liver", [1] + INTRAHEPATIC, sample_priority=1,
        field_note="organ = parenchyma + everything inside it"),
    _ch("vena_porta", [2], thin=True, sample_priority=5,
        field_note="main trunk; the Couinaud bifurcation anchor"),
    _ch("ivc", [3], thin=True, sample_priority=3),
    _ch("aorta", [4], sample_priority=2),
    _ch("pediculo_izquierdo", [13], required=False, thin=True, sample_priority=9),
    _ch("pediculo_derecho", [12], required=False, thin=True, sample_priority=7),
    _ch("pediculo_ant_derecho", [14], required=False, thin=True, sample_priority=8),
    _ch("pediculo_post_derecho", [15], required=False, thin=True, sample_priority=8),
]

# couinaud5: the model that closes the Couinaud loop by *derivation*, not by
# training the tiny classes. The pipeline needs the four pedicles as separate
# masks, but their training labels are remnants (the classBloodVessels contour
# overwrite). Instead this preset trains the structures that are complete in
# all 37 subjects and well learnable -- the portal tree *union* (trunk +
# pedicles) and the vena porta trunk separately -- and the pedicle masks are
# derived at inference time by splitting the predicted union skeleton at the
# portal bifurcation (see predict_pipeline.py). Hepatic veins and gallbladder
# are deliberately absent: the Voronoi division degrades gracefully without
# them. Every channel is required and complete, so nothing is masked.
COUINAUD5 = [
    _ch("liver", [1] + INTRAHEPATIC, sample_priority=1,
        field_note="organ = parenchyma + everything inside it"),
    _ch("vena_porta", [2], thin=True, sample_priority=5,
        field_note="main trunk; anchors the bifurcation at inference"),
    _ch("portal_tree", [2] + PEDICLES, thin=True, sample_priority=4,
        field_note="trunk + pedicles union; the pedicle derivation source"),
    _ch("ivc", [3], thin=True, sample_priority=3),
    _ch("aorta", [4], sample_priority=2),
]

# all15: one channel per raw label id, no unions, nothing merged. Faithful
# transcription of labels.json for ablation against the grouped presets.
ALL15 = [
    _ch("liver", [1], sample_priority=1),
    _ch("vena_porta", [2], thin=True, sample_priority=5),
    _ch("ivc", [3], thin=True, sample_priority=3),
    _ch("aorta", [4], sample_priority=2),
    _ch("gallbladder", [5], sample_priority=2),
    _ch("blood_vessels", [6], thin=True, sample_priority=4),
    _ch("vena_hepatica_derecha", [7], required=False, thin=True, sample_priority=6),
    _ch("vena_hepatica_media", [8], required=False, thin=True, sample_priority=6),
    _ch("vena_hepatica_izquierda", [9], required=False, thin=True, sample_priority=6),
    _ch("arteria_hepatica", [10], required=False, thin=True, sample_priority=7),
    _ch("via_biliar", [11], required=False, thin=True, sample_priority=7),
    _ch("pediculo_derecho", [12], required=False, thin=True, sample_priority=7),
    _ch("pediculo_izquierdo", [13], required=False, thin=True, sample_priority=9),
    _ch("pediculo_ant_derecho", [14], required=False, thin=True, sample_priority=8),
    _ch("pediculo_post_derecho", [15], required=False, thin=True, sample_priority=8),
]

CHANNEL_PRESETS: dict[str, list[Channel]] = {
    "core6": CORE6,
    "core5": CORE5,
    "core7": CORE7,
    "full11": FULL11,
    "couinaud8": COUINAUD8,
    "couinaud5": COUINAUD5,
    "all15": ALL15,
}


def get_channels(preset: str) -> list[Channel]:
    try:
        return CHANNEL_PRESETS[preset]
    except KeyError:
        raise ValueError(
            f"unknown channel preset {preset!r}; choose from {sorted(CHANNEL_PRESETS)}"
        ) from None


def channel_names(channels: list[Channel]) -> list[str]:
    return [c.name for c in channels]


def thin_channel_indices(channels: list[Channel]) -> list[int]:
    return [i for i, c in enumerate(channels) if c.thin]


def channel_counts(per_label_counts: dict[int, int], channels: list[Channel]) -> list[int]:
    """Upper bound on each channel's voxel count from per-label counts.

    Exact, not an estimate: the source label map is a partition, so a union of
    distinct ids has exactly the sum of their counts.
    """
    return [sum(per_label_counts.get(s, 0) for s in c.sources) for c in channels]


def channel_availability(per_label_counts: dict[int, int], channels: list[Channel]) -> list[int]:
    """1 = annotated in this subject (counts as supervision), 0 = mask out."""
    out = []
    for c, n in zip(channels, channel_counts(per_label_counts, channels)):
        out.append(1 if (c.required or n > 0) else 0)
    return out
