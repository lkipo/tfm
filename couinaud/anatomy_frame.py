"""
Recover the anatomical (Right / Anterior / Superior) frame of a case from its own
anatomy, because the NIfTI affine cannot be trusted here.

`continuity/glb_to_nifti.py` writes `affine = np.eye(4)` scaled by the voxel pitch
-- no rotation -- so every case in `data/0_test_nifti` reports axcodes ('R','A','S')
whatever the source GLB's orientation actually was. Measuring the anatomy shows
that claim is wrong: in 33 of 38 cases voxel axis 1 is the cranio-caudal axis, not
axis 2, and the sign conventions vary too. Anything reasoning about "superior",
"anterior" or "left" -- which is most of the Couinaud pipeline -- must derive the
frame rather than read it.

Method: collect direction vectors whose anatomical meaning is known from the label
*names* and from textbook relations, then choose the right-handed triad of signed
voxel axes that best aligns with all of them at once. Fitting the whole triad
jointly, instead of deciding each axis separately, is what makes the estimate
stable: a vector like "RHV is right of LHV" has a real antero-posterior component,
and only the joint fit can tell that component apart from the left-right one.

    R  hepatic vein D is right of hepatic vein I; right pedicles are right of the
       left pedicle; the IVC is right of the aorta; the liver's mass is right of
       the IVC
    A  the right anterior pedicle is anterior to the right posterior pedicle; the
       gallbladder is anterior to the aorta; the porta hepatis is anterior to the IVC
    S  the hepatic veins are superior to the portal pedicles; the liver is superior
       to the gallbladder; the liver is superior to the extrahepatic IVC; and the
       IVC+aorta principal direction names the cranio-caudal axis outright

Right-handedness (R x A = S) is imposed as a hard constraint on the search, so the
result is always a valid frame; `Frame.confidence` reports how well the winning
triad actually explained the evidence, and low-confidence cases should be checked
by eye rather than trusted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# (label A, label B, weight) meaning "centroid(A) - centroid(B) points this way".
# `_` prefixed names are composite regions built in `_centroids`.
RIGHTWARD = [
    ("classVenaHepaticaDerecha", "classVenaHepaticaIzquierda", 1.0),
    ("_right_pedicles", "classPediculoPortalIzquierdo", 1.0),
    ("classIVC", "classAorta", 0.7),
    ("classLiver", "classIVC", 0.7),
]
ANTERIOR = [
    ("classPediculoPortalAnteriorDerecho", "classPediculoPortalPosteriorDerecho", 1.0),
    ("classGallbladder", "classAorta", 0.7),
    ("classVenaPorta", "classIVC", 0.5),
]
SUPERIOR = [
    ("_hepatic_veins", "_pedicles", 1.5),
    ("classLiver", "classGallbladder", 0.7),
    ("classLiver", "_extrahepatic_ivc", 0.5),
]
# weight of the IVC+aorta principal direction, added to the superior evidence
AXIS_PRIOR_WEIGHT = 2.0

LOW_CONFIDENCE = 0.62


@dataclass
class Frame:
    """Signed voxel axis for each anatomical direction, plus how well it fitted."""
    r_axis: int
    r_sign: int
    a_axis: int
    a_sign: int
    s_axis: int
    s_sign: int
    confidence: float = 0.0
    margin: float = 0.0
    evidence: list[tuple[str, str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def unit(self, code: str) -> np.ndarray:
        axis, sign = {"R": (self.r_axis, self.r_sign),
                      "A": (self.a_axis, self.a_sign),
                      "S": (self.s_axis, self.s_sign)}[code]
        v = np.zeros(3)
        v[axis] = sign
        return v

    @property
    def right_handed(self) -> bool:
        return bool(np.allclose(np.cross(self.unit("R"), self.unit("A")), self.unit("S")))

    @property
    def axis_order(self) -> tuple[int, int, int]:
        """Voxel axes in (R, A, S) order -- the transpose for `reorient`."""
        return self.r_axis, self.a_axis, self.s_axis

    def describe(self) -> str:
        parts = []
        for code in "RAS":
            v = self.unit(code)
            ax = int(np.argmax(np.abs(v)))
            parts.append(f"{code}=axis{ax}{'+' if v[ax] > 0 else '-'}")
        return (f"{' '.join(parts)} (confidence {self.confidence:.2f}, "
                f"margin {self.margin:.2f})")

    def reorient(self, volume: np.ndarray) -> np.ndarray:
        """Transpose/flip a volume into (R, A, S) axis order for display."""
        out = np.transpose(volume, self.axis_order)
        for k, code in enumerate("RAS"):
            v = self.unit(code)
            if v[self.axis_order[k]] < 0:
                out = np.flip(out, axis=k)
        return out


def _com(mask: np.ndarray) -> np.ndarray:
    return np.array(ndimage.center_of_mass(mask))


def _centroids(masks: dict[str, np.ndarray], zooms: np.ndarray) -> dict[str, np.ndarray]:
    def live(name):
        m = masks.get(name)
        return m if m is not None and m.any() else None

    out: dict[str, np.ndarray] = {}
    for name, m in masks.items():
        if m is not None and m.any():
            out[name] = _com(m) * zooms

    groups = {
        "_right_pedicles": ["classPediculoPortalDerecho", "classPediculoPortalAnteriorDerecho",
                            "classPediculoPortalPosteriorDerecho"],
        "_pedicles": ["classPediculoPortalIzquierdo", "classPediculoPortalDerecho",
                      "classPediculoPortalAnteriorDerecho", "classPediculoPortalPosteriorDerecho"],
        "_hepatic_veins": ["classVenaHepaticaDerecha", "classVenaHepaticaIzquierda",
                           "classVenaHepaticaMedia"],
    }
    for key, members in groups.items():
        present = [live(n) for n in members]
        present = [m for m in present if m is not None]
        if present:
            out[key] = _com(np.any(present, axis=0)) * zooms

    liver, ivc = live("classLiver"), live("classIVC")
    if liver is not None and ivc is not None:
        outside = ivc & ~liver
        if outside.sum() > 10:
            out["_extrahepatic_ivc"] = _com(outside) * zooms
    return out


def _direction_evidence(spec, centroids) -> list[tuple[str, np.ndarray, float]]:
    ev = []
    for a, b, w in spec:
        if a in centroids and b in centroids:
            v = centroids[a] - centroids[b]
            n = float(np.linalg.norm(v))
            if n > 1e-6:
                ev.append((f"{a} - {b}", v / n, w))
    return ev


def _principal_direction(masks, zooms, max_points=40000, seed=0) -> np.ndarray | None:
    pool = [masks.get("classIVC"), masks.get("classAorta")]
    pool = [m for m in pool if m is not None and m.any()]
    if not pool:
        return None
    pts = np.vstack([np.argwhere(m) for m in pool]).astype(float) * zooms
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
    return vt[0]


def estimate_frame(masks: dict[str, np.ndarray], zooms: np.ndarray | None = None) -> Frame:
    """
    `masks` maps `class*` names (without extension) to boolean volumes. Needs at
    least classLiver, classIVC and classAorta; every further label sharpens the fit.
    """
    zooms = np.ones(3) if zooms is None else np.asarray(zooms, dtype=float)
    if not (masks.get("classLiver") is not None and masks.get("classIVC") is not None):
        raise ValueError("need at least classLiver and classIVC to recover the frame")

    centroids = _centroids(masks, zooms)
    ev_r = _direction_evidence(RIGHTWARD, centroids)
    ev_a = _direction_evidence(ANTERIOR, centroids)
    ev_s = _direction_evidence(SUPERIOR, centroids)
    if not (ev_r and ev_a and ev_s):
        raise ValueError("not enough labels to orient all three anatomical axes")

    principal = _principal_direction(masks, zooms)
    if principal is not None:
        # orient it with whatever superior evidence already exists, then let it vote
        consensus = sum(w * v for _, v, w in ev_s)
        if float(principal @ consensus) < 0:
            principal = -principal
        ev_s = ev_s + [("IVC+aorta principal direction", principal, AXIS_PRIOR_WEIGHT)]

    bundles = (ev_r, ev_a, ev_s)
    total_weight = sum(w for ev in bundles for _, _, w in ev)

    scored = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            triad = [np.eye(3)[perm[k]] * signs[k] for k in range(3)]
            if not (np.allclose(np.cross(triad[0], triad[1]), triad[2]) or
                    np.allclose(np.cross(triad[0], triad[1]), -triad[2])):
                continue   # keep only properly-oriented frames
            score = sum(w * float(v @ triad[k])
                        for k, ev in enumerate(bundles) for _, v, w in ev)
            scored.append((score, perm, signs, triad))
    scored.sort(key=lambda t: -t[0])
    best, runner_up = scored[0], scored[1]
    _, _, _, triad = best

    def axis_of(vec):
        ax = int(np.argmax(np.abs(vec)))
        return ax, int(np.sign(vec[ax]))

    (r_ax, r_sg), (a_ax, a_sg), (s_ax, s_sg) = (axis_of(t) for t in triad)
    confidence = best[0] / total_weight
    margin = (best[0] - runner_up[0]) / total_weight

    evidence = []
    warnings: list[str] = []
    for k, (code, ev) in enumerate(zip("RAS", bundles)):
        for name, v, w in ev:
            alignment = float(v @ triad[k])
            evidence.append((code, name, alignment))
            if alignment < 0:
                warnings.append(f"{code}: '{name}' points the wrong way "
                                f"(alignment {alignment:+.2f}) -- outvoted")
    if confidence < LOW_CONFIDENCE:
        warnings.append(f"low confidence ({confidence:.2f} < {LOW_CONFIDENCE}): the "
                        f"anatomical evidence is weak or contradictory; check by eye")

    return Frame(r_axis=r_ax, r_sign=r_sg, a_axis=a_ax, a_sign=a_sg,
                 s_axis=s_ax, s_sign=s_sg, confidence=confidence, margin=margin,
                 evidence=evidence, warnings=warnings)
