"""
Couinaud segmentation of the liver from the skeletonized vascular graph.

Not a new method. Nearest-seed propagation over labelled portal branches is Selle,
Preim, Schenk & Peitgen, "Analysis of vasculature for liver surgical planning",
IEEE TMI 21(11):1344-1357, 2002 (in `sources/`), where it is called NNSA and is
validated against eight corrosion casts. Two of the three propagation variants in
`growth.py` appear in the same paper as boundary conditions on its Laplacian
alternative. What this directory adds is an audit -- the hepatic veins used as a
held-out test rather than as an input, and the method's dependence on label
completeness measured rather than described. See README.md before claiming novelty.

The idea, in one line: **Couinaud segments are portal territories**, so instead of
cutting the parenchyma with planes we label the *portal skeleton nodes* with a
segment number and let a nearest-seed (Voronoi) assignment propagate those labels
to every liver voxel. Planes are used only to decide which segment a skeleton node
belongs to -- never to slice the liver -- so the resulting boundaries are curved
portal-territory boundaries rather than flat cuts.

What the data gives us (`data/0_test_nifti/<case>/class*.nii.gz`):

    classLiver                        the parenchyma to partition
    classPediculoPortalIzquierdo      left portal pedicle       -> II, III, IV
    classPediculoPortalDerecho        right portal trunk        -> V..VIII (parent)
    classPediculoPortalAnteriorDerecho    right anterior sector -> V, VIII
    classPediculoPortalPosteriorDerecho   right posterior sector-> VI, VII
    classVenaPorta                    main portal trunk (bifurcation landmark)
    classVenaHepaticaMedia            middle HV -> Cantlie plane (cross-check)
    classVenaHepaticaDerecha          right HV  -> right scissura (cross-check/fallback)
    classVenaHepaticaIzquierda        left HV   -> left scissura (cross-check)
    classIVC                          IVC axis: pins down the vein planes, anchors segment I

The four `PediculoPortal*` classes are already the Couinaud *sectors*, which is
what makes this tractable: sector membership is read straight off the label, and
only two further divisions have to be derived.

1. `best_axis_cut`     -- the superior/inferior division inside each sector, found
   by removing the single edge of that pedicle's graph that best separates its
   nodes along the supero-inferior axis. The right anterior pedicle really does
   divide into a segment-V branch and a segment-VIII branch, so cutting the tree at
   that division locates the boundary wherever it sits instead of assuming it lies
   on a plane. `transverse_plane` (PCA through the proximal pedicles) and, failing
   that, the median, are the fallbacks -- a plane threshold alone puts every node on
   one side in roughly 40% of attempts on this data, silently emptying a segment.
2. `umbilical_plane`   -- the medial/lateral division of the left pedicle (IV vs
   II+III). Walking the left pedicle's graph always to the widest neighbour
   (Murray's law makes the parent the widest branch at every bifurcation) recovers
   the left portal vein's trunk; its leftmost point is where the vessel turns into
   the umbilical fissure, and a near-sagittal plane is erected there.
3. `vein_planes`       -- one plane per hepatic vein, each constrained to contain
   the retrohepatic IVC axis, since all three veins converge on it. These are not
   needed for the default assignment; they provide the independent cross-check in
   `checks.py` and the sector boundaries in the optional hybrid mode.

How those seed labels reach the parenchyma is a separate choice, in `growth.py`.
The default is the straight-line nearest seed described above. Three alternatives
are available and all default to off, each addressing a specific known weakness:
a *geodesic* metric that keeps the front inside the liver, *Murray flow weighting*
that lets a pedicle's measured radius rather than its traced extent decide how much
it claims, and a *hepatic-vein barrier* that lets a sector boundary curve onto the
actual vein. `ablate.py` measures what each is worth on this cohort.

Segment I (caudate) has no portal label in this schema and cannot be derived from
the graph. It is filled by an explicit geometric heuristic and flagged as such in
the output -- see `Result.caudate_is_heuristic`.

Everything is computed in millimetre coordinates (voxel index x zooms). The
anatomical axes are recovered from the anatomy by `anatomy_frame.py`, *not* from
the affine -- see the warning in that module and in README.md: every case in this
dataset reports axcodes ('R','A','S') and almost none of them are.
"""

from __future__ import annotations

import sys
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import networkx as nx
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import growth  # noqa: E402
from anatomy_frame import Frame, estimate_frame  # noqa: E402
from reconstruction import load_thinning, estimate_point_radii_from_mask  # noqa: E402

# numpy 2.0 on macOS/Accelerate raises spurious divide/overflow/invalid warnings on
# large matmuls; the results are bit-identical to the loop form (verified), and the
# pipeline does a lot of `points @ normal` on million-row arrays.
_warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

SEG_NAMES = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}
SEG_IDS = list(SEG_NAMES)

SEG_DESCRIPTION = {
    1: "caudate",
    2: "left lateral superior",
    3: "left lateral inferior",
    4: "left medial (quadrate)",
    5: "right anterior inferior",
    6: "right posterior inferior",
    7: "right posterior superior",
    8: "right anterior superior",
}

# Reference volume fraction of total liver, as (mean, SD) in %, from
# Leelaudomlipi et al., "Volumetric analysis of liver segments in 155 living
# donors" (Liver Transpl 2002). The check band below is mean +/- 2 SD, which is
# wide on purpose: inter-subject variation in segment volume is genuinely large,
# so this catches a gross failure and nothing finer.
SEG_REFERENCE_MEAN_SD = {
    1: (1.6, 0.5), 2: (6.4, 2.6), 3: (7.6, 2.7), 4: (16.9, 4.4),
    5: (10.6, 3.4), 6: (8.4, 2.9), 7: (14.9, 4.5), 8: (17.6, 4.7),
}
SEG_REFERENCE_PCT = {s: (max(0.0, m - 2 * sd), m + 2 * sd)
                     for s, (m, sd) in SEG_REFERENCE_MEAN_SD.items()}

# Validated categorical assignment -- see README (`palette` section). The eight
# hues are the dataviz reference palette; the segment->slot mapping was chosen by
# maximising the worst Delta E over the *Couinaud spatial-adjacency* pairlist.
SEG_COLOR_LIGHT = {
    1: "#eda100", 2: "#1baf7a", 3: "#eb6834", 4: "#2a78d6",
    5: "#008300", 6: "#e34948", 7: "#4a3aa7", 8: "#e87ba4",
}
SEG_COLOR_DARK = {
    1: "#c98500", 2: "#199e70", 3: "#d95926", 4: "#3987e5",
    5: "#008300", 6: "#e66767", 7: "#9085e9", 8: "#d55181",
}

PEDICLES = {
    "left": "classPediculoPortalIzquierdo",
    "right": "classPediculoPortalDerecho",
    "right_ant": "classPediculoPortalAnteriorDerecho",
    "right_post": "classPediculoPortalPosteriorDerecho",
}
VEINS = {"mhv": "classVenaHepaticaMedia",
         "rhv": "classVenaHepaticaDerecha",
         "lhv": "classVenaHepaticaIzquierda"}
LIVER = "classLiver"
PORTA = "classVenaPorta"
IVC = "classIVC"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    """Tunables. All distances in mm."""
    trunk_fit_radius: float = 45.0   # radius around the bifurcation used to fit the portal plane
    drop_trunk_mm: float = 12.0      # seeds this close to the bifurcation are dropped (ambiguous trunk)
    max_plane_tilt_deg: float = 50.0 # portal plane falls back to axial past this tilt from horizontal
    max_vein_tilt_deg: float = 55.0  # a scissura normal further than this from expected is rejected
    vein_trunk_mm: float = 70.0      # only vein voxels this close to the IVC define its scissura
    curve_resample: int = 250        # points each curve is resampled to before a joint PCA fit
    trunk_radius_drop: float = 0.55  # largest-radius walk stops below this fraction of the root radius
    umbilical_span_mm: float = 25.0  # trunk length around the turn used to tilt the fissure plane
    umbilical_min_ap: float = 0.55   # that stretch must run this antero-posteriorly to tilt it
    caudate_ivc_mm: float = 20.0     # caudate: max distance to the IVC axis
    caudate_right_mm: float = 12.0   # caudate: how far right of the IVC it may reach
    caudate_below_bif_mm: float = 15.0  # caudate: how far below the bifurcation it may reach
    min_cut_gap_mm: float = 8.0      # a graph cut must separate its two sides by at least this
    min_side_frac: float = 0.08      # neither side of any split may be smaller than this
    min_seed_points: int = 20        # a pedicle with fewer skeleton points is ignored
    min_vein_voxels: int = 150       # a hepatic vein smaller than this cannot define a plane
    min_planarity: float = 0.45      # nor can one whose voxel cloud is not sheet-like
    left_margin_mm: float = 15.0     # left-pedicle nodes further right of the bifurcation
                                     # than this are main-trunk overlap, not left liver
    smooth_iterations: int = 0       # optional majority-vote smoothing passes on the label volume
    largest_liver_cc: bool = True    # drop stray connected components of the liver mask
    mode: str = "voronoi"            # 'voronoi' (portal territories only) or 'hybrid'
    rng_seed: int = 0

    # --- how the seed labels reach the parenchyma (see growth.py) ---------------
    assign: str = "euclidean"        # 'euclidean': nearest seed in a straight line, the
                                     # original rule. 'geodesic': the front must travel
                                     # through the liver, so no seed can claim tissue
                                     # across a fissure it could never branch into.
    flow_alpha: float = 0.0          # Murray weighting, w = r^alpha on the pedicle root
                                     # radius. A weight scales linear extent, so volume
                                     # follows its cube: alpha=1 gives volume ~ r^3,
                                     # which is Murray exactly. 0 = radii ignored, and
                                     # every seed advances equally (today's behaviour,
                                     # and today's coverage bias). Needs 'geodesic'.
    flow_log_clip: float = 0.7       # bound on |log w|, so one bad radius cannot run
                                     # away with the liver (~2x either way)
    vein_barrier: float = 0.0        # how much dearer it is to cross a hepatic vein
                                     # than plain parenchyma. Lets a sector boundary
                                     # curve onto the actual vein instead of taking a
                                     # plane through it. Needs 'geodesic'.
    vein_barrier_mm: float = 12.0    # distance over which that cost ramps up
    vein_barrier_holdout: str = ""   # a vein ('mhv'/'rhv'/'lhv') deliberately kept out
                                     # of the cost, so its plane survives as
                                     # independent evidence for the cross-validation


# --------------------------------------------------------------------------- #
# Case loading & anatomical frame
# --------------------------------------------------------------------------- #

@dataclass
class Case:
    name: str
    path: Path
    masks: dict[str, np.ndarray]
    affine: np.ndarray
    zooms: np.ndarray
    shape: tuple[int, int, int]
    frame: Frame          # anatomical axes recovered from the anatomy, not the affine

    def has(self, key: str) -> bool:
        return key in self.masks and bool(self.masks[key].any())

    def unit(self, code: str) -> np.ndarray:
        """Unit vector (in mm space) pointing anatomically Right / Anterior / Superior."""
        return self.frame.unit(code)

    def voxel_volume_ml(self) -> float:
        return float(np.prod(self.zooms)) / 1000.0


def load_case(case_dir: str | Path) -> Case:
    """Load every `class*.nii.gz` of a case as a boolean mask and recover its frame."""
    case_dir = Path(case_dir)
    files = sorted(case_dir.glob("class*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"no class*.nii.gz under {case_dir}")

    masks, affine, zooms, shape = {}, None, None, None
    for f in files:
        img = nib.load(str(f))
        if affine is None:
            affine, zooms, shape = img.affine, np.asarray(img.header.get_zooms()[:3]), img.shape
        elif img.shape != shape:
            raise ValueError(f"{f.name} shape {img.shape} != case shape {shape}")
        masks[f.name[: -len(".nii.gz")]] = np.asarray(img.dataobj) > 0

    off_diag = np.abs(affine[:3, :3] - np.diag(np.diag(affine[:3, :3]))).max()
    if off_diag > 1e-3:
        print(f"  ! affine is not axis-aligned (off-diagonal {off_diag:.3f}); "
              f"mm coordinates are approximate")

    return Case(name=case_dir.name, path=case_dir, masks=masks, affine=affine,
                zooms=zooms, shape=shape, frame=estimate_frame(masks, zooms))


def missing_labels(case_dir: str | Path) -> list[str]:
    """Which of the labels this pipeline needs are absent from a case folder."""
    names = {f.name[: -len(".nii.gz")] for f in Path(case_dir).glob("class*.nii.gz")}
    need = [LIVER, PORTA, IVC, PEDICLES["left"], *VEINS.values()]
    missing = [n for n in need if n not in names]
    # the right side needs either the trunk or both sub-sectors
    if PEDICLES["right"] not in names and not (
            PEDICLES["right_ant"] in names and PEDICLES["right_post"] in names):
        missing.append("right portal pedicle (trunk or ant+post)")
    return missing


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def crop_skeletonize(mask: np.ndarray) -> np.ndarray:
    """`reconstruction.load_thinning` on the mask's bounding box -> full-volume voxel indices."""
    if not mask.any():
        return np.empty((0, 3), dtype=float)
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    pts = load_thinning(mask[sl])
    if len(pts) == 0:
        return np.empty((0, 3), dtype=float)
    return pts.astype(float) + np.array([s.start for s in sl], dtype=float)


def to_mm(voxel_pts: np.ndarray, case: Case) -> np.ndarray:
    return np.asarray(voxel_pts, dtype=float) * case.zooms


def resample(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random subsample so two curves of very different length weigh equally in a joint PCA."""
    if len(points) <= n:
        return points
    return points[rng.choice(len(points), n, replace=False)]


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA plane. Returns (centroid, unit normal, planarity in [0, 1])."""
    c = points.mean(axis=0)
    x = points - c
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    normal = vt[-1]
    var = sv ** 2 / max(len(points) - 1, 1)
    planarity = 1.0 - var[2] / var[1] if var[1] > 0 else 0.0
    return c, normal / np.linalg.norm(normal), float(planarity)


def fit_plane_through_line(points: np.ndarray, line_point: np.ndarray,
                           line_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Best plane through `points` that *contains* the given line.

    All three hepatic veins converge on the IVC, so every scissura plane has to
    contain the retrohepatic IVC axis. Imposing that leaves a single degree of
    freedom -- rotation about the axis -- which the vein's own points resolve, and
    removes the drift a free 3-parameter fit suffers when a vein's tributary fan is
    broader than its trunk.
    """
    d = line_dir / np.linalg.norm(line_dir)
    helper = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(d, helper); u /= np.linalg.norm(u)
    v = np.cross(d, u)

    rel = np.asarray(points, dtype=float) - line_point
    flat = np.stack([rel @ u, rel @ v], axis=1)      # component perpendicular to the line
    cov = flat.T @ flat / max(len(flat) - 1, 1)
    w, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0][0] * u + vecs[:, 0][1] * v   # minor axis in the perpendicular plane
    planarity = 1.0 - w[0] / w[1] if w[1] > 0 else 0.0
    return np.asarray(line_point, dtype=float), normal / np.linalg.norm(normal), float(planarity)


def orient(normal: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Flip a plane normal so its positive side lies in the `toward` direction."""
    return -normal if float(normal @ toward) < 0 else normal


def signed_distance(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - origin) @ normal


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.clip(np.abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return float(np.degrees(np.arccos(cos)))


@dataclass
class Plane:
    name: str
    origin: np.ndarray
    normal: np.ndarray
    planarity: float = 0.0
    note: str = ""

    def sd(self, points: np.ndarray) -> np.ndarray:
        return signed_distance(points, self.origin, self.normal)


# --------------------------------------------------------------------------- #
# Skeleton graph operations
# --------------------------------------------------------------------------- #

def skeleton_graph(points_mm: np.ndarray, k: int = 8) -> nx.Graph:
    """kNN graph pruned to its MST -- the same construction `reconstruction.py` uses."""
    n = len(points_mm)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    if n < 2:
        return g
    tree = cKDTree(points_mm)
    dists, idxs = tree.query(points_mm, k=min(k + 1, n))
    edges = {}
    for i in range(n):
        for d, j in zip(dists[i][1:], idxs[i][1:]):
            key = (min(i, int(j)), max(i, int(j)))
            edges[key] = float(d)
    g.add_weighted_edges_from(((u, v, w) for (u, v), w in edges.items()), weight="length")
    # MST over each component keeps the graph a forest without disconnecting anything
    return nx.minimum_spanning_tree(g, weight="length")


def best_axis_cut(points_mm: np.ndarray, axis_vec: np.ndarray, min_frac: float = 0.15,
                  min_gap_mm: float = 8.0) -> tuple[np.ndarray, float] | None:
    """
    Split a pedicle's skeleton in two by removing the single graph edge that best
    separates its nodes along `axis_vec`.

    This is the graph-native replacement for thresholding against a plane: the
    right anterior pedicle really does divide into a segment-V branch and a
    segment-VIII branch, and cutting the tree at that division finds the boundary
    wherever it happens to sit, instead of assuming it lies on a fitted plane.

    Returns (boolean mask that is True on the *positive* side of `axis_vec`, gap in
    mm between the two sides' means), or None when no split is convincing enough.
    """
    n = len(points_mm)
    if n < 20:
        return None
    graph = skeleton_graph(points_mm)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    main = graph.subgraph(components[0])
    if main.number_of_nodes() < 20:
        return None

    proj = points_mm @ axis_vec
    root = next(iter(main.nodes))
    parent = {root: None}
    for u, v in nx.dfs_edges(main, root):
        parent[v] = u

    count: dict[int, int] = {}
    total: dict[int, float] = {}
    for u in nx.dfs_postorder_nodes(main, root):
        c, s = 1, float(proj[u])
        for w in main.neighbors(u):
            if parent.get(w) == u:
                c += count[w]
                s += total[w]
        count[u], total[u] = c, s

    all_count, all_total = count[root], total[root]
    best_score, best_node, best_gap = 0.0, None, 0.0
    for node, par in parent.items():
        if par is None:
            continue
        a_c, a_s = count[node], total[node]
        b_c, b_s = all_count - a_c, all_total - a_s
        if min(a_c, b_c) < max(3, int(min_frac * all_count)):
            continue
        gap = abs(a_s / a_c - b_s / b_c)
        score = gap * min(a_c, b_c)
        if score > best_score:
            best_score, best_node, best_gap = score, node, gap
    if best_node is None or best_gap < min_gap_mm:
        return None

    subtree = set(nx.descendants(nx.dfs_tree(main, root), best_node)) | {best_node}
    side = np.zeros(n, dtype=bool)
    idx = np.array(sorted(subtree))
    side[idx] = True
    # nodes in the other (small) components follow their nearest labelled neighbour
    labelled = np.array(sorted(components[0]))
    stray = np.setdiff1d(np.arange(n), labelled)
    if len(stray):
        _, j = cKDTree(points_mm[labelled]).query(points_mm[stray], k=1)
        side[stray] = side[labelled[j]]

    if proj[side].mean() < proj[~side].mean():   # make True the positive-axis side
        side = ~side
    return side, best_gap


def largest_radius_walk(points_mm: np.ndarray, radii: np.ndarray, root: int,
                        drop: float) -> list[int]:
    """
    Walk away from `root` always taking the neighbour with the largest radius.

    Murray's law makes the parent vessel the widest one at every bifurcation, so
    this recovers the main trunk of a pedicle -- for the left portal pedicle that
    is the transverse portion followed by the umbilical portion.
    """
    g = skeleton_graph(points_mm)
    if root not in g:
        return [root]
    order = nx.bfs_tree(g, root)
    root_r = float(radii[root]) if len(radii) else 1.0
    path, current, visited = [root], root, {root}
    while True:
        children = [c for c in order.successors(current) if c not in visited]
        if not children:
            break
        nxt = max(children, key=lambda c: radii[c])
        if root_r > 0 and radii[nxt] < drop * root_r:
            break
        visited.add(nxt)
        path.append(nxt)
        current = nxt
    return path


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    case: str
    labels: np.ndarray                       # uint8 volume, 0 = background, 1..8 = segments
    liver: np.ndarray                        # the liver mask actually partitioned
    seeds_mm: np.ndarray                     # (N, 3) labelled portal skeleton nodes
    seed_labels: np.ndarray                  # (N,) segment id per seed
    seed_source: np.ndarray                  # (N,) pedicle key per seed
    mode: str = "voronoi"
    assign: str = "euclidean"                # metric the labels travelled under
    flow_alpha: float = 0.0                  # Murray weighting actually applied
    vein_barrier: float = 0.0                # vein-crossing cost actually applied
    vein_barrier_holdout: str = ""           # vein deliberately excluded from it
    skeletons: dict[str, np.ndarray] = field(default_factory=dict)   # key -> (N,3) mm
    planes: dict[str, Plane] = field(default_factory=dict)
    bifurcation_mm: np.ndarray | None = None
    umbilical_pt_mm: np.ndarray | None = None
    trunk_path_mm: np.ndarray | None = None
    voxel_volume_ml: float = 1e-3
    caudate_is_heuristic: bool = True
    notes: list[str] = field(default_factory=list)

    def veins_used(self) -> set[str]:
        """
        Which hepatic veins fed into the assignment, and so cannot cross-validate it.

        Hybrid mode takes its sector boundaries straight from the vein planes, and a
        vein barrier puts the vein masks into the growth cost. Comparing either result
        back to those same veins measures nothing. Whatever this returns is spent
        evidence; the rest is still independent.
        """
        if self.mode == "hybrid":
            return set(VEINS)
        if self.vein_barrier > 0:
            return set(VEINS) - {self.vein_barrier_holdout}
        return set()

    def volumes_ml(self) -> dict[int, float]:
        counts = np.bincount(self.labels.ravel(), minlength=9)
        return {s: float(counts[s]) * self.voxel_volume_ml for s in SEG_IDS}

    def volume_pct(self) -> dict[int, float]:
        v = self.volumes_ml()
        total = sum(v.values())
        return {s: (100.0 * v[s] / total if total else 0.0) for s in SEG_IDS}


def _pedicle_skeletons(case: Case, cfg: Config) -> dict[str, np.ndarray]:
    out = {}
    for key, name in PEDICLES.items():
        if case.has(name):
            pts = crop_skeletonize(case.masks[name])
            if len(pts) >= cfg.min_seed_points:
                out[key] = pts
    return out


def _portal_bifurcation(case: Case, ped_vox: dict[str, np.ndarray],
                        porta_vox: np.ndarray, notes: list[str]) -> np.ndarray:
    """
    Midpoint of the closest left-pedicle / right-pedicle pair, snapped onto the
    main portal trunk when that label is available.
    """
    left = to_mm(ped_vox["left"], case)
    right_keys = [k for k in ("right", "right_ant", "right_post") if k in ped_vox]
    right = to_mm(np.vstack([ped_vox[k] for k in right_keys]), case)

    d, j = cKDTree(right).query(left, k=1)
    i = int(np.argmin(d))
    bif = 0.5 * (left[i] + right[int(j[i])])
    notes.append(f"portal bifurcation: left-right pedicle gap {float(d[i]):.1f} mm")

    if len(porta_vox):
        porta = to_mm(porta_vox, case)
        dd, jj = cKDTree(porta).query(bif[None, :], k=1)
        if float(dd[0]) < 25.0:
            bif = porta[int(jj[0])]
            notes.append(f"  snapped to VenaPorta centreline ({float(dd[0]):.1f} mm away)")
    return bif


def _transverse_plane(case: Case, ped_mm: dict[str, np.ndarray], porta_mm: np.ndarray,
                      bif: np.ndarray, cfg: Config, notes: list[str]) -> Plane:
    """The portal plane: PCA through the proximal pedicles around the bifurcation."""
    pool = [p for p in ped_mm.values()]
    if len(porta_mm):
        pool.append(porta_mm)
    pts = np.vstack(pool)
    near = pts[np.linalg.norm(pts - bif, axis=1) <= cfg.trunk_fit_radius]

    s_hat = case.unit("S")
    if len(near) < 20:
        notes.append("transverse plane: too few proximal points -> pure axial through bifurcation")
        return Plane("transverse", bif, s_hat, 0.0, "axial fallback (sparse)")

    _, normal, planarity = fit_plane(near)
    normal = orient(normal, s_hat)
    tilt = angle_deg(normal, s_hat)
    if tilt > cfg.max_plane_tilt_deg:
        notes.append(f"transverse plane: fitted normal tilted {tilt:.0f} deg from vertical "
                     f"(> {cfg.max_plane_tilt_deg:.0f}) -> pure axial fallback")
        return Plane("transverse", bif, s_hat, planarity, f"axial fallback (tilt {tilt:.0f} deg)")

    notes.append(f"transverse plane: {len(near)} proximal points, tilt {tilt:.0f} deg from "
                 f"horizontal, planarity {planarity:.2f}")
    return Plane("transverse", bif, normal, planarity, f"tilt {tilt:.0f} deg")


def _ivc_axis(case: Case, liver: np.ndarray, cfg: Config,
              rng: np.random.Generator, notes: list[str]) -> np.ndarray:
    """
    Retrohepatic IVC centreline, in mm, clipped to the liver's supero-inferior extent.

    Built from the per-slice centroid rather than from `skeletonize`: the IVC is a
    wide tube and 3D thinning leaves a ragged sheet inside it (>1000 points for a
    20 cm vessel), which then dominates any plane fitted through it.
    """
    if not case.has(IVC):
        notes.append("! no IVC label: vein planes are under-determined")
        return np.empty((0, 3))

    s_axis = case.frame.s_axis
    idx = np.argwhere(case.masks[IVC])
    order = np.argsort(idx[:, s_axis])
    idx = idx[order]
    slices, starts = np.unique(idx[:, s_axis], return_index=True)
    centres = []
    for k, start in enumerate(starts):
        stop = starts[k + 1] if k + 1 < len(starts) else len(idx)
        if stop - start < 8:            # too few voxels to trust a centroid
            continue
        centres.append(idx[start:stop].mean(axis=0))
    if len(centres) < 10:
        notes.append("! IVC too thin for a per-slice centreline; falling back to thinning")
        return to_mm(crop_skeletonize(case.masks[IVC]), case)

    pts = to_mm(np.array(centres), case)
    s_hat = case.unit("S")
    liver_pts = to_mm(np.argwhere(liver), case)
    lo, hi = np.percentile(liver_pts @ s_hat, [2, 98])
    proj = pts @ s_hat
    clipped = pts[(proj >= lo - 10) & (proj <= hi + 10)]
    notes.append(f"IVC axis: {len(clipped)}/{len(pts)} per-slice centroids inside the "
                 f"liver's S-range")
    return clipped if len(clipped) >= 10 else pts


def _vein_planes(case: Case, ivc_mm: np.ndarray, liver: np.ndarray, cfg: Config,
                 rng: np.random.Generator, notes: list[str]) -> tuple[dict[str, Plane],
                                                                     dict[str, np.ndarray]]:
    """
    One plane per hepatic vein, each constrained to contain the retrohepatic IVC
    axis and fitted to the vein's proximal trunk (see `fit_plane_through_line`).
    A plane whose normal ends up implausibly far from the direction that scissura
    must separate is rejected rather than used.
    """
    planes: dict[str, Plane] = {}
    skels: dict[str, np.ndarray] = {}
    r_hat, a_hat, s_hat = case.unit("R"), case.unit("A"), case.unit("S")
    # Direction each scissura's normal must roughly point, used both to orient the
    # fitted plane and to reject an implausible one. The left scissura comes out
    # sagittal-ish in every case here (|R| 0.76-0.97), sitting close to the
    # umbilical fissure -- which is what makes it a usable cross-check for the
    # portal-derived IV / II+III boundary rather than a supero-inferior divider.
    toward = {"mhv": r_hat, "rhv": a_hat, "lhv": r_hat}

    liver_pts = to_mm(np.argwhere(liver), case)
    lo = liver_pts.min(axis=0) - 5
    hi = liver_pts.max(axis=0) + 5

    if len(ivc_mm) < 10:
        notes.append("! no IVC axis: hepatic vein planes cannot be anchored, all skipped")
        return planes, skels

    ivc_point = ivc_mm.mean(axis=0)
    ivc_dir = fit_line_direction(ivc_mm)
    ivc_tree = cKDTree(ivc_mm)

    for key, name in VEINS.items():
        if not case.has(name):
            notes.append(f"! {key.upper()} absent")
            continue
        skels[key] = to_mm(crop_skeletonize(case.masks[name]), case)
        # Fit on voxels, not the centreline: a plane needs points rather than a
        # curve, and a small vein thins to a handful of skeleton nodes
        # (`VenaHepaticaDerecha` in RMV2025_0003 gives 4) while still having a
        # perfectly usable voxel cloud.
        pts = to_mm(np.argwhere(case.masks[name]), case)
        inside = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
        pts = inside if len(inside) >= cfg.min_vein_voxels else pts
        # ...and only on the trunk. These labels are whole tributary trees whose
        # peripheral fan is wider than the trunk and pulls the plane away from the
        # scissura; the scissura is defined by the trunk running into the IVC.
        d_ivc, _ = ivc_tree.query(pts, k=1)
        trunk = pts[d_ivc <= cfg.vein_trunk_mm]
        used = trunk if len(trunk) >= cfg.min_vein_voxels else pts
        if len(used) < cfg.min_vein_voxels:
            notes.append(f"! {key.upper()}: only {len(used)} usable voxels, plane skipped")
            continue

        origin, normal, planarity = fit_plane_through_line(
            resample(used, 4 * cfg.curve_resample, rng), ivc_point, ivc_dir)
        normal = orient(normal, toward[key])
        tilt = angle_deg(normal, toward[key])
        if planarity < cfg.min_planarity:
            notes.append(f"! {key.upper()} plane: planarity only {planarity:.2f} "
                         f"(<{cfg.min_planarity}) -- too stubby to define a scissura, skipped")
            continue
        if tilt > cfg.max_vein_tilt_deg:
            notes.append(f"! {key.upper()} plane: normal {tilt:.0f} deg from the expected "
                         f"direction (>{cfg.max_vein_tilt_deg:.0f}) -- implausible scissura, "
                         f"skipped")
            continue
        planes[key] = Plane(key, origin, normal, planarity,
                            f"{len(used)} trunk voxels, anchored on the IVC axis")
        notes.append(f"{key.upper()} plane: {len(used)}/{len(pts)} voxels within "
                     f"{cfg.vein_trunk_mm:.0f} mm of the IVC, planarity {planarity:.2f}, "
                     f"{tilt:.0f} deg from expected")
    return planes, skels


def _umbilical_plane(case: Case, left_vox: np.ndarray, bif: np.ndarray, cfg: Config,
                     notes: list[str]) -> tuple[Plane, np.ndarray, np.ndarray]:
    """
    The umbilical fissure: a vertical plane erected along the umbilical portion of
    the left portal vein, recovered by a largest-radius walk on the left pedicle
    graph starting at the bifurcation.
    """
    left_mm = to_mm(left_vox, case)
    radii = estimate_point_radii_from_mask(case.masks[PEDICLES["left"]], left_vox,
                                           spacing=tuple(case.zooms))
    root = int(np.argmin(np.linalg.norm(left_mm - bif, axis=1)))
    path = largest_radius_walk(left_mm, radii, root, cfg.trunk_radius_drop)
    trunk = left_mm[path]

    r_hat, a_hat, s_hat = case.unit("R"), case.unit("A"), case.unit("S")
    if len(trunk) < 6:
        origin = left_mm.mean(axis=0)
        notes.append("umbilical plane: trunk walk too short -> sagittal through the "
                     "left pedicle centroid")
        return (Plane("umbilical", origin, r_hat, 0.0, "sagittal fallback"),
                origin, trunk)

    # The left portal vein leaves the bifurcation running left (transverse portion)
    # and only then turns forward into the fissure (umbilical portion). Taking the
    # distal half of the walk is not enough -- on a short or truncated pedicle that
    # half is still transverse, and the plane comes out coronal. Anchor on the
    # trunk's leftmost point instead, which is where the turn happens.
    umb_idx = int(np.argmin(trunk @ r_hat))
    umb_pt = trunk[umb_idx]
    near = trunk[np.linalg.norm(trunk - umb_pt, axis=1) <= cfg.umbilical_span_mm]

    normal, note = r_hat, "sagittal through the umbilical point"
    if len(near) >= 5:
        direction = fit_line_direction(near)
        if abs(float(direction @ a_hat)) >= cfg.umbilical_min_ap:
            candidate = np.cross(direction, s_hat)
            n_norm = float(np.linalg.norm(candidate))
            if n_norm > 0.2:
                normal = orient(candidate / n_norm, r_hat)
                note = (f"tilted along the umbilical portion "
                        f"({len(near)} nodes, |A| {abs(float(direction @ a_hat)):.2f})")

    tilt = angle_deg(normal, r_hat)
    notes.append(f"umbilical plane: trunk walk {len(trunk)} nodes "
                 f"({np.linalg.norm(trunk[-1] - trunk[0]):.0f} mm), turn at node "
                 f"{umb_idx}, {note}, {tilt:.0f} deg off sagittal")
    return Plane("umbilical", umb_pt, normal, 0.0, note), umb_pt, trunk


def fit_line_direction(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    return vt[0] / np.linalg.norm(vt[0])


def _label_seeds(case: Case, ped_mm: dict[str, np.ndarray], planes: dict[str, Plane],
                 transverse: Plane, umbilical: Plane, bif: np.ndarray, cfg: Config,
                 notes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Give every portal skeleton node a Couinaud segment number.

    left pedicle : medial -> IV; lateral -> II (superior) / III (inferior)
    right ant.   : VIII (superior) / V (inferior)
    right post.  : VII (superior) / VI (inferior)
    right trunk  : anterior/posterior taken from the nearest labelled sub-sector node,
                   or from the RHV plane when the sub-sectors are missing

    Both divisions come from `split` below, which prefers a cut of the pedicle's own
    graph and only falls back to a plane. The corresponding plane -- umbilical for
    medial/lateral, transverse for superior/inferior -- is the second choice, not
    the first.
    """
    pts_list, lab_list, src_list = [], [], []
    r_hat, s_hat = case.unit("R"), case.unit("S")

    def add(points: np.ndarray, labels: np.ndarray, src: str) -> None:
        keep = np.linalg.norm(points - bif, axis=1) > cfg.drop_trunk_mm
        if keep.sum() == 0:
            return
        pts_list.append(points[keep])
        lab_list.append(labels[keep])
        src_list.append(np.full(int(keep.sum()), src, dtype=object))

    def balanced(side: np.ndarray) -> bool:
        return min(int(side.sum()), int((~side).sum())) >= max(3, int(cfg.min_side_frac
                                                                     * len(side)))

    def split(points: np.ndarray, axis_vec: np.ndarray, plane: Plane, what: str,
              axis_name: str) -> np.ndarray:
        """
        Divide a pedicle's nodes in two along `axis_vec`, preferring a cut of its own
        graph and never returning a degenerate split.

        A plane threshold on its own puts every node on one side surprisingly often
        (30 of 72 attempts across this dataset), which silently produces an empty
        segment. The cascade tries the graph cut, then a relaxed graph cut, then the
        plane, and finally the median -- so the sector always has both halves.
        """
        for gap, frac, tag in ((cfg.min_cut_gap_mm, 0.15, "graph cut"),
                               (cfg.min_cut_gap_mm / 2, 0.08, "relaxed graph cut")):
            cut = best_axis_cut(points, axis_vec, min_frac=frac, min_gap_mm=gap)
            if cut is not None and balanced(cut[0]):
                side, gap_mm = cut
                notes.append(f"  {what}: {tag} splits {int(side.sum())}/"
                             f"{int((~side).sum())} nodes, {gap_mm:.0f} mm apart in "
                             f"{axis_name}")
                return side
        if plane is not None:
            side = plane.sd(points) > 0
            if balanced(side):
                notes.append(f"  {what}: no usable graph cut -> {plane.name} plane "
                             f"({int(side.sum())}/{int((~side).sum())})")
                return side
        proj = points @ axis_vec
        side = proj > np.median(proj)
        notes.append(f"  ! {what}: no graph cut and the {plane.name if plane else 'plane'} "
                     f"split is degenerate -> median {axis_name} "
                     f"({int(side.sum())}/{int((~side).sum())})")
        return side

    # --- left pedicle -----------------------------------------------------
    left = ped_mm["left"]
    medial = split(left, r_hat, umbilical, "left medial/lateral (IV vs II+III)", "R")
    lab = np.full(len(left), 4, dtype=np.uint8)
    if (~medial).sum() >= cfg.min_seed_points:
        lateral_pts = left[~medial]
        sup = split(lateral_pts, s_hat, transverse, "left lateral (II/III)", "S")
        lab[~medial] = np.where(sup, 2, 3).astype(np.uint8)
    else:
        notes.append("  ! left lateral territory has too few nodes to split II/III")
        lab[~medial] = 3
    add(left, lab, "left")
    notes.append(f"left pedicle seeds: IV={int((lab == 4).sum())} "
                 f"II={int((lab == 2).sum())} III={int((lab == 3).sum())}")

    # --- right sub-sectors ------------------------------------------------
    sub = {}
    for key, sup_id, inf_id in (("right_ant", 8, 5), ("right_post", 7, 6)):
        if key not in ped_mm:
            continue
        pts = ped_mm[key]
        sup = split(pts, s_hat, transverse,
                    f"{key} ({SEG_NAMES[sup_id]}/{SEG_NAMES[inf_id]})", "S")
        lab = np.where(sup, sup_id, inf_id).astype(np.uint8)
        sub[key] = pts
        add(pts, lab, key)
        notes.append(f"{key} seeds: {SEG_NAMES[sup_id]}={int((lab == sup_id).sum())} "
                     f"{SEG_NAMES[inf_id]}={int((lab == inf_id).sum())}")

    # --- right trunk ------------------------------------------------------
    if "right" in ped_mm:
        pts = ped_mm["right"]
        superior = transverse.sd(pts) > 0
        if sub:
            anchor = np.vstack([sub[k] for k in sub])
            anchor_is_ant = np.concatenate([np.full(len(sub[k]), k == "right_ant")
                                            for k in sub])
            _, j = cKDTree(anchor).query(pts, k=1)
            is_ant = anchor_is_ant[j]
            how = "nearest labelled sub-sector node"
        elif "rhv" in planes:
            is_ant = planes["rhv"].sd(pts) > 0
            how = "RHV plane (no sub-sector labels available)"
        else:
            is_ant = np.ones(len(pts), dtype=bool)
            how = "! defaulted to anterior (no sub-sectors, no RHV)"
        lab = np.where(is_ant, np.where(superior, 8, 5), np.where(superior, 7, 6)).astype(np.uint8)
        add(pts, lab, "right")
        notes.append(f"right trunk seeds split by {how}: "
                     + " ".join(f"{SEG_NAMES[s]}={int((lab == s).sum())}" for s in (5, 6, 7, 8)))

    if not pts_list:
        raise RuntimeError("no portal seeds survived labelling")
    return np.vstack(pts_list), np.concatenate(lab_list), np.concatenate(src_list)


def _caudate(case: Case, liver: np.ndarray, labels: np.ndarray, ivc_mm: np.ndarray,
             bif: np.ndarray, cfg: Config, notes: list[str]) -> np.ndarray:
    """
    Segment I, by geometry rather than by portal territory.

    The caudate lobe has no dedicated portal branch in this label schema, so it is
    approximated as the parenchyma hugging the retrohepatic IVC, posterior to the
    portal bifurcation, from just below the bifurcation upward. Largest connected
    component only. This is the least trustworthy part of the pipeline.
    """
    if not len(ivc_mm):
        notes.append("! segment I skipped (no IVC axis)")
        return labels

    vox = np.argwhere(liver)
    pts = to_mm(vox, case)
    r_hat, a_hat, s_hat = case.unit("R"), case.unit("A"), case.unit("S")

    tree = cKDTree(ivc_mm)
    d_ivc, nearest = tree.query(pts, k=1)
    # the caudate hugs the IVC on its left; the caudate process reaches a little to
    # the right of it but never into the right lobe proper
    right_of_ivc = (pts - ivc_mm[nearest]) @ r_hat
    posterior = (pts - bif) @ a_hat < 0
    above = (pts - bif) @ s_hat > -cfg.caudate_below_bif_mm
    cand = ((d_ivc <= cfg.caudate_ivc_mm) & posterior & above
            & (right_of_ivc <= cfg.caudate_right_mm)
            # only its true neighbours may be reassigned; V/VI are never caudate
            & np.isin(labels[tuple(vox.T)], (2, 4, 7, 8)))

    if cand.sum() == 0:
        notes.append("! segment I: no candidate voxels")
        return labels

    vol = np.zeros(case.shape, dtype=bool)
    vol[tuple(vox[cand].T)] = True
    cc, n = ndimage.label(vol)
    if n > 1:
        sizes = ndimage.sum_labels(vol, cc, index=range(1, n + 1))
        vol = cc == (int(np.argmax(sizes)) + 1)
    labels[vol] = 1
    notes.append(f"segment I (heuristic): {int(vol.sum())} voxels from {int(cand.sum())} "
                 f"candidates, {n} components")
    return labels


def _pedicle_root_radii(case: Case, ped_vox: dict[str, np.ndarray],
                        notes: list[str]) -> dict[str, float]:
    """
    Radius of each pedicle near its root, in mm.

    A high percentile of the whole pedicle's point radii rather than the value at the
    root node itself: the root is exactly where the distance transform is least
    reliable, because the mask there abuts the main portal trunk and the inscribed
    sphere leaks into it.
    """
    out: dict[str, float] = {}
    for key, cls in PEDICLES.items():
        mask, pts = case.masks.get(cls), ped_vox.get(key)
        if mask is None or pts is None or len(pts) < 5:
            continue
        r = np.asarray(estimate_point_radii_from_mask(mask, pts, spacing=tuple(case.zooms)),
                       dtype=float)
        r = r[np.isfinite(r) & (r > 0)]
        if len(r) >= 5:
            out[key] = float(np.percentile(r, 90))
    if out:
        notes.append("pedicle root radii: "
                     + "  ".join(f"{k}={v:.2f}mm" for k, v in sorted(out.items())))
    return out


def _flow_weights(radii: dict[str, float], seed_labels: np.ndarray,
                  seed_source: np.ndarray, cfg: Config,
                  notes: list[str]) -> dict[int, float] | None:
    """Per-segment propagation weights, derived from the feeding pedicle's radius."""
    if cfg.flow_alpha <= 0:
        return None
    per_pedicle = growth.murray_weights(radii, cfg.flow_alpha, cfg.flow_log_clip, notes)

    # Segments the trunk seeds were handed to may not appear in SEGMENT_PEDICLE's
    # static map with the right pedicle, so trust the seed's own recorded source and
    # fall back to the map only when a segment has no seeds of its own.
    weights: dict[int, float] = {}
    for seg in SEG_IDS:
        pick = seed_labels == seg
        if pick.any():
            srcs, counts = np.unique(seed_source[pick].astype(str), return_counts=True)
            key = str(srcs[int(np.argmax(counts))])
        else:
            key = growth.SEGMENT_PEDICLE.get(seg, "")
        weights[seg] = per_pedicle.get(key, 1.0)
    return weights


def _growth_cost(case: Case, cfg: Config, notes: list[str]) -> np.ndarray | None:
    """Elevation image for the front, or None when the barrier is switched off."""
    if cfg.vein_barrier <= 0:
        return None
    veins = {k: case.masks[cls] for k, cls in VEINS.items()
             if k != cfg.vein_barrier_holdout and case.has(cls)}
    if cfg.vein_barrier_holdout:
        notes.append(f"  vein barrier holds out the {cfg.vein_barrier_holdout.upper()}, "
                     f"so its plane stays independent evidence")
    return growth.vein_barrier_cost(case.masks[LIVER], veins, case.zooms,
                                    cfg.vein_barrier, cfg.vein_barrier_mm, notes)


def _assign_voxels(case: Case, liver: np.ndarray, seeds: np.ndarray, seed_labels: np.ndarray,
                   seed_source: np.ndarray, planes: dict[str, Plane], umbilical: Plane,
                   radii: dict[str, float], cfg: Config,
                   notes: list[str]) -> np.ndarray:
    """
    Propagate the seed labels to every liver voxel.

    'voronoi' is the pure portal-territory answer: nearest labelled skeleton node,
    full stop. Its weakness is that it inherits the *labelling completeness* of the
    pedicle masks -- a sector whose pedicle was traced further into the periphery
    claims more parenchyma, whether or not it should.

    'hybrid' removes that bias exactly where it bites hardest, at the sector
    boundaries, by taking them from the hepatic vein planes (fitted from labels the
    portal side knows nothing about) and letting the portal graph decide only the
    superior/inferior division *inside* each sector, where both candidate labels
    come from the same pedicle and the bias cancels.

    Orthogonal to that choice, `cfg.assign` selects the metric the labels travel
    under -- straight-line or through the parenchyma -- and the geodesic option
    carries the flow weighting and the vein barrier with it. See `growth.py`.
    """
    vox = np.argwhere(liver)
    pts = to_mm(vox, case)
    labels = np.zeros(case.shape, dtype=np.uint8)
    mode = cfg.mode
    geodesic = cfg.assign == "geodesic"
    if cfg.assign not in ("euclidean", "geodesic"):
        raise ValueError(f"unknown assign rule {cfg.assign!r}")
    if not geodesic and (cfg.flow_alpha > 0 or cfg.vein_barrier > 0):
        raise ValueError("flow weighting and the vein barrier need assign='geodesic'; "
                         "neither has any meaning under a straight-line metric")

    seed_vox = np.rint(seeds / case.zooms).astype(int) if geodesic else None
    weights = _flow_weights(radii, seed_labels, seed_source, cfg, notes) if geodesic else None
    cost = _growth_cost(case, cfg, notes) if geodesic else None

    def propagate(pick: np.ndarray, restrict: np.ndarray | None, what: str) -> np.ndarray:
        """Label the voxels of one growth domain using the chosen subset of seeds."""
        if geodesic:
            grown = growth.grow(liver, seed_vox[pick], seed_labels[pick], case.zooms,
                                weights=weights, cost=cost, restrict=restrict, notes=notes)
            domain = liver if restrict is None else (liver & restrict)
            return growth.fill_unreached(grown, domain)
        target = np.ones(len(pts), dtype=bool) if restrict is None else restrict[tuple(vox.T)]
        out = np.zeros(case.shape, dtype=np.uint8)
        if not target.any():
            notes.append(f"  {what}: empty")
            return out
        _, nearest = cKDTree(seeds[pick]).query(pts[target], k=1)
        out[tuple(vox[target].T)] = seed_labels[pick][nearest]
        return out

    if mode == "voronoi":
        labels = propagate(np.ones(len(seeds), dtype=bool), None, "liver")
        notes.append(f"{'geodesic growth' if geodesic else 'Voronoi'}: {len(pts):,} "
                     f"liver voxels from {len(seeds):,} seeds")
        return labels

    def split_by(plane_key: str, groups: tuple[tuple[int, ...], tuple[int, ...]],
                 within: np.ndarray, what: str) -> np.ndarray:
        """
        True on the first group's side. Uses the named vein plane when it exists and
        otherwise falls back to the portal seeds themselves, so a case missing one
        hepatic vein still gets every other boundary from the veins.
        """
        plane = planes.get(plane_key)
        if plane is not None:
            return plane.sd(pts) > 0
        pick = np.isin(seed_labels, groups[0] + groups[1])
        if not pick.any():
            notes.append(f"  ! {what}: neither a {plane_key.upper()} plane nor seeds")
            return within
        _, nearest = cKDTree(seeds[pick]).query(pts, k=1)
        notes.append(f"  ! {what}: no {plane_key.upper()} plane, falling back to the "
                     f"portal seeds")
        return np.isin(seed_labels[pick][nearest], groups[0])

    everywhere = np.ones(len(pts), dtype=bool)
    right = split_by("mhv", ((5, 6, 7, 8), (2, 3, 4)), everywhere, "left/right split")
    anterior = split_by("rhv", ((5, 8), (6, 7)), everywhere, "right sector split")
    medial = umbilical.sd(pts) > 0

    regions: list[tuple[str, np.ndarray, tuple[int, ...]]] = [
        ("right anterior sector", right & anterior, (5, 8)),
        ("right posterior sector", right & ~anterior, (6, 7)),
        ("left medial (IV)", ~right & medial, (4,)),
        ("left lateral", ~right & ~medial, (2, 3)),
    ]

    for name, mask, group in regions:
        if not mask.any():
            notes.append(f"  {name}: empty")
            continue
        pick = np.isin(seed_labels, group)
        if not pick.any():
            pick = np.ones(len(seed_labels), dtype=bool)
            notes.append(f"  ! {name}: no seeds of its own, using all seeds")
        restrict = np.zeros(case.shape, dtype=bool)
        restrict[tuple(vox[mask].T)] = True
        part = propagate(pick, restrict, name)
        labels[restrict] = part[restrict]
        notes.append(f"  {name}: {int(mask.sum()):,} voxels from "
                     f"{int(pick.sum())} seeds ({'/'.join(SEG_NAMES[g] for g in group)})")
    notes.append(f"hybrid: sector boundaries from the vein planes, "
                 f"{len(pts):,} liver voxels assigned")
    return labels


def _smooth(labels: np.ndarray, liver: np.ndarray, iterations: int) -> np.ndarray:
    """Majority vote in a 3x3x3 neighbourhood -- removes speckle on Voronoi boundaries."""
    for _ in range(iterations):
        votes = np.zeros((9,) + labels.shape, dtype=np.uint16)
        for s in SEG_IDS:
            votes[s] = ndimage.uniform_filter((labels == s).astype(np.float32),
                                              size=3) * 27
        new = np.argmax(votes, axis=0).astype(np.uint8)
        labels = np.where(liver, new, 0).astype(np.uint8)
    return labels


def segment_liver(case: Case, cfg: Config | None = None, verbose: bool = True) -> Result:
    """Run the whole pipeline on one loaded case."""
    cfg = cfg or Config()
    rng = np.random.default_rng(cfg.rng_seed)
    notes: list[str] = []

    if not case.has(LIVER):
        raise ValueError(f"{case.name}: no liver mask")
    liver = case.masks[LIVER]
    if cfg.largest_liver_cc:
        cc, n = ndimage.label(liver)
        if n > 1:
            sizes = ndimage.sum_labels(liver, cc, index=range(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            dropped = int(liver.sum() - sizes[keep - 1])
            liver = cc == keep
            notes.append(f"liver mask: kept the largest of {n} components, dropped "
                         f"{dropped:,} stray voxels ({100 * dropped / (dropped + liver.sum()):.2f}%)")

    ped_vox = _pedicle_skeletons(case, cfg)
    if "left" not in ped_vox:
        raise ValueError(f"{case.name}: left portal pedicle missing -- cannot segment")
    if not any(k in ped_vox for k in ("right", "right_ant", "right_post")):
        raise ValueError(f"{case.name}: no right portal pedicle -- cannot segment")
    notes.append("pedicle skeletons: "
                 + ", ".join(f"{k}={len(v)}" for k, v in ped_vox.items()))

    porta_vox = crop_skeletonize(case.masks[PORTA]) if case.has(PORTA) else np.empty((0, 3))
    ped_mm = {k: to_mm(v, case) for k, v in ped_vox.items()}
    porta_mm = to_mm(porta_vox, case) if len(porta_vox) else np.empty((0, 3))

    bif = _portal_bifurcation(case, ped_vox, porta_vox, notes)

    # `classPediculoPortalIzquierdo` overlaps the main portal trunk near the
    # bifurcation, sometimes by tens of millimetres (65 mm in RMV2026_0022). Left
    # there, those nodes send the trunk walk backwards along the main portal vein
    # and let segment IV claim right-liver parenchyma.
    right_of_bif = (ped_mm["left"] - bif) @ case.unit("R")
    keep_left = right_of_bif <= cfg.left_margin_mm
    if keep_left.sum() < cfg.min_seed_points:
        notes.append("! left pedicle is almost entirely right of the bifurcation; "
                     "keeping it unfiltered")
    elif not keep_left.all():
        notes.append(f"left pedicle: dropped {int((~keep_left).sum())}/{len(keep_left)} "
                     f"nodes lying >{cfg.left_margin_mm:.0f} mm right of the bifurcation "
                     f"(main-trunk overlap)")
        ped_mm["left"] = ped_mm["left"][keep_left]
        ped_vox["left"] = ped_vox["left"][keep_left]

    transverse = _transverse_plane(case, ped_mm, porta_mm, bif, cfg, notes)
    ivc_mm = _ivc_axis(case, liver, cfg, rng, notes)
    vein_planes, vein_skels = _vein_planes(case, ivc_mm, liver, cfg, rng, notes)
    umbilical, umb_pt, trunk = _umbilical_plane(case, ped_vox["left"], bif, cfg, notes)

    seeds, seed_labels, seed_src = _label_seeds(case, ped_mm, vein_planes, transverse,
                                                umbilical, bif, cfg, notes)

    radii = _pedicle_root_radii(case, ped_vox, notes) if cfg.flow_alpha > 0 else {}
    labels = _assign_voxels(case, liver, seeds, seed_labels, seed_src, vein_planes,
                            umbilical, radii, cfg, notes)
    labels = _caudate(case, liver, labels, ivc_mm, bif, cfg, notes)
    if cfg.smooth_iterations:
        labels = _smooth(labels, liver, cfg.smooth_iterations)
        notes.append(f"majority smoothing: {cfg.smooth_iterations} pass(es)")

    planes = dict(vein_planes)
    planes["transverse"] = transverse
    planes["umbilical"] = umbilical

    skeletons = {f"portal_{k}": v for k, v in ped_mm.items()}
    skeletons.update({f"hv_{k}": v for k, v in vein_skels.items()})
    if len(porta_mm):
        skeletons["portal_trunk"] = porta_mm
    if len(ivc_mm):
        skeletons["ivc"] = ivc_mm

    if verbose:
        for n in notes:
            print("  " + n)

    return Result(case=case.name, labels=labels, liver=liver, seeds_mm=seeds,
                  seed_labels=seed_labels, mode=cfg.mode, assign=cfg.assign,
                  flow_alpha=cfg.flow_alpha, vein_barrier=cfg.vein_barrier,
                  vein_barrier_holdout=cfg.vein_barrier_holdout,
                  seed_source=seed_src, skeletons=skeletons, planes=planes,
                  bifurcation_mm=bif, umbilical_pt_mm=umb_pt, trunk_path_mm=trunk,
                  voxel_volume_ml=case.voxel_volume_ml(), notes=notes)


def segment_case_dir(case_dir: str | Path, cfg: Config | None = None,
                     verbose: bool = True) -> tuple[Case, Result]:
    case = load_case(case_dir)
    if verbose:
        print(f"[{case.name}] shape={case.shape} zooms={np.round(case.zooms, 2).tolist()}")
        print(f"  header axcodes {nib.aff2axcodes(case.affine)} (nominal); "
              f"recovered frame {case.frame.describe()}")
        for w in case.frame.warnings:
            print(f"  ! {w}")
    return case, segment_liver(case, cfg, verbose)
