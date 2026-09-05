# Couinaud Segmentation — Mathematical Process

This document explains the mathematics implemented in `continuity/couinaud/` . All references are of the form `file:line`.

## 1. Premise

Couinaud segments are **portal territories** (`couinaud.py:4-9`, `README.md:6-11`). The pipeline does not cut the liver parenchyma with planes. It **labels portal skeleton nodes** with a segment id `1…8` and propagates those labels to every liver voxel by **nearest-seed (Voronoi) assignment** (`couinaud.py:867-893`). Plane geometry is only used to decide which segment a skeleton node belongs to, so final boundaries are curved Voronoi boundaries, not flat cuts.

Four `PediculoPortal*` masks are already Couinaud **sectors** (`couinaud.py:24-26`):

* `left` → II, III, IV
* `right` → V–VIII (parent)
* `right_ant` → V, VIII
* `right_post` → VI, VII

Sector membership is read off the label. Only two further subdivisions must be derived: superior/inferior inside each sector, and medial/lateral inside the left pedicle (`couinaud.py:28-40`).

All geometry is in **millimetre coordinates**: `p_mm = p_vox * zooms` (`couinaud.py:245-246`), with `zooms = header.get_zooms()[:3]`.

---

## 2. Anatomical Frame Recovery — `anatomy_frame.py`

### 2.1 Why

`continuity/glb_to_nifti.py` writes `affine = diag(zooms)` with no rotation, so every case reports axcodes `('R','A','S')` while 33/38 cases actually store the cranio-caudal axis on voxel axis 1 (`anatomy_frame.py:1-11`, `couinaud.py:51-53`, `README.md:95-102`). The frame must be recovered from anatomy.

### 2.2 Evidence vectors

Centroids are `center_of_mass(mask) * zooms` (`anatomy_frame.py:117-118`). Composite centroids are built for unions (`anatomy_frame.py:131-143`).

Each evidence item is a normalized direction `v = (com(A)-com(B))/||·||` with weight `w` (`anatomy_frame.py:153-161`):

```python
RIGHTWARD (anatomy_frame.py:44-49):  RHV - LHV (1.0), right_pedicles - left_pedicle (1.0),
                                     IVC - Aorta (0.7), Liver - IVC (0.7)
ANTERIOR  (anatomy_frame.py:51-54):  right_ant - right_post (1.0), Gallbladder - Aorta (0.7),
                                     VenaPorta - IVC (0.5)
SUPERIOR  (anatomy_frame.py:56-60):  hepatic_veins - pedicles (1.5), Liver - Gallbladder (0.7),
                                     Liver - extrahepatic_IVC (0.5)
```

The **IVC+aorta principal direction** is added to the superior evidence with weight `AXIS_PRIOR_WEIGHT = 2.0` (`anatomy_frame.py:62,164-198`). It is the first right singular vector `v0` of the centred point cloud `X ∈ R^{N×3}`:

```
X = pts - mean(pts),  X = U Σ V^T  (SVD),  principal = V[0]
```

oriented to agree with the existing superior consensus (`anatomy_frame.py:194-197`).

### 2.3 Joint triad fit

Search space: all axis permutations `perm ∈ S3` (6) × all sign choices `s ∈ {±1}^3` (8) = 48 triads `T_k = s_k·e_{perm(k)}` (`anatomy_frame.py:204-205`). Only right-handed triads `T0 × T1 = T2` are kept (24 candidates) (`anatomy_frame.py:207`).

Score of a triad `T = (T_R, T_A, T_S)`:

```
score(T) = Σ_{k∈{R,A,S}} Σ_{(name,v,w)∈evidence_k}  w · (v · T_k)    (anatomy_frame.py:209-210)
```

This is a weighted sum of cosines between evidence directions and triad axes. Maximising it is equivalent to a weighted Procrustes alignment restricted to signed permutations.

Let `W = Σ w`. Then:

```
confidence = best_score / W            ∈ [-1,1]   (anatomy_frame.py:221)
margin     = (best_score - runner_up) / W         (anatomy_frame.py:222)
```

`confidence` = mean weighted alignment of the winner; `margin` = separation from second-best. `LOW_CONFIDENCE = 0.62` flags suspect cases (`anatomy_frame.py:64,233-235`). Per-evidence alignments `v·T_k` and wrong-way warnings are stored (`anatomy_frame.py:224-232`).

`Frame.unit(code)` returns the unit vector in mm space (`anatomy_frame.py:81-86`); `signed_distance` and `angle_deg` use it throughout `couinaud.py:292-304`.

---

## 3. Skeletonisation — `couinaud.py:234-242`, `reconstruction.py`

Each pedicle/vein/IVC mask is cropped to its bounding box `find_objects` then thinned via `reconstruction.load_thinning` (Lee 3-D thinning / `skimage.morphology.skeletonize`). Output is voxel indices shifted back to full-volume coordinates. Radii for the left trunk walk are estimated by `estimate_point_radii_from_mask` (Euclidean distance transform sampled at skeleton points).

---

## 4. Skeleton Graph — `couinaud.py:322-339`

For `N` points `P ∈ R^{N×3}` (mm):

1. `k=8` nearest neighbours via `cKDTree` (excluding self).
2. Undirected weighted graph `G` with edge weight = Euclidean length.
3. **Minimum Spanning Tree** per connected component (`networkx.minimum_spanning_tree`).

This reproduces the construction in `reconstruction.py` and yields a forest that approximates the vessel tree without cycles. `skeleton_graph` is the substrate for both `best_axis_cut` and `largest_radius_walk`.

---

## 5. Portal Bifurcation — `couinaud.py:482-503`

```
left_mm  = left pedicle skeleton (mm)
right_mm = union(right, right_ant, right_post) (mm)
(i*, j*) = argmin_{i,j} ||left[i]-right[j]||   via cKDTree
bif0     = 0.5*(left[i*]+right[j*])
```

If `classVenaPorta` skeleton exists and `min_j ||bif0 - porta[j]|| < 25 mm`, snap: `bif = porta[argmin]` (`couinaud.py:498-502`). This anchors the anatomical origin for all subsequent planes.

---

## 6. Left-Pedicle Overlap Trim — `couinaud.py:988-1002`

`classPediculoPortalIzquierdo` overlaps the main portal trunk (up to 65 mm in `RMV2026_0022`). Nodes with `(p - bif)·R̂ > left_margin_mm (15 mm)` are dropped as main-trunk overlap. If too few nodes would remain (`< min_seed_points`), the filter is skipped.

---

## 7. Transverse (Portal) Plane — `couinaud.py:506-530`

Pool: pedicle skeletons + `classVenaPorta` skeleton. Keep only points within `trunk_fit_radius = 45 mm` of `bif` (`couinaud.py:513`). Fit a free PCA plane:

### 7.1 `fit_plane` — `couinaud.py:256-264`

```
c = mean(P)                          centroid
X = P - c
X = U Σ V^T  (SVD),  V = [v0; v1; v2] rows of Vt
normal = v2                          smallest singular vector
var_k  = σ_k²/(N-1)
planarity = 1 - var2/var1            ∈ [0,1]
```

`planarity → 1` for a flat cloud, `→ 0` for isotropic. Normal is oriented toward `Ŝ` (`couinaud.py:521`, `292-294`). Tilt `θ = arccos(|n·Ŝ|)` (`couinaud.py:301-303`). If `θ > max_plane_tilt_deg (50°)` or `N < 20`, fall back to a pure axial plane `n = Ŝ, origin = bif`.

---

## 8. IVC Axis — `couinaud.py:532-569`

The IVC is a wide tube; 3-D thinning produces a ragged sheet (>1000 points) that would dominate plane fits. Instead:

* For each distinct `S`-slice `s`, compute the centroid of voxels in that slice (`centre = mean(idx[start:stop])`) keeping only slices with ≥8 voxels (`couinaud.py:551-553`).
* Convert to mm, clip to liver's S-range `[p2, p98] ± 10 mm` (`couinaud.py:562-565`).

Result: `ivc_mm ∈ R^{M×3}`, `M≈tube length / voxel pitch`, a clean centreline. Fallback to thinning if `<10` centroids.

`fit_line_direction` (`couinaud.py:695-698`) gives the IVC direction `d_ivc = V[0]` (first principal component).

---

## 9. Hepatic Vein (Scissura) Planes — `couinaud.py:572-643`

Each scissura must **contain the retrohepatic IVC axis** (all three hepatic veins converge on the IVC). This removes 2 DOF, leaving rotation about the axis as the only free parameter.

### 9.1 `fit_plane_through_line` — `couinaud.py:267-289`

Given line `(L0, d)` and points `P`:

```
d̂ = d/||d||
build orthonormal basis (u, v) ⊥ d̂
rel = P - L0
flat = [rel·u, rel·v] ∈ R^{N×2}          projection ⊥ d
Cov  = flat^T flat / (N-1) ∈ R^{2×2}
Cov = Q diag(w0,w1) Q^T,  w0 ≤ w1  (eigh)
n = Q[:,0]_0 · u + Q[:,0]_1 · v           minor axis in ⊥ plane
planarity = 1 - w0/w1
origin = L0,  normal = n/||n||
```

Geometrically: intersect the unknown plane with the plane ⊥ d → a line through the origin of `flat`; the best plane minimises squared distances of `flat` to that line, i.e. the minor eigenvector of `Cov`.

### 9.2 Fitting protocol

* Points = **voxel cloud** `argwhere(mask) * zooms` (not skeleton) — a small vein may give only 4 skeleton nodes (`RMV2025_0003`) but still has a usable voxel cloud (`couinaud.py:608-613`).
* Keep only voxels within `vein_trunk_mm = 70 mm` of the IVC centreline (`cKDTree` query) to isolate the trunk from the peripheral fan that would pull the plane off the scissura (`couinaud.py:618-620`).
* Resample to `4*curve_resample = 1000` points so curves of different length weigh equally (`couinaud.py:624-626`).
* Expected normal directions (`couinaud.py:587-589`): `MHV→R̂`, `RHV→Â`, `LHV→R̂` (the LHV scissura is sagittal, `|R| 0.76–0.97`, sitting near the umbilical fissure). Each fitted normal is oriented toward its expected direction and rejected if `planarity < 0.45` or `tilt > 55°` (`couinaud.py:626-638`).

---

## 10. Umbilical Fissure Plane — `couinaud.py:646-692`

### 10.1 Trunk recovery — `largest_radius_walk` (`couinaud.py:412-437`)

```
root = argmin ||left_mm - bif||
G    = skeleton_graph(left_mm)
Walk from root, always stepping to the neighbour (in BFS tree) with largest radius;
stop when radius < drop*root_radius (drop=0.55).
```

By Murray's law the parent vessel is the widest at each bifurcation, so greedy widest-neighbour walk recovers the main trunk (transverse + umbilical portions).

### 10.2 Plane erection

* `umb_pt = trunk[argmin trunk·R̂]` — the leftmost point, where the transverse portion turns into the fissure (`couinaud.py:673-674`). Using the distal half fails on short pedicles (still transverse → coronal plane).
* `near = { p ∈ trunk : ||p - umb_pt|| ≤ umbilical_span_mm (25 mm) }`
* If `|d_near·Â| ≥ 0.55` (stretch runs antero-posteriorly), set `n = (d_near × Ŝ)/||·||` oriented toward `R̂` — a vertical plane tilted along the umbilical portion (`couinaud.py:678-686`).
* Otherwise `n = R̂` (pure sagittal through `umb_pt`).

---

## 11. Graph Cut — `best_axis_cut` (`couinaud.py:341-410`)

This is the **core innovation** replacing plane thresholds. A plane threshold alone puts every node on one side in ~40% of attempts (30/72), silently emptying a segment (`couinaud.py:734-742`).

Goal: split a pedicle's skeleton into superior/inferior (along `Ŝ`) or medial/lateral (along `R̂`) by removing the single graph edge that best separates nodes along the axis.

Algorithm on the largest connected component of `G`:

```
proj[i] = P[i]·axis_vec                                     projection
Root G at arbitrary node, DFS to get parent[] and postorder
For each node u:  count[u] = subtree size,  total[u] = Σ proj in subtree
                  via bottom-up accumulation (couinaud.py:371-378)

For each edge (parent[u], u):
    a = subtree(u)          b = rest
    gap = |mean(proj[a]) - mean(proj[b])|
    score = gap * min(|a|,|b|)                              (couinaud.py:389-390)
    feasible if  min(|a|,|b|) ≥ max(3, min_frac·N)  and  gap ≥ min_gap_mm
Best edge = argmax score
```

`side = True` on the positive-axis side (flipped if needed, `couinaud.py:407-408`). Stray components inherit the label of their nearest labelled neighbour (`couinaud.py:401-405`). Returns `None` if no feasible cut.

Two attempts are made: `(gap≥8 mm, 15%)` then relaxed `(≥4 mm, 8%)` (`couinaud.py:744-752`).

---

## 12. Seed Labelling — `_label_seeds` (`couinaud.py:701-819`)

Every retained portal skeleton node receives a segment id. A `split(points, axis_vec, plane, ...)` helper cascades (`couinaud.py:733-764`):

```
1. best_axis_cut (strict)  → 2. best_axis_cut (relaxed)  → 3. plane threshold (if balanced)
→ 4. median of proj  (always balanced, last resort)
```

`balanced` means `min(|side|,|¬side|) ≥ max(3, min_side_frac·N)` with `min_side_frac=0.08`.

* **Left pedicle**: `medial = split(left, R̂, umbilical)` → `IV` if `True`, else lateral territory split by `split(lateral, Ŝ, transverse)` → `II` (superior) / `III` (inferior) (`couinaud.py:767-779`).
* **Right sub-sectors** (`right_ant`, `right_post`): each split along `Ŝ` via `transverse` → `VIII/V` and `VII/VI` respectively (when thresholds degenerate, `8` vs `5` etc. are chosen by `where(sup, sup_id, inf_id)`) (`couinaud.py:782-793`).
* **Right trunk** (`right`): anterior/posterior from nearest labelled sub-sector node (`cKDTree` on `right_ant ∪ right_post`) or from `RHV` plane if sub-sectors absent; superior/inferior from `transverse.sd > 0` → 4-way assignment `where(is_ant, where(sup,8,5), where(sup,7,6))` (`couinaud.py:796-815`).

Nodes within `drop_trunk_mm = 12 mm` of `bif` are dropped as ambiguous trunk (`couinaud.py:722`).

---

## 13. Voxel Assignment — `_assign_voxels` (`couinaud.py:868-940`)

Let `Liver = {vox : liver[vox]=1}`, `Seeds = {(s_i, label_i)}`.

### 13.1 Voronoi (default, `mode="voronoi"`)

```
label[vox] = label_{ argmin_i ||pts[vox] - s_i||_2 }    via cKDTree  (couinaud.py:889-892)
```

Pure portal-territory answer. Boundaries are Voronoi cell interfaces → curved. Documents note the weakness: a sector traced further peripherally claims more parenchyma (`couinaud.py:873-877`, `run_couinaud.py:42-53`).

### 13.2 Hybrid (`mode="hybrid"`)

Sector boundaries from hepatic vein planes; superior/inferior division inside each sector still from portal seeds (bias cancels within-sector):

```
right    = (pts·n_mhv > 0)   else portal seeds vote   (couinaud.py:915)
anterior = (pts·n_rhv > 0)   else portal seeds vote   (couinaud.py:916)
medial   = (pts·n_umbilical > 0)                       (couinaud.py:917)

Regions:
  right &  anterior → {5,8},   right & ¬anterior → {6,7}
  ¬right &  medial  → {4},     ¬right & ¬medial  → {2,3}
Within each region: nearest seed among that region's labels (couinaud.py:926-937)
```

`run_couinaud.py:99-118` always computes both modes; `mode_agreement = |{vox: label_vor==label_hyb}|/|Liver|` and per-segment Dice are reported.

### 13.3 Propagation rules — `growth.py`

The mode above decides *which* seeds compete for a voxel. Orthogonally, `Config.assign` and two weights decide *how*. All default to off, so `Config()` reproduces 13.1 exactly. General form:

```
label[vox] = argmin_L  d_L(pts[vox]) / w_L

d_L(p) = min over paths γ from any seed of label L to p, γ ⊂ Liver, of ∫_γ c(x)|dx|
```

`c ≡ 1`, `w ≡ 1`, paths unconstrained  ⇒  13.1.

Implementation: one `skimage.graph.MCP_Geometric` pass per label over the liver's bounding box, then `argmin` of the eight cost fields. Weights are a post-hoc division (`argmin` is invariant to a global scale on `w`), and `sampling=zooms` keeps voxel anisotropy honest.

**Geodesic** (`assign="geodesic"`) — paths confined to the liver. A straight line can leave the organ and re-enter across the gallbladder fossa or the IVC groove, letting a seed claim parenchyma no branch of its pedicle could reach. Side effect that is really the main effect: a monotone front inside a connected mask **cannot** leave a basin in two pieces, so "every segment is one connected blob" becomes a theorem.

**Murray flow weighting** (`flow_alpha=α`):

```
w_L = r_{P(L)}^α        P(L) = pedicle feeding segment L
                        r_P  = 90th percentile of point radii along P
```

The exponent is linear, *not* cubic, and this is the easy mistake. A weight divides distance, so it scales linear extent and claimed **volume follows its cube** (measured: `w=2` → 8.1× volume). Murray asks for `V ∝ r³`, so `w ∝ r`, i.e. `α=1`. Setting `w = r³` delivers `V ∝ r⁹`. Segments fed by the same pedicle share a weight — their boundary is a cut inside one pedicle's graph where the bias already cancels. Weights normalised to geometric mean 1; radii `< 1.5 mm` refused as sitting on the voxel quantisation floor; `|log w|` clipped by `flow_log_clip`.

**Vein barrier** (`vein_barrier=s`, `vein_barrier_mm=L`):

```
c(x) = 1 + s · max(0, 1 - d_HV(x)/L)      d_HV = distance to the retained vein masks
```

A ramp, not a wall — a hard barrier is a plane by another name and shatters where a vein label is fragmentary. Boundaries are *pulled* onto a vein where one exists and left portal-decided where none does.

**This one spends evidence.** Once a vein enters `c`, its Dice check is circular. `Result.veins_used()` records which veins were consumed and `checks.py` **refuses** those comparisons rather than reporting a number that is high by construction. `vein_barrier_holdout` keeps one vein out of `c` so its plane stays independent.

Measured outcomes are in `README.md` and §19 below; `ablate.py` regenerates them.

---

## 14. Segment I (Caudate) — `_caudate` (`couinaud.py:822-865`)

No portal branch exists for segment I in this schema, so it is a **geometric heuristic** (`Result.caudate_is_heuristic = True`):

```
Candidates = { vox∈Liver :  dist(vox, IVC) ≤ 20 mm
                          ∧ (vox-bif)·Â < 0              posterior to bifurcation
                          ∧ (vox-bif)·Ŝ > -15 mm         not far below bif
                          ∧ (vox - ivc_nearest)·R̂ ≤ 12 mm  not far right of IVC
                          ∧ label[vox] ∈ {2,4,7,8} }      only neighbours, never V/VI
Keep largest connected component (ndimage.label), relabel to 1
```

Flagged as the least trustworthy step (`couinaud.py:828-831`).

---

## 15. Smoothing — `_smooth` (`couinaud.py:943-952`)

Optional `smooth_iterations` majority-vote in a `3×3×3` neighbourhood:

```
votes_s = uniform_filter( [label==s], size=3 ) * 27
new_label = argmax_s votes_s   where liver==1
```

Removes Voronoi speckle on boundaries.

---

## 16. Validation — `checks.py`

No segment-level ground truth exists. Checks catch definitively wrong results (`checks.py:1-21`):

| Category | Test | Mathematics |
|---|---|---|
| **Structural** | Partition exactly | `unlabelled==0 ∧ outside==0` (`checks.py:84-87`) |
| | All 8 present | `∀s: ∃vox label==s` (`checks.py:89-91`) |
| | One blob per segment | `max_cc_size / total ≥ 0.85` (`checks.py:93-98`, `54-61`) |
| **Anatomical** | II above III, VIII>V, VII>VI | `(centroid(sup)-centroid(inf))·Ŝ > 0` (`checks.py:101-109`) |
| | Anterior sector anterior | `(mean(V,VIII)-mean(VI,VII))·Â > 0` (`checks.py:115-119`) |
| | Left left of right | `(mean(V-VIII)-mean(II-IV))·R̂ > 0` (`checks.py:121-125`) |
| **Independent** | Portal left/right vs MHV plane | `Dice( {vox:label∈{2,3,4}}, {vox: sd_mhv<0} ) ≥ 0.70` (`checks.py:128-143`) |
| | Anterior/posterior vs RHV | `Dice( {V,VIII}, {sd_rhv>0}∩RightLiver ) ≥ 0.70` (`checks.py:145-157`) |
| | Segment IV vs LHV | `Dice( {IV}, {sd_lhv>0}∩LeftLiver ) ≥ 0.70` (`checks.py:159-171`) |
| **Volumetric** | Leelaudomlipi 2002 | `pct[s] ∈ mean±2SD` for ≥6/8 segments (`checks.py:174-180`, `couinaud.py:98-108`) |
| | Left hemiliver share | `Σ_{2,3,4} pct ∈ [25,45]%` (`checks.py:182-185`) |
| | Frame confidence | `confidence ≥ 0.62` (`checks.py:187-189`) |

Cross-validation is always against the **Voronoi** labelling (`checks.py:64-72`, `run_couinaud.py:113-118`): in hybrid mode sector boundaries *are* the vein planes, so Dice would be 1.0 by construction and measure nothing.

`Dice(A,B) = 2|A∩B|/(|A|+|B|)` (`checks.py:48-51`).

A batch-level diagnostic `coverage_bias` (`run_couinaud.py:42-84`) correlates `log(post/ant seed ratio)` vs `log(post/ant volume ratio)` across cases; strong correlation evidences that Voronoi volumes are driven by label completeness rather than anatomy.

---

## 17. Pipeline Order — `segment_liver` (`couinaud.py:955-1038`)

```
load_case → largest liver CC → pedicle skeletons → to_mm
→ bifurcation → trim left pedicle → transverse plane → IVC axis
→ vein planes → umbilical plane → label seeds → assign voxels
→ caudate heuristic → optional smoothing → package Result
```

`Config` (`couinaud.py:140-166`) holds all thresholds in mm/degrees/counts; `Result` (`couinaud.py:444-470`) carries `labels`, `seeds_mm`, `planes`, `bifurcation_mm`, `notes`, and `volumes_ml/pct`.

---

## 18. Known Limits (README.md)

* Segment I is heuristic, not a portal territory.
* IVa/IVb not separated (IV reported whole).
* Voronoi inherits label-completeness bias — and §19 shows it cannot be corrected at the propagation stage.
* Hepatic vein masks are tributary fans of variable completeness; plane fits are guarded and may be rejected.
* Boundaries are voxel-accurate, unsmoothed by default.
* **No ground truth.** Every number here is a consistency or plausibility measure, not accuracy.

---

## 19. Prior work, and what is actually new

The method is not new and the document should not imply otherwise.

**Selle, Preim, Schenk & Peitgen, "Analysis of vasculature for liver surgical planning," IEEE TMI 21(11):1344–1357, 2002** — in `sources/` — contains:

| here | Selle 2002 |
|---|---|
| `assign="euclidean"` (§13.1) | **NNSA**, §III-C-2. Identical. |
| `assign="geodesic"` (§13.3) | **LASA**, §III-C-3: solve `∇²φ_i = 0` with `φ_i = 1` on branch `B_i`, `φ_i = 0` on other branches **and outside the liver**; assign `argmax φ_i`. The exterior boundary condition already confines the field to the organ — and more completely than a geodesic does, since the potential depends on the whole configuration, not one shortest path. |
| `vein_barrier` (§13.3) | Eq. (10): hepatic vein as a **sink**, `φ_i = 0` on it. Proposed *and validated*. |
| `flow_alpha` (§13.3) | Eq. (11): branch boundary condition scaled by **local radius**, "the influence of smaller branches is reduced in comparison to the larger branches". Finer than ours — a radius per position, where we use one per pedicle. |

Their validation, which we cannot match: eight human livers cast in resin, corroded, CT-scanned at 1 mm, giving portal trees of 10–18 m accumulated length against 1–1.5 m *in vivo*. Trees then **systematically pruned** to three levels to simulate clinical incompleteness, scored by volume overlap against cast-derived truth:

```
pruning level        A            B            C
NNSA          79.0% (3.6)  89.9% (2.9)  93.4% (2.9)
LASA          77.7% (3.5)  88.6% (2.6)  91.7% (2.9)
Ext. LASA     81.7% (3.5)  90.8% (2.8)  92.7% (2.9)
```

Level A is "the main branches of the Couinaud subtrees", radiologist-selected — roughly where this dataset's sector masks sit, possibly below. Note the vein sink gains 2.7 points at A and **loses** 0.7 at C: the benefit decays as the tree improves, which is exactly why our barrier arm is inert.

### What survives as ours

1. **Hepatic veins held out as a test rather than used as an input.** Selle puts the vein *into* the model (Eq. 10); we fit its plane from labels the assignment never sees and ask whether portal anatomy *predicts* venous anatomy — with `veins_used()` / `checks.py` refusing the comparison once that evidence is spent. An improvement in **applicability**, not rigour: a corrosion cast beats all of it.
2. **The coverage bias as a number** (ρ = 0.79, within-case control). Eq. (11) exists *because* the sensitivity was known; they reported qualitative robustness, never a figure.
3. **`best_axis_cut`** (§11). Not in Selle — but only because a connected portal tree plus a radiologist made it unnecessary. It reconstructs structure this dataset lacks; it is not a better idea than having the structure.
4. **The measured failure of radius weighting on accuracy.** Selle claims Eq. (11) improves *robustness*, never accuracy, so this fills a gap rather than contradicting them.

### The gap the comparison exposes

Selle turned completeness into a **controlled variable** by pruning complete trees against known truth. Our ρ compares two pedicles within a case: it detects the bias but cannot calibrate it. That design needs complete trees, which we do not have — a limitation of the data, not a choice.

**Defensible framing:** not a new segmentation method, but *a known vascular-territory method deployed on data lacking the tree structure it assumes, with an audit protocol for the no-ground-truth case*.

