# Couinaud segments from the skeletonized portal graph

Can the liver's eight Couinaud segments be recovered from the vascular skeletons in
`data/0_test_nifti`? This directory is the experiment that answers it.

The premise is that **Couinaud segments are portal territories**. So rather than
slicing the parenchyma with planes, the pipeline labels the *portal skeleton's
nodes* with a segment number and lets a nearest-seed assignment carry those labels
out to every liver voxel. Boundaries then come out as curved territory boundaries
instead of flat cuts, and the geometry is only ever used to decide which segment a
skeleton node belongs to.

**That approach is not new** — it is Selle et al.'s NNSA (2002), and two of the
three propagation variants here appear in that same paper. See
[Prior work](#prior-work-and-what-is-actually-new) before claiming novelty for any
of it. What this directory adds is an *audit*: hepatic veins held out as a test
rather than consumed as an input, and the method's dependence on label completeness
measured rather than described.

```
python run_couinaud.py --list                    # which cases can run, and why not
python run_couinaud.py RMV2025_0001_CT_NCT_HBP_X # one case -> out/<case>/*.png
python run_couinaud.py --all                     # every usable case + batch summary
python run_couinaud.py --all --nifti             # also write label volumes
python ablate.py                                 # compare the propagation rules
python test_couinaud.py                          # 70 unit + real-data checks
python export_latex.py                           # regenerate ../../couinaud_results.tex
```

Run with `vascular-lab/.venv` or `graph-learning/.venv` (both carry nibabel,
scikit-image, networkx, scipy). `reconstruction.load_thinning` and
`estimate_point_radii_from_mask` are imported from `continuity/`, so nothing is
duplicated.

| file | role |
|---|---|
| `couinaud.py` | the pipeline: geometry primitives, seed labelling, assignment |
| `growth.py` | the three propagation rules (geodesic, flow weighting, vein barrier) |
| `anatomy_frame.py` | recovers R/A/S from the anatomy, because the affines are wrong |
| `checks.py` | 12 sanity checks per case, including the spent-evidence guard |
| `ablate.py` | runs the propagation rules across the cohort and tabulates them |
| `viz.py` | seven figure generators |
| `export_latex.py` | emits `couinaud_results.tex` so the write-up cannot drift |
| `explanation.md` | the mathematics, step by step, with `file:line` references |
| `FINETUNING.md` | training a network to produce these labels, and what to copy to a cluster |

## What the labels give us

| label | role |
|---|---|
| `classLiver` | the parenchyma to partition |
| `classPediculoPortalIzquierdo` | left portal pedicle → II, III, IV |
| `classPediculoPortalDerecho` | right portal trunk → V–VIII (parent) |
| `classPediculoPortalAnteriorDerecho` | right anterior sector → V, VIII |
| `classPediculoPortalPosteriorDerecho` | right posterior sector → VI, VII |
| `classVenaPorta` | main trunk; locates the bifurcation |
| `classVenaHepatica{Media,Derecha,Izquierda}` | the three scissura planes |
| `classIVC` | anchors the vein planes and segment I |
| `classAorta`, `classGallbladder` | only used to recover the anatomical frame |

The four `PediculoPortal*` classes are already the Couinaud *sectors*, which is why
this works at all: sector membership is read off the label, and only the
superior/inferior division inside each sector has to be derived.

26 of the 38 cases in the legacy `continuity/data/0_test_nifti` carry every label the
pipeline needs (`--list` prints the rest). After the `couinaud_nifti` rebuild
(`continuity/couinaud/tools/build_couinaud_case.py` onto `data/0_test_nifti/imagesTs`
grid), **29/38 pass QA**, of which **28 are CT** (`RMV2025_0002_MR` excluded, see
`FINETUNING.md` Sec.0/4) and **22/28 have all 4 stage-1 channels** (`liver`+`portal_tree`+
`hepatic_veins`+`ivc`), **7/28 miss `hepatic_veins`** but are still usable via masked
loss (`LightningMedSeg3D/src/lightning_medseg3d/losses/__init__.py:20`,
`segmentation_datamodule.py:97`). For training, `28` CT cases after QA are the
effective cohort (legacy `26/38` number is deprecated).

## The pipeline

1. **Recover the anatomical frame** (`anatomy_frame.py`) — see the warning below.
2. **Skeletonize** each portal pedicle with `reconstruction.load_thinning`, cropped
   to its bounding box.
3. **Locate the portal bifurcation** as the midpoint of the closest left/right
   pedicle pair, snapped onto the `classVenaPorta` centreline.
4. **Trim the left pedicle.** `classPediculoPortalIzquierdo` overlaps the main
   portal trunk near the bifurcation — by 65 mm in `RMV2026_0022`. Nodes more than
   15 mm right of the bifurcation are dropped as trunk overlap.
5. **Find the umbilical fissure.** Walk the left pedicle's graph always taking the
   widest neighbour (Murray's law makes the parent the widest branch at every
   bifurcation) to recover the left portal vein's trunk, take the trunk's leftmost
   point as the umbilical turn, and erect a near-sagittal plane there.
6. **Split the pedicles with graph cuts.** Both remaining divisions — medial vs
   lateral in the left pedicle (IV vs II+III), and superior vs inferior inside each
   sector — come from removing the single graph edge that best separates that
   pedicle's nodes along the relevant axis. The right anterior pedicle really does
   divide into a segment-V branch and a segment-VIII branch, so cutting the tree at
   that division finds the boundary wherever it sits rather than assuming it lies
   on a plane. The corresponding plane (umbilical or transverse) is the *second*
   choice, and the median a last resort: a plane threshold alone puts every node on
   one side in roughly 40% of attempts on this data, which silently empties a
   segment.
7. **Fit the scissura planes** (used for cross-validation and for the hybrid mode).
   Each is constrained to *contain* the retrohepatic IVC axis — all three hepatic
   veins converge there — leaving one degree of freedom that the vein's proximal
   trunk resolves.
8. **Propagate** the seed labels to the parenchyma (see the two modes below).
9. **Segment I** by an explicit geometric heuristic, flagged as such: parenchyma
   hugging the IVC, posterior to the bifurcation, largest component only. The
   caudate has no dedicated portal branch in this label schema, so it cannot come
   out of the graph.

### Two assignment modes

- **`voronoi`** (default) — pure portal territory: each liver voxel takes the
  segment of its nearest labelled skeleton node. Its weakness is that it inherits
  the *labelling completeness* of the pedicle masks: a sector traced further into
  the periphery claims more parenchyma whether or not it should.
- **`hybrid`** — takes the sector boundaries from the hepatic vein planes and lets
  the portal graph decide only the superior/inferior division inside each sector,
  where both candidate labels come from the same pedicle and the bias cancels.

Both are always computed, because only the Voronoi labelling can be compared
against the vein planes without circularity, and because the agreement between the
two is a useful stability signal in itself. See `out/batch_summary.json` for which
one wins on this data.

### Three propagation rules (`growth.py`)

The mode above decides *which seeds* compete for a voxel. Orthogonal to it,
`Config.assign` and two weights decide *how they compete*. All three default to
off, so `Config()` reproduces the original nearest-seed result exactly.

| option | what it changes | why |
|---|---|---|
| `assign="geodesic"` | the front may only travel through the liver | A straight line can leave the organ and re-enter across the gallbladder fossa or the IVC groove, letting a seed claim parenchyma no branch of its pedicle could reach. It also makes every territory connected *by construction* — a monotone front inside a connected mask cannot leave a basin in two pieces. |
| `flow_alpha=a` | territory *L* advances at rate `r_L^a` on its pedicle's root radius | The coverage bias. A pedicle traced further into the periphery has more seeds and wins more parenchyma under any nearest-seed rule — but its **radius** does not grow because the segmentation followed it further. Across 23 cases the seed-count ratio correlates with the produced volume ratio at **0.79** and with the Murray ratio at only **0.19**, so the radius is close to uncontaminated evidence. |
| `vein_barrier=s` | crossing a hepatic vein costs `1+s` | Couinaud's scissurae are not planes. This is the only option that can put a *curved* boundary on the actual vein. |

**The weight is linear in the radius, not cubic.** A weight divides distance, so it
scales a territory's *linear* extent and the claimed volume follows its **cube**
(measured: `w=2` claims 8.1× the volume). Murray wants volume ∝ `r³`, so the weight
that delivers it is `w ∝ r` — i.e. `flow_alpha=1`. Setting `w = r³` would deliver
volume ∝ `r⁹`. That was the parameterisation in the first version of this module,
and the cohort ablation caught it: ρ fell as intended while the sector ratio drifted
*away* from published volumetry, which is the signature of an over-applied
correction rather than a wrong one. `test_flow_weighted_growth` now pins the cubic
response so it cannot regress silently.

`flow_alpha` is a shrinkage knob, not a switch. Cubing a radius triples its relative
error — ±0.5 mm on a 3 mm radius is a factor 1.6 in flow — and on 1 mm data 46
pedicles take only 19 distinct radius values, two of them at the quantisation floor.
Radii below 1.5 mm are refused outright and `flow_log_clip` bounds the rest.

**The barrier spends evidence.** Putting the vein masks into the growth cost makes
the Dice-versus-vein-plane checks circular, which is why `Result.veins_used()`
exists and `checks.py` *refuses* those comparisons instead of reporting a number
that is high by construction. Use `vein_barrier_holdout` to keep one vein out of
the cost so its plane survives as independent evidence.

```
python ablate.py                                    # measure all of it on the cohort
python ablate.py --arms base geo --cases 6          # a quick subset while iterating
python ablate.py --figure RMV2025_0011_CT_UTV_VEP_S # see where the boundaries moved
python run_couinaud.py CASE --assign geodesic --flow-alpha 1.0
```

### What the ablation found (26 cases)

```
arm          sector err   ratio    rho  DiceRHV  DiceMHV  DiceLHV  8 segs  1 blob   band
base              0.508    1.24   0.79    0.765    0.892    0.720   21/26   22/26  10/26
geo               0.529    1.21   0.78    0.747    0.888    0.722   20/26   24/26  12/26
geo+flow50        0.535    1.26   0.75    0.745    0.863    0.721   20/26   25/26  10/26
geo+flow100       0.561    1.31   0.71    0.709    0.863    0.724   20/26   25/26  10/26
geo+flow150       0.555    1.36   0.65    0.707    0.845    0.733   20/26   25/26   9/26
geo+vein2         0.526    1.22   0.78    0.741    spent    spent   20/26   24/26  11/26
all               0.528    1.27   0.76    0.733    spent    spent   20/26   25/26   9/26
```

`sector err` is the median |log| distance of (VI+VII)/(V+VIII) from the published
0.83; `rho` is the coverage bias. All three options remain **off by default**, and
this table is why.

**The geodesic metric earns its keep structurally, not numerically.** Connectivity
22/26 → 24/26 and volumetry 10/26 → 12/26, while Dice and the sector ratio move by
less than the spread between cases. It buys correctness by construction — a monotone
front inside a connected mask cannot leave a basin in two pieces — rather than
accuracy. That is the one worth turning on.

**The flow weighting works as a mechanism and fails as a fix.** ρ falls monotonically
with α (0.79 → 0.75 → 0.71 → 0.65), so the weighting really does decouple claimed
volume from seed count, exactly as designed. But the sector ratio moves *away* from
published volumetry at every α, including α=1 where the weighting is Murray exactly.

The reason is that **the coverage bias is positional, not a matter of rate.** At α=1
the per-case volume ratios still correlate with the unweighted ones at 0.91, and ρ
only drops to 0.71. A pedicle traced deep into the periphery has seeds sitting near
the far capsule; no weighting of the *speed* of a front can take that territory back,
because the seed is already there. The radius is clean evidence — it correlates with
trace completeness at only 0.19 — but injecting it at the propagation stage is too
late. Correcting this properly means pruning or reweighting the seeds themselves, or
matching each pedicle's traced extent before propagation.

**The vein barrier is close to inert**, moving the sector ratio 0.508 → 0.526 while
spending two of the three independent checks to do it. On individual cases it visibly
disrupts the II/III split where the LHV crosses it.

## ⚠ The NIfTI affines in this dataset are wrong (legacy) / fixed in `couinaud_nifti`

`continuity/glb_to_nifti.py` writes `affine = np.eye(4)` scaled by the voxel pitch,
with no rotation, so **every case in `continuity/data/0_test_nifti` reports axcodes
`('R','A','S')` regardless of the source GLB's actual orientation**. Measuring the
anatomy shows that claim is false: in 33 of 38 cases voxel axis 1 is the cranio-caudal
axis, not axis 2, and the sign conventions vary between cases too.

For the training dataset `data/couinaud_nifti` (built by
`continuity/couinaud/tools/build_couinaud_case.py` onto `data/0_test_nifti/imagesTs`),
the affine is **correct** - `imagesTs.affine` is copied verbatim
(`build_couinaud_case.py:399`), verified via `X_LPS=-T_R+a1` etc., and `imagesTs`
itself is `0.76-0.97×0.76-0.97×0.40-0.70mm` with proper `(-R,+A,+S)` encoding (e.g.
`RMV2025_0001 [-0.765, 0.765, 0.701]`). `anatomy_frame.py` still recovers the frame
from anatomy for robustness on legacy data and as a sanity check on `couinaud_nifti`,
but on `couinaud_nifti` its `confidence` should be high.

Anything that reasons about "superior"/"anterior"/"left" from the **legacy**
`0_test_nifti` files is reading a header that does not describe the data.
`anatomy_frame.py` recovers the frame instead, from direction vectors whose meaning
is known from the label names and from textbook relations (hepatic vein *Derecha* is
right of *Izquierda*, the right *Anterior* pedicle is anterior to the *Posterior* one,
the hepatic veins are superior to the portal pedicles, the aorta is left of the IVC,
and so on). The whole right-handed triad is fitted at once rather than axis by axis,
because a vector like "RHV is right of LHV" has a real antero-posterior component that
only a joint fit can separate out. `Frame.confidence` reports how well the winning
triad explained the evidence.

## Checks

`checks.py` cannot prove a segmentation correct — there is no segment-level ground
truth here — but it catches one that is definitely wrong:

- **structural** — the labelling partitions the liver exactly, all eight segments
  are present, each is a single connected blob
- **anatomical** — II above III, VIII above V, VII above VI, the posterior sector
  behind the anterior one, the left segments to the patient's left. A violation
  means an axis or a graph cut went the wrong way.
- **independent** — the portal-derived left/right boundary is compared against the
  MHV (Cantlie) plane, the anterior/posterior boundary against the RHV plane, and
  the segment IV boundary against the LHV plane, by Dice. These planes come from
  *different labels entirely*, so portal anatomy predicting venous anatomy is the
  closest thing to external validation available. The comparison is always run
  against the **Voronoi** labelling — in hybrid mode the sector boundaries *are*
  the vein planes, so comparing that result back to them returns 1.0 and measures
  nothing. Where that dodge is not available — a vein barrier run, where the vein
  entered the growth cost — `Result.veins_used()` reports the spent evidence and
  the check is **refused** rather than reported. A validation metric that cannot
  fail is not a validation metric.
- **volumetric** — each segment's share against mean ± 2 SD from Leelaudomlipi et
  al., *Volumetric analysis of liver segments in 155 living donors* (Liver Transpl
  2002). A wide band that only flags gross failure.

## Figures

`out/<case>/` holds six figures per case, plus a seventh from `ablate.py --figure`:

| file | what it shows |
|---|---|
| `1_orientation.png` | the recovered frame, with the landmarks that fixed each axis |
| `2_geometry.png` | skeletons, bifurcation, trunk walk, and the fitted planes |
| `3_seeds_territories.png` | the labelled portal skeleton and the parenchyma it produced |
| `4_segments.png` | axial montage plus coronal and sagittal, every region labelled |
| `5_volumes.png` | volumetry against the reference band, with the check table |
| `6_modes.png` | the two assignment modes side by side, and where they disagree |
| `7_assignment.png` | the propagation rules compared, baseline boundary overlaid |

`out/batch_summary.png` and `out/batch_summary.json` summarise a `--all` run;
`out/ablation*.json` and `out/ablation*.log` hold the propagation-rule comparison.
`export_latex.py` turns both into macros for `couinaud.tex`, so no number in the
write-up is typed by hand.

### Palette

The eight segments use the dataviz reference categorical palette. The segment → slot
mapping was chosen by maximising the worst OKLab ΔE over the pairs of segments that
are actually *spatially adjacent* in the liver — a map only needs neighbours to be
separable — rather than over slot order. The best achievable worst adjacent pair is
CVD ΔE 7.2, inside the 6–8 warn band, so secondary encoding is **mandatory** here
and is always present: direct roman-numeral labels at each region's in-slice
centroid, a legend on every figure, and a volume table repeating the numbers as
text.

## Known limits

- **Segment I is a heuristic**, not a portal territory. There is no caudate label
  and no caudate portal branch in this schema.
- **IVa/IVb are not separated.** Segment IV is reported whole; splitting it needs
  the same superior/inferior cut applied to the segment IV branches specifically.
- **Voronoi inherits label completeness bias** — see the two modes above. The
  Murray flow weighting in `growth.py` was built to attack it and does not fix it:
  the bias is in where the seeds *are*, not how fast they grow. See the ablation
  above for the measurement, and for what would have to be tried instead.
- **The hepatic vein labels are tributary fans**, not trunks, and vary a lot in
  completeness between cases. The plane fits are constrained and guarded against
  implausible orientations, and rejected planes are reported, but a case with a
  fragmentary vein label simply cannot supply that scissura.
- Boundaries are voxel-accurate but unsmoothed. `--smooth N` applies N majority-vote
  passes if a cleaner surface is wanted.
- **There is no ground truth here.** Every number in this README is a consistency or
  plausibility measure. Selle et al. scored volume overlap against eight corrosion
  casts and expect 80–90% for clinical-quality trees; nothing here is comparable to
  that figure and none of it should be quoted as if it were.
- **The coverage bias is detected but not calibrated.** ρ compares two pedicles
  within a case, which shows the bias exists and how strongly it acts but not how
  much volume error it causes. Calibrating it needs the pruning experiment below,
  which needs complete trees we do not have.

## Prior work, and what is actually new

The assignment rule is not new. `sources/Analysis_of_vasculature_for_liver_surgical_planning.pdf`
is Selle, Preim, Schenk & Peitgen, *IEEE TMI* 21(11):1344–1357, 2002, and it already
contains most of what is here:

| this directory | Selle 2002 |
|---|---|
| `assign="euclidean"` (the default) | **NNSA**, §III-C-2 — identical |
| `assign="geodesic"` | **LASA**, §III-C-3 — solves `∇²φ=0` with `φ=0` outside the liver, so the field is already confined to the organ, and by a better route than a shortest path |
| `vein_barrier` | Eq. (10) — the hepatic vein as a **sink** |
| `flow_alpha` | Eq. (11) — branch weight scaled by **local radius**, finer than our one-radius-per-pedicle version |

They also validated far better than is possible here: eight resin corrosion casts,
CT at 1 mm, portal trees of 10–18 m (vs 1–1.5 m *in vivo*), then **systematically
pruned** to three levels to simulate clinical incompleteness and scored by volume
overlap against cast-derived truth.

```
pruning level        A            B            C
NNSA          79.0% (3.6)  89.9% (2.9)  93.4% (2.9)
LASA          77.7% (3.5)  88.6% (2.6)  91.7% (2.9)
Ext. LASA     81.7% (3.5)  90.8% (2.8)  92.7% (2.9)
```

Level A is "the main branches of the Couinaud subtrees", radiologist-selected —
roughly where this dataset's sector masks sit, possibly below. The vein sink gains
2.7 points at A and **loses** 0.7 at C: its benefit decays as the tree improves,
which predicts the inert `geo+vein2` row in our ablation.

**What survives as ours**, ranked by how much weight it bears:

1. **Hepatic veins held out as a test, not used as an input.** Selle puts the vein
   *into* the model; we fit its plane from labels the assignment never sees and ask
   whether portal anatomy *predicts* venous anatomy — with `Result.veins_used()` and
   `checks.py` refusing the comparison once that evidence is spent. This is an
   improvement in **applicability**, not rigour: a corrosion cast beats all of it.
2. **The coverage bias as a number** (ρ = 0.79, within-case control). Eq. (11) exists
   *because* the sensitivity was known; the reported outcome there is qualitative
   robustness, never a figure.
3. **`best_axis_cut`.** Not in Selle — but only because a connected portal tree plus
   a radiologist made it unnecessary. It reconstructs structure this dataset lacks.
4. **The measured failure of radius weighting on accuracy.** Selle claims Eq. (11)
   improves *robustness*, never accuracy, so this fills a gap rather than
   contradicting them.

**Defensible framing:** not a new segmentation method, but *a known vascular-territory
method deployed on data that lacks the tree structure the method assumes, with an
audit protocol for the case where no ground truth exists.*
