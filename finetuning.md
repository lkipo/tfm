# Training a network to produce the labels this pipeline needs

The Couinaud pipeline in this directory consumes per-class binary masks. Its two
dominant weaknesses are both *label* problems, not algorithm problems: 10 of 38 cases
lack a label the pipeline requires, and where the labels exist they are traced to
inconsistent depth, which biases every volume it reports (ρ = 0.79). A network that
fixes those two things is worth more here than any change to the segmentation method.

Every number below was measured on the actual masks, not estimated.

---

## 1. The classes to train

Twelve.

| # | class | present | why it is in the list |
|---|---|---|---|
| 1 | `classLiver` | 38/38 | mandatory — the domain to partition; also conditions the portal head |
| 2 | `classPediculoPortalIzquierdo` | 28/38 | **mandatory** — → II, III, IV |
| 3 | `classPediculoPortalAnteriorDerecho` | 28/38 | → V, VIII |
| 4 | `classPediculoPortalPosteriorDerecho` | 28/38 | → VI, VII |
| 5 | `classPediculoPortalDerecho` | 23/38 | right trunk; fallback when 3 and 4 are absent |
| 6 | `classVenaPorta` | 38/38 | extrahepatic trunk — locates the bifurcation |
| 7 | `classIVC` | 38/38 | segment I, and all three scissura fits |
| 8 | `classVenaHepaticaMedia` | 28/38 | cross-validation only |
| 9 | `classVenaHepaticaDerecha` | 28/38 | cross-validation only |
| 10 | `classVenaHepaticaIzquierda` | 27/38 | cross-validation only |
| 11 | `classAorta` | 38/38 | frame recovery |
| 12 | `classGallbladder` | 38/38 | frame recovery |

**Classes 2–5 are the job.** They *are* the Couinaud sectors — that is why this
pipeline works at all. Everything else is either easy (1, 7, 11, 12), a landmark (6),
or a held-out test (8–10). Without 3 and 4 you get six segments, not eight.

### Not classes

| class | present | what to do instead |
|---|---|---|
| `classArteriaHepatica` | 18/38 | median 440 voxels. Too sparse to learn, but the hepatic artery runs inside the same pedicle sheath as the portal vein, so it is the nearest thing to a hard negative you have. **Mask out; do not let it count as background** on the 20 cases where it is unlabelled. |
| `classViaBiliar` | 6/38 | median 210 voxels. Same reasoning, more so. |
| `classBloodVessels` | 38/38 | an *intrahepatic vessel union* — 84.5% inside the liver, 0% of aorta, artery or duct — but **partial**, covering only 51–76% of each pedicle and 17% of `VenaPorta`. Auxiliary or pretraining target for a binary vesselness head; not ground truth. |
| `classTumour` | 12/38 | unused by the pipeline. |

### Sigmoid, not softmax

The classes overlap. Pairwise overlap as % of the smaller class (median/max, 12 cases):

```
           Porta   Ped-R   Ped-L  Ped-RA  Ped-RP     MHV     RHV     LHV     IVC
Porta          -    5/28     0/4       .     0/2     0/5       .     0/3     0/0
Ped-R       5/28       -    0/10    6/24    4/21       .       .       .     0/2
Ped-L        0/4    0/10       -    0/20     0/1       .       .     0/0       .
Ped-RA         .    6/24    0/20       -    1/10     0/0       .       .       .
Ped-RP       0/2    4/21     0/1    1/10       -       .     0/0       .       .
RHV            .       .       .       .     0/0     0/1       -       .    4/29
```

Medians near zero, **maxima of 20–29%**, and every one sits at a junction: the portal
bifurcation and the caval confluence — the two places this pipeline is most
sensitive. A softmax forces a decision the annotation never made, exactly there.

### It is a tree-partitioning problem, not an appearance problem

Classes 2–6 are **one connected vessel**: same contrast, same texture, same calibre
progression. `Izquierdo` vs `AnteriorDerecho` is not a visual distinction — it is a
question about which side of the bifurcation a branch connects back to. The class
boundary is a **cut point on a tree**, and a 64³ patch cannot see the bifurcation.

So use **two heads**:

1. **Binary vesselness** — portal tree, hepatic veins, IVC. Local, well-posed, ample
   signal; this is where a pretrained vessel network's weights actually transfer.
2. **Tree partitioning** — skeletonise the portal prediction, locate the bifurcation,
   label subtrees by connectivity. `skeleton_graph()`, `largest_radius_walk()` and
   `best_axis_cut()` in `couinaud.py` already do this and are covered by tests.

One usable exception to "appearance carries nothing": the **fraction of a branch
lying inside the liver** separates three of the five portal classes geometrically —
`VenaPorta` 0.8%, `PediculoPortalDerecho` 25%, the three working pedicles 95–98%.
Feed the network the liver mask and it has real evidence for a distinction it
otherwise cannot see.

A direct 5-class portal head is worth building only as a baseline, and only if it is
conditioned on the liver mask and given a receptive field that reaches the hilum.
Patch-based multi-class prediction produces laterally inconsistent labels that look
plausible slice-by-slice and break the graph cut.

---

## 2. What the pipeline actually requires

Measured by deleting each mask and re-running, not by reading the gatekeeper.

**Hard failures — only these three:**

| label | without it |
|---|---|
| `classLiver` | nothing to partition |
| `classPediculoPortalIzquierdo` | no left liver, no umbilical fissure |
| any right pedicle (`Derecho`, **or** `AnteriorDerecho`+`PosteriorDerecho`) | no right liver |

**Everything else degrades gracefully:**

| dropped | result |
|---|---|
| `AnteriorDerecho` + `PosteriorDerecho` (trunk kept) | **6/8 segments** — these two *are* the right sectors |
| `classIVC` | **7/8 segments** — segment I lost, and all three vein planes |
| all three hepatic veins | 8/8 segments, **no cross-validation** |
| `classVenaPorta` | 8/8 segments — only snaps the bifurcation to the trunk centreline |
| `classAorta`, `classGallbladder` | 8/8 segments — frame confidence drops |

Case counts under each requirement:

```
runs at all (minimal set)          28/38
can yield all 8 segments           27/38
passes missing_labels() today      27/38
```

**A known false negative.** `missing_labels()` demands all three hepatic veins, which
is stricter than the pipeline is. `RMV2025_0005_CT_UTV_VEP_L` has complete portal
anatomy and would give eight segments, but is refused for having no hepatic veins.
One case today — relevant because a network that predicts portal well and veins
poorly would hit this gate unnecessarily.

---

## 3. What "good" means here — it is not Dice

The measured weakness is that a pedicle traced further into the periphery claims more
parenchyma whether or not it should (ρ = 0.79). **A network trained on Dice against
these annotations will reproduce the annotators' inconsistency**, because that is
precisely what it is being asked to imitate. Plain Dice also converges happily to a
solution that captures the proximal trunk and drops every distal branch — the same
failure by a different route.

Optimise and evaluate for these instead:

1. **Centreline recall / clDice.** A centreline-aware loss (clDice, or a soft skeleton
   term added to Dice). Overlap losses under-weight thin distal branches by
   construction; clDice weights topology instead of volume. Highest-leverage single
   choice in this document.
2. **Traced depth per pedicle.** Skeletonise the prediction and measure branch
   generations or accumulated centreline length. A prediction reaching *further* than
   the annotation is better for this pipeline and will score *worse* on Dice. Do not
   let Dice veto it.
3. **Inter-pedicle depth consistency — the metric that matters most.** Per case, the
   ratio of skeleton size between the right posterior and right anterior pedicles. In
   the current annotations that ratio has median 1.11 and ranges **0.44 to 3.61**
   across 23 cases: one pedicle traced three and a half times further than its
   neighbour in the same liver. **A model closer to 1.0 across the cohort has attacked
   the pipeline's dominant error, even at lower Dice.** Measure it the way
   `coverage_bias()` in `run_couinaud.py` does, so the numbers are comparable.
4. **Connectivity.** Fraction of predicted pedicle voxels in one component reaching
   the bifurcation. Fragmented predictions break the graph cut.
5. **Case usability.** Cases where all required classes are non-empty. Baseline 28/38.

### The end-to-end acceptance test

The real evaluation already exists. Write predictions in the same `class*.nii.gz`
layout, point the pipeline at them, compare:

```bash
python run_couinaud.py --all --no-figures
python ablate.py --arms base geo
```

| metric | annotation baseline | goal |
|---|---|---|
| cases that run | 28/38 | ≥ 34 |
| cases yielding 8 segments | 27/38 | ≥ 34 |
| ρ (coverage bias) | 0.79 | < 0.5 |
| sector ratio (VI+VII)/(V+VIII) | 1.24 | → 0.83 |
| all eight segments present | 21/26 | ≥ 24 |
| Dice vs MHV plane | 0.892 | ≥ 0.85, do not regress |

**One caveat on the last row.** If a single multi-task network predicts both the
portal and the venous classes, their errors are coupled and the vein cross-validation
weakens. `checks.py` cannot detect this — its guard knows about the vein *barrier*,
not about shared network weights. If you train them together, say so when reporting
those Dice numbers.

---

## 4. Traps

**Never flip left–right.** The classes are laterality-specific (`Izquierdo`/`Derecho`,
`VenaHepaticaDerecha`/`Izquierda`). A mirror teaches the wrong laterality — and
because the pipeline recovers its anatomical frame *from these very labels*, a lateral
error propagates into the coordinate system and corrupts every downstream decision.
A–P and S–I flips are forbidden for the same reason. On nnU-Net this means
`-tr nnUNetTrainerNoMirroring` or your version's equivalent; it is not optional.
**Safe:** rotations ≤ 15°, elastic deformation, scaling, intensity/noise/bias-field,
gamma.

**Class imbalance is the whole difficulty.** A pedicle is 4–6.5k voxels in a ~1.5M
voxel liver: **0.3% of the liver, ~0.03% of the volume.** Sample patches centred on
foreground at ≥ 1/3 positive ratio. Report per-class metrics only — a mean over
classes is dominated by `classLiver`.

**Write a correct affine on export.** Every mask in this dataset declares
`axcodes (R,A,S)` and almost none of them are; `anatomy_frame.py` exists solely to
work around it. Do not recreate that bug in your predictions.

**Do not resample.** Every case is already 1.0 × 1.0 × 1.0 mm. Resampling gains
nothing and blurs exactly the thin distal branches you are trying to recover.

**Split by patient, not by case.** Check for repeated patients across `RMV2025_*` and
`RMV2026_*` before splitting. With n=38 a leak is easy and would invalidate
everything.

**n=38 is small.** 5-fold cross-validation, not a single held-out split; report
variance. Consider pretraining on `datagen_pipeline/hepatic_dvn_dataset` — 10
synthetic samples in DeepVesselNet's 7-channel format from Whitehead-style CCO trees.
Binary-vessel only, so it pretrains the vesselness backbone and nothing else. Generate
more via `datagen_pipeline/hepatic_dvn_datagen_guide.md`.

**Cases the pipeline refuses are informative.** `RMV2025_0009_CT_UTV_VEP_D` has no
left portal pedicle at all. If the network produces one, verify against the image
before celebrating.

---

## 5. Recipe

Nothing exotic; the judgement is in §1–4.

- **Framework.** nnU-Net v2 is the pragmatic default — patch sampling, foreground
  oversampling, augmentation and 5-fold CV out of the box, and `3d_cascade_fullres`
  gives the receptive field a direct multi-class head would need. Disable mirroring.
- **Alternative.** `deepvesselnet/` in this repo (Keras/TF: `FCN` with cross-hair
  convolutions, plus `unet.py`, `vnet.py`) is small and built for whole-volume vessel
  segmentation, which suits head 1. It is not suited to semantic branch labelling.
- **Loss.** `Dice + clDice` on the vessel classes; add Tversky (β > α) if distal
  recall is still short. Cross-entropy alone will not work at 0.03% prevalence.
- **Patches.** 128³ or larger — receptive field is the binding constraint.
- **Fine-tuning.** Replace the output head, freeze the encoder ~10 epochs, unfreeze at
  1/10 the learning rate. At n=38 the encoder is what you cannot afford to retrain.

---

## 6. What to copy to the datacenter

### Essential

| what | size | why |
|---|---|---|
| `continuity/data/0_test_nifti/` | 58 MB | the 38 cases of labels |
| the paired image volumes | — | not in this repo |
| `continuity/couinaud/` | 6.3 MB | the pipeline, for the §3 acceptance test |
| `continuity/reconstruction.py` | 108 KB | `load_thinning`, `estimate_point_radii_from_mask` — imported by `couinaud.py`, which will not run without it |

### Recommended

| what | size | why |
|---|---|---|
| `datagen_pipeline/` | 40 MB | 10 synthetic pretraining samples + generators |
| `deepvesselnet/` | 256 KB | reference implementation |
| `sources/Analysis_of_vasculature_for_liver_surgical_planning.pdf` | 1 MB | Selle et al. 2002, the method being audited |

### Do not copy

`continuity/couinaud/out/` (regenerated), `.venv` directories (rebuild against the
cluster's CUDA stack), `figures/`, and the unrelated subtrees (`bezier-skeletons`,
`zenodo_skeletons`, `viewer-samples`, `graph-learning`).

### Commands

```bash
cd /Users/lois/removirt

# note: on BSD/macOS tar the --exclude flags must come *before* the paths
tar czf couinaud_train.tar.gz \
    --exclude='out' --exclude='__pycache__' --exclude='.venv' \
    continuity/data/0_test_nifti \
    continuity/couinaud \
    continuity/reconstruction.py \
    datagen_pipeline \
    deepvesselnet \
    sources/Analysis_of_vasculature_for_liver_surgical_planning.pdf

# 53 MB, 833 files (verified). The image volumes are not in it.
scp couinaud_train.tar.gz user@datacenter:/path/
```

Verify on arrival, before queuing anything:

```bash
pip install nibabel numpy scipy scikit-image networkx
python continuity/couinaud/test_couinaud.py --quick     # 50 checks, no data needed
python continuity/couinaud/run_couinaud.py --list       # expect 27/38
```

27 have every label `missing_labels()` demands; 26 complete (`RMV2025_0009` is refused
at runtime for a left pedicle too thin to skeletonise); 28 would run under the true
minimal requirement of §2. All three are expected — if `--list` says anything other
than 27, the data did not copy correctly.
