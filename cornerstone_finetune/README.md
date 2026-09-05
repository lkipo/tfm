# Fine-tuning SwinUNETR on `data/cornerstone_masks`

Fine-tunes the BTCV SwinUNETR checkpoint
(`data/nn_model_weights/swin_unetr-epoch=483-val_loss=0.07.ckpt`) onto the liver
structures annotated in `data/cornerstone_masks`, with a multi-label sigmoid
head, checkpointing that survives anything short of disk loss, and TensorBoard
throughout.

```bash
./cornerstone_finetune/run.sh prepare      # one-off, ~7 min, writes data/cornerstone_prepared
./cornerstone_finetune/run.sh smoke        # 2 epochs on 2 subjects: proves the wiring
./cornerstone_finetune/run.sh start        # core6, detached, survives SSH loss
./cornerstone_finetune/run.sh tb           # TensorBoard on :6010
./cornerstone_finetune/run.sh status       # after re-connecting
```

---

## 1. What runs

| file | role |
|---|---|
| `prepare_data.py` | reorients to RAS, resamples to 1 mm isotropic, crops to the liver, writes NIfTI + `manifest.json` |
| `labels.py` | the 15 source ids and the channel presets built from them |
| `datamodule.py` | class-balanced patch sampling, mirror-free augmentation, per-subject channel masks |
| `losses.py` | Dice + `pos_weight`-ed BCE + clDice, all channel-masked |
| `module.py` | SwinUNETR `feature_size=36`, head swapped, bias initialised to the class prior |
| `callbacks.py` | encoder freezing, TensorBoard slice previews, checkpoint-on-SIGTERM |
| `train.py` | entry point; auto-resumes from `checkpoints/last.ckpt` |
| `predict.py` | writes `class*.nii.gz` back on the original CT grid |
| `run.sh` | detached launcher: `prepare / smoke / start / status / logs / attach / stop / kill / tb` |

The architecture change is one layer. 157 of the checkpoint's 159 tensors load
verbatim into a `feature_size=36` SwinUNETR; the two that do not are
`out.conv.conv.{weight,bias}` — the 14-class BTCV head, replaced by
`len(channels)` sigmoid outputs. Encoder and decoder transfer untouched.

---

## 2. Read this before choosing a preset

The masks were measured before any of the above was written, and two properties
of them decide what is worth training.

**Nine of the fifteen classes come from GLB meshes, and the mask builder lets
the coarse contour classes overwrite them.** `data/cornerstone_masks/README.md`
states the rule: where a mesh class and a contour class collide, the contour
wins. `classBloodVessels` (id 6) is a contour class *and* an intrahepatic
vessel union, so it overwrites precisely the portal pedicles that ids 12–15 are
supposed to carry. Across the 18 subjects that have both, `log(BloodVessels)`
against `log(sum of pedicles)` correlates at **−0.55**: the more thorough the
vessel contour, the less of the pedicles survives.

| class | subjects | median voxels (native grid) | expected (finetuning.md §4) |
|---|---|---|---|
| `classLiver` | 38/38 | 2 773 016 | — |
| `classVenaPorta` | 38/38 | 37 121 | — |
| `classIVC` / `classAorta` / `classGallbladder` / `classBloodVessels` | 38/38 | 121 k / 317 k / 34 k / 43 k | — |
| `classPediculoPortalIzquierdo` | 28/38 | **1 329** | 4 000–6 500 |
| `classPediculoPortalDerecho` | 22/38 | **395** | 4 000–6 500 |
| `classPediculoPortalAnteriorDerecho` | 26/38 | **208** | 4 000–6 500 |
| `classPediculoPortalPosteriorDerecho` | 27/38 | **105** | 4 000–6 500 |
| `classVenaHepatica{Derecha,Media,Izquierda}` | 25/22/19 of 38 | 151 / 164 / 461 | — |
| `classArteriaHepatica` / `classViaBiliar` | 18/38, 6/38 | 875 / 338 | too sparse to learn |

Those counts are on the native grid; on the prepared 1 mm grid they roughly
halve again (`classPediculoPortalPosteriorDerecho` has a median of **49**
voxels), because the native voxels are ~0.6 mm³.

Eight subjects have no portal meshes at all — `RMV2025_0017` ships six meshes,
none of them portal — so those channels are *unannotated* there, not empty.

**Consequences, both handled in code:**

* A channel defined as a single pedicle id is mostly missing. A channel defined
  as the union with the contour class that ate it is intact: `portal_tree` =
  `classVenaPorta ∪ classPediculo*` is 37–80 k voxels in **all 38** subjects.
  That is why the presets group.
* Where a channel's sources are all empty in a subject, it is masked out of
  that subject's loss and metrics rather than treated as background —
  finetuning.md §1: "mask out; do not let it count as background".

**Presets** (`data.channel_preset`):

| preset | channels | all present? | use |
|---|---|---|---|
| `core6` | liver, portal_tree, ivc, aorta, gallbladder, vessels | 38/38, ≥7 k voxels each | **start here** |
| `core5` | `core6` minus gallbladder | 38/38 | gallbladder is only a soft frame-recovery signal and was the noisiest channel of the core6 run |
| `core7` | + hepatic_veins | 25/38, masked elsewhere | after core6 |
| `full11` | + the four named pedicles | 22–28/38, remnant-sized | the Couinaud sectors; expect low numbers and read them per channel |
| `all15` | one channel per raw id, nothing unioned | — | ablation, to price the grouping |

If the pedicles matter downstream — and for Couinaud they are the whole job —
the productive fix is upstream of this directory: re-rasterise the masks with
mesh classes winning over `classBloodVessels`, or drop `classBloodVessels` from
the collision rule. `build_couinaud_hybrid.py` is already heading that way.
Nothing here can recover voxels the label map does not contain.

---

## 3. Preprocessing, and why it is not optional

The source volumes are 150–500 MB of float64 each, 13 GB for the cohort, and
their spacing runs **0.63–0.98 mm in-plane and 0.40–1.50 mm through-plane**.
Two things follow:

* A fixed voxel ROI would be a **3.75×-varying physical receptive field** across
  subjects. `prepare_data.py` resamples to 1 mm isotropic so a 96³ patch is
  96 mm everywhere. finetuning.md's "do not resample" applies to the already-
  isotropic cohort in `continuity/`; these are the native scans.
* Cropping to the liver plus a 32 mm margin cuts each case to ~29 MB, so the
  whole training cohort caches in RAM and epochs are bounded by the GPU rather
  than by gzip.

Reorientation to RAS moves image and label together, so it does not touch
laterality — unlike a flip, which finetuning.md §4 forbids outright.

The aorta is deliberately excluded from the bounding box (it runs the length of
the scan and would drag the thorax back in); aorta labels *inside* the liver box
are kept.

**What the resampling costs, measured.** Pushing the ground truth through the
whole round trip — native → 1 mm crop → back onto the original grid via
`predict.py` — and scoring it against the source mask on `RMV2026_0030`:

| class | Dice after round trip |
|---|---|
| `classLiver` | 0.992 |
| `classGallbladder` | 0.984 |
| `classVenaPorta` | 0.969 |
| `classPediculoPortalIzquierdo` | 0.941 |

That is the **ceiling** the preprocessing imposes: no prediction exported this
way can score above ~0.94 on a thin structure, however good the network is.
Shape and affine come back byte-identical to the source volume. Structures
fully inside the crop keep their world coordinates to <0.2 mm; the liver voxel
count changes by 0.24%. If that ceiling ever binds, `--spacing 0.8 0.8 0.8`
raises it at ~2× the memory.

---

## 4. The choices that come from the previous runs

`training_diary.md` records six attempts that ended with vessel Dice at ~0.01
and liver at 0.18. The specific failures, and what is done differently:

| failure | fix here |
|---|---|
| All-background collapse: unweighted BCE+Dice drove every vessel logit to −∞ over 300 epochs | `pos_weight` = capped inverse prevalence, computed from the manifest; **and** the head bias initialised to `log(p/(1−p))`, so the head starts at the base rate instead of at p=0.5 (Lin et al. 2017 §3.3) |
| Sampling from the foreground union drew liver-only crops 30–100× too often | `RandCropByLabelClassesd` over a priority map where the rarest channel wins every overlap; 70% of crop centres land on a tubular channel |
| LR trap: freeze 10 + warmup 5 + lr 1e-4 gave an effective 6e-6 for ten epochs | freeze 3, warmup 2, lr 3e-4, encoder ×0.3 — and the LR is logged every epoch, so check it on epoch 1 |
| `monitor: val/dice` picked checkpoints on a liver-dominated mean that hid vessel Dice of 0.002 | `monitor: val/dice_thin`; every channel is logged separately and there is a `val/dice_bulk` for the easy ones |
| clDice at weight 1.0 dominated Dice+BCE early | `lambda_cldice: 0.5`, on tubular channels only |
| Random crops cached after epoch 1 | the crop is the first random transform, so `CacheDataset` caches only the deterministic prefix — `cache_rate_train: 1.0` is safe and is the reason epochs are fast |
| fp16 instability capped `pos_weight` | `bf16-mixed` on the RTX 5090 |

**No mirroring, ever.** No `RandFlipd`, no `RandRotate90d` — the classes are
laterality-specific and the Couinaud pipeline recovers its anatomical frame from
them. Augmentation is rotation ≤15°, scaling ±10%, elastic, noise, gamma, and
intensity shift.

**Splits.** Grouped, seeded, and reproducible; `split.json` is written into the
run directory. finetuning.md §4 warns about repeated patients across `RMV2025_*`
and `RMV2026_*` — checked against the DICOM headers, **all 38 `PatientID`s are
distinct**, so there is no leak to avoid here. `group_by: suffix` remains
available if that ever changes. For a real number rather than one noisy split,
set `n_folds: 5` and sweep `fold: 0..4`.

---

## 5. What it costs to run

Measured on this machine (RTX 5090, 26 training subjects, `core6`, 96³, 4
patches per step):

| | |
|---|---|
| `prepare` | ~7 min with 5 workers, once; 1.1 GB on disk |
| epoch | ~43 s (104 optimiser steps + 6 sliding-window validations) |
| GPU | 15.9 GB of 32 GB |
| host RAM | ~20 GB (the whole cohort is cached; drop `cache_rate_train` if that is tight) |
| checkpoint | 374 MB each — top-3 + `last` + `periodic` + `interrupted` is ~2.2 GB per run |

300 epochs is therefore about 3.5 hours.

---

## 6. Monitoring

```bash
./cornerstone_finetune/run.sh tb                       # serves runs/cornerstone/*
ssh -N -L 6010:localhost:6010 lois@lab-doutorandos     # from your laptop
```

The log file also carries one line per epoch (the detached run has no progress
bar), which is what `run.sh status` reports. The format — values here are
placeholders, not a result:

```
[epoch <n>]  <wall>  lr=<lr>  train_loss=  val_loss=  liver=  portal_tree=  ivc=  aorta=  gallbladder=  vessels=  thin=
```

Scalars: `train/loss`, `val/loss`, `val/dice_<channel>` for every channel,
`val/dice_thin` (tubular mean — the one to watch), `val/dice_bulk`, `val/dice`
(all-channel mean; ignore it, `liver` owns it), and `lr-AdamW/pg1` (encoder,
scaled by `encoder_lr_scale`) against `lr-AdamW/pg2` (everything else) — check
these on epoch 1 against the LR you meant to set. Images: a ground-truth-vs-prediction strip every 5 epochs at the
most foreground-dense axial slice.

**What good looks like by ~epoch 30 on `core6`:** `liver` > 0.90, `aorta` and
`ivc` > 0.80, `portal_tree` climbing past 0.30. If `liver` is still near 0.2
after ten epochs, stop — the previous runs died exactly there, and the cause is
upstream of any hyper-parameter.

Dice is not the acceptance test. finetuning.md §3: a prediction that traces
*further* than the annotation is better for the pipeline and scores *worse* on
Dice. The real evaluation is end-to-end:

```bash
python cornerstone_finetune/predict.py \
    --checkpoint runs/cornerstone/core6/checkpoints/last.ckpt \
    --split test --out runs/cornerstone/core6/predictions
python continuity/couinaud/run_couinaud.py --all --no-figures
```

---

## 7. Surviving the SSH session

Three independent mechanisms, because each alone has a hole:

1. **`setsid`** — the trainer leads its own session and process group, so the
   `SIGHUP` that follows a dropped connection is never delivered. `nohup` alone
   only masks `SIGHUP` for the shell's child; it does not detach the group.
2. **`nohup` + redirection** — output goes to `logs/<run>_<timestamp>.log`, so
   nothing blocks on a vanished pty.
3. **Auto-resume** — `train.py` restarts from `checkpoints/last.ckpt` with
   optimiser, scheduler and epoch intact. `run.sh start` *is* the resume
   command; there is no separate one.

Checkpoints in `runs/cornerstone/<run>/checkpoints/`: the top 3 by
`val/dice_thin`, `last.ckpt` every epoch, `periodic.ckpt` every 30 minutes
regardless of the metric, and `interrupted.ckpt` written from the SIGTERM
handler. `run.sh stop` sends SIGTERM and waits; `run.sh kill` takes out the
whole process group if dataloader workers are ever orphaned holding GPU memory.

A host reboot costs at most one epoch. Re-running `run.sh start` picks it up.

---

## 8. Knobs worth turning first

```bash
./cornerstone_finetune/run.sh start core6                              # default
PYTHON=/path/to/python ./cornerstone_finetune/run.sh start full11      # stage 2

# one-offs, in the foreground
python cornerstone_finetune/train.py --config cornerstone_finetune/configs/core6.yaml \
    --set model.learning_rate=1e-4 --set data.roi_size='[128,128,128]' \
    --set run_name=core6_lr1e-4
```

* `data.roi_size` — 128³ costs ~2.4× the activation memory of 96³; halve
  `samples_per_volume` with it. finetuning.md §5 wants ≥128 for anything that
  has to see the hilum.
* `model.encoder_lr_scale` — 0.3 here. At n=38 the encoder is what you cannot
  afford to retrain; 0.1 is the more conservative reading of finetuning.md §5.
* `data.crop_thin_share` — share of crop centres on tubular channels. 0.7 for
  `core6`, 0.8 for the presets with sparser channels.
* `model.pos_weight_cap` — 50. Uncapped inverse prevalence reaches ~3000 for the
  thinnest channels and swamps the Dice term.
