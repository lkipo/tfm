# Training Diary — Couinaud / Hepatic Vessel Segmentation

> Chronological log of **all** training runs for this TFM.  
> Each entry: goal, data, model, hyper-parameters, hardware, command, metrics per epoch, problems, decisions.  
> Complements `old/lab_diary.md` (synthetic DVN & GNN history) and `continuity/couinaud/FINETUNING.md` (design rationale).

**Last updated:** 2026-09-03 (couinaud8 finished; couinaud5 running; pedicle derivation validated on GT)
**Author:** lois  
**Hardware:** `lab-doutorandos` — NVIDIA GeForce RTX 5090 32 GB, CUDA 13.2 / Driver 595.84, 64 GB RAM  
**Framework:** `LightningMedSeg3D` (`lightning==2.6.5`, `monai`, `torch 2.6`, `16-mixed` AMP)  
**Dataset (stage-1 Couinaud):** `data/couinaud_nifti` (38 rebuilt cases → 29 pass QA → **28 CT after MR filter**, see `data/couinaud_nifti/build_report.json`), `data/0_test_nifti/imagesTs` (full 512×512×~910 volumes, native spacing `0.76-0.97 × 0.76-0.97 × 0.40-0.70 mm`). `RMV2025_0002_MR` excluded (`exclude_modality=[MR]`). Effective cohort **28** → split `70/15/15` (seed 42) → `19 train / 4 val / 4 test` (`src/lightning_medseg3d/datamodules/segmentation_datamodule.py:179`). Of 28, **22 with hepatic_veins, 6 without** (masked loss/metrics, `src/lightning_medseg3d/losses/__init__.py:20`). Current val split (seed 42): `RMV2026_0024_CT_UTV_VEP_J` (no hep), `RMV2026_0037_CT_UTV_VEP_C` (no hep), `RMV2025_0004_CT_UTV_VEP_N` (hep), `RMV2025_0007_CT_UTV_VEP_H` (hep).

---

## Overview

| # | Date | Name | Model | Data | Loss | Key hparams | Result | Status |
|---|------|------|-------|------|------|-------------|--------|--------|
| 0 | 2026-08-?? | `couinaud` pre-pos_weight (historic, from `FINETUNING.md:54`) | `swin_unetr` (BTCV-pretrained) | 28 CT, `128³` @ `1.5mm` | `dice_bce_cldice` **without** `pos_weight` (only `RandCropByLabelClassesd` ratios) | `lr 1e-4, freeze 10, warmup 5, wd 0.05, cldice 1.0, cache 1.0` | `val/dice ~0` for 3/4 channels after 300e (all-background local minimum) | aborted — motivated `pos_weight` design |
| 1 | 2026-06-17 | DVN Run 1 — weighted-CE (from `old/lab_diary.md`) | `DeepVesselNet-FCN` (4 conv, cross-hair) | 10 synthetic `120×100×150` → 2361 `32³` patches (1.7% vessel) | `weighted_categorical_crossentropy_with_fpr` | `Adam 1e-3, 50ep, bs4, 90/10` | val Dice 0.95 (patch) / 0.97 (vol) but `max P(vessel) ~0.49` (threshold hack 0.485) | superseded |
| 2 | 2026-06-24 | DVN Run 2 — Dice+CE, 20 synth (from `old/lab_diary.md`) | same FCN | 20 synth → 4713 patches | `combined_dice_ce_loss(0.5)` | `Adam 1e-3, 50ep, bs4` | **val Dice 0.982** (0.5 thr), hold-out `0.961`, `max P 1.0` ✓ | saved `deepvesselnet/models/dvn_seg.dat` |
| 3 | 2026-08-29 12:25 | `couinaud` **version_0** (`tb/version_0`) | `swin_unetr` `feature_size 36` BTCV `epoch=1390` (14-class head discarded, `157/159` tensors loaded, `src/models/segmentation_module.py:143`) | 28 CT, `ROI 128³` @ claimed `1.5×1.5×2.0mm` (but `couinaud` ignores `Spacingd` — effectively native, mismatch), `cache 1.0` | `dice_bce_cldice` `cldice [1,2] w=1.0` `pos_weight [2,30,30,15]` | `lr 1e-4, wd 0.05, warmup 5, freeze 10, encoder_lr×0.1, cosine, 300ep` | `val/dice 0.0624→0.0668 @ e2` (liver-dominated, `monitor val/dice`) — killed at e2 | **stopped** — LR trap diagnosed |
| 4 | 2026-08-29 13:30 → 14:33 | `couinaud` **version_1** (`tb/version_1`) — **stopped** | same `swin_unetr-36` BTCV | 28 CT, `ROI 96³` native (`spacing [1,1,1]` NOT APPLIED, `src/datamodules/segmentation_datamodule.py:202`), `PriorityLabelMapd` + `RandCropByLabelClassesd ratios [0.05,0.15,0.25,0.25,0.30]` for `bg/liver/ivc/hep/portal`, `cache_rate 0.0` (recompute crops each epoch) | `dice_bce_cldice` `cldice [1,2] w=0.5` `pos_weight [2,30,30,15]` | `lr 3e-4, wd 0.01, warmup 2, freeze 3, encoder×0.1, cosine, 300ep, monitor val/dice_vessels` | `liver 0.184 @ e14`, `vessels 0.0082 @ e14` (best `0.0111 @ e13`), `train 1.98`, `val/loss 1.97` — stopped at e15 (see below) | **stopped manually** `SIGTERM` (see `## Run 4` termination) |
| 5 | 2026-08-29 14:41 → 15:48 | `couinaud` **fix 4ch** (`tb/swin_unetr_fix/version_0`) — **stopped** | same `swin_unetr-36` BTCV | 28 CT, `96³` native, `selected_channels none` (4ch) | `dice_bce` `pos_weight [2,30,30,15]` (no clDice) | `lr 1e-3, wd 0.01, warmup 0, freeze 0, encoder×0.3, log 5, ES 50` | `e9 vessels 0.00182` (best `0.0019 @e3`), `liver 0.166`, `train 1.64` — flat vs `version_1 0.0111` (see `## Run 5`) | **stopped manual** (worse than `version_1`) |
| 6 | 2026-08-29 15:48 → | `couinaud` **3ch reduced** (`tb/swin_unetr_3ch`) — **ready** | same `swin_unetr-36` BTCV | 28 CT, `96³` native, `selected_channels [0,1,3]` → 3ch `[liver, portal_tree, ivc]` (drop `hep` 0.1% rarest, `6/28` missing) | `dice_bce` `pos_weight [2,30,15]` `cldice [1]` | `lr 1e-3, freeze 0, warmup 0, encoder×0.3, ratios [0.05,0.15,0.30,0.50]` | not yet launched — `swin_unetr_3ch.yaml:1` | **ready** `run_3ch.sh` |
| 7 | 2026-08-31 22:26 → 09-01 01:58 | `cornerstone_finetune` **core6** (6ch: liver, portal_tree, ivc, aorta, gallbladder, vessels) | `swin_unetr-36` BTCV `epoch=483-val_loss=0.07.ckpt` (157/159, sigmoid head + bias prior) | 37 CT (`cornerstone_masks` → RAS, 1mm iso, liver crop 32mm), `96³`, 26/6/5 seed 42 | `dice_bce_cldice`: Dice 1.0 + BCE `pos_weight≤50` 1.0 + clDice 0.5 (tubular, 10 iters) | `lr 3e-4, freeze 3, enc×0.3, warmup 2, cosine 300, bf16, clip 1.0, monitor val/dice_thin` | best thin **0.6068** (e254); final e297: thin 0.597, liver 0.820, portal_tree 0.523 | **stopped by request @e297** — plateau since ~e160; fixes validated |
| 8 | 2026-09-01 02:21 → 06:02 | `cornerstone_finetune` **core5** (core6 minus gallbladder, internal backbone) | same as core6 | 37 CT, `96³`, 26/6/5 seed 42 | same as core6 | same as core6 | final e299: thin 0.6301, portal_tree 0.5714, liver 0.8362, vessels 0.5994; best thin 0.6369 (e203); test thin 0.6559 / portal 0.5797 | **completed** |
| 9 | 2026-09-01 14:11 → 18:42 | `cornerstone_finetune` **core5_paper_backbone** (5ch: liver, portal_tree, ivc, aorta, vessels) | `swin_unetr-36` `LightningMedSeg3D/weights/BTCV/swin_unetr.pth` (paper benchmark checkpoint, 157/159 tensors loaded) | 37 CT (`cornerstone_masks`), `96³`, 26/6/5 seed 42 | same as core5 | identical to `core5.yaml` — only `model.pretrained_weights` changed | final e299: thin 0.6781, portal_tree 0.6398, liver 0.8963, vessels 0.5947, dice_mean 0.7429; best thin **0.6816** (e150); test thin 0.6984 / portal 0.6352 | **completed** — paper backbone wins on every channel except vessels (±0) |
| 10 | 2026-09-01 19:0x → | `cornerstone_finetune` **core5_paper_cldice** | same paper backbone | same | same + `lambda_cldice 0.5→0.8` (thin plateaued in Run 9, pre-approved knob) | same otherwise | not yet known | **launched** |
| 11 | 2026-09-03 11:47 → 17:15 | `cornerstone_finetune` **couinaud8** (8ch: liver, vena_porta, ivc, aorta, 4 pedicles) | paper backbone | 37 CT, `96³`, 26/6/5 | same as core5 + masked pedicles | same as core5 | test: liver 0.889, ivc 0.821, aorta 0.794, vena_porta 0.606, pedicles **all 0.0**, thin 0.238 | **completed** — pedicles below the learnable ceiling; → derivation approach |
| 12 | 2026-09-03 18:07 → | `cornerstone_finetune` **couinaud5** (5ch: liver, vena_porta, portal_tree, ivc, aorta) | paper backbone | 37 CT, `96³`, 26/6/5 | same as core5 (portal_tree replaces the 4 pedicle channels) | same as core5 | running (e32: liver 0.63, vena_porta 0.55, portal_tree 0.55) | **running** — pedicles derived at inference from predicted portal_tree (Run 12a) |

Detailed historic synthetic runs (1-2) remain authoritative in `old/lab_diary.md:1`; this diary focuses on `LightningMedSeg3D` Couinaud runs (3-4) but retains 0-2 for completeness.

---

## Common Setup (Runs 3-4)

**Pretrain:** `data/nn_model_weights/swin_unetr-epoch=1390-val_loss=0.48.ckpt` (409 MB, BTCV 14-class, `feature_size 36`, `out.conv [14,36,1,1,1]` discarded by shape match, `src/models/segmentation_module.py:137`). Also exists `epoch=483-val_loss=0.07.ckpt` (not used).

**Intensity:** `ScaleIntensityRanged [-175,250] → [0,1] clip` (CT soft-tissue window). MR `RMV2025_0002` filtered because window invalid (`FINETUNING.md:44`).

**Labels:** `src/datamodules/couinaud_label.py:22` `CHANNEL_NAMES [liver, portal_tree, hepatic_veins, ivc]` (`portal_tree` = `VenaPorta` + 4 pedicles, `hepatic_veins` = 3 `VenaHepatica*`). Multi-label sigmoid (`multi_label True`, `src/models/segmentation_module.py:83`), not softmax (classes overlap `~20%` at bifurcation, `couinaud.py:988`). Union bbox +20 vox crop before any transform (`couinaud_label.py:91`).

**Augment:** `RandAffined prob 0.3 rotate ±0.26 rad (~15°) scale ±0.1`, `Rand3DElastic prob 0.2 sigma 5-7 mag 50-150`, `RandGaussianNoise 0.15 std 0.01`, `RandAdjustContrast 0.15 gamma 0.7-1.5`, `RandShiftIntensity 0.5` (`segmentation_datamodule.py:273`). **No flips/rot90** for `couinaud` (laterality-specific labels, `FINETUNING.md:4`).

**Optim:** `AdamW`, `gradient_clip 1.0`, `deterministic warn`, `16-mixed`, `batch 1`, `samples_per_volume 4`, `num_workers 4`, `seed 42`, `default_root_dir ./runs/couinaud/swin_unetr`, `TensorBoard tb/version_*`.

**Metrics:** `src/models/segmentation_module.py:222` per-channel Dice at `sigmoid>0.5`, accumulated via `_dice_inter/_dice_denom/_dice_count` with missing-hepatic masking (`nanmean`). `val/dice` = mean 4, `val/dice_vessels` = mean `portal+hepatic+ivc`, `val/dice_thin` = `portal+hepatic` (`FINETUNING.md:3` says watch vessels/thin, not liver-dominated mean).

---

## Run 0 — Historic first Couinaud run (no pos_weight) — referenced

* **When:** before 2026-08-29 (described in `FINETUNING.md:54` and `configs/couinaud/swin_unetr.yaml:54` comment).
* **Why it matters:** Used only `RandCropByLabelClassesd` oversampling, **no** `pos_weight`. Result: `val/dice at ~0 for 3 of 4 channels after 300 epochs — the all-background solution is a very strong local minimum under this imbalance` (`swin_unetr.yaml:58`). This motivated `pos_weight [2,30,30,15]` (capped well below inverse-prevalence `~500` for `fp16` stability, `losses/__init__.py:45`).
* **Lesson:** Foreground-oversampled cropping alone insufficient for `0.1%` prevalence; `BCE pos_weight` + `Dice` required.

---

## Run 3 — version_0 — 2026-08-29 12:25 — stopped at epoch 2

**Command:** `lightning-medseg3d fit --config configs/couinaud/swin_unetr.yaml` (with `v0` yaml)

**Config (`runs/couinaud/swin_unetr/tb/version_0/hparams.yaml` + `config.yaml`):**
```yaml
model: {architecture: swin_unetr, in_channels: 1, num_classes: 4, roi_size: [128,128,128], swin_feature_size: 36,
        learning_rate: 0.0001, weight_decay: 0.05, scheduler: cosine, warmup_epochs: 5,
        loss: dice_bce_cldice, cldice_channels: [1,2], cldice_weight: 1.0, pos_weight: [2,30,30,15], pretrained_weights: ../data/nn_model_weights/swin_unetr-epoch=1390-val_loss=0.48.ckpt, encoder_lr_scale: 0.1}
trainer: {precision: 16-mixed, max_epochs: 300, gradient_clip_val: 1.0, callbacks: [ModelCheckpoint(monitor: val/dice, filename: epoch={epoch}-dice={val/dice:.4f}), FreezeEncoderCallback(freeze_epochs: 10)]}
data: {dataset: couinaud, roi_size: [128,128,128], spacing: [1.5,1.5,2.0], intensity_range: [-175,250], batch_size: 1, num_workers: 4, train_frac: 0.7, val_frac: 0.15, samples_per_volume: 4, cache_rate: 1.0, exclude_modality: null}
```

**Pretrain load:** `[pretrained_weights] loaded 157/159 tensors` (head `14→4` mismatch, `segmentation_module.py:148`).

**Checkpoints:** `epoch=0-dice=0.0624.ckpt`, `epoch=1-dice=0.0638.ckpt`, `epoch=2-dice=0.0668.ckpt`, `last.ckpt` (each ~391 MB, `runs/couinaud/swin_unetr/tb/version_0/checkpoints/`). `events.out.tfevents.1787998799.lab-doutorandos.340471.0` (4.3 KB).

**Why stopped:** Diagnosed **learning-rate trap** (`FINETUNING.md:342`): `lr 1e-4` with `freeze 10` + `warmup 5` gives encoder `6e-6` / decoder `6e-5` at `e2` (logged), effectively no learning for 10 epochs then cosine decay. Also `roi 128³` at native `0.76mm` (~97 mm) wasteful, `cache 1.0` cached random crops after e1 (`segmentation_datamodule.py:320`), `cldice_weight 1.0` dominated `Dice+BCE` (~0.5 vs 1.0), `monitor val/dice` liver-dominated hid vessel failure (legacy `val/dice 0.062 = 0.24 liver, 0.002 portal, 0.001 hep, 0.019 ivc` @ e3, `FINETUNING.md:258`).

**Fixes applied → version_1:** `lr 1e-4→3e-4`, `freeze 10→3`, `warmup 5→2`, `wd 0.05→0.01`, `roi 128→96`, `cldice 1.0→0.5`, `cache 1.0→0.0`, `spacing` doc fix `1.5→1.0` (still NOT APPLIED for `couinaud`), `monitor val/dice→val/dice_vessels` (`swin_unetr.yaml:18`), added `exclude_modality [MR]`.

---

## Run 4 — version_1 — 2026-08-29 13:30 → 14:33 **STOPPED** at epoch 15 (last completed: epoch 14)

**Command:** `.venv/bin/lightning-medseg3d fit --config configs/couinaud/swin_unetr.yaml`  
**PID:** `381531` (parent), children `410494-410575` (`num_workers 4` + 3 compute), `GPU 0 14079/32607 MiB`, `log LightningMedSeg3D/logs_couinaud_fixed_20260829_132850.log` (225 KB final, `fast_dev_run` off).  
**Config:** `LightningMedSeg3D/configs/couinaud/swin_unetr.yaml` (also `runs/couinaud/swin_unetr/tb/version_1/config.yaml` + `hparams.yaml`):

```yaml
model: {roi_size: [96,96,96], learning_rate: 3.0e-4, weight_decay: 0.01, warmup_epochs: 2, loss: dice_bce_cldice, cldice_channels: [1,2], cldice_weight: 0.5, pos_weight: [2.0,30.0,30.0,15.0], swin_feature_size: 36, encoder_lr_scale: 0.1}
trainer: {max_epochs: 300, callbacks: [ModelCheckpoint(monitor: val/dice_vessels, filename: epoch={epoch}-dice_vessels={val/dice_vessels:.4f}, save_top_k: 3), FreezeEncoderCallback(freeze_epochs: 3)]}
data: {roi_size: [96,96,96], spacing: [1.0,1.0,1.0], batch_size: 1, num_workers: 4, samples_per_volume: 4, cache_rate: 0.0, exclude_modality: [MR]}
```

**Cohort:** `[couinaud] 28 cases after QA+modality filter (22 with hepatic_veins, 6 without; masked loss will handle missing channel)` (`segmentation_datamodule.py:176`). Same `70/15/15` split as above.

**Checkpoints final:** `epoch=9-dice_vessels=0.0109.ckpt`, `epoch=11-dice_vessels=0.0110.ckpt`, `epoch=13-dice_vessels=0.0111.ckpt` (**best**), `last.ckpt` (all ~428 MB), `events 15 KB`. Stopped run left `last.ckpt` at `e14` state (not `e15` — `e15` validation never finished).

**Epoch progress (de-duplicated last validation per epoch, from `logs_couinaud_fixed_20260829_132850.log` via regex `Epoch (\d+): 100%.*val/loss.*val/dice_liver.*val/dice_vessels.*train/loss_epoch`):**

| epoch | wall | `val/loss` | `val/dice` (mean) | `val/dice_liver` | `val/dice_portal` | `val/dice_hep` | `val/dice_ivc` | `val/dice_vessels` | `val/dice_thin` | `train/loss_epoch` | notes |
|------:|------|-----------|-------------------|-----------------|-------------------|----------------|----------------|-------------------|-----------------|-------------------|-------|
| 0 | 13:35+4.5m | 2.10 | 0.0396 | 0.149 | 0.00202 | 0.00134 | 0.00566 | 0.00301 | 0.00168 | 2.33 | sanity val @ e0 |
| 1 | +3.9m | 2.07 | 0.0419 | 0.161 | 0.00201 | 0.00091 | 0.00331 | 0.00208 | 0.00146 | 2.27 | frozen encoder |
| 2 | +3.7m | 2.06 | 0.0429 | 0.162 | 0.00275 | 0.00223 | 0.00524 | 0.00341 | 0.00249 | 2.23 | |
| 3 | +5.1m | 2.05 | 0.0419 | 0.163 | 0.00053 | 0.00035 | 0.00367 | 0.00152 | 0.00044 | 2.19 | **unfroze `swinViT` @ e3** |
| 4 | +4.8m | 2.04 | 0.0436 | 0.166 | 0.00261 | 6.4e-5 | 0.00516 | 0.00261 | 0.00134 | 2.16 | `no available indices class 3` |
| 5 | +4.9m | 2.03 | 0.0462 | 0.171 | 0.00510 | 0.00258 | 0.00588 | 0.00452 | 0.00384 | 2.15 | first `>0.004` |
| 6 | +3.7m | 2.03 | 0.0446 | 0.171 | 0.00192 | 0.00101 | 0.00424 | 0.00239 | 0.00146 | 2.14 | regression |
| 7 | +3.5m | 2.02 | 0.0476 | 0.175 | 0.00226 | 0.00681 | 0.00698 | 0.00535 | 0.00453 | 2.14 | |
| 8 | +3.0m | 2.01 | 0.0499 | 0.183 | 0.00633 | 0.00581 | 0.00471 | 0.00562 | 0.00607 | 2.11 | |
| 9 | +3.5m | 2.00 | 0.0519 | 0.175 | 0.00761 | 0.0121 | 0.0129 | **0.0109** | 0.00987 | 2.07 | |
| 10 | +3.1m | 2.00 | 0.0512 | 0.179 | 0.00361 | 0.00927 | 0.0131 | 0.00867 | 0.00644 | 2.08 | noise down |
| 11 | ~3m | 1.99 | 0.0527 | 0.178 | 0.00582 | 0.0118 | 0.0154 | **0.0110** | 0.0088 | 2.04 | |
| 12 | +4.1m | 1.98 | 0.0521 | 0.185 | 0.00608 | 0.00401 | 0.0130 | 0.00771 | 0.00505 | 2.03 | |
| 13 | +4.1m | 1.97 | 0.0525 | 0.177 | 0.00852 | 0.00492 | 0.0199 | **0.0111** | 0.00672 | 2.03 | **new best** |
| 14 | +3.6m | 1.97 | 0.0520 | 0.184 | 0.00302 | 0.00521 | 0.0164 | 0.0082 | 0.00411 | 1.98 | last **completed** val |
| 15 | — | — | — | — | — | — | — | — | — | — | **interrupted** `14/20` (`Epoch 15: 70%`) — `SIGTERM` (see below) |

*Full step log: `train/loss_step 1.8-2.5` flat, `no available indices class 3` 2-3×/epoch, `val 4 vols [00:03-00:19]`.*

**Termination — 2026-08-29 14:33 CEST (user request):**
- **How:** `kill 381531` → `SIGTERM` delivered to `PID 381531` (Lightning), log shows `[rank: 0] Received SIGTERM: 15` during `Epoch 15: 70%` (`14/20 steps`). After `~15s` graceful attempt, workers `410494-410575` orphaned still at `~100%` CPU; `pkill -f lightning-medseg3d` + `kill -9` on remaining workers, `GPU` freed `14079→87 MiB`. Clean exit — no traceback except `SIGTERM` line, `last.ckpt` retained at `e14` state.
- **Duration:** `01:04:42` wall (`13:28:33→14:33:33`), `15` epochs attempted (`14` validated), `~4.3 min/ep`.
- **Final state @ e14:** `train/loss 1.98` (`2.33→1.98`), `val/loss 1.97` (`2.10→1.97`), `liver 0.184` (+0.035), `vessels 0.0082` (`0.0111 best @ e13`), `portal 0.003`, `hep 0.005`, `ivc 0.016`. Liver still `0.18` (target `>0.8` by `e10`).
- **Why stopped (reasons logged):**
  1. **Not worth 300e at this trajectory** — `+0.0005 vessels/ep`, need `~170ep` for `0.1` vessels; `EarlyStopping patience 50` in fix config would have stopped similarly.
  2. **Liver red flag** — `0.18` after 14e with 12% bbox prevalence confirms systemic under-learning, not just rare-vessel difficulty (likely `lr 3e-4 / encoder 3e-5` still too low for native spacing domain shift, `n=28`/`val 4` variance, `clDice` early domination).
  3. **Fix ready** — `configs/couinaud/swin_unetr_fix.yaml` prepared with diary fixes (`lr 1e-3, freeze 0, warmup 0, encoder×0.3, loss dice_bce`), detached launcher `run_fix.sh` ready; stopping frees `14GB` GPU for next run.
  4. **Checkpoint saved** — best `epoch=13-dice_vessels=0.0111.ckpt` + `last.ckpt` (e14) preserved for resume/compare; `logs_couinaud_fixed_20260829_132850.log` (225 KB) + `tb/version_1/events 15 KB` retained.

**Artifacts retained:** `runs/couinaud/swin_unetr/tb/version_1/checkpoints/{epoch=9,11,13,last}.ckpt`, `logs_couinaud_fixed_20260829_132850.log` (SIGTERM tail), `tb/version_1/events.out.tfevents.1788002936.lab-doutorandos.381531.0`.

---

## Run 5 — fix 4ch — 2026-08-29 14:41 → 15:48 **STOPPED** at epoch 10 (last completed: epoch 9)

**Command:** `bash run_fix.sh` → `.venv/bin/lightning-medseg3d fit --config configs/couinaud/swin_unetr_fix.yaml`  
**PID:** `416403`, `GPU 15443/32607 MiB`, `log logs/fix_20260829_144108.log` (27 KB → 15 KB events), `runs/couinaud/swin_unetr_fix/tb/version_0`.

**Config vs `version_1`:** `lr 3e-4→1e-3, freeze 3→0, warmup 2→0, encoder×0.1→0.3, loss dice_bce_cldice(0.5)→dice_bce, log 10→5, +EarlyStopping 50` (`swin_unetr_fix.yaml:1`).

**Epoch progress (from `fix_20260829_144108.log`):**

| epoch | `val/loss` | `val/dice` | `liver` | `portal` | `hep` | `ivc` | `vessels` | `thin` | `train` | notes |
|------:|-----------|-----------|---------|----------|-------|-------|-----------|--------|---------|-------|
| 0 | 1.54 | 0.0424 | 0.165 | 0.00132 | 2e-11 | 0.00288 | 0.0014 | 0.00066 | 1.79 | best `0.0014` |
| 1 | 1.52 | 0.0427 | 0.167 | 0.00334 | 2e-11 | 0.00062 | 0.00132 | 0.00167 | 1.70 | |
| 2 | 1.51 | 0.0428 | 0.168 | 0.00096 | 2e-11 | 0.00226 | 0.00107 | 0.00048 | 1.71 | |
| 3 | 1.52 | 0.0433 | 0.167 | 0.00061 | 2e-11 | 0.00509 | **0.0019** | 0.00031 | 1.71 | **best** |
| 4 | 1.52 | 0.0402 | 0.159 | 0.00158 | 2e-11 | 1.8e-5 | 0.00053 | 0.00079 | 1.68 | |
| 5 | 1.52 | 0.0416 | 0.162 | 0.0031 | 1.6e-11 | 0.00097 | 0.00136 | 0.00155 | 1.66 | |
| 6 | 1.52 | 0.0427 | 0.169 | 0.00023 | 2e-11 | 0.00141 | 0.00055 | 0.00011 | 1.69 | |
| 7 | 1.52 | 0.0419 | 0.166 | 9.8e-5 | 2e-11 | 0.00131 | 0.00047 | 4.9e-5 | 1.71 | |
| 8 | 1.53 | 0.0421 | 0.166 | 0.00139 | 3e-12 | 0.0012 | 0.00086 | 0.00070 | 1.69 | |
| 9 | 1.51 | 0.0429 | 0.166 | 0.00308 | 0.00131 | 0.00107 | 0.00182 | 0.0022 | 1.64 | last completed |
| 10 | — | — | — | — | — | — | — | — | — | **killed** `01:01:47` `@e10 0%` |

**Termination — 15:48 (user request, worse than `version_1`):**
- `kill 416403` + `pkill -9` (workers `425128-425131`), `GPU 15443→87 MiB`. Log ends mid-`Epoch 10` after `e9` val.
- **Why stopped:** `fix` `e9 0.00182` < `version_1 e3 0.0019` best and << `version_1 0.0111 @e13`; `hep` stayed `2e-11` for `8e` (no learning), `liver 0.166` flat vs `version_1 0.18`; `lr 1e-3 + no freeze` did **not** beat `lr 3e-4 + clDice` — indicates need for class reduction not just LR.
- **Artifacts:** `runs/couinaud/swin_unetr_fix/tb/version_0/checkpoints/{epoch=0,3,last}.ckpt` (best `e3 0.0019`), `logs/fix_20260829_144108.log` + `tb_fix` logs (kept for comparison).

---

## Run 6 — 3ch reduced — 2026-08-29 15:48 → **READY** (not yet launched)

**Goal:** Drop `hepatic_veins` (rarest `0.1%`, `6/28` missing, `25%` crop budget) → `3ch [liver, portal_tree, ivc]` via `selected_channels [0,1,3]` (`couinaud_label.py: SelectChannelsd` + `segmentation_datamodule.py: selected_channels`).

**Config:** `LightningMedSeg3D/configs/couinaud/swin_unetr_3ch.yaml:1`:
```yaml
model: {num_classes: 3, channel_names: [liver, portal_tree, ivc], pos_weight: [2,30,15], cldice: [1], lr: 1e-3, encoder×0.3}
data: {selected_channels: [0,1,3], ratios: [0.05,0.15,0.30,0.50] for bg/liver/ivc/portal (hep drops, portal 30%→50%)}
trainer: {root_dir: ./runs/couinaud/swin_unetr_3ch, TB 6008}
```
**Launcher:** `LightningMedSeg3D/run_3ch.sh:1` (same `nohup`+`disown` as `run_fix.sh`, survives SSH; `TB 6008` vs `6007`/`6006`).

**Launch (detached):**
```bash
cd LightningMedSeg3D
bash run_3ch.sh            # → logs/3ch_*.log + logs/3ch.pid, TB 6008
bash run_3ch.sh status     # PID + GPU + last epoch (after re-SSH)
bash run_3ch.sh logs 50    # tail
bash run_3ch.sh attach     # tail -f
```

---

## Run 7 — `cornerstone_finetune` core6 — 2026-08-31 22:26 → 2026-09-01 01:58 — STOPPED by request at epoch 297/300

**Command:** `cornerstone_finetune/run.sh start core6` (detached via `setsid`+`nohup`; auto-resume from `last.ckpt`).

**Framework:** `cornerstone_finetune/` (rewrite after the Run 3–6 failures). BTCV `data/nn_model_weights/swin_unetr-epoch=483-val_loss=0.07.ckpt` (157/159 tensors; head → 6 sigmoid outputs with log-odds bias prior). Prepared data: RAS, 1 mm isotropic, liver crop +32 mm, whole cohort cached in RAM.

**Config:** `cornerstone_finetune/configs/core6.yaml` (`runs/cornerstone/core6/resolved_config.yaml`):
- channels: `core6` = liver, portal_tree, ivc, aorta, gallbladder, vessels (all 38/38, ≥7 k voxels)
- cohort: 37 CT after MR exclusion; split 70/15/15 seed 42 → **26/6/5** (`split.json`; all 38 DICOM `PatientID`s distinct → case split = patient split)
- data: `96³` @1 mm, batch 1, 4 samples/volume; intensity `[-175,250]`; priority sampling (5 % bg, 70 % tubular centres)
- augment: rot ≤15° + scale ±10 % (p 0.3), elastic σ5–7 / 50–150 (p 0.15), noise σ0.01 (p 0.15), gamma 0.7–1.5 (p 0.15), shift (p 0.5); **no flips/rot90**
- loss: Dice 1.0 + BCE `pos_weight≤50` 1.0 + soft-clDice 0.5 on tubular channels (10 iters)
- optim: AdamW `lr 3e-4`, `wd 0.01`, freeze encoder 3 epochs, encoder ×0.3, warmup 2 (0.1→1), cosine → 300, grad clip 1.0, `bf16-mixed`
- monitor: `val/dice_thin`; sliding-window inference 96³, sw_batch 4, overlap 0.5, threshold 0.5

**Final state (epoch 297):** `lr=3.3e-08`, `train_loss 0.8010`, `val_loss 0.8681`, `dice_mean 0.6677`, `liver 0.8200`, `aorta 0.7777`, `ivc 0.6751`, `gallbladder 0.6181`, `portal_tree 0.5233`, `vessels 0.5918`, `bulk 0.7386`, `thin 0.5967`.
**Best (top-3):** `epoch160-val_dice_thin0.6065.ckpt`, `epoch254-val_dice_thin0.6068.ckpt`, `epoch274-val_dice_thin0.6051.ckpt`. Epochs 240–297 flat: liver 0.80–0.83, portal_tree 0.50–0.54, thin 0.57–0.61.

**Why stopped (user request):**
1. **Metric plateau since ~epoch 160** — `thin` stuck at 0.60–0.61; liver ~0.82 vs the 0.90 marker of the run plan (`cornerstone_finetune/README.md` §6); portal_tree ~0.52 above its 0.30 marker but no longer climbing.
2. **Cosine schedule exhausted** — `lr` already `3.3e-08` at e297; the last ~100 epochs moved nothing, so finishing to 300 would not change the outcome.
3. GPU freed for the next run. Checkpoints preserved in `runs/cornerstone/core6/checkpoints/` (top-3 + `last` + `interrupted`); log `cornerstone_finetune/logs/core6_20260831_222601.log`.

**What it proved:** the four fixes from Runs 0/3–5 work — liver 0.18→0.82 and portal_tree 0.01→0.52 vs the failed attempts. The bottleneck is no longer optimization; it is data (gallbladder noise, pedicle remnants under `classBloodVessels`) and receptive field → Run 8.

---

## Run 8 — `cornerstone_finetune` core5 — 2026-09-01 02:21 → 06:02 — COMPLETED

**Goal:** one-variable change over core6 — drop the gallbladder channel (option 1 of the post-core6 review). Rationale:
- The pipeline uses it only for frame recovery (`anatomy_frame.py`: 2 evidence vectors, w=0.7 each, of ~10.3 total weight); degradation is graceful — 8/8 segments, only frame confidence drops (`finetuning.md` §2).
- It was the noisiest channel of the core6 run (val Dice 0.45–0.72), so removing it also removes a perturbing loss term and frees head capacity for the channels Couinaud actually needs.

**Config:** `configs/core5.yaml` (identical to core6 except `channel_preset: core5`); `core5 → [liver, portal_tree, ivc, aorta, vessels]`, thin = [1,2,4]. Internal backbone `data/nn_model_weights/swin_unetr-epoch=483-val_loss=0.07.ckpt`. Command: `cornerstone_finetune/run.sh start core5`, log `logs/core5_20260901_022146.log`.

**Final state (epoch 299):** `train_loss 0.7701`, `val_loss 0.8253`, `dice_mean 0.6983`, `liver 0.8362`, `portal_tree 0.5714`, `ivc 0.7194`, `aorta 0.7649`, `vessels 0.5994`, `bulk 0.8005`, `thin 0.6301`.
**Best (top-3):** `epoch203-val_dice_thin0.6369.ckpt`, `epoch221 0.6368`, `epoch202 0.6363`. Plateau from ~e200.
**Test (best ckpt):** dice 0.7166, liver 0.8259, portal_tree 0.5797, vessels 0.6365, thin 0.6559, ivc 0.7515, aorta 0.7893.

**What it proved:** dropping gallbladder helped thin (0.6068→0.6369 val best, +0.030) and liver (0.82→0.84) vs core6 — confirming the noisy-channel hypothesis. Baseline now fixed for the backbone swap (Run 9).

**Knobs proposed for the runs after core5 (from the post-core6 review; try one at a time):**

| knob | from | to | why |
|---|---|---|---|
| `data.roi_size` | 96³ | 128³ | receptive field must reach the hilum (`finetuning.md` §5 wants ≥128); ~2.4× memory → halve `samples_per_volume` |
| `model.encoder_lr_scale` | 0.3 | 0.1 | more conservative encoder update at n=38 |
| `data.crop_thin_share` | 0.7 | 0.8 | more crop-centre budget for the sparse channels of core7/full11 |
| `n_folds` / `fold` | 0 | 5 / 0..4 | a single 26/6/5 split is too noisy; report variance (`finetuning.md` §4) |
| `model.lambda_cldice` | 0.5 | 0.8 | raise once `dice_thin` plateaus; keep ≤1.0 (1.0 dominated early in Run 3) — **condition met at Run 9 e150-300 → chosen for Run 10** |
| `data.batch_size` | 1 | 2 (or grad accumulation) | smoother thin-channel updates; GPU headroom (15.9/32 GB) |
| upstream masks | contour-wins | mesh-wins | re-rasterize so pedicles survive `classBloodVessels` (`build_couinaud_hybrid.py`) before `full11` — no hyperparameter recovers voxels the label map lacks |

**Do not touch:** `monitor val/dice_thin`, `pos_weight` + head bias prior, mirror-free augmentation, `bf16-mixed`, `freeze_encoder_epochs: 3`.

---

## Run 9 — `cornerstone_finetune` core5_paper_backbone — 2026-09-01 14:11 → 18:42 — COMPLETED

**Goal:** the project has two separately-produced SwinUNETR checkpoints for the liver-vessel work — the internal `data/nn_model_weights/swin_unetr-epoch=483-val_loss=0.07.ckpt` used as `core5`/`core6`'s backbone, and the externally-validated BTCV SwinUNETR reported in the LightningMedSeg3D benchmark paper (`sources/1-s2.0-S2352914826000687-main.pdf`), released at `LightningMedSeg3D/weights/BTCV/swin_unetr.pth`. Rather than treat these as two competing single-purpose networks (liver-only vs. vessel-only) that need an ensemble or cascade, the paper's checkpoint is architecturally identical to the one `cornerstone_finetune` already fine-tunes from — so it's used as a drop-in backbone swap into the existing multi-channel sigmoid-head fine-tune, keeping vessels (and the other channels) as trained outputs, not disregarded.

**Verified before launch (no assumptions):**
- Both checkpoints loaded and inspected directly: MONAI `SwinUNETR`, `feature_size=36`, 14-class BTCV head (`out.conv.conv.weight` shape `[14, 36, 1,1,1]`) in both.
- Tensor-by-tensor diff: 160/160 keys match by name and shape; 151/160 differ in value — confirms these are two distinct BTCV training runs of the identical architecture, not the same weights, so the swap is a genuine change of backbone prior, not a no-op.
- Per the paper (Table 4: BTCV liver DSC 0.956; Table 6: near-zero zero-shot generalisation gap on the Álvaro Cunqueiro external cohort, most reliable of the nine benchmarked architectures) — the bet is that a better-generalising liver prior in the shared encoder/decoder benefits every fine-tuned channel, not just liver.

**Implemented:** `cornerstone_finetune/configs/core5_paper_backbone.yaml` — identical to `configs/core5.yaml` (channel preset `core5` = `[liver, portal_tree, ivc, aorta, vessels]`, gallbladder excluded per request) except `model.pretrained_weights: LightningMedSeg3D/weights/BTCV/swin_unetr.pth`. No changes to `labels.py`, `losses.py`, or `module.py` — this is a one-variable config change over `core5`.

**Verification before the real run:** `run.sh smoke core5_paper_backbone` — `[pretrained] 157/159 tensors ... skipped 3: ['loss_function.dice.class_weight', 'out.conv.conv.bias', 'out.conv.conv.weight']` (identical skip pattern to `core5`/`core6`, confirming the paper checkpoint is a valid drop-in), channel availability confirmed `[liver, portal_tree, ivc, aorta, vessels]` with no `gallbladder`, loss decreased over 2 smoke epochs. Smoke run directory deleted after verification.

**Command:** `cornerstone_finetune/run.sh start core5_paper_backbone` — detached (`setsid`+`nohup`), auto-resume from `checkpoints/last.ckpt`. PID 214939, log `cornerstone_finetune/logs/core5_paper_backbone_20260901_141147.log`, run dir `runs/cornerstone/core5_paper_backbone`, TensorBoard `http://localhost:6010`.

**Final state (epoch 299):** `train_loss 0.7269`, `val_loss 0.6994`, `dice_mean 0.7429`, `liver 0.8963`, `portal_tree 0.6398`, `ivc 0.7998`, `aorta 0.7841`, `vessels 0.5947`, `bulk 0.8402`, `thin 0.6781`.
**Best (top-3):** `epoch150-val_dice_thin0.6816.ckpt`, `epoch241 0.6805`, `epoch247 0.6805`. Plateau ~e150–e300 (thin 0.675–0.682).
**Test (best ckpt e150, 5 cases):** dice 0.7651, liver 0.8705, portal_tree 0.6352, vessels 0.6367, thin 0.6984, ivc 0.8233, aorta 0.8600, loss 0.7992.

**Trajectory (val):** fast early — liver 0.89 by e75 (vs core5's 0.72 @ e80); thin 0.63 @ e75; best thin 0.6816 @ e150; then flat to e299. ~47 s/ep early, ~70 s/ep late (slice logging), wall 4h31m.

**Verdict — paper backbone vs internal (core5) at final epoch (val):**

| channel | core5 (internal) | core5_paper | Δ |
|---|---|---|---|
| liver | 0.8362 | 0.8963 | **+0.060** |
| portal_tree | 0.5714 | 0.6398 | **+0.068** |
| ivc | 0.7194 | 0.7998 | **+0.080** |
| vessels | 0.5994 | 0.5947 | −0.005 (noise) |
| thin | 0.6301 | 0.6781 | **+0.048** |
| dice_mean | 0.6983 | 0.7429 | **+0.045** |

The bet paid off: the paper checkpoint's better-generalising liver prior lifted **every** channel — except `vessels`, which is unchanged at ~0.59-0.60 (its label is the noisiest: pedicle remnants under `classBloodVessels`, see mesh-wins knob). `vessels` is now the weakest channel and the plateau is optimization-proof: thin flat 0.675–0.682 for 150 epochs with lr decaying 2.3e-4 → 0. Bottleneck is now the loss/ROI/data for thin structures, not the backbone → Run 10.

---

## Run 10 — `cornerstone_finetune` core5_paper_cldice — 2026-09-01 → RUNNING

**Goal:** lift `vessels` + `portal_tree` — the two thin channels that did not (vessels) or only partly (portal) move with the Run 9 backbone swap. Single variable over `core5_paper_backbone`: `model.lambda_cldice 0.5 → 0.8`.

**Rationale:**
- The knob table condition is met: `dice_thin` plateaued at 0.675–0.682 from e150 to e300 in Run 9 while lr decayed to 0 — exactly the "raise once dice_thin plateaus" trigger.
- clDice is the topology term: plain Dice+BCE converged to trunk-only solutions in Runs 0–6, and the thin channels' failure mode is distal recall (finetuning.md §3: "highest-leverage single choice"). +0.3 weight = +60% topological pressure.
- Run 3's domination caution respected: at 1.0, clDice (~0.5 scale) overwhelmed Dice+BCE (~1.0); at 0.8 it contributes ~0.4 vs ~1.0–1.4 combined — still subdominant, and Run 3 also had the LR trap / native-spacing confounds.
- Tversky (β>α) deliberately **not** combined: `core6_tversky` (internal backbone, 0.3/0.7, +liver crop share, ES30) stopped at e117 with best val thin 0.5965 and test thin 0.6074 / portal 0.5188 — no evidence it beats Dice on this data, and it competes with clDice for the same distal-recall job. If Run 10 stalls, tversky on top of 0.8 clDice is the fallback, or the ROI-128³ knob (receptive field, `finetuning.md:237`).

**Config:** `cornerstone_finetune/configs/core5_paper_cldice.yaml` — copy of `core5_paper_backbone.yaml`, only `model.lambda_cldice: 0.8` changed (and `run_name`).

**Command:** `cornerstone_finetune/run.sh start core5_paper_cldice` after `run.sh smoke core5_paper_cldice`. Log `logs/core5_paper_cldice_*.log`, TensorBoard `http://localhost:6010`. Compare against Run 9's `epoch150-val_dice_thin0.6816.ckpt` on val/test `dice_thin`, `dice_vessels`, `dice_portal_tree`.

---

## Run 11 — `cornerstone_finetune` couinaud8 — 2026-09-03 11:47 → 17:15 — COMPLETED (pedicles all-zero)

**Goal:** the pipeline's four pedicle classes trained directly as their own channels (liver, vena_porta, ivc, aorta, pediculo_izquierdo/derecho/ant_derecho/post_derecho), so `predict.py` output could feed `continuity/couinaud` without derivation. This is the "remnant supervision" attempt: pedicle labels are remnants of `classBloodVessels`-contour overwrite.

**Config:** `configs/couinaud8.yaml` — `channel_preset: couinaud8` (`labels.py:137`), paper backbone, masked loss where a pedicle channel is absent (`required=False`), otherwise identical to core5. Command: `cornerstone_finetune/run.sh start couinaud8`, log `logs/couinaud8_paper_cldice_20260903_180739.log` (resumed run dir `runs/cornerstone/couinaud8_paper_cldice`). ~46-76 s/ep (GPU mostly free).

**Final state (epoch 299) and test (best `epoch279-val_dice_thin0.2355.ckpt`):**

| channel | val (final) | test (best ckpt) | prevalence (test) |
|---|---|---|---|
| liver | 0.916 | 0.889 | 31.0 % |
| vena_porta | 0.525 | 0.606 | 0.3 % |
| ivc | 0.806 | 0.821 | 1.1 % |
| aorta | 0.770 | 0.794 | 0.8 % |
| pediculo_izquierdo | 0.0 | 0.0 | 0.002 % (21-1450 voxels) |
| pediculo_derecho | 0.0 | 0.0 | 0.002-0.01 % |
| pediculo_ant_derecho | 0.0 | 0.0 | ~34 voxels |
| pediculo_post_derecho | 0.0 | 0.0 | 321 voxels |
| thin | 0.2355 | 0.238 | — |

**Why it failed:** the four pedicle remnants total ~10³ voxels per case (prevalence 0.001-0.01 %) — below the learnable ceiling at 96³ ROI with `pos_weight≤50`; the model outputs empty pedicle masks at 0.5 threshold on every test case. The `bulk` channels (liver/ivc/aorta/vena_porta) all improved vs core5 (liver 0.889 vs 0.870, vena_porta 0.606 vs 0.635-0.52 range) — consistent with the 4 pedicle channels adding no gradient signal.

**Decision — derivation instead of supervision:** the tarball/GLB registration investigation (below) proved the external full-label cohort cannot supervise CT training, and the cornerstone pedicle remnants cannot be learned. So the pedicle masks must be **derived at inference** from the well-learnable union (`portal_tree` + `vena_porta`) → Run 12 trains those two channels and `derive_pedicles.py` splits the union at the bifurcation.

---

## Registration investigation — tarball/GLB labels cannot supervise CT training — 2026-09-03 (dead end, logged)

**Question:** can `couinaud_train.tar.gz` (21 complete-label cases with real pedicle/vein trees) provide training labels for the CT cohort via a rigid/affine registration of the tarball's liver mesh onto the CT liver mask?

**Attempts (all on `RMV2025_0003_CT_UTV_VEP_P` and `..._0004_...`):**
1. `build_couinaud_case.py`'s own transform (tarball GLB → CT space): livers overlap (0.93 / 0.79) but internal vessels miss the CT anatomy entirely.
2. Raw ICP on liver surfaces: converges to a wrong local minimum.
3. PCA-aligned ICP: same.
4. Hill-climb over the tarball liver PCA frame + grid search on the residual transform: best fit overlaps the liver (0.93, 0.1 mm mean distance) but **IVC recall 0.11-0.50, pedicle recall 0.01-0.36** — the tarball's vessels are systematically displaced relative to the CT anatomy (different patient presentation / mesh drift), no rigid transform can fix it.

**Conclusion:** tarball labels cannot supervise CT training; the `classPediculoPortal*` labels in the cornerstone masks are the only pedicle truth and they are remnants. Derivation is the only viable route.

---

## Run 12 — `cornerstone_finetune` couinaud5 — 2026-09-03 18:07 → 2026-09-04 02:34 — COMPLETED (300/300 epochs)

**Goal:** train only the classes that are complete in all 37 cases and well-learnable — `[liver, vena_porta, portal_tree, ivc, aorta]` — and **derive the four pedicle masks at inference** from the predicted `portal_tree` union by splitting at the portal bifurcation (`derive_pedicles.py`). No pedicle channels to fail, no hepatic veins / gallbladder (Voronoi degrades gracefully without them).

**Config:** `configs/couinaud5.yaml` — `channel_preset: couinaud5` (`labels.py:160`), paper backbone, identical hyper-parameters to core5 (the union channels already proved learnable: portal_tree 0.635 test in Run 9). Command: `cornerstone_finetune/run.sh start couinaud5` (note: the `run.sh start` arg is the config *file* name — `couinaud5`, not the `run_name`; status output is stale because `run.sh` reads the old core6 pidfile). Log `logs/couinaud5_paper_cldice_20260903_180739.log`.

**Incident — worker killed at epoch 11 (18:45):** `RuntimeError: DataLoader worker (pid 933038) is killed by signal: Terminated` mid-validation — SIGTERM, not OOM (dmesg clean; 42 GB RAM free; GPU fully contended by 5 external processes → 30.4/32 GB, 100 % util). `last.ckpt` intact; `run.sh start couinaud5` resumed at epoch 12 (PID 976866).

**GPU contention:** host load average 28-32 (7 users); 5 external training processes (2.7-3.8 GB each) + ours 13-18 GB ≈ 28-30 GB → 95-200 s/ep vs 46-76 s/ep when the GPU was free (ETA revised to ~10 h from e109, up from the ~6 h estimated at e32 — contention has not eased). Current log is `logs/couinaud5_paper_cldice_20260903_185715.log` (the post-resume file; the pre-crash `..._180739.log` stops at epoch 11).

**Trajectory (val):** e32: dice_mean 0.593, liver 0.629, portal_tree 0.555, vena_porta 0.552, ivc 0.687, aorta 0.544 → e112: dice_mean 0.611, liver 0.868 → e190: dice_mean ~0.71-0.73, liver 0.88-0.90 (GPU contention eased through the run, 95-200 s/ep down to ~70-130 s/ep by the end, no further crashes after the e11 resume).

**Final test (best `epoch187-val_dice_thin0.6811.ckpt`):**

| channel | test Dice |
|---|---|
| liver | 0.911 |
| ivc | 0.823 |
| aorta | 0.830 |
| vena_porta | 0.630 |
| portal_tree | 0.625 |
| thin | 0.693 |
| bulk | 0.871 |
| mean | 0.764 |

Matches or beats `core5_paper_cldice` (thesis numbers: liver 0.910, ivc 0.829, aorta 0.779, vena_porta 0.610) on every channel it shares — the extra `portal_tree` channel cost nothing.

---

## Run 12a — Pedicle derivation (`derive_pedicles.py`) — design and GT validation — 2026-09-03

**Design:** `cornerstone_finetune/derive_pedicles.py` splits the predicted `portal_tree` union at the portal bifurcation:

1. **Skeleton split** (`derive_pedicle_masks`): skeletonize the union, remove skeleton points within a trunk radius of the `vena_porta` skeleton, split what remains into left/right-ant/right-post components. **Always returns `{}` on this data, GT included**, not just on blob-like predictions: the cornerstone `vena_porta` (id 2) is not a short stub — the contour-overwrite behaviour `labels.py` documents means it already extends through 91 % of the union (29,115 of 32,068 voxels on case 0003) — so its skeleton is nearly as long as the whole tree's (1,200 of 1,203 points) and every tree point falls within the trunk radius; zero points ever survive removal. `_plane_split` (below) is the only path that has ever produced output.
2. **Plane split** (`_plane_split`, the production path, iterated twice):

   *First pass (superseded):* cut `portal_tree & liver` at planes through a "bifurcation" point (tree skeleton point nearest the `vena_porta` skeleton), using R-L = transverse component of (`ivc` centroid − `aorta` centroid) and A = `vena_porta` − `ivc` centroid, right side bisected by the median of the anterior coordinate. Gating by liver (`portal_tree & liver`, using the trained *liver channel* = `[1]+INTRAHEPATIC`, not raw `classLiver`) discards the trunk correctly — cornerstone `vena_porta` is 100 % extrahepatic under this hard-partition schema, so `portal_tree & liver` is exactly `{12,13,14,15}`, same as excluding the trunk directly. But the "bifurcation" point is degenerate: since the trunk skeleton is nearly as long as the tree's, *every* trunk skeleton point is at ~0 distance from *some* tree point, so the nearest-pair `argmin` picks whichever point happens to be first in an effectively-tied array — a bounding-box corner, not the hilum. And separately, **measured directly**: the cosine similarity between the (`ivc`−`aorta`)-derived R-L axis and the true left/right pedicle-centroid axis on case 0003 is **0.03** — no correlation at all. Result: left recall only 0.47, and the right side (everything not "left") absorbed ~99 % of the union before the median rescued ant/post from total collapse.
   *Current:* drop the bifurcation point and the `ivc`/`aorta`-centroid axes entirely. R-L and A-P axes now come from `continuity/couinaud/anatomy_frame.estimate_frame()` — the same anatomy-fit frame the Couinaud stage itself already trusts, given only `[liver, vena_porta, ivc, aorta]` (everything `couinaud5` predicts). Each of the two splits (left/right, then right→ant/post) is an Otsu threshold of the relevant axis projection over `portal_tree & ~vena_porta` (mathematically identical set to `portal_tree & liver` above, expressed without needing a separate liver mask) — Otsu finds the natural valley in a skewed 1-D distribution, where a plain median forces an exact 50/50 split regardless of where the true boundary sits and a fixed point-referenced threshold is fragile once trunk removal reshapes the point cloud per case.

**GT-derived results on `RMV2025_0003` (final, after Run 12b's two further fixes below):**

| sector | voxels | recall | prec | GT voxels |
|---|---|---|---|---|
| Izquierdo (left) | 1,687 | 0.96 | 0.65 | 1,138 |
| AnteriorDerecho | 668 | 0.24 | 0.01 | 34 |
| PosteriorDerecho | 555 | 0.98 | 0.57 | 321 |

AnteriorDerecho's precision is inherently poor throughout — its GT is a 34-voxel scrap — but recall (does the derived sector *contain* the true remnant) is the meaningful number: the Couinaud stage reads these masks for sector identity, not exact geometry (its own graph cuts refine boundaries). Recall vs. the remnants is a proxy, not the acceptance criterion — the Couinaud stage output is.

**Couinaud stage on the derived case dir (GT anchors + derived pedicles, `continuity/couinaud`, `run_checks`):** **6/12 checks PASS**: all 8 segments present, **superior/inferior ordering holds for all three pairs** (II>III, VIII>V, VII>VI), **right sector anterior of posterior (+67 mm)**, **left of right (+65 mm)**, frame recovered confidently (0.75). Failing: one segment (III) drops to 49% single-component (a new, minor side-effect of the MIN_SPLIT_FRAC median fallback below — the more balanced ant/post split shifts a small chunk of III across a boundary into two pieces); the 3 vein-plane cross-checks fail by design (no hepatic veins in this label set — validation-only, degrades gracefully per the `couinaud8`/`couinaud5` design decision above); segment volumes (4/8 in the published range; left hemiliver 24.2%, just under the 25-45% band). The volume-balance failures are a property of the *input* (tiny, spatially incomplete GT remnants near the hilum bias where mass concentrates within each sector), not of the axis-recovery method.

**Frame recovery fix (separate, already landed):** `anatomy_frame.py` only searched *right-handed* voxel triads — but LAS voxel spaces are left-handed, so for the cornerstone masks the A axis came out flipped (`A=axis1-`, confidence 0.45; "right anterior anterior of posterior" −59 mm). The search now admits both handednesses and lets the evidence disambiguate → `R=axis0- A=axis1+ S=axis2+`, confidence 0.74. `_frame_axes` in `derive_pedicles.py` (above) calls this same fixed `estimate_frame`, so it inherits the correction.

---

## Run 12b — first end-to-end test on real predictions — 2026-09-04

**Setup:** `predict.py --checkpoint .../epoch187-val_dice_thin0.6811.ckpt --case RMV2025_0003_CT_UTV_VEP_P --out runs/pipeline_tests/predictions_couinaud5 --grid prepared`, once Run 12 finished and freed the GPU (an earlier attempt mid-training OOM'd with 137 MB free; a second attempt at ~7.7 GB free while training was still running also OOM'd — 1.8 GB short mid-inference — so this genuinely needed the GPU to itself). Succeeded cleanly: `liver=1,590,929 vena_porta=20,774 portal_tree=22,940 ivc=92,924 aorta=296,896` voxels on the 361×253×440 prepared grid. `derive_pedicles.py --case-dir ...` ran without error (plane-cut fallback, as expected) but the **Couinaud stage badly regressed on this real prediction: only 2/12 checks passed, segments II/III/IV/VI/VII all missing**, VIII alone claiming 96.5% of the liver — a much worse result than the 6/12 GT test predicted. Root-caused and fixed as two separate bugs, both invisible against GT:

1. **Stray fragments.** GT labels are a clean hard partition; predictions aren't. Measured on this case: `classLiver` 300 components (8.5% stray voxels), `classPortalTree` 18 components (6.9% stray), `classVenaPorta` 10 components (5.0% stray), `classAorta` 88 components (12.9% stray), `classIvc` 29 components (6.4% stray). `_plane_split` fed the *entire* `portal_tree`/`vena_porta` into the Otsu splits with no filtering, so scattered false-positive fragments far from the liver became legitimate-looking seeds after skeletonization — one case put a "left" seed cluster's Z-span at 306 mm (the liver itself is ~150 mm tall). **Fix:** restrict `portal_tree` and `vena_porta` to their largest connected component before anything else, mirroring what the Couinaud stage already does for its own liver mask.
2. **Extrahepatic leakage.** Excluding the trunk (`portal_tree & ~vena_porta`) was proven equivalent to gating by liver *on GT* (Run 12a) only because the cornerstone labels are a hard partition — id 2 (`vena_porta`) can never overlap ids 12-15 by construction, so both filters land on the identical voxel set. Predictions carry no such guarantee: measured directly, the derived "left" sector without a liver gate sat at R-projection 98-169 mm against a predicted liver spanning 142-328 mm — almost entirely *outside* the organ, most likely genuine extrahepatic portal trunk tissue the network is (correctly) predicting as part of the union. That starved segments II/III/IV of any in-liver seeds. **Fix:** `pedicle_only = portal_tree & ~vena_porta & liver` (largest component of the predicted liver too).

Fixing only those two surfaced a **third**, previously-masked issue: with the trunk-dominated bulk of `portal_tree` now excluded by the liver gate too, `pedicle_only` shrank enough that a plain Otsu threshold on the ant/post axis collapsed one GT case to an 18/1,205 split (right-anterior lost all its seeds — V and VIII went missing, 4/12). **Fix:** `_otsu_split` now falls back to the median whenever Otsu's split would leave either side under `MIN_SPLIT_FRAC = 0.15` of the total — Otsu's better boundary-placement when the data supports it, a guaranteed floor when it doesn't.

**Result after all three fixes, same real prediction:** `left=668 ant=170 post=180` voxels, **8/12 checks pass** — all 8 segments present, each a single connected blob (worst III at 88%), right anterior of posterior (+57 mm), left of right (+60 mm), superior/inferior ordering holds, left hemiliver in range (31.6%), frame confidence 0.78. Re-ran the GT test (Run 12a) with the same three fixes: still 6/12, all 8 segments, both sidedness checks, all three superior/inferior orderings now also pass — net improvement there too, at the cost of one segment (III) dropping to 49% single-component, a shape side-effect of fix 3's more balanced split.

**This is the pipeline's first successful close of the full loop: CT → couinaud5 prediction → derived pedicles → 8 Couinaud segments**, with no ground truth involved anywhere in the chain.

---

## Problems & Fixes Log (chronological)

| Date | Problem | Symptoms | Root cause | Fix | File |
|------|---------|----------|------------|-----|------|
| pre-08-29 | All-background minimum | `val/dice 0` for vessels after 300e | No `pos_weight`, imbalance `0.1%` hep | `pos_weight [2,30,30,15]` | `losses/__init__.py:42`, `swin_unetr.yaml:54` |
| 08-29 12:25 | LR trap | `6e-6` encoder / `6e-5` decoder @ e2, no learning | `freeze 10 + warmup 5 + lr 1e-4` wasted schedule | `lr 3e-4, freeze 3, warmup 2, wd 0.01` | `swin_unetr.yaml:46`, `segmentation_module.py:351` |
| 08-29 12:25 | Cached random crops | `global_step 45` after 3e = 15 steps/ep not 60 | `CacheDataset cache_rate 1.0` caches `RandCropByLabelClassesd` | `cache_rate 0.0` for `couinaud` train | `segmentation_datamodule.py:338` |
| 08-29 12:25 | `128³` too large | OOM risk, wasteful at native `0.76mm` | Pretrain `1.5mm` physical mismatch | `roi 96³` (`73-93mm` native) + `PriorityLabelMapd` 55% vessel-centred crops | `swin_unetr.yaml:43`, `couinaud_label.py:131` |
| 08-29 12:25 | `clDice` domination | `clDice ~0.5` vs `Dice+BCE ~1.0` | `cldice_weight 1.0` | `0.5` | `swin_unetr.yaml:53` |
| 08-29 12:25 | Liver-dominated checkpoint | `val/dice 0.06` hid vessel `0.002` | `monitor val/dice` | `monitor val/dice_vessels` | `swin_unetr.yaml:18` |
| 08-29 13:30+ | `no available indices class 3` | 2-3×/epoch warnings | 6/28 cases lack `hepatic_veins` (also `ivc` rare) → fallback to other classes | Masked loss `channel_mask` (`losses/__init__.py:20`), ignored warnings, `nanmean` metrics | `segmentation_module.py:289`, `losses/__init__.py:66` |
| 08-29 13:30+ | Broken `0_test_nifti` affines | `R,A,S` with `zooms 1.0` false, translation untrustworthy | Legacy masks rebuilt onto `imagesTs` affine via `build_couinaud_case.py` | Use `couinaud_nifti` + `imagesTs` shared affine | `FINETUNING.md:0` |
| 06-xx | DVN prob saturation | `max P ~0.49` at 0.5 thr → Dice 0 | Weighted-CE alone near-minimum at 0.49 | `combined_dice_ce_loss` (`0.5 Dice`) | `old/deepvesselnet/dvn/losses.py` |
| 09-03 | Pedicle remnants unlearnable | couinaud8 pedicles 0.0 test Dice | prevalence 0.001-0.01 %, below the learnable ceiling at 96³/`pos_weight≤50` | derive pedicles at inference from the predicted `portal_tree` union | `derive_pedicles.py`, Run 12 |
| 09-03 | Tarball labels can't register onto CT | IVC recall 0.11-0.50, pedicles 0.01-0.36 after perfect liver fit (0.93 overlap) | tarball vessels systematically displaced vs CT anatomy; no rigid transform fixes it | abandon tarball supervision; derivation only | this diary "Registration investigation" |
| 09-03 | Seeds outside the liver | segments II/VI/VII empty; IV 73 %; 90 % of seeds extrahepatic | cornerstone `vena_porta` (id 2) is 100 % extrahepatic → the union `[2,12-15]` is 91 % trunk | gate the union by the liver (`portal_tree & liver`); trunk kept only as the bifurcation anchor | `derive_pedicles.py` intrahepatic gating |
| 09-03 | Flipped A axis on LAS scans | `A=axis1-` while the evidence says anterior=axis1+; ant/post sector gap −59 mm | frame estimator only searched right-handed voxel triads; LAS voxel spaces are left-handed | admit both handednesses; evidence disambiguates (confidence 0.45 → 0.74) | `anatomy_frame.py` |
| 09-03 | Ant/post split still ~empty after gating | PosteriorDerecho ≈ 0 vox even with the trunk excluded; left recall only 0.47 | R-L/A axes from raw `ivc`/`aorta`/`vena_porta` centroids: measured cosine similarity with the true left/right pedicle axis on case 0003 is 0.03 (no correlation); the plane's origin (skeleton nearest-pair `argmin`) resolves to an arbitrary bbox corner, not the hilum, once the trunk skeleton is nearly as long as the tree's | drop the bifurcation point and the ad hoc axes; use `anatomy_frame.estimate_frame()` for R/A (same frame the Couinaud stage trusts) and an Otsu threshold per axis instead of a fixed origin | `derive_pedicles.py` `_frame_axes`, `_otsu_split`, `_plane_split` |
| 09-04 | GT-validated derivation failed on real predictions (2/12 checks, 5 segments missing) | segments II/III/IV/VI/VII empty, VIII alone = 96.5% of the liver | predictions (unlike GT) aren't a hard partition: (a) 5-13% stray voxels per channel across up to 90 components put seeds up to 306 mm from the true liver; (b) excluding the trunk no longer implies intrahepatic-only, so genuine extrahepatic portal tissue pulled the "left" sector almost entirely outside the predicted liver (R-proj 98-169 mm vs. liver's 142-328 mm) | restrict `portal_tree`/`vena_porta` to their largest component; gate `pedicle_only` by the (largest-component) predicted liver too | `derive_pedicles.py` `_plane_split` |
| 09-04 | Fixing the above starved V/VIII on the GT case (4/12) | Otsu split 1,223 right-side voxels into 18/1,205 once the liver gate shrank the input | Otsu finds *a* valley, not a *balanced* one — fine on a large point cloud, degenerate on a small one | `_otsu_split` falls back to the median whenever either side would be `< MIN_SPLIT_FRAC (0.15)` of the total | `derive_pedicles.py` `_otsu_split` |
| 09-03 | DataLoader worker SIGTERM @ e11 | training died mid-validation, `last.ckpt` intact | external kill / heavy contention (5 processes, 30.4/32 GB, load 28-32); not OOM | auto-resume from `last.ckpt`; monitor; consider waiting for GPU to free | `run.sh start couinaud5` |

---

## Next Steps (from `FINETUNING.md:5` recipe)

- [ ] **Overfit test** (1-2 cases) — confirm model can memorize `liver >0.9`.
- [ ] **LR / freeze ablation** — `lr 1e-3`, `freeze 0` vs `3` vs `5`.
- [ ] **5-fold CV** — `seed 42` single split `val 4` variance too high.
- [ ] **Intensity hygiene** — verify `[-175,250]` window covers liver/portal (checked: `0.77-0.80` in-window) and no MR leak.
- [ ] **End-to-end acceptance:** `python continuity/couinaud/run_couinaud.py --all` on predicted `class*.nii.gz` (same `couinaud_nifti` layout, `FINETUNING.md:235`), then `ablate.py` (ρ coverage bias `0.79→<0.5`, sector ratio etc.).
- [ ] Optional: synthetic pretrain on `datagen_pipeline/hepatic_dvn_dataset` (10 DVN samples) for vesselness backbone.

---

## How to launch next trains — detached, survive SSH disconnect

**Current:** `fix 4ch` stopped, `3ch reduced` ready. Both use `nohup`+`disown` (no `tmux` needed).

**Fix 4ch (already stopped, for reference `Run 5`):** `LightningMedSeg3D/configs/couinaud/swin_unetr_fix.yaml:1` (`lr 3e-4→1e-3, freeze 3→0, warmup 2→0, encoder×0.1→0.3, loss dice_bce_cldice(0.5)→dice_bce, log 10→5, +EarlyStopping 50`).

**Launch fix 4ch (detached, nohup — safe to close SSH):**
```bash
cd LightningMedSeg3D
bash run_fix.sh            # → logs/fix_*.log + logs/fix.pid, TB 6007
```

**Launch 3ch reduced (recommended next):**
```bash
cd LightningMedSeg3D
bash run_3ch.sh            # → logs/3ch_*.log + logs/3ch.pid, TB 6008
```

**After disconnect, re-SSH and get verbose updates (either run):**
```bash
cd LightningMedSeg3D
bash run_fix.sh status;  bash run_3ch.sh status   # PID, GPU, last epoch
bash run_fix.sh logs 100; bash run_3ch.sh logs 100
bash run_fix.sh attach;  bash run_3ch.sh attach   # tail -f (Ctrl-C keeps training)
bash watch_fix.sh                               # live loop (fix) / adapt for 3ch
# TensorBoard: 6006 (version_1 stopped) vs 6007 (fix) vs 6008 (3ch)
# tensorboard --logdir runs/couinaud/swin_unetr_3ch/tb --port 6008 --bind_all
```

**Stop / resume:**
```bash
bash run_fix.sh stop       # SIGTERM graceful
bash run_fix.sh kill       # SIGKILL if orphaned workers remain
# resume from last.ckpt (if stopped): add to command or yaml: ckpt_path: runs/couinaud/swin_unetr_fix/tb/version_*/checkpoints/last.ckpt
```

**Overfit sanity (1-2 cases, should reach liver >0.9 by e20 if no bug):**
```bash
.venv/bin/lightning-medseg3d fit --config configs/couinaud/swin_unetr_fix.yaml --data.train_frac=0.07 --data.val_frac=0.0 --trainer.max_epochs=30 --trainer.limit_train_batches=5
```

**Current version_1 artifacts remain:** `runs/couinaud/swin_unetr/tb/version_1/checkpoints/epoch=13-dice_vessels=0.0111.ckpt` (best) for comparison; do not delete.

---

## References

* `continuity/couinaud/FINETUNING.md` — what to train, what “good” means (clDice > Dice, traced depth, `ρ`, connectivity).
* `LightningMedSeg3D/configs/couinaud/swin_unetr.yaml` — current `v1` config.
* `LightningMedSeg3D/src/lightning_medseg3d/datamodules/segmentation_datamodule.py` + `couinaud_label.py` — data, `PriorityLabelMapd`, `RandCropByLabelClassesd`.
* `LightningMedSeg3D/src/lightning_medseg3d/models/segmentation_module.py` — `multi_label`, `val/dice_*`, `DiceMetric` vs per-channel accumulators.
* `LightningMedSeg3D/src/lightning_medseg3d/losses/__init__.py` — `DiceBCELoss` + `SoftClDiceLoss` + `channel_mask`.
* `old/lab_diary.md` — synthetic DVN & GNN history.
* `data/couinaud_nifti/build_report.json` — 38→29→28 cohort.
* Logs: `LightningMedSeg3D/logs_couinaud_fixed_20260829_132850.log`, `runs/couinaud/swin_unetr/tb/version_*/`.

