# Data Structure and Properties

## Overview

**38 cases** (37 CT, 1 MR) in `data/0_test_nifti/imagesTs/` and `data/0_test_glb/`, with contour annotations in `data/0_test_nifti/cornerstone_annotations/`.

All annotations are **axial contours in DICOM LPS coordinates**. Rasterization requires `LPS→RAS` transform: negate both x and y. A single y-flip (which still lands 100% of points inside the CT box due to near-symmetric anatomy) mirrors left/right and must be rejected.

## Base CT Volumes (`imagesTs/`)

512×512×910 slices. Modality: 37 CT, 1 MR (`RMV2025_0002_MR_NCT_ARP_N`).

Native spacing varies per case: 0.76–0.97 mm in-plane, 0.40–0.70 mm slice thickness. Examples:
- `RMV2025_0001`: 0.765×0.765×0.701 mm
- `RMV2025_0008`: 0.906×0.906×0.40 mm

**Affine matrix note:** All 38 share the same RAS+ orientation (`L`, `A`, `S`) and near-identical scale/rotation (diagonal rotation is negated x and y for LPS→RAS). Slice axis is z. The DICOM `z_shift` (world z-offset to the skull base) varies per case but can be solved analytically from the contours.

## Annotations (`cornerstone_annotations/`)

**One identical class-set across all 38 files:**
1. `classLiver` — ~2.9M voxels median
2. `classVenaPorta` — ~10.9k voxels
3. `classIVC` — ~95k voxels
4. `classAorta` — ~403.6k voxels
5. `classGallbladder` — ~63.9k voxels
6. `classBloodVessels` — ~133.4k voxels

**No `PediculoPortal*` or `VenaHepatica*` classes in cornerstone.** Those exist only as GLB meshes in `data/0_test_glb/`.

Format: JSON with `classes` dict (class name → UUID list) and `state` array of contours. Each contour is a polyline (list of [x, y, z] in mm, LPS frame).

### Contour Density and Gaps

Per-class coverage is highly variable:
- **Liver:** 225–261 traced slices across 353–909 z range (span 557 slices). Typically two clusters: main body (353–577) and spurious shoulder-level cluster (874–909). The shoulder cluster holds ~0.66% of liver volume but 21% of contours (before volume-based outlier rejection).
- **VenaPorta:** 65–79 slices. Gaps up to 340 slices; minimal traceable volume clusters.
- **Aorta, Gallbladder:** Denser tracing, few stray fragments.

**Outlier rejection (volume-based):** Clusters holding <5% of a class's voxels are dropped. This catches the liver shoulder artefact (19,356 voxels on case 0) and a 2.6% VenaPorta cluster at 869–882.

### Laterality and Registration

**LPS→RAS transform:** Negate both x and y. Verified by:
- 100% of contour points land inside CT bounding box (all 38 cases)
- IVC centroid patient-right (+10.6 mm world-space), aorta left (−9.8 mm) — correct anatomy
- Liver HU stats: 151.9 ± 42 (homogeneous parenchyma, <0.1% air)
- Aorta HU: 236.8 ± 68 (enhanced vessel, brighter than liver)

A y-only flip (old mistake) produced:
- Liver 51.8 ± 251 HU with 6.2% air voxels (sitting on stomach/bowel)
- Anatomy flipped (IVC left of aorta — wrong)

## Rasterization

Cell 7 in `data_preview.ipynb` rasterizes contours onto the imagesTs grid.

**Algorithm:**
1. Transform polylines: `LPS→RAS = [-1, -1, 1]`
2. Per-slice even-odd polygon fill (XOR, so holes survive)
3. Outlier rejection: drop z-clusters with <5% of class volume
4. Gap filling: signed-distance blending between adjacent slices, capped at 5-slice gaps (longer gaps left empty — they signal end of annotation, not solid structure)

**Validation:** Liver Dice **0.9754** vs `annotations2niftimask` label 6 in the valid region (z < 874). The reference also contains the shoulder artefact (19,356 voxels), so full-volume Dice is 0.9722 after volume-based rejection drops it.

## GLB Meshes (`0_test_glb/`)

**15 anatomical classes per case, but only 6 classes in all cases:**

Universal (38/38): Aorta, BloodVessels, Gallbladder, IVC, Liver, VenaPorta

Partial coverage:
- PediculoPortal* (4 branches): 23–28/38 cases
- VenaHepatica* (3 branches): 27–28/38 cases
- ArteriaHepatica, Tumour, ViaBiliar: 6–18/38 cases

**Two mesh sets per case:**
- Case root: decimated, simplified meshes (~18k verts for liver)
- `full_model/`: original high-resolution meshes (~230k verts for liver)

Only 4 classes differ between sets: Aorta, Gallbladder, IVC, Liver. The other 11 are byte-identical (verified with `cmp`).

**Coordinate frame:** Meshes are **NOT in the CT world frame.** Bounds: x ∈ [−481.9, 0], y ∈ [0, 763.7], z ∈ [19.6, 623.7] (never positive x, never negative y). Attempts to match via rigid transforms yield 18.6 mm bounding-box residual at best. **Meshes derive from manual annotations, not directly from imagesTs.**

### Z-Calibration from Contours

The liver mesh's axial extent matches the contour z-span. Using this to solve `z_shift` analytically:

```
z_shift = contour_z_min - mesh_bounds[axis]
```

**Results on all 38 cases:**
- **32/38 exact** (`|residual| < 1 mm`, mostly 0.00 mm exactly)
- **6/38 on mesh axis 2** instead of axis 1 (cases 0015, 0016, 0017, 0018, 0019, 0029)
- **6/38 with residual 1–31 mm** (0014, 0021, 0022, 0023, 0029, 0030, 0033)
- **Case 0014** is genuinely broken (−209 mm on best axis; no axis matches extent)

Mesh axis usage: 32 cases use axis 1, 6 use axis 2. Two export conventions in one dataset.

**Comparison to peak-anchored calibration** (in `build_couinaud_case.py`): Peak cross-section matching yields ~20 mm residual and 9/38 QA failures. The contour-derived z_shift is analytical, more accurate, and explains most of the QA failures (wrong axis choice).

## Quality Issues

### Artifacts in Existing Conversions

`annotations2niftimask/` (2 cases, legacy):
- **Mirrored left-right** (likely from a y-only flip)
- Contains shoulder artefact (same 19.3k–19.4k voxels)
- Not suitable as training reference

### Known Per-Case Issues

**Shoulder/axillary contamination:** Liver contours extend above the diaphragm (z > 874) in all cases, labelling clavicle/shoulder soft tissue as liver. 0.66% of voxel volume but 21% of contours. Volume-based outlier rejection removes it.

**Axis confusion:** Cases 0015–0019, 0029 export GLB meshes with non-standard z-axis mapping.

**QA gate failures (9/38 in legacy `build_couinaud_case.py`):** Mostly explained by z-shift residuals >20 mm (now resolved by contour-derived calibration) or axis mismatch.

## All 38 Case IDs

```
RMV2025_0001_CT_NCT_HBP_X    RMV2025_0009_CT_UTV_VEP_D     RMV2026_0027_CT_UTV_VEP_D
RMV2025_0002_MR_NCT_ARP_N    RMV2025_0010_CT_UTV_VEP_U     RMV2026_0028_CT_UTV_VEP_B
RMV2025_0003_CT_UTV_VEP_P    RMV2025_0011_CT_UTV_VEP_S     RMV2026_0029_CT_UTV_VEP_9
RMV2025_0004_CT_UTV_VEP_N    RMV2025_0012_CT_UTV_VEP_Q     RMV2026_0030_CT_UTV_VEP_Q
RMV2025_0005_CT_UTV_VEP_L    RMV2025_0013_CT_UTV_VEP_O     RMV2026_0031_CT_UTV_VEP_O
RMV2025_0006_CT_UTV_VEP_J    RMV2025_0014_CT_UTV_VEP_M     RMV2026_0032_CT_UTV_VEP_M
RMV2025_0007_CT_UTV_VEP_H    RMV2025_0015_CT_UTV_VEP_K     RMV2026_0033_CT_UTV_VEP_K
RMV2025_0008_CT_UTV_VEP_F    RMV2025_0016_CT_UTV_ARP_C     RMV2026_0034_CT_UTV_VEP_I
                              RMV2025_0017_CT_UTV_VEP_G     RMV2026_0035_CT_UTV_VEP_G
RMV2026_0020_CT_UTV_VEP_R    RMV2025_0018_CT_UTV_VEP_E     RMV2026_0036_CT_UTV_VEP_E
RMV2026_0021_CT_UTV_VEP_P    RMV2025_0019_CT_UTV_VEP_C     RMV2026_0037_CT_UTV_VEP_C
RMV2026_0022_CT_UTV_VEP_N                                   RMV2026_0038_CT_UTV_VEP_A
RMV2026_0023_CT_UTV_VEP_L
RMV2026_0024_CT_UTV_VEP_J
RMV2026_0025_CT_UTV_VEP_H
RMV2026_0026_CT_UTV_VEP_F
```

## Data Access Tools

See `data_preview.ipynb` (12 cells):

- `list_cases()` — enumerate case IDs
- `load_volume(directory, case)` — load a NIfTI from any subdirectory
- `load_case(case)` — returns (ct_array, affine, layers_dict)
- `contours_to_mask(case)` — rasterize cornerstone contours onto imagesTs grid
- `overlay_ortho(case)` and `overlay_axial(case)` — visualize annotations over CT
- `write_all_cases()` — batch-rasterize all 38 to disk (~35 s/case, ~20 min total)
- `solve_z_shift(case)` — analytical z-calibration from contours
