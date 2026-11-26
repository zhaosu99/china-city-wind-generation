# Final M4 workflow: code provenance

This repository contains the final M4 calibration and reconstruction
workflow for the manuscript.  The earlier M2/M1 calibration scripts are
retained under `legacy/` for transparency; the final calibration path
does not import them.  A deprecated illustrative figure
(`figures/fig04_optimal_kb_example/`) keeps a local M2 helper copy and is
not part of the calibration pipeline.

## 1. Final calibration method

For each province and season the wind-speed calibration is selected as:

1. build the wind-speed SMAPE surface against the CDMet-constrained
   benchmark;
2. follow the one-dimensional valley `b*(k) = argmin_b W(k,b)`;
3. select the point on this valley with the minimum **wind-speed**
   SMAPE.

The generation-error surface is **not** used to choose `(k, b)`.  The
implementation is `pipeline/m4_selection.py`; the historical compatibility
name `pipeline/find_optimal_kb_full_data.py` re-exports it.

The production generation calibration `G_corr = K G + B` is fitted
separately on official province-month generation for 2013–2023 with
the non-negative-intercept constraint.  Cross-validation uses the
2013–2025 pool (143 periods), with 50 seasonally stratified splits
(29 test periods per split).

## 2. HPC production provenance

HPC root:

`/geogfs1/groups/geog_liulb/hongyu`

| Item | HPC path |
|---|---|
| M4/M2 comparison and M4 selection | `cv_v143/out_m2m4_2b/hpc_run_m2_m4_2b.py` |
| 143-period base/CV utilities | `cv_v143/out_m2m4_2b/run_cv_143.py` |
| M4 selection table | `cv_v143/out_m2m4_2b/M4/selection.csv` |
| Final production `(k,b)` | `m4prod/results/optimal_kb/optimal_kb.csv` |
| Final released `(K,B)` (post-Feb-fix) | `m4prod/data/03_params_and_validation/gen_correction_M4prod.csv` |
| Final city dataset | `m4prod/results/city_final/` |
| Final production job | `run_full.slurm` |
| Final production log | `m4prod/logs/m4prod_39884.out` |

The final production log (from the HPC 12-step panel version) shows:

```text
Enabled steps: 9:calibrated_gen
Skipped steps: 1..8, 10, 11, 12
Step 8/12 skipped: find_optimal_kb_full_data
Step 9/12 completed: calibrated_gen
JOB END ... rc=0
```

That is, the production run used the **precomputed final M4 parameters**
and did not execute the legacy generation-surface selection.  The public
`pipeline/panel.py` now has 11 steps (the unused validation step was
removed); `--steps 9` has the same meaning.

The released city dataset was then produced by
`drivers/build_final_city.py`, which applies:

- 1989–2004: no generation correction;
- 2005–2012: the national annual scaling in `parameters/pre2012_official.csv`;
- 2013–2023: the per-province affine `(K,B)`.

## 3. Parameter checksums

| File | SHA256 (prefix) | Note |
|---|---|---|
| `parameters/optimal_kb_M4.csv` | `d4778452f54582ee` | 120 rows; exact `k`, `b`, `w_smape` match HPC `M4/selection.csv` (max abs diff 0.0) |
| `parameters/gen_correction_M4prod_final.csv` | `8cbdd8b3111edfe0` | released production `(K,B)`; byte-identical to the post-Feb-fix HPC file |
| HPC `m4prod/gen_correction_M4prod.csv` | `03f716ece73140e0` | **pre-Feb-fix** file; differs in the constrained rows (Jiangxi/Henan negative intercepts) |
| HPC `M4/gen_correction.csv` | `75295276dd5de7c7` | production-style unconstrained fit from the M2/M4 comparison run; **not** the released file and not a CV-fold output |

The released production file is the post-Feb-fix
`m4prod/data/03_params_and_validation/gen_correction_M4prod.csv`
(`8cbdd8b3...`).  The older `m4prod/gen_correction_M4prod.csv` and the
CV output should not be substituted for it.  The Feb-alignment fix
updated the official province-month alignment; running
`drivers/fit_generation_calibration.py` on the older pre-Feb inputs
reproduces the pre-Feb `3128...` table, not the released `8CBD...`
table, so the documented input version is part of the provenance.

## 4. Code map

| Manuscript step | Public entry point |
|---|---|
| Build wind-valley paths | `drivers/build_wind_paths.py` |
| M4 wind-only `(k,b)` selection | `pipeline/m4_selection.py` |
| 143-period CV and selection validation | `drivers/cv_m4.py` |
| Production `(K,B)` fit | `drivers/fit_generation_calibration.py` |
| Calibrated generation (intermediate; stage 9) | `pipeline/panel.py --steps 9` |
| Final 1989–2023 city dataset | `drivers/build_final_city.py` |
| Final correction job | `workflows/run_finalize.slurm` |
| 2024–2025 calibrated generation | `workflows/run_2425.slurm` |
| Wind-calibrated production job | `workflows/run_full.slurm` |

## 5. Legacy code

The following files are deliberately **not** on the final calibration
path.  They preserve the old M2 workflows for reviewers who want to
inspect the difference:

- `legacy/m2_find_optimal_kb_full_data.py`
- `legacy/m2_kb_validation.py`
- `legacy/m2_kb_evaluation.py`
- `legacy/find_optimal_kb_with_m2.py`
- `legacy/refit_and_cv_2013_2023.py`
- `legacy/refit_constrained_2013_2023.py`
- `legacy/patch_cv_constraint_2013_2023.py`

They should not be cited as the final method.  The figure script
`figures/fig04_optimal_kb_example/fig04_province_optimal_kb.py` still
uses the local `figures/_lib` copy of the M2 helper for an illustrative
surface; it is not part of the calibration pipeline.

## 6. Validation performed

- M4 selection on all 120 wind-valley paths: exact match to
  `parameters/optimal_kb_M4.csv` for `k`, `b` and `w_smape`.
- The M4 CLI on the HPC wind-SMAPE surfaces reproduced all 120 final
  `(k,b)` values (max `w_smape` difference ~2e-9).
- One full 2013–2025 CV split on the HPC enumeration data completed
  with 30 provinces and 120 province-season selections.
- Unit tests cover the wind-only selector, the non-negative intercept
  constraint, the 143-period grid, and a synthetic end-to-end CV.
