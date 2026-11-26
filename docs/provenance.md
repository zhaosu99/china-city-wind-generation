# Final M4 workflow: code provenance

This repository contains the final M4 calibration and reconstruction
workflow for the manuscript.  The earlier M2/M1 scripts are retained
only under `legacy/` for transparency; they are not imported by the
final pipeline.

## 1. Final calibration method

For each province and season the wind-speed calibration is selected as:

1. build the wind-speed SMAPE surface against the CDMet-constrained
   benchmark;
2. follow the one-dimensional valley `b*(k) = argmin_b W(k,b)`;
3. select the point on this valley with the minimum **wind-speed**
   SMAPE.

The generation-error surface is **not** used to choose `(k, b)`.  The
implementation is `pipeline/m4_selection.py`; the compatibility name
`pipeline/find_optimal_kb_full_data.py` re-exports it.

The production generation calibration `G_corr = K G + B` is fitted
separately on official province-month generation for 2013–2023 with
the non-negative-intercept constraint.  Cross-validation uses the
2013–2025 pool (143 periods), with 50 seasonal 80/20 splits.

## 2. HPC production provenance

HPC root:

`/geogfs1/groups/geog_liulb/hongyu`

| Item | HPC path |
|---|---|
| M4/M2 comparison and M4 selection | `cv_v143/out_m2m4_2b/hpc_run_m2_m4_2b.py` |
| 143-period base/CV utilities | `cv_v143/out_m2m4_2b/run_cv_143.py` |
| M4 selection table | `cv_v143/out_m2m4_2b/M4/selection.csv` |
| Final production `(k,b)` | `m4prod/results/optimal_kb/optimal_kb.csv` |
| Final production `(K,B)` | `m4prod/gen_correction_M4prod.csv` |
| Final production job | `run_full.slurm` |
| Final production log | `m4prod/logs/m4prod_39884.out` |

The final production log shows:

```text
Enabled steps: 9:calibrated_gen
Skipped steps: 1..8, 10, 11, 12
Step 8/12 skipped: find_optimal_kb_full_data
Step 9/12 completed: calibrated_gen
JOB END ... rc=0
```

That is, the production run used the **precomputed final M4 parameters**
and did not execute the legacy generation-surface selection.

## 3. Parameter checksums

| File | SHA256 (prefix) | Note |
|---|---|---|
| `parameters/optimal_kb_M4.csv` | `d4778452f54582ee` | 120 rows; exact `k`, `b`, `w_smape` match HPC `M4/selection.csv` (max abs diff 0.0) |
| `parameters/gen_correction_M4prod_final.csv` | `3128ffb1e24717f8` | production `(K,B)`; equals HPC `m4prod/gen_correction_M4prod.csv` plus the `mode` column |
| HPC `M4/gen_correction.csv` | `75295276dd5de7c7` | CV run output; **not** the production file |

The production parameters are the ones used for the released
1989–2023 series; the separate `M4/gen_correction.csv` is a CV output
and should not be confused with `gen_correction_M4prod_final.csv`.

## 4. Code map

| Manuscript step | Public entry point |
|---|---|
| M4 wind-only `(k,b)` selection | `pipeline/m4_selection.py` |
| 143-period CV and selection validation | `drivers/cv_m4.py` |
| Production `(K,B)` fit | `drivers/fit_generation_calibration.py` |
| Calibrated generation (production stage 9) | `pipeline/panel.py --steps 9` |
| 2024–2025 application | `workflows/run_2425.slurm` |
| Final 1989–2023 production | `workflows/run_full.slurm` |

## 5. Legacy code

The following files are deliberately **not** on the final code path.
They preserve the old M2 workflows for reviewers who want to inspect
the difference:

- `legacy/m2_find_optimal_kb_full_data.py`
- `legacy/m2_kb_validation.py`
- `legacy/refit_and_cv_2013_2023.py`
- `legacy/refit_constrained_2013_2023.py`

They should not be cited as the final method.

## 6. Validation performed

- M4 selection on all 120 wind-valley paths: exact match to
  `parameters/optimal_kb_M4.csv` for `k`, `b` and `w_smape`.
- One full 2013–2025 CV split on the HPC enumeration data completed
  with 30 provinces and 120 province-season selections.
- Unit tests cover the wind-only selector, the non-negative intercept
  constraint, the 143-period grid and the production fit.
