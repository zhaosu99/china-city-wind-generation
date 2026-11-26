# China city-level hourly wind power generation reconstruction (1989–2023)

Code accompanying the manuscript **“Hourly Reconstruction of Historical
Wind Power Generation for China's Cities”**.

The reconstruction produces hourly, city-level wind generation for
mainland China (1989–2023) from ERA5, the CDMet gridded meteorological
data, a wind-farm inventory, monthly city capacity records and official
provincial generation statistics.

## Calibration: final M4 method

The final calibration has two separate stages.

### 1. Wind-speed calibration (M4, wind-only)

For each province and season, a smooth wind-speed SMAPE surface
`W(k,b)` is built against the CDMet-constrained benchmark.  The
one-dimensional valley `b*(k) = argmin_b W(k,b)` is followed and the
point with the minimum **wind-speed** SMAPE is selected.

**Generation data are not used to select `(k,b)`.**  The old M2
generation-surface selector is archived under `legacy/`; the final
calibration path does not import it.  One deprecated illustrative figure
(`figures/fig04_optimal_kb_example/`) still contains a local copy of the
M2 helper; it is not part of the calibration pipeline.

Implementation:

```text
pipeline/m4_selection.py
pipeline/find_optimal_kb_full_data.py   # compatibility re-export
```

Final table: `parameters/optimal_kb_M4.csv` (120 rows = 30 calibration
units × 4 seasons).

### 2. Generation-level calibration (2B)

For each province, `G_corr = K G + B` is fitted to official
province-month generation over 2013–2023.  The intercept is constrained
to be non-negative.  The pre-2013 and 2005–2012 branches are described
in the manuscript.

Implementation:

```text
drivers/fit_generation_calibration.py
```

Final table: `parameters/gen_correction_M4prod_final.csv`.

### Cross-validation

The CV uses the final 143-period grid (2013–2025; January–February
merged) and 50 seasonal 80/20 splits.  The M4 selection is fixed before
each split; `(K,B)` is re-fitted on each training fold.

```text
drivers/cv_m4.py
drivers/cv_base.py
```

## Repository layout

```text
pipeline/       core reconstruction modules and the M4 selector
drivers/        CV, production generation fit and city-level drivers
validation/     §4.4 out-of-sample validation and Figures 11-15 code
figures/        final figure-rendering scripts (Figures 1-15)
workflows/      calibration, validation and production entry points
parameters/     final M4 parameter tables
tests/          unit tests for the M4 selector, CV grid and fit
legacy/         old M2 scripts, kept only for audit; not on the final path
docs/           provenance and code map
```

## Reproducing the calibration

The expected input layout and commands are in
[`workflows/README.md`](workflows/README.md).  In short:

```bash
export WINDPAPER_ROOT="$PWD"
bash workflows/run_calibration.sh
```

The important checks are:

1. `pipeline/m4_selection.py` reads only the wind-error surface and
   derives the valley from it; generation CSVs are used only to
   enumerate province/season pairs.
2. `drivers/cv_m4.py` uses 2013–2025 and 50 seasonally stratified
   splits (29 test periods each).
3. `drivers/fit_generation_calibration.py` uses the production
   2013–2023 fit and enforces `B >= 0`.

## Production

The final production has two stages.  First, `pipeline/panel.py --steps 9`
applies the M4 wind calibration and writes `calibrated_gen`:

```bash
sbatch workflows/run_full.slurm
```

Second, `drivers/build_final_city.py` applies the generation-level
calibration to produce `city_final`:

```bash
python drivers/build_final_city.py
```

The 2024–2025 application is `workflows/run_2425.slurm`.

## Validation and figures

The §4.4 out-of-sample validation (2013–2023, 2024, 2025) and the
Figures 11–15 code are in `validation/`; the complete final figure code
for Figures 1–15 is under `figures/`.  Run the validation/figures with:

```bash
bash workflows/run_validation.sh
```

Inputs and output locations are in `validation/README.md` and
`figures/README.md`.

## Final parameter provenance

| File | Meaning |
|---|---|
| `parameters/optimal_kb_M4.csv` | final M4 `(k,b)`; matches the HPC M4 selection table exactly |
| `parameters/gen_correction_M4prod_final.csv` | released production `(K,B)` for 2013–2023 (post-Feb-fix file) |
| `parameters/pre2012_official.csv` | national annual totals used for 2005–2012 |

See [`docs/provenance.md`](docs/provenance.md) for the HPC paths,
final-production log, SHA256 checksums and the distinction between the
released production `(K,B)` table and the separate CV output.

## Validation

- The M4 selector reproduces all 120 final `(k,b)` values from the
  wind-valley paths (max difference 0.0 for `k`, `b`, `w_smape`).
- A full 2013–2025 CV split on the HPC enumeration data completed with
  30 provinces and 120 province-season selections.
- `python -m pytest -q tests` covers the wind-only selection,
  non-negative intercept, 143-period grid and production fit.

## Data and code availability

The reconstructed dataset is archived separately; this repository does
not redistribute ERA5, CDMet, the wind-farm inventory or official
statistics.  Input paths are configured through `WINDPAPER_*`
environment variables (`config.example.env`).

## License and citation

Code: MIT (`LICENSE`).  Dataset: CC BY 4.0.  Citation metadata:
`CITATION.cff`.
