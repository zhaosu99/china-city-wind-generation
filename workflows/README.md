# Workflows

The final M4 workflow has five stages.  Stages 0–3 are the calibration;
stage 4 produces the released city-level dataset.

## Inputs

Set these variables (all are plain files/directories):

| Variable | Content |
|---|---|
| `WINDPAPER_WIND_SMAPE_DIR` | `<prov>_<season>_w_smape_enums.csv` wind-error surfaces |
| `WINDPAPER_GEN_SMAPE_DIR` | `<prov>_<season>_full_gen_smape.csv` files, used only to enumerate province/season pairs |
| `WINDPAPER_SURF_DIR` | optional `<prov>_<season>.npz` wind-surface files with `periods`, `acc`, `cnt` |
| `WINDPAPER_PATH_DIR` | `<prov>_<season>.npz` wind-valley paths (`k`, `b`, `wind_at_path`); built from `WINDPAPER_SURF_DIR` if absent |
| `WINDPAPER_ENUM_DIR` | `<prov>_<season>_monthly_gen_enums.csv` generation enumerations |
| `WINDPAPER_MODEL_MONTHLY` | production model monthly CSV (`prov`, `ym`, `gen_MWh`) |
| `WINDPAPER_OFFICIAL_MONTHLY` | official monthly CSV (`cn`, `year`, `month`, `value_MWh`) |
| `WINDPAPER_RESULTS` | reconstruction output root |
| `WINDPAPER_RESULTS_2425` | output root for 2024–2025 |

## Calibration

```bash
export WINDPAPER_ROOT=/path/to/repo
bash workflows/run_calibration.sh
```

Stages:

0. optional path build: `drivers/build_wind_paths.py` (when
   `WINDPAPER_SURF_DIR` is set and `WINDPAPER_PATH_DIR` is absent);
1. M4 wind-only selection: `pipeline/m4_selection.py`;
2. 143-period CV: `drivers/cv_m4.py`;
3. production `(K,B)` fit: `drivers/fit_generation_calibration.py`.

This produces:

- `$WINDPAPER_CALIBRATION_OUT/selection/optimal_kb.csv` — M4 wind-only
  `(k,b)`; the paper table is `parameters/optimal_kb_M4.csv`;
- `$WINDPAPER_CALIBRATION_OUT/cv/` — 143-period CV outputs;
- `$WINDPAPER_CALIBRATION_OUT/gen_correction.csv` — production `(K,B)`;
  the released table is `parameters/gen_correction_M4prod_final.csv`
  (post-Feb-fix).

## Production

The production consists of two stages.  First, place the final selection
table at:

```text
$WINDPAPER_RESULTS/optimal_kb/optimal_kb.csv
```

and run the wind-calibrated generation:

```bash
sbatch workflows/run_full.slurm
```

This executes `pipeline/panel.py --steps 9` and produces
`calibrated_gen`; it does **not** apply `(K,B)` or the 2005–2012 national
scaling.  The final city dataset is then built with:

```bash
python drivers/build_final_city.py \
  --src "$WINDPAPER_RESULTS/calibrated_gen/gen/city" \
  --dst "$WINDPAPER_RESULTS/city_final"
```

`build_final_city.py` reads
`parameters/gen_correction_M4prod_final.csv` and
`parameters/pre2012_official.csv`.  On Slurm, the same stage is
`workflows/run_finalize.slurm`.

The steps 1–3 outputs (`farm_raw_w`, `farm_power_curves` and the
inventory/ERA5/CDMet inputs) must exist before `run_full.slurm` is
submitted.  For 2024–2025, generate the 2024–25 raw wind series and set
the paths in `workflows/run_2425.slurm`, then submit it.

## Why this layout

The old M2 scripts used the generation-error surface to choose the
candidate path.  They are preserved under `legacy/` only for
transparency.  The final M4 path is wind-only and is implemented in
`pipeline/m4_selection.py`.

See `docs/provenance.md` for the HPC evidence and checksums.
