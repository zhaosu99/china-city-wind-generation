# Workflows

The final M4 workflow has four stages.  The first three are independent
of the city-level reconstruction; the fourth consumes the resulting
parameters.

## Inputs

Set these variables (all are plain files/directories):

| Variable | Content |
|---|---|
| `WINDPAPER_WIND_SMAPE_DIR` | `<prov>_<season>_w_smape_enums.csv` wind-error surfaces |
| `WINDPAPER_GEN_SMAPE_DIR` | `<prov>_<season>_full_gen_smape.csv` files, used only to enumerate province/season pairs |
| `WINDPAPER_PATH_DIR` | `<prov>_<season>.npz` wind-valley paths (`k`, `b`, `wind_at_path`) |
| `WINDPAPER_ENUM_DIR` | `<prov>_<season>_monthly_gen_enums.csv` generation enumerations |
| `WINDPAPER_MODEL_MONTHLY` | production model monthly CSV (`prov`, `ym`, `gen_MWh`) |
| `WINDPAPER_OFFICIAL_MONTHLY` | official monthly CSV (`cn`, `year`, `month`, `value_MWh`) |
| `WINDPAPER_RESULTS` | reconstruction output root |

## Calibration

```bash
export WINDPAPER_ROOT=/path/to/repo
bash workflows/run_calibration.sh
```

This produces:

- `$WINDPAPER_CALIBRATION_OUT/selection/optimal_kb.csv` — M4 wind-only
  `(k,b)`; the paper table is `parameters/optimal_kb_M4.csv`;
- `$WINDPAPER_CALIBRATION_OUT/cv/` — 143-period CV outputs;
- `$WINDPAPER_CALIBRATION_OUT/gen_correction.csv` — production `(K,B)`;
  the paper table is `parameters/gen_correction_M4prod_final.csv`.

## Production

Place the final selection table at:

```text
$WINDPAPER_RESULTS/optimal_kb/optimal_kb.csv
```

and run:

```bash
sbatch workflows/run_full.slurm
```

The production run executes `pipeline/panel.py --steps 9`; it does not
execute the legacy step-8 selector.  For 2024–2025, set the paths in
`workflows/run_2425.slurm` and submit it.

## Why this layout

The old M2 scripts used the generation-error surface to choose the
candidate path.  They are preserved under `legacy/` only for
transparency.  The final M4 path is wind-only and is implemented in
`pipeline/m4_selection.py`.

See `docs/provenance.md` for the HPC evidence and checksums.
