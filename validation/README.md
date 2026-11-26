# Out-of-sample validation (§4.4) and Figures 11–15

This directory contains the code used for the 2024–2025 out-of-sample
validation and the associated figures.

## Inputs

All paths are resolved by `config_paths.py`; override with environment
variables:

| Variable | Meaning |
|---|---|
| `WINDPAPER_M4PROD_DIR` | directory with `prov_month_from_city.csv`, `prov_month_2425.csv` |
| `WINDPAPER_MODEL_MONTHLY` | 2013–2023 model monthly CSV |
| `WINDPAPER_MODEL_MONTHLY_2425` | 2024–2025 model monthly CSV |
| `WINDPAPER_KB` | released production `(K,B)` table |
| `WINDPAPER_OFFICIAL_MONTHLY` | official province-month generation |
| `WINDPAPER_CAP_ADJ` | 2024–2025 capacity adjustment table |
| `WINDPAPER_UTILIZATION` | 2024–2025 utilization table |
| `WINDPAPER_VALIDATION_OUT` | output directory |
| `WINDPAPER_FIG_OUT` | figure output directory |

The official monthly and capacity/utilization inputs are not
redistributed in this repository.

## Scripts

| Script | Role |
|---|---|
| `compute_validation.py` | 2013–2023 / 2024 / 2025 validation table (noKB vs full calibration) |
| `fig11_data.py` | data assembly for the Figure 11 panel |
| `make_fig11_12.py` | Figures 11–12: in-sample and 2024–2025 scatters |
| `make_fig11_nokb.py` | Figure 11/12 no-K/B comparison |
| `make_fig13_14.py` | Figures 13–14: wind and generation parameter stability |
| `make_fig15.py` | Figure 15: holdout SMAPE boxplots |
| `honest_monthly_errors.py` | monthly error diagnostics |
| `temporal_diagnostics.py` | temporal validation diagnostics |
| `kb_last5y.py` | last-5-year K/B sensitivity |
| `holdout_check.py` | pseudo-out-of-sample 2013–2020 → 2021–2023 check |

Run from the repository root with:

```bash
export PYTHONPATH="$PWD/validation:$PYTHONPATH"
python validation/compute_validation.py
python validation/fig11_data.py
python validation/make_fig11_12.py
python validation/make_fig13_14.py
python validation/make_fig15.py
```

The scripts write to `$WINDPAPER_VALIDATION_OUT` and
`$WINDPAPER_FIG_OUT`; they do not write into the repository.
