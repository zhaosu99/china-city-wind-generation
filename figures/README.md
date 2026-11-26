# Figure code

Final figure scripts are grouped by manuscript figure.  They are
independent of the central reconstruction pipeline and read the
post-processed city/province datasets.

| Directory | Manuscript figure | Notes |
|---|---|---|
| `fig01_wind_farm_map/` | Figure 1 | capacity map |
| `fig02_workflow/` | Figure 2 | workflow PPT builder |
| `fig03_power_curves/` | Figure 3 | power curves |
| `fig04_optimal_kb_example/` | Figure 4 | final M4 wind-surface examples |
| `fig05_generation_dynamics/` | Figures 5–6 | generation dynamics / provincial shares |
| `fig07_city_in_province/` | Figure 7 | city shares within provinces |
| `fig08_city_ranking/` | Figure 8 | city ranking |
| `fig09_sunburst_grid/`, `fig10_sunburst_type/` | Figures 9–10 | sunbursts |
| `fig11_15_final/` | Figures 11–15 | validation scatters and parameter stability |
| `_lib/` | shared | `config_paths.py`, `figure_paths.py`, `grid_map.py`, `indices.py` |

The figure scripts set paths through `_lib/config_paths.py` and
`_lib/figure_paths.py`; override with `WINDPAPER_PLOT_DATA`,
`WINDPAPER_PAPER_ROOT`, `WINDPAPER_M4PROD_DIR`, `WINDPAPER_FIG_OUT`,
`WINDPAPER_ADMIN_DATA` and related variables.  Input data, administrative
boundaries and proprietary inventory are not redistributed.

The old M2 Figure 4 script is archived in
`legacy/fig04_m2_province_optimal_kb.py`; it is not the final figure.
