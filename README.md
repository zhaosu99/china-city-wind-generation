# China city-level hourly wind power generation reconstruction (1989-2023)

Code repository accompanying the manuscript **"Hourly Reconstruction of Historical Wind Power Generation for China's Cities"**.

The pipeline reconstructs hourly, city-level wind power generation for mainland China from 1989 to 2023 by combining ERA5 reanalysis, the CDMet gridded meteorological dataset, a wind-farm inventory with turbine specifications, and official capacity and generation statistics. The reconstruction has two calibration stages: (i) a province- and season-specific affine calibration of hub-height wind speed against a CDMet-constrained benchmark (2013-2020), and (ii) a generation-level affine calibration against official national annual totals (2005-2012) and provincial monthly totals (2013-2023).

## Reconstruction at a glance

- 304 cities, 31 provincial-level units (32 statistical units after splitting Inner Mongolia into west and east)
- 1989-2023, hourly resolution, onshore and offshore wind
- 2023 national generation: 755.26 TWh
- Calibration parameters: 30 calibration units with official monthly generation statistics

## Repository structure

```
pipeline/      Core reconstruction modules (wind interpolation, density correction,
               power curves, farm/city aggregation, calibration, validation)
drivers/       Driver and post-processing scripts (city series, 2B scaling,
               constrained refit, cross-validation, number checks)
figures/       Scripts used to produce the figures in the manuscript
parameters/    Calibration parameter tables reported in the appendix
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The code was developed with Python 3.10+ and tested on Windows and Linux. `geopandas`, `netCDF4` and `xarray` require their usual system dependencies (GDAL/GEOS/PROJ, HDF5, netCDF).

## Input data

The repository does **not** redistribute input data. The pipeline expects the following inputs (paths are configured through environment variables, see `pipeline/config.py`):

- ERA5 hourly 10 m and 100 m wind, 2 m temperature and dew point, surface pressure and geopotential
- CDMet daily gridded 10 m wind (used as the independent wind-speed benchmark)
- Wind-farm inventory: coordinates, commissioning year, capacity, turbine model
- City-level monthly cumulative installed capacity
- Provincial monthly wind generation and national annual generation statistics
- Provincial annual wind curtailment rates

Data sources and processing are described in the manuscript and its appendices. The reconstructed dataset is archived separately (Zenodo DOI to be added here upon publication).

## Configuration

`pipeline/config.py` reads six environment variables and falls back to cluster defaults:

```
WINDPAPER_ROOT        project root
WINDPAPER_DATA        input data root
WINDPAPER_RESULTS     output root
WINDPAPER_CURVE_DIR   power-curve directory
WINDPAPER_ERA5_DIR    ERA5 input directory
WINDPAPER_CDMET_DIR   CDMet input directory (optional)
```

For a local run, set `WINDPAPER_DATA`, `WINDPAPER_RESULTS`, `WINDPAPER_CURVE_DIR`, `WINDPAPER_ERA5_DIR` and (optionally) `WINDPAPER_CDMET_DIR`. See `config.example.env`.

## Pipeline outline

1. `pipeline/horizontal_interp.py`, `pipeline/vars_computation.py` - interpolate ERA5 fields to farm sites, vertical extrapolation and air-density correction.
2. `pipeline/power_curve.py`, `pipeline/farm_power_curves.py` - build and smooth turbine-model power curves.
3. `pipeline/aggr_farm.py`, `pipeline/aggr_farm_aux.py`, `pipeline/capacity.py` - convert wind speed to generation, apply technical lifetime and capacity factors.
4. `pipeline/find_optimal_kb.py`, `pipeline/find_optimal_kb_full_data.py`, `pipeline/kb_batch_aggr.py`, `pipeline/kb_validation.py` - estimate wind-speed calibration parameters (k, b) and the response-surface path.
5. `pipeline/final_gen.py`, `pipeline/no_kb_gen.py` - produce calibrated hourly generation.
6. `drivers/build_final_city.py`, `drivers/apply_2b_city.py` - assemble city-level final files and apply the generation calibration.
7. `drivers/refit_and_cv.py`, `drivers/refit_constrained.py` - refit generation calibration with the non-negative intercept constraint and run the resampling/cross-validation experiments.

Each module exposes command-line help (`python pipeline/<module>.py --help`).

## Calibration parameters

`parameters/optimal_kb_M4.csv` contains the province-season wind-speed parameters (k, b) and `parameters/gen_correction_M4prod_final.csv` the generation calibration parameters (K*, B*) for the 30 calibration units. `parameters/pre2012_official.csv` contains the national annual totals used for the 2005-2012 scaling.

## Validation

The manuscript reports in-sample (2013-2023), out-of-sample (2024-2025), and resampling validation. The scripts used for the resampling experiment and the parameter refits are in `drivers/`.

## License

Code: MIT License (see `LICENSE`). The reconstructed dataset is released separately under CC BY 4.0.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`). The archived dataset DOI will be added here once available.

## Contact

Issues and questions: please open a GitHub issue.
