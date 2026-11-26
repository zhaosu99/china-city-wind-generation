# Parameter tables

| File | Status |
|---|---|
| `optimal_kb_M4.csv` | final M4 wind-only `(k,b)`; 120 rows; SHA256 prefix `D4778452F54582EE` |
| `gen_correction_M4prod_final.csv` | **released** production `(K,B)` after the Feb-alignment fix; SHA256 prefix `8CBDD8B3111EDFE0`; 30 rows |
| `pre2012_official.csv` | national annual wind generation for 2005–2012, in 亿 kWh; used by `drivers/build_final_city.py` |

The released `(K,B)` table is the post-Feb-fix file
`m4prod/data/03_params_and_validation/gen_correction_M4prod.csv`.
The older pre-Feb constrained table has prefix `3128FFB1...`; it is
**not** the released table.  `drivers/fit_generation_calibration.py`
implements the same constrained fit, but its inputs must use the
Feb-aligned official province-month data to reproduce `8CBD...`.
