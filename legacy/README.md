# Legacy M2 scripts (not used in the final paper)

This directory is preserved for audit only.  The final manuscript uses
the M4 wind-only selection in `pipeline/m4_selection.py`.

| File | Original role |
|---|---|
| `m2_find_optimal_kb_full_data.py` | M2: build a candidate path from the **generation-error surface**, then choose the minimum wind error on that path |
| `m2_kb_validation.py` | old M2 validation / resampling script |
| `m2_kb_evaluation.py` | old M2 evaluation helpers |
| `refit_and_cv_2013_2023.py` | old 2013–2023 refit/CV script with hard-coded local paths |
| `refit_constrained_2013_2023.py` | old constrained `(K,B)` fit with hard-coded local paths |
| `patch_cv_constraint_2013_2023.py` | helper that patched the old refit script; obsolete |

These scripts are not imported by `pipeline/panel.py`, `pipeline/m4_selection.py`
or `drivers/cv_m4.py`.  Do not cite them as the final method.
