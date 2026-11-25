# -*- coding: utf-8 -*-
"""Final numbers: new dataset 2023 breakdown vs old figure, plus fig9 coverage."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

MOD = Path(r"D:\HKU PhD\wind data paper\plot\sunburst\fig10_generation_sunburst.py")
ROOT = Path(r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\sunburst_root")
CITY = Path(r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\data\01_city_final_1989_2023")
ALT = Path(r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\data\02_city_2024_2025")
RUN = Path(r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod")

spec = importlib.util.spec_from_file_location("fig10_mod", MOD)
m = importlib.util.module_from_spec(spec)
sys.modules["fig10_mod"] = m
spec.loader.exec_module(m)

attr_map = {}
for p in ALT.glob("*/full_year/gen.nc"):
    ds = xr.open_dataset(p)
    if ds.attrs.get("province_short_name") and ds.attrs.get("city_short_name"):
        attr_map[p.parents[1].name] = (str(ds.attrs["province_short_name"]), str(ds.attrs["city_short_name"]))
    ds.close()

_orig = m._open_dataset_lazy


def patched(path):
    ds = _orig(path)
    if "province_short_name" not in ds.attrs:
        prov, city = attr_map[Path(path).parent.parent.name]
        ds.attrs["province_short_name"] = prov
        ds.attrs["city_short_name"] = city
    return ds


m._open_dataset_lazy = patched

new = m.build_generation_sunburst_dataframe(root_dir=ROOT, time_selector=2023, language="en")
typ = m.build_generation_type_sunburst_dataframe(root_dir=ROOT, time_selector=2023, language="en")

old_grid = pd.read_csv(RUN / "old_nodes_fig10_generation_sunburst_en_year_2023.csv", encoding="utf-8-sig")
old_typ = pd.read_csv(RUN / "old_nodes_fig10_generation_type_sunburst_en_year_2023.csv", encoding="utf-8-sig")
# empty-string parents become NaN on CSV round-trip
old_grid["parent"] = old_grid["parent"].fillna("")
old_typ["parent"] = old_typ["parent"].fillna("")

tot_new = new["generation_mwh"].sum() / 1e6
tot_old = old_grid.loc[old_grid["parent"] == "", "value"].sum() / 1e6
print("=== 2023 total ===")
print(f"NEW {tot_new:,.1f} TWh ({len(new)} cities) | OLD {tot_old:,.1f} TWh -> {100*(tot_new/tot_old-1):+.2f}%")

g_new = (new.groupby("grid")["generation_mwh"].sum() / 1e6)
old_roots = old_grid[old_grid["parent"] == ""].copy()
old_roots["grid"] = old_roots["label"].str.replace("<br>", " ", regex=False)
g_old = old_roots.set_index("grid")["value"] / 1e6
cmp = pd.DataFrame({"new": g_new, "old": g_old})
cmp["rel%"] = (cmp["new"] / cmp["old"] - 1) * 100
cmp = cmp.sort_values("new", ascending=False)
print("\n=== grid (TWh) ===")
print(cmp.round(1).to_string())
print("\nnew grid shares %:")
print((100 * g_new / g_new.sum()).round(1).sort_values(ascending=False).to_string())

t_new = typ.groupby("type")["generation_mwh"].sum() / 1e6
t_old = old_typ[old_typ["parent"] == ""].set_index("label")["value"] / 1e6
print("\n=== type (TWh) ===")
print(pd.DataFrame({"new": t_new, "old": t_old}).round(1).to_string())
print("offshore share: new %.1f%% | old %.1f%%" % (100 * t_new["Offshore"] / t_new.sum(), 100 * t_old["Offshore"] / t_old.sum()))

p_new = (new.groupby("province")["generation_mwh"].sum() / 1e6).sort_values(ascending=False)
print("\n=== provinces top 12 (new, TWh) ===")
print(p_new.head(12).round(1).to_string())

print("\n=== top 20 cities (fig9 basis, GWh 2023) ===")
top = new.sort_values("generation_mwh", ascending=False).head(20)
for i, r in enumerate(top.itertuples(), 1):
    print(f"{i:2d}. {r.city:<16s} {r.province:<18s} {r.generation_mwh/1000:9,.0f} GWh")

print("\n=== fig9: cities with data per year ===")
rows = {}
for p in sorted(CITY.glob("*/full_year/gen.nc")):
    ds = xr.open_dataset(p)
    s = ds["total_monthly_gen"].to_series().resample("YE").sum()
    ds.close()
    for idx, val in s.items():
        rows.setdefault(idx.year, 0)
        if pd.notna(val) and val > 0:
            rows[idx.year] += 1
for y in sorted(rows):
    print(f"  {y}: {rows[y]} cities")

cmp.round(2).to_csv(RUN / "final_grid_compare.csv", encoding="utf-8-sig")
