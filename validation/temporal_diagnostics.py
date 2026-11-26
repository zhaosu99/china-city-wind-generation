# -*- coding: utf-8 -*-
"""Rich diagnostics for the temporal-validation subsection (4.1.1).

Outputs:
  D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\validation44\temporal_diag_report.txt
"""
import io
import numpy as np
import pandas as pd
from config_paths import (
    M4, OUTD, OUTDIR, OUT, PROD23, PROD24, KB, KB_FULL, OFF, CAP, UTIL, CV_WIND, FULL_WIND, CV_GEN,
)


PY2CN = {"anhui":"安徽","beijing":"北京","chongqing":"重庆","fujian":"福建","gansu":"甘肃","guangdong":"广东",
 "guangxi":"广西","guizhou":"贵州","hainan":"海南","hebei":"河北","heilongjiang":"黑龙江","henan":"河南",
 "hubei":"湖北","hunan":"湖南","jiangsu":"江苏","jiangxi":"江西","jilin":"吉林","liaoning":"辽宁","neimenggu":"内蒙古",
 "ningxia":"宁夏","qinghai":"青海","shaanxi":"陕西","shandong":"山东","shanghai":"上海","shanxi":"山西",
 "sichuan":"四川","tianjin":"天津","xinjiang":"新疆","yunnan":"云南","zhejiang":"浙江"}
REGION = {
 "华北": ["北京","天津","河北","山西","内蒙古"],
 "东北": ["辽宁","吉林","黑龙江"],
 "华东": ["上海","江苏","浙江","安徽","福建","江西","山东"],
 "华中": ["河南","湖北","湖南"],
 "华南": ["广东","广西","海南"],
 "西南": ["重庆","四川","贵州","云南"],
 "西北": ["陕西","甘肃","青海","宁夏","新疆"],
}
SEASON_OF_MK = {1: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
                9: "SON", 10: "SON", 11: "SON", 12: "DJF"}
lines = []


def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat); m = den > 0
    return 100.0 * np.mean(np.abs(yhat[m] - y[m]) / den[m])


def r2(y, pred):
    return float(np.corrcoef(y, pred)[0, 1] ** 2)


def load_prod(path, factor=False):
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["cn"] = d["prov"].map(PY2CN); d = d.dropna(subset=["cn"]).copy()
    d["year"] = d["ym"].astype(str).str[:4].astype(int)
    d["month"] = d["ym"].astype(str).str[4:6].astype(int)
    if factor:
        cap = pd.read_csv(CAP, encoding="utf-8-sig")
        capf = cap[["省简称","年","月","cap_ratio"]].rename(columns={"省简称":"cn","年":"year","月":"month","cap_ratio":"cap_f"})
        util = pd.read_csv(UTIL, encoding="utf-8-sig")
        ucol = "factor" if "factor" in util.columns else "util_applied"
        uf = util[["省简称","年","月",ucol]].rename(columns={"省简称":"cn","年":"year","月":"month",ucol:"util_f"})
        f = capf.merge(uf, on=["cn","year","month"], how="outer")
        f["factor"] = f["cap_f"].fillna(1.0) * f["util_f"].fillna(1.0)
        d = d.merge(f[["cn","year","month","factor"]], on=["cn","year","month"], how="left")
        d["factor"] = d["factor"].fillna(1.0); d["gen_MWh"] *= d["factor"]
    d["mk"] = d["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return d.groupby(["cn","year","mk"], as_index=False)["gen_MWh"].sum()


m23 = load_prod(PROD23).rename(columns={"gen_MWh": "x"})
m24 = load_prod(PROD24, factor=True).rename(columns={"gen_MWh": "x"})
o = pd.read_csv(OFF, encoding="utf-8-sig"); o = o[o["value_MWh"] > 0]
o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else int(m))
obs = o.groupby(["cn","year","mk"], as_index=False)["value_MWh"].sum().rename(columns={"value_MWh": "y"})
kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")
for g in (m23, m24):
    g["K"] = g.cn.map(kb.K); g["B"] = g.cn.map(kb.B)

d23 = m23.merge(obs, on=["cn","year","mk"]).dropna(subset=["K"])
d23 = d23[(d23.year >= 2013) & (d23.year <= 2023)]
d24 = m24.merge(obs, on=["cn","year","mk"]).dropna(subset=["K"])
d24 = d24[d24.year >= 2024]
for d in (d23, d24):
    d["sim"] = (d.K * d.x + d.B) / 1e3
    d["obs"] = d.y / 1e3
    d["season"] = d.mk.map(SEASON_OF_MK)
    d["region"] = d.cn.map({c: r for r, cs in REGION.items() for c in cs})
EX = ["北京", "海南"]

# ---------------- 1. per-province tables
def prov_table(d, label):
    rows = []
    for cn, g in d.groupby("cn"):
        small = g[g.year <= 2015]
        rows.append(dict(prov=cn, n=len(g),
                         smape=smape(g.obs, g.sim), r2=r2(g.obs, g.sim),
                         ratio=g.sim.sum() / g.obs.sum(),
                         obs_mean=g.obs.mean(),
                         smape_early=smape(small.obs, small.sim) if len(small) > 2 else np.nan))
    return pd.DataFrame(rows).sort_values("smape")


IN = prov_table(d23, "in")
OOS = prov_table(d24[d24.cn.isin(set(d23.cn) - set(EX))], "oos")
lines.append("=== 1. Per-province, calibration window 2013-2023 (Fig. 11, 30 provinces) ===")
lines.append("SMAPE median %.2f mean %.2f min %.2f max %.2f" % (IN.smape.median(), IN.smape.mean(), IN.smape.min(), IN.smape.max()))
lines.append("R2 median %.3f mean %.3f min %.3f max %.3f" % (IN.r2.median(), IN.r2.mean(), IN.r2.min(), IN.r2.max()))
lines.append("ratio median %.4f min %.4f max %.4f" % (IN.ratio.median(), IN.ratio.min(), IN.ratio.max()))
lines.append("Best 6:\n" + IN.head(6).to_string(index=False, float_format=lambda v: "%.3f" % v))
lines.append("Worst 6:\n" + IN.tail(6).to_string(index=False, float_format=lambda v: "%.3f" % v))
lines.append("corr(mean monthly generation, SMAPE) = %.3f" % IN[["obs_mean", "smape"]].corr().iloc[0, 1])
lines.append("corr(mean monthly generation, SMAPE), log-size = %.3f" % np.corrcoef(np.log(IN.obs_mean), IN.smape)[0, 1])
lines.append("")
lines.append("=== 2. Per-province, out-of-sample 2024-2025 (Fig. 12, 28 provinces) ===")
lines.append("SMAPE median %.2f mean %.2f min %.2f max %.2f" % (OOS.smape.median(), OOS.smape.mean(), OOS.smape.min(), OOS.smape.max()))
lines.append("R2 median %.3f mean %.3f min %.3f max %.3f" % (OOS.r2.median(), OOS.r2.mean(), OOS.r2.min(), OOS.r2.max()))
lines.append("ratio median %.4f min %.4f max %.4f" % (OOS.ratio.median(), OOS.ratio.min(), OOS.ratio.max()))
lines.append("Best 6:\n" + OOS.head(6).to_string(index=False, float_format=lambda v: "%.3f" % v))
lines.append("Worst 6:\n" + OOS.tail(6).to_string(index=False, float_format=lambda v: "%.3f" % v))
lines.append("ratio extremes (|log ratio|):")
OOS["logratio"] = np.log(OOS.ratio)
lines.append(OOS.reindex(OOS.logratio.abs().sort_values(ascending=False).index).head(6)[["prov", "smape", "ratio", "r2"]].to_string(index=False, float_format=lambda v: "%.3f" % v))
lines.append("corr(log mean generation, OOS SMAPE) = %.3f" % np.corrcoef(np.log(OOS.obs_mean), OOS.smape)[0, 1])
lines.append("")
m = IN.merge(OOS, on="prov", suffixes=("_in", "_oos"))
m["dsmape"] = m.smape_oos - m.smape_in
m["dr2"] = m.r2_oos - m.r2_in
lines.append("=== 3. In-sample vs out-of-sample (same 28 provinces) ===")
lines.append("OOS SMAPE lower than in-sample: %d/28" % (m.dsmape < 0).sum())
lines.append("median delta SMAPE %.2f pp; IQR %.2f..%.2f" % (m.dsmape.median(), m.dsmape.quantile(.25), m.dsmape.quantile(.75)))
lines.append("largest OOS deterioration:\n" + m.sort_values("dsmape", ascending=False).head(6)[["prov", "smape_in", "smape_oos", "dsmape"]].to_string(index=False, float_format=lambda v: "%.2f" % v))
lines.append("largest OOS improvement:\n" + m.sort_values("dsmape").head(6)[["prov", "smape_in", "smape_oos", "dsmape"]].to_string(index=False, float_format=lambda v: "%.2f" % v))
lines.append("")
lines.append("=== 4. Annual pooled SMAPE / ratio (28-provinces set, 2013-2025) ===")
for y in range(2013, 2026):
    d = pd.concat([d23, d24]); d = d[(d.year == y)]
    if d.empty:
        continue
    d = d[~d.cn.isin(EX)] if y >= 2024 else d
    lines.append("%d: n=%3d  pooled SMAPE %6.2f  sim/obs %.3f" % (y, len(d), smape(d.obs, d.sim), d.sim.sum() / d.obs.sum()))
lines.append("")
lines.append("=== 5. Seasonal pooled SMAPE / ratio ===")
for tag, d in (("2013-2023", d23), ("2024", d24[d24.year == 2024]), ("2025", d24[d24.year == 2025])):
    dd = d[~d.cn.isin(EX)] if tag != "2013-2023" else d
    lines.append("-- %s --" % tag)
    for s in ("MAM", "JJA", "SON", "DJF"):
        x = dd[dd.season == s]
        lines.append("   %s: n=%4d  pooled SMAPE %6.2f  sim/obs %.3f" % (s, len(x), smape(x.obs, x.sim), x.sim.sum() / x.obs.sum()))
lines.append("")
lines.append("=== 6. Regional pooled SMAPE / ratio ===")
for tag, d in (("2013-2023", d23), ("2024-2025", d24[~d24.cn.isin(EX)])):
    lines.append("-- %s --" % tag)
    for r in REGION:
        x = d[d.region == r]
        if x.empty:
            continue
        lines.append("   %-4s: n=%4d  pooled SMAPE %6.2f  sim/obs %.3f  mean_n_prov %d" % (
            r, len(x), smape(x.obs, x.sim), x.sim.sum() / x.obs.sum(), x.cn.nunique()))
lines.append("")
lines.append("=== 7. Per-season per-province variability (in-sample) ===")
piv = d23.groupby(["cn", "season"]).apply(lambda g: smape(g.obs, g.sim), include_groups=False).unstack()
piv = piv[["MAM", "JJA", "SON", "DJF"]]
lines.append("seasonal SMAPE, median across provinces: " + ", ".join("%s %.1f" % (c, piv[c].median()) for c in piv.columns))
lines.append("provinces whose worst season is JJA: %d/30; SON: %d/30; DJF: %d/30; MAM: %d/30" % tuple(
    (piv.idxmax(axis=1) == s).sum() for s in ["JJA", "SON", "DJF", "MAM"]))
lines.append("")
lines.append("=== 8. Per-season per-province variability (out-of-sample) ===")
piv2 = d24[~d24.cn.isin(EX)].groupby(["cn", "season"]).apply(lambda g: smape(g.obs, g.sim), include_groups=False).unstack()
piv2 = piv2[["MAM", "JJA", "SON", "DJF"]]
lines.append("seasonal SMAPE, median across provinces: " + ", ".join("%s %.1f" % (c, piv2[c].median()) for c in piv2.columns))
lines.append("provinces whose worst season is JJA: %d/28; SON: %d/28; DJF: %d/28; MAM: %d/28" % tuple(
    (piv2.idxmax(axis=1) == s).sum() for s in ["JJA", "SON", "DJF", "MAM"]))
lines.append("")
lines.append("=== 9. 2024 vs 2025 per-province change (28 provinces) ===")
for y in (2024, 2025):
    d = d24[(d24.year == y) & (~d24.cn.isin(EX))]
    lines.append("%d: pooled SMAPE %.2f  sim/obs %.3f  n_prov %d" % (y, smape(d.obs, d.sim), d.sim.sum() / d.obs.sum(), d.cn.nunique()))
p24 = d24[(d24.year == 2024) & (~d24.cn.isin(EX))].groupby("cn").apply(lambda g: smape(g.obs, g.sim), include_groups=False)
p25 = d24[(d24.year == 2025) & (~d24.cn.isin(EX))].groupby("cn").apply(lambda g: smape(g.obs, g.sim), include_groups=False)
ch = (p25 - p24).sort_values()
lines.append("SMAPE 2025 < 2024 in %d/28 provinces; median change %.2f pp" % ((p25 < p24).sum(), (p25 - p24).median()))
lines.append("largest 2025 deterioration: " + ", ".join("%s %+.1f" % (k, v) for k, v in ch.tail(4).items()))
lines.append("largest 2025 improvement:  " + ", ".join("%s %+.1f" % (k, v) for k, v in ch.head(4).items()))

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("wrote", OUT)
print("\n".join(lines[:40]))
