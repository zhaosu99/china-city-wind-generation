# -*- coding: utf-8 -*-
"""Honest check: in-sample (2013-2023) monthly errors, province by province / month by month,
vs the pooled totals used in panel (d). Plus the same stats for 2024/2025 out-of-sample."""
import os
import numpy as np
import pandas as pd
from config_paths import (
    M4, OUTD, OUTDIR, OUT, PROD23, PROD24, KB, KB_FULL, OFF, CAP, UTIL, CV_WIND, FULL_WIND, CV_GEN,
)

OOS_EXCLUDE = ["海南", "北京"]

PY2CN = {"anhui":"安徽","beijing":"北京","chongqing":"重庆","fujian":"福建","gansu":"甘肃","guangdong":"广东",
 "guangxi":"广西","guizhou":"贵州","hainan":"海南","hebei":"河北","heilongjiang":"黑龙江","henan":"河南",
 "hubei":"湖北","hunan":"湖南","jiangsu":"江苏","jiangxi":"江西","jilin":"吉林","liaoning":"辽宁","neimenggu":"内蒙古",
 "ningxia":"宁夏","qinghai":"青海","shaanxi":"陕西","shandong":"山东","shanghai":"上海","shanxi":"山西",
 "sichuan":"四川","tianjin":"天津","xinjiang":"新疆","yunnan":"云南","zhejiang":"浙江"}


def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat); m = den > 0
    return 100.0 * np.mean(np.abs(yhat[m] - y[m]) / den[m]) if m.any() else np.nan


def load_prod(path, factor=False):
    d = pd.read_csv(path, encoding="utf-8-sig")
    d["cn"] = d["prov"].map(PY2CN)
    d = d.dropna(subset=["cn"]).copy()
    d["year"] = d["ym"].astype(str).str[:4].astype(int)
    d["month"] = d["ym"].astype(str).str[4:6].astype(int)
    if factor:
        cap = pd.read_csv(CAP, encoding="utf-8-sig")
        capf = cap[["省简称","年","月","cap_ratio"]].rename(columns={"省简称":"cn","年":"year","月":"month","cap_ratio":"cap_f"})
        util = pd.read_csv(UTIL, encoding="utf-8-sig")
        ucol = "factor" if "factor" in util.columns else ("util_applied" if "util_applied" in util.columns else util.columns[-1])
        uf = util[["省简称","年","月",ucol]].rename(columns={"省简称":"cn","年":"year","月":"month",ucol:"util_f"})
        f = capf.merge(uf, on=["cn","year","month"], how="outer")
        f["factor"] = f["cap_f"].fillna(1.0) * f["util_f"].fillna(1.0)
        d = d.merge(f[["cn","year","month","factor"]], on=["cn","year","month"], how="left")
        d["factor"] = d["factor"].fillna(1.0)
        d["gen_MWh"] = d["gen_MWh"] * d["factor"]
    d["mk"] = d["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return d.groupby(["cn","year","mk"], as_index=False)["gen_MWh"].sum()


def load_obs():
    o = pd.read_csv(OFF, encoding="utf-8-sig")
    o = o[o["value_MWh"] > 0].copy()
    o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return o.groupby(["cn","year","mk"], as_index=False)["value_MWh"].sum()


def err_stats(relerr):
    rel = np.abs(np.asarray(relerr, float))
    return dict(n=len(rel), mean=100*rel.mean(), median=100*np.median(rel),
                p90=100*np.percentile(rel, 90), p99=100*np.percentile(rel, 99), max=100*rel.max(),
                w10=100*np.mean(rel <= 0.10), w20=100*np.mean(rel <= 0.20),
                w50=100*np.mean(rel <= 0.50), gt100=100*np.mean(rel > 1.0))


def main():
    kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")
    obs = load_obs().rename(columns={"value_MWh": "y"})
    lines = []

    # ---------- in-sample 2013-2023, 30 provinces ----------
    m = load_prod(PROD23).rename(columns={"gen_MWh": "x"})
    mm = m[(m.year >= 2013) & (m.year <= 2023)].merge(obs, on=["cn","year","mk"], how="inner")
    mm["K"] = mm.cn.map(kb["K"]); mm["B"] = mm.cn.map(kb["B"])
    mm = mm.dropna(subset=["K"])
    mm["pred"] = mm.K * mm.x + mm.B
    mm["rel"] = (mm.pred - mm.y) / mm.y
    lines.append("== A. IN-SAMPLE 2013-2023, 30 provinces: %d province-months ==" % len(mm))
    st = err_stats(mm.rel)
    lines.append("monthly |relative error|: mean %.1f%%  median %.1f%%  p90 %.1f%%  p99 %.1f%%  max %.0f%%"
                 % (st["mean"], st["median"], st["p90"], st["p99"], st["max"]))
    lines.append("share within +/-10%%: %.1f%%   +/-20%%: %.1f%%   +/-50%%: %.1f%%   >100%%: %.1f%%"
                 % (st["w10"], st["w20"], st["w50"], st["gt100"]))
    lines.append("province-month sMAPE (all): %.2f%%" % smape(mm.y, mm.pred))

    lines.append("\n-- per-year (30 provinces) --")
    for yr, g in mm.groupby("year"):
        s = err_stats(g.rel)
        lines.append("  %d  n=%3d  median|err| %5.1f%%  mean|err| %5.1f%%  sMAPE %5.2f%%  pooled ratio %.3f"
                     % (yr, len(g), s["median"], s["mean"], smape(g.y, g.pred), g.pred.sum()/g.y.sum()))

    lines.append("\n-- per-province sMAPE & median |err| (worst 8) --")
    rows = []
    for cn, g in mm.groupby("cn"):
        rows.append(dict(cn=cn, n=len(g), smape=smape(g.y, g.pred),
                         med_abs_rel=100*np.median(np.abs(g.rel)),
                         pooled_ratio=g.pred.sum()/g.y.sum()))
    pr = pd.DataFrame(rows).sort_values("smape", ascending=False)
    lines.append(pr.head(8).to_string(index=False, float_format=lambda v: "%.2f" % v))

    lines.append("\n-- provinces where pooled ratio is NOT exactly 1 (B constrained to 0) --")
    nonid = pr[(pr.pooled_ratio - 1).abs() > 1e-6]
    lines.append(nonid.to_string(index=False, float_format=lambda v: "%.6f" % v))

    # exact check for B>0
    exact = 0; tot = 0
    for cn, g in mm.groupby("cn"):
        if abs(float(kb.loc[cn,"B"])) > 1e-9:
            tot += 1
            if abs(g.pred.sum()-g.y.sum())/g.y.sum() < 1e-12:
                exact += 1
    lines.append("B>0 provinces with |sum(pred)-sum(y)|/sum(y) < 1e-12: %d / %d" % (exact, tot))

    # ---------- out-of-sample ----------
    lines.append("\n== B. OUT-OF-SAMPLE 2024 / 2025 (28 provinces) ==")
    m24 = load_prod(PROD24, factor=True).rename(columns={"gen_MWh": "x"})
    for yr in (2024, 2025):
        g = m24[m24.year == yr].merge(obs, on=["cn","year","mk"], how="inner")
        g = g[~g.cn.isin(OOS_EXCLUDE)]
        g["K"] = g.cn.map(kb["K"]); g["B"] = g.cn.map(kb["B"])
        g = g.dropna(subset=["K"])
        g["pred"] = g.K*g.x + g.B
        g["rel"] = (g.pred - g.y)/g.y
        s = err_stats(g.rel)
        lines.append("  %d  n=%3d  median|err| %5.1f%%  mean|err| %5.1f%%  p90 %5.1f%%  max %.0f%%  sMAPE %5.2f%%  pooled ratio %.3f"
                     % (yr, len(g), s["median"], s["mean"], s["p90"], s["max"], smape(g.y, g.pred), g.pred.sum()/g.y.sum()))

    # ---------- monthly distribution table (histogram) ----------
    lines.append("\n== C. distribution of in-sample monthly relative errors ==")
    bins = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 99]
    labels = ["<=5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-100%", ">100%"]
    cnt = pd.cut(np.abs(mm.rel), bins=bins, labels=labels, right=True).value_counts().reindex(labels)
    for k, v in cnt.items():
        lines.append("  %-8s %5d  (%5.1f%%)" % (k, v, 100*v/len(mm)))

    # examples
    lines.append("\n== D. examples: biggest in-sample monthly misses (30 prov) ==")
    top = mm.reindex(mm.rel.abs().sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        lines.append("  %s %d-%02d  official %12.0f  model %12.0f  rel %+8.1f%%"
                     % (r.cn, r.year, r.mk, r.y, r.pred, 100*r.rel))

    open(os.path.join(OUTD, "honest_monthly_errors.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", os.path.join(OUTD, "honest_monthly_errors.txt"))


if __name__ == "__main__":
    main()
