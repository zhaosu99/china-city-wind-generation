# -*- coding: utf-8 -*-
"""Pseudo-out-of-sample check: fit K,B on 2013-2020 only, validate on 2021-2023.
Compare with the production K,B (fitted 2013-2023) evaluated on 2013-2023 and on 2024/2025.
"""
import os
import numpy as np
import pandas as pd
from config_paths import (
    M4, OUTD, OUTDIR, OUT, PROD23, PROD24, KB, KB_FULL, OFF, CAP, UTIL, CV_WIND, FULL_WIND, CV_GEN,
)

EXCLUDE = ["海南", "北京"]

PY2CN = {"anhui":"安徽","beijing":"北京","chongqing":"重庆","fujian":"福建","gansu":"甘肃","guangdong":"广东",
 "guangxi":"广西","guizhou":"贵州","hainan":"海南","hebei":"河北","heilongjiang":"黑龙江","henan":"河南",
 "hubei":"湖北","hunan":"湖南","jiangsu":"江苏","jiangxi":"江西","jilin":"吉林","liaoning":"辽宁","neimenggu":"内蒙古",
 "ningxia":"宁夏","qinghai":"青海","shaanxi":"陕西","shandong":"山东","shanghai":"上海","shanxi":"山西",
 "sichuan":"四川","tianjin":"天津","xinjiang":"新疆","yunnan":"云南","zhejiang":"浙江"}


def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat); m = den > 0
    return 100.0 * np.mean(np.abs(yhat[m] - y[m]) / den[m]) if m.any() else np.nan


def load_prod():
    d = pd.read_csv(PROD23, encoding="utf-8-sig")
    d["cn"] = d["prov"].map(PY2CN)
    d = d.dropna(subset=["cn"]).copy()
    d["year"] = d["ym"].astype(str).str[:4].astype(int)
    d["month"] = d["ym"].astype(str).str[4:6].astype(int)
    d["mk"] = d["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return d.groupby(["cn","year","mk"], as_index=False)["gen_MWh"].sum()


def load_obs():
    o = pd.read_csv(OFF, encoding="utf-8-sig")
    o = o[(o["value_MWh"] > 0) & (o["year"] <= 2025)].copy()
    o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return o.groupby(["cn","year","mk"], as_index=False)["value_MWh"].sum()


def fit_kb(m, y0, y1):
    sub = m[(m.year >= y0) & (m.year <= y1)]
    rows = []
    for cn, g in sub.groupby("cn"):
        a = g.x.values.astype(float); b = g.y.values.astype(float)
        if len(a) < 3 or np.var(a, ddof=1) <= 0:
            continue
        K = np.cov(a, b, ddof=1)[0, 1] / np.var(a, ddof=1)
        B = b.mean() - K * a.mean()
        if B < 0:
            K = float((a * b).sum() / (a * a).sum()); B = 0.0
        rows.append(dict(cn=cn, K=K, B=B))
    return pd.DataFrame(rows).set_index("cn")


def evaluate(m, obs, kb, y0, y1, tag, exclude=()):
    ob = obs[(obs.year >= y0) & (obs.year <= y1)].rename(columns={"value_MWh": "y"})
    mm = m[(m.year >= y0) & (m.year <= y1)].rename(columns={"gen_MWh": "x"}).merge(ob, on=["cn","year","mk"], how="inner")
    mm = mm[~mm.cn.isin(exclude)]
    recs = []
    for cn, g in mm.groupby("cn"):
        if cn not in kb.index:
            continue
        K = float(kb.loc[cn,"K"]); B = float(kb.loc[cn,"B"])
        y = g.y.values; x = g.x.values
        recs.append(dict(cn=cn, smape=smape(y, K*x+B), ratio=(K*x+B).sum()/y.sum()))
    r = pd.DataFrame(recs)
    return r, "%-34s n=%2d  sMAPE mean %6.2f med %6.2f | ratio mean %6.3f med %6.3f | ratio in[0.85,1.15] %d/%d" % (
        tag, len(r), r.smape.mean(), r.smape.median(), r.ratio.mean(), r.ratio.median(),
        int(((r.ratio >= 0.85) & (r.ratio <= 1.15)).sum()), len(r))


def main():
    m = load_prod()
    obs = load_obs().rename(columns={"value_MWh": "y"})
    mf = m.rename(columns={"gen_MWh": "x"}).merge(obs, on=["cn","year","mk"], how="inner")
    lines = []
    lines.append("== A. K/B fitted on 2013-2020 (constraint B>=0), evaluated 2021-2023 ==")
    kb20 = fit_kb(mf, 2013, 2020)
    r, s = evaluate(m, obs, kb20, 2021, 2023, "full calib (fit 2013-2020) 2021-2023")
    lines.append(s)
    lines.append("   K median %.4f | B median %.0f" % (kb20.K.median(), kb20.B.median()))
    # noKB on the same window
    ob = obs[(obs.year >= 2021) & (obs.year <= 2023)]
    mm = m[(m.year >= 2021) & (m.year <= 2023)].rename(columns={"gen_MWh": "x"}).merge(ob, on=["cn","year","mk"], how="inner")
    mm = mm[~mm.cn.isin(EXCLUDE)]
    recs = []
    for cn, g in mm.groupby("cn"):
        recs.append(dict(cn=cn, smape=smape(g.y.values, g.x.values), ratio=g.x.sum()/g.y.sum()))
    nr = pd.DataFrame(recs)
    lines.append("noKB 2021-2023: n=%d sMAPE mean %.2f med %.2f | ratio mean %.3f med %.3f"
                 % (len(nr), nr.smape.mean(), nr.smape.median(), nr.ratio.mean(), nr.ratio.median()))

    lines.append("\n== B. production K/B (fitted 2013-2023) on the same 2021-2023 window (for reference) ==")
    kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")
    r2, s2 = evaluate(m, obs, kb, 2021, 2023, "full calib (fit 2013-2023) 2021-2023")
    lines.append(s2)

    lines.append("\n== C. production K/B on 2013-2023 (pooled, identity) ==")
    r3, s3 = evaluate(m, obs, kb, 2013, 2023, "full calib 2013-2023 pooled")
    lines.append(s3)

    lines.append("\n== D. annual ratio with production K/B (28 provinces, mean across provinces) ==")
    mm = m.rename(columns={"gen_MWh": "x"}).merge(obs, on=["cn","year","mk"], how="inner")
    mm = mm[(mm.year >= 2013) & (mm.year <= 2023)]
    for yr, g in mm.groupby("year"):
        rr = []
        for cn, gg in g.groupby("cn"):
            if cn not in kb.index or cn in EXCLUDE:
                continue
            K = float(kb.loc[cn,"K"]); B = float(kb.loc[cn,"B"])
            y = gg.y.values; x = gg.x.values
            rr.append((K*x+B).sum()/y.sum())
        lines.append("  %d  ratio mean %.3f  med %.3f" % (yr, np.mean(rr), np.median(rr)))

    open(os.path.join(OUTD, "audit_holdout_2021_2023.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", os.path.join(OUTD, "audit_holdout_2021_2023.txt"))


if __name__ == "__main__":
    main()
