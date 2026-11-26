# -*- coding: utf-8 -*-
"""Prepare (NOT plot) the last-5-year calibration variant:
fit K,B per province on 2019-2023 official monthly data, apply to 2024 and 2025,
and compare with the production K,B (fitted on 2013-2023).

Outputs in D:\\HongyuEnergy Copilot\\_run\\essd_survey\\m4prod\\validation44\\kb_last5y\\:
  gen_correction_last5y.csv     per-province K,B fitted on 2019-2023
  forward_2425_last5y.csv       per-province metrics on 2024, 2025, and 2024+2025
  compare_last5y_vs_full.txt    summary comparison
"""
import os
import numpy as np
import pandas as pd
from config_paths import (
    M4, OUTD, OUTDIR, OUT, PROD23, PROD24, KB, KB_FULL, OFF, CAP, UTIL, CV_WIND, FULL_WIND, CV_GEN,
)

os.makedirs(OUTD, exist_ok=True)
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


def r2score(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    ss = ((y - y.mean())**2).sum()
    return 1.0 - ((y - pred)**2).sum() / ss if ss > 0 else np.nan


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


def fit_kb(train):
    rows = []
    for cn, g in train.groupby("cn"):
        x = g.x.values.astype(float); y = g.y.values.astype(float)
        if len(x) < 6 or np.var(x, ddof=1) <= 0:
            continue
        K = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        B = y.mean() - K * x.mean()
        if B < 0:
            K = float((x*y).sum() / (x*x).sum()); B = 0.0
        res = y - (K*x + B)
        r2 = 1 - (res**2).sum() / ((y-y.mean())**2).sum()
        rows.append(dict(prov=cn, K=K, B=B, n=len(x), r2=r2, sum_x=x.sum(), sum_y=y.sum()))
    return pd.DataFrame(rows).set_index("prov")


def evaluate(kb, test):
    recs = []
    for cn, g in test.groupby("cn"):
        if cn not in kb.index:
            continue
        K = float(kb.loc[cn, "K"]); B = float(kb.loc[cn, "B"])
        pred = K * g.x.values.astype(float) + B
        y = g.y.values.astype(float)
        recs.append(dict(prov=cn, n=len(y), smape=smape(y, pred), r2=r2score(y, pred),
                         ratio=pred.sum()/y.sum(),
                         pred_TWh=pred.sum()/1e6, obs_TWh=y.sum()/1e6))
    return pd.DataFrame(recs).set_index("prov")


def summarize(r, tag):
    return ("%-28s n=%2d  sMAPE mean %6.2f  med %6.2f | R2 median %6.3f | ratio mean %6.3f  med %6.3f"
            % (tag, len(r), r.smape.mean(), r.smape.median(), r.r2.median(), r.ratio.mean(), r.ratio.median()))


def main():
    m23 = load_prod(PROD23).rename(columns={"gen_MWh": "x"})
    m24 = load_prod(PROD24, factor=True).rename(columns={"gen_MWh": "x"})
    obs = load_obs().rename(columns={"value_MWh": "y"})
    d23 = m23.merge(obs, on=["cn","year","mk"], how="inner")
    d23 = d23[(d23.year >= 2013) & (d23.year <= 2023)]
    d24 = m24.merge(obs, on=["cn","year","mk"], how="inner")
    d24 = d24[~d24.cn.isin(EXCLUDE)]

    kb5 = fit_kb(d23[(d23.year >= 2019) & (d23.year <= 2023)].copy())
    kb5.reset_index().to_csv(os.path.join(OUTD, "gen_correction_last5y.csv"), index=False, encoding="utf-8-sig")
    kb_full = pd.read_csv(KB_FULL, encoding="utf-8-sig").set_index("prov")

    lines = ["Calibration window comparison: K,B fitted on 2019-2023 ('last 5 years') vs 2013-2023 ('full').",
             "Evaluation on 28 provinces (Hainan, Beijing excluded).", ""]
    lines.append("== fitted coefficients ==")
    lines.append("last5y: K median %.4f  B median %.0f  (provinces %d)" % (kb5.K.median(), kb5.B.median(), len(kb5)))
    lines.append("full  : K median %.4f  B median %.0f  (provinces %d)" % (kb_full.K.median(), kb_full.B.median(), len(kb_full)))
    j = kb5.join(kb_full, rsuffix="_full")
    lines.append("K correlation last5y vs full: %.4f ; median |dK| %.4f ; median |dB| %.0f"
                 % (j.K.corr(j.K_full), (j.K - j.K_full).abs().median(), (j.B - j.B_full).abs().median()))

    for yr in (2024, 2025):
        te = d24[d24.year == yr]
        r5 = evaluate(kb5, te); rf = evaluate(kb_full, te)
        lines.append("")
        lines.append("== test %d ==" % yr)
        lines.append(summarize(r5, "last5y (2019-2023 fit)"))
        lines.append(summarize(rf, "full   (2013-2023 fit)"))
        cmp = pd.DataFrame({"smape_last5y": r5.smape, "smape_full": rf.smape,
                            "ratio_last5y": r5.ratio, "ratio_full": rf.ratio})
        cmp["d_smape"] = cmp.smape_last5y - cmp.smape_full
        lines.append("   mean d_sMAPE (last5y - full): %+.2f pp ; last5y better in %d/%d provinces"
                     % (cmp.d_smape.mean(), int((cmp.d_smape < 0).sum()), len(cmp)))
        lines.append("   worst regressions (last5y worse):")
        lines.append(cmp.sort_values("d_smape", ascending=False).head(5).to_string(float_format=lambda v: "%.2f" % v))

    # combined 2024+2025 per province (as in the earlier forward file)
    te = d24[d24.year >= 2024]
    r5 = evaluate(kb5, te); rf = evaluate(kb_full, te)
    lines.append("")
    lines.append("== test 2024+2025 combined ==")
    lines.append(summarize(r5, "last5y (2019-2023 fit)"))
    lines.append(summarize(rf, "full   (2013-2023 fit)"))

    out5 = r5.join(rf, rsuffix="_full")
    out5.to_csv(os.path.join(OUTD, "forward_2425_last5y.csv"), encoding="utf-8-sig")
    open(os.path.join(OUTD, "compare_last5y_vs_full.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", OUTD)


if __name__ == "__main__":
    main()
