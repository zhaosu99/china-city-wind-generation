# -*- coding: utf-8 -*-
"""§4.4 validation table v2 (period convention = refit_and_cv: Jan-Feb merged into 'mk=1').

States per province-period:
  raw   : model series as-is (no capacity/util factor, no K/B)         [only meaningful for 2024/25 vs noKB]
  noKB  : model series after capacity/util alignment, before K/B       (2013-2023: same as raw)
  full  : K * noKB + B
Metrics: monthly sMAPE, generation ratio over matched province-months.
Periods: A=2013-2023, B=2024, C=2025.
"""
import os
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


def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat); m = den > 0
    return 100.0 * np.mean(np.abs(yhat[m] - y[m]) / den[m]) if m.any() else np.nan


def month_key(m):
    return 1 if m <= 2 else int(m)


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
    d["mk"] = d["month"].apply(month_key)
    return d.groupby(["cn","year","mk"], as_index=False)["gen_MWh"].sum()


def load_obs():
    o = pd.read_csv(OFF, encoding="utf-8-sig")
    o = o[(o["value_MWh"] > 0) & (o["year"] <= 2025)].copy()
    o["mk"] = o["month"].apply(month_key)
    return o.groupby(["cn","year","mk"], as_index=False)["value_MWh"].sum()


def main():
    m23 = load_prod(PROD23)
    m24a = load_prod(PROD24, factor=True)   # adjusted
    m24r = load_prod(PROD24, factor=False)  # raw
    obs = load_obs()
    kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")
    print("KB provinces:", len(kb))

    recs = []
    audit = []
    periods = {"A_2013_2023": (2013, 2023), "B_2024": (2024, 2024), "C_2025": (2025, 2025)}
    for period, (y0, y1) in periods.items():
        if y1 <= 2023:
            m_no = m23[(m23.year >= y0) & (m23.year <= y1)].rename(columns={"gen_MWh": "x"})
            m_raw = m_no.copy()
        else:
            m_no = m24a[(m24a.year >= y0) & (m24a.year <= y1)].rename(columns={"gen_MWh": "x"})
            m_raw = m24r[(m24r.year >= y0) & (m24r.year <= y1)].rename(columns={"gen_MWh": "x"})
        ob = obs[(obs.year >= y0) & (obs.year <= y1)].rename(columns={"value_MWh": "y"})
        mm = m_no.merge(ob, on=["cn","year","mk"], how="inner")
        mm_raw = m_raw.merge(ob, on=["cn","year","mk"], how="inner")
        provs = sorted(set(mm.cn) & set(kb.index))
        for cn in provs:
            g = mm[mm.cn == cn]; gr = mm_raw[mm_raw.cn == cn]
            K = float(kb.loc[cn,"K"]); B = float(kb.loc[cn,"B"])
            y = g.y.values; x = g.x.values; xr = gr.x.values
            yhat = K*x + B
            recs.append(dict(prov=cn, period=period, state="noKB", n=len(g),
                             smape=smape(y,x), ratio=x.sum()/y.sum(), pred_TWh=x.sum()/1e6, obs_TWh=y.sum()/1e6))
            recs.append(dict(prov=cn, period=period, state="full", n=len(g),
                             smape=smape(y,yhat), ratio=yhat.sum()/y.sum(), pred_TWh=yhat.sum()/1e6, obs_TWh=y.sum()/1e6))
            if y1 > 2023:
                recs.append(dict(prov=cn, period=period, state="raw", n=len(gr),
                                 smape=smape(y,xr), ratio=xr.sum()/y.sum(), pred_TWh=xr.sum()/1e6, obs_TWh=y.sum()/1e6))
        audit.append(dict(period=period, prov_months=len(mm), provinces=len(provs),
                          model_TWh=mm.x.sum()/1e6, obs_TWh=mm.y.sum()/1e6))
    t = pd.DataFrame(recs)
    t.to_csv(os.path.join(OUTD, "validation44_table_v2.csv"), index=False, encoding="utf-8-sig")

    lines = ["== matched audit (Jan-Feb merged) =="]
    lines.append(pd.DataFrame(audit).to_string(index=False, float_format=lambda v: "%.2f" % v))
    lines.append("\n== summary across 30 provinces (all) ==")
    for period in periods:
        for state in ("raw", "noKB", "full"):
            s = t[(t.period == period) & (t.state == state)]
            if s.empty:
                continue
            lines.append("%-11s %-5s n=%2d  sMAPE mean %6.2f med %6.2f  ratio mean %6.3f med %6.3f"
                         % (period, state, len(s), s.smape.mean(), s.smape.median(), s.ratio.mean(), s.ratio.median()))
    # robust view excluding two provinces with known official-data anomalies
    bad = ["海南", "北京"]
    lines.append("\n== summary excluding %s (official-data anomalies) ==" % ("/".join(bad)))
    for period in periods:
        for state in ("raw", "noKB", "full"):
            s = t[(t.period == period) & (t.state == state) & (~t.prov.isin(bad))]
            if s.empty:
                continue
            lines.append("%-11s %-5s n=%2d  sMAPE mean %6.2f med %6.2f  ratio mean %6.3f med %6.3f"
                         % (period, state, len(s), s.smape.mean(), s.smape.median(), s.ratio.mean(), s.ratio.median()))
    lines.append("\n== per-province sMAPE ==")
    lines.append(t.pivot_table(index="prov", columns=["period","state"], values="smape").round(2).to_string())
    lines.append("\n== per-province ratio ==")
    lines.append(t.pivot_table(index="prov", columns=["period","state"], values="ratio").round(3).to_string())
    out = os.path.join(OUTD, "validation44_summary_v2.txt")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", out, "| rows", len(t))


if __name__ == "__main__":
    main()
