# -*- coding: utf-8 -*-
"""Figure 11 (new) data assembly: 4 rows x 2 cols, one box per cell.

Rows (user design, 2026-11-24):
  R1 = no calibration (no wind-speed adjustment, no generation K/B), window 2013-2023
  R2 = full calibration, window 2013-2023
  R3 = full calibration, 2024 only (out-of-sample)
  R4 = full calibration, 2025 only (out-of-sample)
Sample: provinces with complete monthly records -> exclude Hainan and Beijing (official-source anomalies).
Outputs: validation44_fig11_data.csv, validation44_fig11_stats.txt
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
    o = o[(o["value_MWh"] > 0) & (o["year"] <= 2025)].copy()
    o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else int(m))
    return o.groupby(["cn","year","mk"], as_index=False)["value_MWh"].sum()


def series_for(period):
    if period in ("2013_2023",):
        m = load_prod(PROD23); m = m[(m.year >= 2013) & (m.year <= 2023)]
        return m.rename(columns={"gen_MWh": "x"}), False
    y = int(period)
    m = load_prod(PROD24, factor=True); m = m[m.year == y]
    return m.rename(columns={"gen_MWh": "x"}), True


def main():
    obs = load_obs()
    kb = pd.read_csv(KB, encoding="utf-8-sig").set_index("prov")
    rows = [
        ("R1_noKB_2013_2023", "noKB", "2013_2023"),
        ("R2_full_2013_2023", "full", "2013_2023"),
        ("R3_full_2024", "full", "2024"),
        ("R4_full_2025", "full", "2025"),
    ]
    recs = []
    for label, state, period in rows:
        m, _ = series_for(period)
        y0, y1 = (2013, 2023) if period == "2013_2023" else (int(period), int(period))
        ob = obs[(obs.year >= y0) & (obs.year <= y1)].rename(columns={"value_MWh": "y"})
        mm = m.merge(ob, on=["cn","year","mk"], how="inner")
        exclude = set(EXCLUDE) if period in ("2024", "2025") else set()
        provs = sorted((set(mm.cn) & set(kb.index)) - exclude)
        for cn in provs:
            g = mm[mm.cn == cn]
            K = float(kb.loc[cn,"K"]); B = float(kb.loc[cn,"B"])
            y = g.y.values; x = g.x.values
            pred = K * x + B if state == "full" else x
            recs.append(dict(row=label, prov=cn, n=len(g), smape=smape(y, pred),
                             ratio=pred.sum() / y.sum()))
    t = pd.DataFrame(recs)
    t.to_csv(os.path.join(OUTD, "validation44_fig11_data.csv"), index=False, encoding="utf-8-sig")

    lines = []
    lines.append("== Figure 11 panel data: n per row ==")
    lines.append(t.groupby("row").agg(provinces=("prov","nunique"),
                                      n_min=("n","min"), n_med=("n","median"), n_max=("n","max")).to_string())
    lines.append("\n== per-row summary (28 provinces) ==")
    for label in t.row.unique():
        s = t[t.row == label]
        q = s.smape.quantile([0.25, 0.5, 0.75])
        qr = s.ratio.quantile([0.25, 0.5, 0.75])
        lines.append("%-22s SMAPE mean %6.2f  med %6.2f  IQR %5.2f-%5.2f  |  ratio mean %6.3f med %6.3f IQR %5.3f-%5.3f"
                     % (label, s.smape.mean(), q[0.5], q[0.25], q[0.75],
                        s.ratio.mean(), qr[0.5], qr[0.25], qr[0.75]))
        lines.append("   SMAPE range %6.2f-%6.2f | ratio range %6.3f-%6.3f | ratio in [0.85,1.15]: %d/%d"
                     % (s.smape.min(), s.smape.max(), s.ratio.min(), s.ratio.max(),
                        int(((s.ratio >= 0.85) & (s.ratio <= 1.15)).sum()), len(s)))

    # paired changes
    w = t.pivot_table(index="prov", columns="row", values="smape")
    lines.append("\n== paired changes (28 provinces) ==")
    d1 = w["R2_full_2013_2023"] - w["R1_noKB_2013_2023"]
    lines.append("SMAPE noKB -> full (in-sample): mean change %+.2f pp, median %+.2f; improved %d/28"
                 % (d1.mean(), d1.median(), int((d1 < 0).sum())))
    for r in ("R3_full_2024", "R4_full_2025"):
        d2 = w[r] - w["R2_full_2013_2023"]
        lines.append("SMAPE full in-sample -> %s: mean change %+.2f pp, median %+.2f; within +/-3 pp %d/28; within +/-5 pp %d/28"
                     % (r, d2.mean(), d2.median(), int(d2.abs().le(3).sum()), int(d2.abs().le(5).sum())))
    wr = t.pivot_table(index="prov", columns="row", values="ratio")
    lines.append("ratio noKB -> full (in-sample): median %+.3f (%.3f -> %.3f)"
                 % ((wr["R2_full_2013_2023"] - wr["R1_noKB_2013_2023"]).median(),
                    wr["R1_noKB_2013_2023"].median(), wr["R2_full_2013_2023"].median()))
    for r in ("R3_full_2024", "R4_full_2025"):
        lines.append("ratio in-sample -> %s: median change %+.3f (to %.3f)"
                     % (r, (wr[r] - wr["R2_full_2013_2023"]).median(), wr[r].median()))
    lines.append("\n== per-province table ==")
    lines.append(t.pivot_table(index="prov", columns="row", values="smape").round(2).to_string())
    lines.append("")
    lines.append(t.pivot_table(index="prov", columns="row", values="ratio").round(3).to_string())
    out = os.path.join(OUTD, "validation44_fig11_stats.txt")
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", out, "rows", len(t))


if __name__ == "__main__":
    main()
