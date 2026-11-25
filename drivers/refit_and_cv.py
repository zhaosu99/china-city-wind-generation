# -*- coding: utf-8 -*-
"""在【新生产序列】上：
   1) 重新拟合逐省 (K,B)（2013–2023 全部期点）
   2) 重跑 50 次按季分层 80/20 交叉验证
口径与 run_cv_143 一致：期点 = 年×月，1–2 月合并；按季分层抽测。
"""
import os

import numpy as np
import pandas as pd

PROD = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\prov_month_from_city.csv"
OFF = r"D:\HongyuEnergy Copilot\_run\prod_2b\work\official_prov_month.csv"
OUTD = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\refit"
os.makedirs(OUTD, exist_ok=True)

PY2CN = {
    "anhui": "安徽", "beijing": "北京", "chongqing": "重庆", "fujian": "福建",
    "gansu": "甘肃", "guangdong": "广东", "guangxi": "广西", "guizhou": "贵州",
    "hainan": "海南", "hebei": "河北", "heilongjiang": "黑龙江", "henan": "河南",
    "hubei": "湖北", "hunan": "湖南", "jiangsu": "江苏", "jiangxi": "江西",
    "jilin": "吉林", "liaoning": "辽宁", "neimenggu": "内蒙古", "ningxia": "宁夏",
    "qinghai": "青海", "shaanxi": "陕西", "shandong": "山东", "shanghai": "上海",
    "shanxi": "山西", "sichuan": "四川", "tianjin": "天津", "xinjiang": "新疆",
    "yunnan": "云南", "zhejiang": "浙江",
}
SEASONS = ("spring", "summer", "autumn", "winter")
SEASON_OF_MONTH = {3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn",
                   12: "winter", 1: "winter", 2: "winter"}


def smape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat)
    m = den > 0
    if not m.any():
        return np.nan
    return 100.0 * np.mean(np.abs(yhat[m] - y[m]) / den[m])


# ---------------- 1. 构造期点表（2013–2023，1–2 月合并） ----------------
d = pd.read_csv(PROD, encoding="utf-8-sig")
d["cn"] = d["prov"].map(PY2CN)
d = d.dropna(subset=["cn"])
d["year"] = d["ym"].astype(str).str[:4].astype(int)
d["month"] = d["ym"].astype(str).str[4:6].astype(int)

# 期点键：1–2 月合并为 mk=1
d["mk"] = d["month"].apply(lambda m: 1 if m <= 2 else m)
x = d.groupby(["cn", "year", "mk"], as_index=False)["gen_MWh"].sum()
x = x[(x["year"] >= 2013) & (x["year"] <= 2023)]
x["season"] = x["mk"].map({1: "winter", 3: "spring", 4: "spring", 5: "spring",
                          6: "summer", 7: "summer", 8: "summer",
                          9: "autumn", 10: "autumn", 11: "autumn", 12: "winter"})

o = pd.read_csv(OFF, encoding="utf-8-sig")
o = o[o["value_MWh"] > 0].copy()
o["mk"] = o["month"].apply(lambda m: 1 if m <= 2 else m)
y = o.groupby(["cn", "year", "mk"], as_index=False)["value_MWh"].sum()
y = y[(y["year"] >= 2013) & (y["year"] <= 2023)]

m = x.merge(y, on=["cn", "year", "mk"], how="inner")
m["period"] = m["year"].astype(str) + "-" + m["mk"].astype(str).str.zfill(2)
print("对齐后的省-期点: %d 行 ｜ 省 %d ｜ 期点 %d"
      % (len(m), m["cn"].nunique(), m["period"].nunique()))
m.to_csv(os.path.join(OUTD, "prov_period_matched.csv"), index=False,
         encoding="utf-8-sig")

# ---------------- 2. 拟合逐省 (K,B) ----------------
rows = []
for prov, g in m.groupby("cn"):
    a = g["gen_MWh"].values.astype(float)
    b_ = g["value_MWh"].values.astype(float)
    if len(a) < 3 or np.var(a, ddof=1) <= 0:
        continue
    K = np.cov(a, b_, ddof=1)[0, 1] / np.var(a, ddof=1)
    B = b_.mean() - K * a.mean()
    if B < 0:                      # 截距非负约束：避免小水量月份出现负发电
        K = float((a * b_).sum() / (a * a).sum())
        B = 0.0
    res = b_ - (K * a + B)
    r2 = 1 - (res ** 2).sum() / ((b_ - b_.mean()) ** 2).sum()
    rows.append(dict(prov=prov, K=K, B=B, n=len(a), r2=r2,
                     sum_x=a.sum(), sum_y=b_.sum()))
fit = pd.DataFrame(rows).sort_values("prov")
fit.to_csv(os.path.join(OUTD, "gen_correction_M4prod.csv"), index=False,
           encoding="utf-8-sig")
print()
print("== 新 (K,B)（30 省）==")
print("K: 均值 %.4f ｜ 中位 %.4f ｜ 范围 %.4f–%.4f"
      % (fit.K.mean(), fit.K.median(), fit.K.min(), fit.K.max()))
print("B: 中位 %.4g ｜ r2 中位 %.4f" % (fit.B.median(), fit.r2.median()))

# ---------------- 3. 50 次按季分层 80/20 ----------------
periods = sorted(m[["year", "mk", "period", "season"]]
                 .drop_duplicates(["year", "mk"]).itertuples(index=False))
pool = {s: [p.period for p in periods if p.season == s] for s in SEASONS}
n_test = {s: max(1, int(round(0.20 * len(pool[s])))) for s in SEASONS}
print()
print("期点池:", {s: len(pool[s]) for s in SEASONS}, "｜ 每次抽测:", n_test,
      "｜ 测试占比 %.1f%%" % (100 * sum(n_test.values()) / len(periods)))

rng = np.random.default_rng(20261116)
seen, splits = set(), []
while len(splits) < 50:
    test = set()
    for s in SEASONS:
        pick = rng.choice(pool[s], size=n_test[s], replace=False)
        test.update(pick.tolist())
    key = frozenset(test)
    if key in seen:
        continue
    seen.add(key)
    splits.append(test)

recs = []
for si, test in enumerate(splits, 1):
    tr = m[~m["period"].isin(test)]
    te = m[m["period"].isin(test)]
    kb = {}
    for prov, g in tr.groupby("cn"):
        a = g["gen_MWh"].values.astype(float)
        b_ = g["value_MWh"].values.astype(float)
        if len(a) < 3 or np.var(a, ddof=1) <= 0:
            kb[prov] = (1.0, 0.0)
        else:
            K = np.cov(a, b_, ddof=1)[0, 1] / np.var(a, ddof=1)
            Bp = b_.mean() - K * a.mean()
            if Bp < 0:                # 同上：截距非负约束
                K = float((a * b_).sum() / (a * a).sum())
                Bp = 0.0
            kb[prov] = (K, Bp)
    for prov, g in te.groupby("cn"):
        K, B = kb.get(prov, (1.0, 0.0))
        gh = K * g["gen_MWh"].values + B
        yv = g["value_MWh"].values
        recs.append(dict(split=si, prov=prov,
                         smape=smape(yv, gh),
                         ratio=gh.sum() / yv.sum() if yv.sum() else np.nan,
                         n=len(yv), K=K, B=B))
run = pd.DataFrame(recs)
run.to_csv(os.path.join(OUTD, "cv_per_run.csv"), index=False,
           encoding="utf-8-sig")

per_prov = (run.groupby("prov")
            .agg(smape=("smape", "mean"), ratio=("ratio", "mean"))
            .reset_index())
per_prov.to_csv(os.path.join(OUTD, "cv_per_prov.csv"), index=False,
                encoding="utf-8-sig")

print()
print("== 新 CV 结果（50 次 × %d 省）==" % per_prov.shape[0])
print("  test SMAPE: 均值 %.3f ｜ 中位 %.3f ｜ 省间 sd %.3f"
      % (per_prov.smape.mean(), per_prov.smape.median(), per_prov.smape.std()))
print("  test ratio: 均值 %.4f ｜ 中位 %.4f"
      % (per_prov.ratio.mean(), per_prov.ratio.median()))
print("  （旧口径：test SMAPE 均值 14.868）")
print()
print("最差 5 省:")
print(per_prov.sort_values("smape", ascending=False).head(5)
      .to_string(index=False, float_format=lambda v: "%.2f" % v))
print()
print("输出目录:", OUTD)
