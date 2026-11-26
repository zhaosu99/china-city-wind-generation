# -*- coding: utf-8 -*-
"""给 refit_and_cv.py 加上 B>=0 约束（与最终生产参数一致），再重跑 CV。"""
import io

P = r"D:\HongyuEnergy Copilot\_run\essd_survey\m4prod\refit_and_cv.py"
s = io.open(P, encoding="utf-8").read()

old1 = """    K = np.cov(a, b_, ddof=1)[0, 1] / np.var(a, ddof=1)
    B = b_.mean() - K * a.mean()
    res = b_ - (K * a + B)"""
new1 = """    K = np.cov(a, b_, ddof=1)[0, 1] / np.var(a, ddof=1)
    B = b_.mean() - K * a.mean()
    if B < 0:                      # 截距非负约束：避免小水量月份出现负发电
        K = float((a * b_).sum() / (a * a).sum())
        B = 0.0
    res = b_ - (K * a + B)"""

old2 = """            K = np.cov(a, b_, ddof=1)[0, 1] / np.var(a, ddof=1)
            kb[prov] = (K, b_.mean() - K * a.mean())"""
new2 = """            K = np.cov(a, b_, ddof=1)[0, 1] / np.var(a, ddof=1)
            Bp = b_.mean() - K * a.mean()
            if Bp < 0:                # 同上：截距非负约束
                K = float((a * b_).sum() / (a * a).sum())
                Bp = 0.0
            kb[prov] = (K, Bp)"""

n = 0
if old1 in s:
    s = s.replace(old1, new1); n += 1
if old2 in s:
    s = s.replace(old2, new2); n += 1
io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("应用补丁处数:", n)
