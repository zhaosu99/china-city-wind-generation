# -*- coding: utf-8 -*-
"""Workflow figure (plain language, v17 final) → fig3_workflow_v17.pptx

修订依据（两轮空白读者测试）：
  · 纵向箭头必须落在框上，不能落在框缝里 → SPINE 取内容区左侧、第 1 框中心
  · 第 2/4 步各合并为一个框，避免"公式→拟合"方向反了
  · 第 4 步不能写 "each monthly total matches"（一对 K,B 做不到），改为最小二乘
  · 数据行内部不画箭头（五个输入是并列的）
  · 补：装机容量取"当年在运容量"；校正拟合期为 2013–2023
"""
import os
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

OUT = Path(os.environ.get(
    "WINDPAPER_FIG_OUT", Path(__file__).resolve().parent / "output"
)) / "fig3_workflow_v18.pptx"
OUT.parent.mkdir(parents=True, exist_ok=True)
W, H = 18.00, 23.16

C_TX = RGBColor(0x1A, 0x1A, 0x1A)
C_MUTE = RGBColor(0x50, 0x50, 0x50)
C_IN = RGBColor(0xFF, 0xF2, 0xCC)
C_IN_LN = RGBColor(0xBF, 0x8F, 0x00)
C_WHT = RGBColor(0xFF, 0xFF, 0xFF)
C_WHT_LN = RGBColor(0x80, 0x80, 0x80)
C_COR = RGBColor(0xDE, 0xEA, 0xF6)
C_COR_LN = RGBColor(0x1F, 0x4E, 0x79)
C_OUT = RGBColor(0xE2, 0xEF, 0xDA)
C_OUT_LN = RGBColor(0x54, 0x7B, 0x2D)
C_ARR = RGBColor(0x33, 0x33, 0x33)
C_TITLE = RGBColor(0x1F, 0x4E, 0x79)

prs = Presentation()
prs.slide_width = Cm(W)
prs.slide_height = Cm(H)
slide = prs.slides.add_slide(prs.slide_layouts[6])


def box(x, y, w, h, text, fill, line, size=9.0, bold=False, lw=1.0):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = line
    sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.12)
    tf.margin_right = Cm(0.12)
    tf.margin_top = Cm(0.05)
    tf.margin_bottom = Cm(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.name = "Arial"
        r.font.color.rgb = C_TX
    return sh


def arrow(x1, y1, x2, y2, lw=1.25):
    cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                    Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    cn.line.color.rgb = C_ARR
    cn.line.width = Pt(lw)
    ln = cn.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"),
                             {"type": "triangle", "w": "med", "len": "med"}))
    return cn


def htext(x, y, w, h, txt, size=9.0, bold=False, color=C_TX):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.05)
    tf.margin_right = Cm(0.05)
    tf.margin_top = Cm(0.02)
    tf.margin_bottom = Cm(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(txt.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.name = "Arial"
        r.font.color.rgb = color
    return tb


TITLE_W = 3.25
LBL_X = 0.62
BOX_X = 4.10
BOX_R = 17.25
ROW_H = 3.10
GAP = 0.72
Y0 = 0.55
PAD = 0.15
SPINE = 5.30                         # 落在第 1 个框内部，而不是框缝


def row(i):
    return Y0 + i * (ROW_H + GAP)


def row_boxes(i, items, fill, line, size=9.0, gap=0.28, chain=False):
    y = row(i)
    n = len(items)
    w = ((BOX_R - BOX_X) - 2 * PAD - (n - 1) * gap) / n
    for k, t in enumerate(items):
        x = BOX_X + PAD + k * (w + gap)
        box(x, y + 0.15, w, ROW_H - 0.30, t, fill, line, size=size)
        if chain and k:
            arrow(x - gap + 0.02, y + ROW_H / 2, x - 0.02, y + ROW_H / 2, lw=1.1)


def row_title(i, txt):
    htext(LBL_X, row(i), TITLE_W, ROW_H, txt, size=10.5, bold=True,
          color=C_TITLE)


# ------------------------------------------------------------ Data
row_title(0, "Data")
row_boxes(0, ["ERA5 reanalysis\nwind, temperature,\npressure\n(step 1)",
              "CDMet\nobserved wind speed\n(steps 1 and 2)",
              "Wind farm list\nlocation, turbine type,\ncapacity, commissioning\n"
              "(step 3)",
              "Official statistics\nprovincial monthly\ngeneration\n(step 4)",
              "Official statistics\nprovincial annual\ncurtailment rate\n"
              "(step 3)"],
          C_IN, C_IN_LN, size=8.4, chain=False)

# ------------------------------------------------------------ 1
row_title(1, "1. Wind speed\nat hub height")
row_boxes(1, ["ERA5 wind at 10 m and 100 m\n\u2192 wind speed at hub height\n"
              "(wind shear and air density)",
              "Observed wind speed\nat hub height (from CDMet)"],
          C_WHT, C_WHT_LN, chain=False)

# ------------------------------------------------------------ 2
row_title(2, "2. Wind-speed\ncorrection")
row_boxes(2, ["applied to the wind speed, before conversion \u2014 one "
              "correction for each\nprovince and season:  corrected wind = "
              "k \u00d7 hub-height wind + b\nk and b fitted to the CDMet "
              "observations by least squares"],
          C_COR, C_COR_LN, chain=False)

# ------------------------------------------------------------ 3
row_title(3, "3. From wind\nto generation")
row_boxes(3, ["wind speed \u2192 power curve of each\nturbine type \u2192 "
              "capacity factor",
              "\u00d7 installed capacity in service\nin that year\n"
              "\u00d7 (1 \u2212 curtailment rate)",
              "sum over farms \u2192 city \u2192 province\nhourly series"],
          C_WHT, C_WHT_LN, chain=True)

# ------------------------------------------------------------ 4
row_title(4, "4. Generation-\nlevel correction")
row_boxes(4, ["applied to the hourly series as one constant factor for each "
              "province and month:\ncorrected generation = K \u00d7 modelled "
              "+ B\nK and B fitted to the official monthly totals by least "
              "squares; the hourly\npattern of step 3 is rescaled but not "
              "reshaped"],
          C_COR, C_COR_LN, chain=False)

# ------------------------------------------------------------ 主箭头（尖端进入框内）
BX3 = 14.80                              # 第 3 步出口 = 第 3 个框
arrow(SPINE, row(1) + ROW_H - 0.10, SPINE, row(2) + 0.17)
arrow(SPINE, row(2) + ROW_H - 0.10, SPINE, row(3) + 0.17)
arrow(BX3, row(3) + ROW_H - 0.10, BX3, row(4) + 0.17)
arrow(SPINE, row(4) + ROW_H - 0.10, SPINE, 19.45)

# ------------------------------------------------------------ 输出
OUT_Y, OUT_H = 19.38, 2.42
box(LBL_X, OUT_Y, BOX_R - LBL_X, OUT_H,
    "Hourly wind power generation for every city in China, 1989\u20132023\n"
    "(onshore and offshore)",
    C_OUT, C_OUT_LN, size=11.5, bold=True, lw=1.5)

# ------------------------------------------------------------ 图注说明
htext(0.75, 22.02, 16.30, 0.92,
      "Both corrections are fitted over 2013\u20132023 and then held fixed. "
      "They use different information: the first fits the wind speed to the "
      "observations, the second fits the monthly totals to the official "
      "statistics.",
      size=8.6, color=C_MUTE)

prs.save(OUT)
print("已生成:", OUT, os.path.getsize(OUT), "B  形状:", len(slide.shapes))
