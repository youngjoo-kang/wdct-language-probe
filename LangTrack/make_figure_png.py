"""Figure 1 as PNG (PIL only, for the Google Docs template). 2026-11-15.

usage: py make_figure_png.py <runs_dir> <out.png>
Same data as fig_shift.svg in make_figures.py, drawn at 2x for legibility.
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import MODELS, shifts  # noqa: E402

S = 2
W, H, L, B, T = 760 * S, 380 * S, 70 * S, 60 * S, 50 * S
YMAX = 0.45
COL = {"ko": (43, 108, 176), "en2": (160, 174, 192)}


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts", name), size * S)


def y(v):
    return H - B - (v / YMAX) * (H - B - T)


def main(d, path):
    im = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(im)
    f12, f11, f13b = font(13), font(11), font(13, True)
    for t in (0, 0.1, 0.2, 0.3, 0.4):
        dr.line([(L, y(t)), (W - 20 * S, y(t))], fill=(226, 232, 240), width=S)
        dr.text((L - 8 * S, y(t)), f"{t:.1f}", fill=(74, 85, 104), font=f12, anchor="rm")
    lab = Image.new("RGBA", (300 * S, 20 * S), (255, 255, 255, 0))
    ImageDraw.Draw(lab).text((150 * S, 10 * S), "mean |change in P(A)|", fill=(45, 55, 72), font=f12, anchor="mm")
    lab = lab.rotate(90, expand=True)
    im.paste(lab, (int(8 * S), int((H - B + T) / 2 - lab.size[1] / 2)), lab)
    gw = (W - L - 20 * S) / 4
    k = 0
    for pre, name in MODELS:
        sh = shifts(d, pre)
        for sd in ("speak", "act"):
            x0 = L + k * gw + gw * 0.2
            for j, cond in enumerate(("ko", "en2")):
                m, lo, hi = sh[sd][cond]
                x = x0 + j * gw * 0.32
                w = gw * 0.28
                dr.rectangle([x, y(m), x + w, y(0)], fill=COL[cond])
                cx = x + w / 2
                dr.line([(cx, y(lo)), (cx, y(hi))], fill=(26, 32, 44), width=int(1.5 * S))
                dr.text((cx, y(hi) - 6 * S), f"{m:.2f}", fill=(26, 32, 44), font=f11, anchor="mb")
            dr.text((x0 + gw * 0.3, H - B + 12 * S), sd, fill=(45, 55, 72), font=f12, anchor="mt")
            k += 1
        dr.text((L + (k - 1) * gw, H - B + 32 * S), name, fill=(45, 55, 72), font=f13b, anchor="mt")
    for r, (cond, txt) in enumerate((("ko", "English → Korean (|Δ|)"), ("en2", "English → English paraphrase (|Δ′|)"))):
        yy = (14 + 18 * r) * S
        dr.rectangle([W - 300 * S, yy, W - 288 * S, yy + 12 * S], fill=COL[cond])
        dr.text((W - 282 * S, yy + 6 * S), txt, fill=(0, 0, 0), font=f12, anchor="lm")
    im.save(path)
    print("ok", path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
