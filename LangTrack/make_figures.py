"""Report figures as plain SVG (no plotting library). 2026-11-14.

usage: py make_figures.py <runs_dir> <out_dir>
fig_shift.svg  = mean |Δ| (English → Korean) and mean |Δ′| (English → English paraphrase), speak vs act, per model, 95% bootstrap CI
fig_design.svg = design schematic
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_lang import boot_mean, cell, load, pbar  # noqa: E402

MODELS = [("full_", "Qwen2.5-1.5B-Instruct"), ("exa_full_", "EXAONE-4.0-1.2B")]
COL = {"ko": "#2b6cb0", "en2": "#a0aec0"}


def shifts(d, pre):
    data = {lg: load(os.path.join(d, f"{pre}{lg}.csv")) for lg in ("en", "en2", "ko")}
    out = {}
    for sd in ("speak", "act"):
        en, ko, en2 = (pbar(cell(data[lg], sd)[0]) for lg in ("en", "ko", "en2"))
        ids = [i for i in en if i in ko and i in en2]
        out[sd] = {"ko": boot_mean([abs(ko[i] - en[i]) for i in ids]),
                   "en2": boot_mean([abs(en2[i] - en[i]) for i in ids])}
    return out


def fig_shift(d, path):
    W, H, L, B, T = 760, 380, 70, 60, 50
    ymax = 0.45
    y = lambda v: H - B - (v / ymax) * (H - B - T)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for t in (0, 0.1, 0.2, 0.3, 0.4):
        s.append(f'<line x1="{L}" x2="{W-20}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#e2e8f0"/>')
        s.append(f'<text x="{L-8}" y="{y(t)+4:.1f}" text-anchor="end" fill="#4a5568">{t:.1f}</text>')
    s.append(f'<text x="18" y="{(H-B+T)/2:.0f}" transform="rotate(-90 18 {(H-B+T)/2:.0f})" text-anchor="middle" fill="#2d3748">mean |change in P(A)|</text>')
    gw = (W - L - 20) / 4
    k = 0
    for pre, name in MODELS:
        sh = shifts(d, pre)
        for sd in ("speak", "act"):
            x0 = L + k * gw + gw * 0.2
            for j, cond in enumerate(("ko", "en2")):
                m, lo, hi = sh[sd][cond]
                x = x0 + j * gw * 0.32
                w = gw * 0.28
                s.append(f'<rect x="{x:.1f}" y="{y(m):.1f}" width="{w:.1f}" height="{y(0)-y(m):.1f}" fill="{COL[cond]}"/>')
                cx = x + w / 2
                s.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" stroke="#1a202c" stroke-width="1.5"/>')
                s.append(f'<text x="{cx:.1f}" y="{y(hi)-6:.1f}" text-anchor="middle" font-size="11" fill="#1a202c">{m:.2f}</text>')
            s.append(f'<text x="{x0+gw*0.3:.1f}" y="{H-B+18}" text-anchor="middle" fill="#2d3748">{sd}</text>')
            k += 1
        s.append(f'<text x="{L + (k-1)*gw:.1f}" y="{H-B+38}" text-anchor="middle" font-weight="bold" fill="#2d3748">{name}</text>')
    s.append(f'<rect x="{W-300}" y="14" width="12" height="12" fill="{COL["ko"]}"/><text x="{W-282}" y="25">English → Korean (|Δ|)</text>')
    s.append(f'<rect x="{W-300}" y="32" width="12" height="12" fill="{COL["en2"]}"/><text x="{W-282}" y="43">English → English paraphrase (|Δ′|)</text>')
    s.append("</svg>")
    open(path, "w", encoding="utf-8").write("\n".join(s))


def fig_design(path):
    W, H = 760, 250
    box = lambda x, y, w, h, t, f="#edf2f7": (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{f}" stroke="#4a5568"/>'
                                              + "".join(f'<text x="{x+w/2}" y="{y+20+16*i}" text-anchor="middle">{ln}</text>' for i, ln in enumerate(t)))
    arrow = lambda x1, y1, x2, y2: f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4a5568" stroke-width="1.5" marker-end="url(#a)"/>'
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12">',
         '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#4a5568"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         box(10, 85, 150, 70, ["90 WDCT pairs", "(WVS Wave 7 items)", "speak + act"]),
         box(220, 15, 170, 56, ["English (WDCT original)"], "#ebf8ff"),
         box(220, 92, 170, 56, ["English paraphrase", "(wording baseline)"], "#f7fafc"),
         box(220, 169, 170, 56, ["Korean translation", "(owner-reviewed)"], "#ebf8ff"),
         box(450, 92, 140, 56, ["P(A) per option order", "(next token, 2 models)"]),
         box(630, 60, 120, 120, ["0: order agreement", "≥ 60%", "1: corrected shift", "c = |Δ| − |Δ′|", "2: speak–act link"], "#fffaf0"),
         arrow(160, 120, 220, 43), arrow(160, 120, 220, 120), arrow(160, 120, 220, 197),
         arrow(390, 43, 450, 112), arrow(390, 120, 450, 120), arrow(390, 197, 450, 128), arrow(590, 120, 630, 120),
         "</svg>"]
    open(path, "w", encoding="utf-8").write("\n".join(s))


if __name__ == "__main__":
    d, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    fig_shift(d, os.path.join(out, "fig_shift.svg"))
    fig_design(os.path.join(out, "fig_design.svg"))
    print("ok")
