"""Pre-registered analysis for the language-swap probe (REPORT_SKELETON.md 「사전 고정」 r7, 2026-11-14).

usage: py analyze_lang.py <runs_dir> <prefix>      e.g. prefix "full_" (Qwen) or "exa_full_" (EXAONE)
reads <prefix>en.csv, <prefix>en2.csv (paraphrase baseline) and, if present, <prefix>ko.csv.
Every number the rules use is printed; verdict lines follow the block wording.
"""
import csv
import os
import sys

import numpy as np

B, SEED, GATE, MOVE, GAP = 10000, 0, 0.60, 0.10, 0.10
SIDES = ("speak", "act")


def load(path):
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        rows[r["id"]] = r
    return rows


def cell(rows, side):
    """{id: (p_orig, p_swap)} after exclusions (mass < 0.5 in either order, NaN swap)."""
    out, excl = {}, 0
    for i, r in rows.items():
        po, ps = float(r[f"{side}_pA_orig"]), float(r[f"{side}_pA_swap"])
        mo, ms = float(r[f"{side}_mass_orig"]), float(r[f"{side}_mass_swap"])
        if np.isnan(po) or np.isnan(ps) or mo < 0.5 or ms < 0.5:
            excl += 1
            continue
        out[i] = (po, ps)
    return out, excl


def pick(p):
    return "A" if p > 0.5 else "B" if p < 0.5 else "tie"


def agreement(c):
    same = sum(1 for po, ps in c.values() if pick(po) == pick(ps) and pick(po) != "tie")
    return same / len(c) if c else float("nan")


def kappa(c):
    a = [pick(po) == "A" for po, ps in c.values()]
    b = [pick(ps) == "A" for po, ps in c.values()]
    n = len(a)
    if not n:
        return float("nan")
    po_ = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po_ - pe) / (1 - pe) if pe < 1 else float("nan")


def pbar(c):
    return {i: (po + ps) / 2 for i, (po, ps) in c.items()}


def boot_mean(x):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    m = rng.choice(x, (B, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def ranks(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    v = np.asarray(v)
    for u in np.unique(v):  # average ranks for ties
        idx = v == u
        r[idx] = r[idx].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def boot_spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(SEED)
    n = len(x)
    bs = []
    for _ in range(B):
        k = rng.integers(0, n, n)
        if np.std(x[k]) == 0 or np.std(y[k]) == 0:
            continue
        bs.append(spearman(x[k], y[k]))
    return spearman(x, y), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def where(lo, hi):
    return "above0" if lo > 0 else "below0" if hi < 0 else "contains0"


def main():
    d, pre = sys.argv[1], sys.argv[2]
    data = {lg: load(os.path.join(d, f"{pre}{lg}.csv")) for lg in ("en", "en2", "ko")
            if os.path.exists(os.path.join(d, f"{pre}{lg}.csv"))}
    cells, agr = {}, {}
    print(f"== {pre} | conditions: {', '.join(data)}")
    print("-- 0단계: order agreement (gate 60%), kappa reference, exclusions")
    for lg in data:
        for sd in SIDES:
            c, ex = cell(data[lg], sd)
            cells[lg, sd], agr[lg, sd] = c, agreement(c)
            print(f"{lg:3} {sd:5} n={len(c):3} excl={ex} agree={agr[lg, sd]:.3f} "
                  f"{'PASS' if agr[lg, sd] >= GATE else 'FAIL'} kappa={kappa(c):.3f}")
    en2_ok = all(agr.get(("en2", sd), 0) >= GATE for sd in SIDES)
    print(f"baseline en′ usable: {en2_ok}")
    if "ko" not in data:
        return
    print("-- 0단계 prediction (KO lower than EN by ≥ 5pp on each side)")
    diffs = {sd: agr["en", sd] - agr["ko", sd] for sd in SIDES}
    if all(v >= 0.05 for v in diffs.values()):
        v = "맞음"
    elif any(agr["ko", sd] >= agr["en", sd] for sd in SIDES):
        v = "틀림"
    else:
        v = "차이 작음"
    print(f"EN−KO speak {diffs['speak']:+.3f} act {diffs['act']:+.3f} → {v}")

    side_ok = {sd: agr["en", sd] >= GATE and agr["ko", sd] >= GATE for sd in SIDES}
    absd, c_ = {}, {}
    print("-- 이동 (|Δ| = |p̄ko − p̄en|, |Δ′| = |p̄en′ − p̄en|, c = |Δ| − |Δ′|)")
    moved = {}
    for sd in SIDES:
        en, ko, en2 = pbar(cells["en", sd]), pbar(cells["ko", sd]), pbar(cells.get(("en2", sd), {}))
        ids = [i for i in en if i in ko and i in en2]
        absd[sd] = {i: abs(ko[i] - en[i]) for i in ids}
        c_[sd] = {i: absd[sd][i] - abs(en2[i] - en[i]) for i in ids}
        a = np.array(list(absd[sd].values()))
        m, lo, hi = boot_mean(list(c_[sd].values()))
        print(f"{sd:5} n={len(ids)} mean|Δ|={a.mean():.3f} SD={a.std():.3f} IQR={np.percentile(a, 75) - np.percentile(a, 25):.3f} "
              f"mean|Δ′|={np.mean([abs(en2[i] - en[i]) for i in ids]):.3f} mean c={m:.3f} [{lo:.3f}, {hi:.3f}]")
        if not side_ok[sd] or not en2_ok:
            moved[sd] = "판정 불가"
        else:
            moved[sd] = "움직임" if a.mean() >= MOVE and lo > 0 else "안 움직임"
        print(f"      3×3 axis = {moved[sd]}")
        # descriptive flip_conf: confident (|p̄ − 0.5| > 0.2) in both languages
        conf = [i for i in ids if abs(en[i] - 0.5) > 0.2 and abs(ko[i] - 0.5) > 0.2]
        flips = sum(1 for i in conf if (en[i] > 0.5) != (ko[i] > 0.5))
        print(f"      flip_conf = {flips}/{len(conf)}")

    print("-- H2: mean(act c − speak c)")
    ids = [i for i in c_["speak"] if i in c_["act"]]
    gate2 = all(side_ok.values()) and en2_ok and all(abs(agr[lg, "speak"] - agr[lg, "act"]) < GAP for lg in ("en", "ko"))
    m, lo, hi = boot_mean([c_["act"][i] - c_["speak"][i] for i in ids])
    m0, lo0, hi0 = boot_mean([absd["act"][i] - absd["speak"][i] for i in ids])
    verdict = {"above0": "성립", "contains0": "행동 > 말 증거 없음", "below0": "반증(말 > 행동)"}[where(lo, hi)]
    print(f"n={len(ids)} corrected {m:.3f} [{lo:.3f}, {hi:.3f}] | uncorrected {m0:.3f} [{lo0:.3f}, {hi0:.3f}]")
    print(f"H2 → {verdict if gate2 else '판정 불가'} (gate {'pass' if gate2 else 'fail'})")

    print("-- H1: Spearman(|Δ| speak, |Δ| act)")
    r, lo, hi = boot_spearman([absd["speak"][i] for i in ids], [absd["act"][i] for i in ids])
    gate1 = moved["speak"] == "움직임" and moved["act"] == "움직임"
    verdict = {"contains0": "연동 증거 없음", "above0": "반증(같이 감)", "below0": "반증(반대로 감)"}[where(lo, hi)]
    print(f"n={len(ids)} r={r:.3f} [{lo:.3f}, {hi:.3f}]")
    print(f"H1 → {verdict if gate1 else '판정 불가(움직임이 작아 연동을 볼 수 없음)'}")


if __name__ == "__main__":
    main()
