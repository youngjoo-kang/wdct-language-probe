# direction_korea.py
# EXPLORATORY analysis, defined AFTER the language-condition results were seen (2026-11-14).
# Not preregistered. Question: when Qwen2.5-1.5B-Instruct and EXAONE-4.0-1.2B answer the WDCT
# "speak" (stated-value) items in Korean instead of English, does P(option A) move toward the
# Korean public's answer share relative to the US public's (GlobalOpinionQA / WVS-7)?
#
# usage: py -I -X utf8 direction_korea.py   (writes runs/direction_korea.txt)
# Inputs (read only): runs/{full,exa_full}_{en,ko}.csv, WVS7/wdct_wvs_ko_pairs.csv,
#   GlobalOpinionQA raw global_opinions.csv (HF Anthropic/llm_global_opinions).
# Index note: goqa_idx in the pairs file is the row index into the WVS-source subset of
#   global_opinions.csv (353 rows, file order), NOT the full 2556-row file. Verified by reading
#   the GOQA question tail against master_label for all used items (plus a token-overlap check
#   printed in the output).
import ast
import csv
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
# The pairing file (WDCT id -> WVS question, GlobalOpinionQA index, WVS label) contains WVS text and is
# available on request; GlobalOpinionQA is at huggingface.co/datasets/Anthropic/llm_global_opinions.
PAIRS = os.environ.get("PAIRS_CSV", "wdct_wvs_ko_pairs.csv")
GOQA = os.environ.get("GOQA_CSV", "global_opinions.csv")
OUT = os.path.join(RUNS, "direction_korea.txt")
MODELS = {"Qwen2.5-1.5B-Instruct": ("full_en.csv", "full_ko.csv"),
          "EXAONE-4.0-1.2B": ("exa_full_en.csv", "exa_full_ko.csv")}
THR = 0.05          # |g| and |delta| threshold for test (a)
NBOOT, SEED = 10_000, 0
AMBIGUOUS = {"wav_92": "speak_q asks 'a VERY good way?' but WVS scale is very/fairly good/bad",
             "wav_93": "speak_q asks 'a VERY good way?' but WVS scale is very/fairly good/bad"}

csv.field_size_limit(10**9)


# ---------- option collapse ----------
def is_missing(o):
    o = o.lower()
    return o in ("don't know", "no answer") or o.startswith("missing") or o.startswith("other missing")


def item_type(opts):
    s = " | ".join(opts).lower()
    if "justifiable" in s:
        return "justif10"
    if "completely disagree" in s and "completely agree" in s:
        return "agree10"
    if "trust completely" in s:
        return "trust4"
    if "a great deal" in s:
        return "confidence4"
    if "should have the right" in s:
        return "gov_right4"
    if opts[0] == "Freedom":
        return "freedom_vs"
    if "very good" in s:
        return "goodway4"
    if "agree" in s:
        return "agree_neither" if "neither" in s else "agree4"
    raise ValueError(opts)


def collapse(opts, typ, neither="drop", justif="1-5|6-10"):
    """Return list of 'A'/'B'/None per GOQA option (None = dropped, renormalised away)."""
    out = []
    for i, o in enumerate(opts):
        lo = o.lower()
        if is_missing(o):
            out.append(None)
        elif typ in ("agree10", "justif10"):
            # options[0..9] = scale points 1..10; 10 = completely agree / always justifiable = model 'Yes'
            if typ == "justif10" and justif == "1|2-10":
                out.append("B" if i == 0 else "A")
            else:
                out.append("B" if i <= 4 else "A")
        elif typ in ("agree4", "agree_neither"):
            if "neither" in lo:
                out.append(None if neither == "drop" else "B")
            else:
                out.append("B" if "disagree" in lo else "A")
        elif typ == "trust4":
            out.append("B" if lo.startswith("do not") else "A")
        elif typ == "confidence4":
            out.append("A" if lo in ("a great deal", "quite a lot") else "B")
        elif typ == "gov_right4":
            out.append("B" if "should not" in lo else "A")
        elif typ == "freedom_vs":
            out.append("A" if i == 0 else "B")
        elif typ == "goodway4":
            out.append("A" if "good" in lo else "B")
    return out


def share_A(dist, amap):
    a = sum(p for p, m in zip(dist, amap) if m == "A")
    b = sum(p for p, m in zip(dist, amap) if m == "B")
    return a / (a + b)


# ---------- stats ----------
def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def binom_two_sided(k, n):
    pmf = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9)))


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(fn, *arrs):
    rng = np.random.default_rng(SEED)
    n = len(arrs[0])
    vals = []
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrs])
        if not math.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def toks(s):
    return set(re.findall(r"[a-z]+", s.lower())) - {"the", "a", "of", "to", "and", "for", "in", "is", "than"}


# ---------- load ----------
def load_goqa():
    rows = [r for r in csv.DictReader(open(GOQA, encoding="utf-8")) if r["source"] == "WVS"]
    for r in rows:
        s = r["selections"]
        if s.startswith("defaultdict("):
            s = s[s.index("{"):-1]
        r["sel"] = ast.literal_eval(s)
        r["opts"] = ast.literal_eval(r["options"])
    return rows


def load_run(fn):
    d = {}
    for r in csv.DictReader(open(os.path.join(RUNS, fn), encoding="utf-8")):
        d[r["id"]] = (float(r["speak_pA_orig"]) + float(r["speak_pA_swap"])) / 2
    return d


def build_items(goqa, ran, pairs, neither="drop", justif="1-5|6-10", include_ambiguous=False):
    items, excl = [], []
    for p in pairs:
        i = p["wdct_id"]
        if i not in ran or p["korea_dist"] != "True":
            continue
        g = goqa[int(p["goqa_idx"])]
        if "United States" not in g["sel"] or "South Korea" not in g["sel"]:
            excl.append((i, "no US or KR distribution"))
            continue
        if i in AMBIGUOUS and not include_ambiguous:
            excl.append((i, "ambiguous mapping: " + AMBIGUOUS[i]))
            continue
        typ = item_type(g["opts"])
        amap = collapse(g["opts"], typ, neither, justif)
        kr = share_A(g["sel"]["South Korea"], amap)
        us = share_A(g["sel"]["United States"], amap)
        ml = toks(p["master_label"])
        tail = toks(g["question"][-160:])
        overlap = len(ml & tail) / max(1, len(ml))
        items.append(dict(id=i, mq=p["master_q"], typ=typ, kr=kr, us=us, g=kr - us, overlap=overlap,
                          amap=amap, opts=g["opts"]))
    return items, excl


def summarize(items, run_en, run_ko, lines, label, full=True):
    ids = [it["id"] for it in items]
    g = np.array([it["g"] for it in items])
    pen = np.array([run_en[i] for i in ids])
    pko = np.array([run_ko[i] for i in ids])
    d = pko - pen
    kr = np.array([it["kr"] for it in items])
    sel = (np.abs(g) >= THR) & (np.abs(d) >= THR)
    k = int(np.sum(np.sign(d[sel]) == np.sign(g[sel])))
    n = int(sel.sum())
    lo, hi = wilson(k, n)
    rho = spearman(d, g)
    rlo, rhi = boot(spearman, d, g)
    md = float(d.mean())
    mlo, mhi = boot(lambda x: float(x.mean()), d)
    closer = int(np.sum(np.abs(pko - kr) < np.abs(pen - kr)))
    if not full:
        lines.append(f"  {label}: n={len(items)} | (a) {k}/{n} toward = {k/n if n else float('nan'):.2f} "
                     f"[{lo:.2f},{hi:.2f}] | (b) rho={rho:+.2f} [{rlo:+.2f},{rhi:+.2f}] | "
                     f"(c) mean d={md:+.3f} [{mlo:+.3f},{mhi:+.3f}] | (d) closer {closer}/{len(items)}")
        return
    lines.append(f"  n items = {len(items)}")
    lines.append(f"  (a) toward Korea, among |g|>={THR} and |delta|>={THR}: {k}/{n} = "
                 f"{k/n if n else float('nan'):.3f}, Wilson 95% CI [{lo:.3f}, {hi:.3f}], chance 0.50, "
                 f"exact two-sided binomial p = {binom_two_sided(k, n) if n else float('nan'):.3f}")
    lines.append(f"  (b) Spearman(delta, g) = {rho:+.3f}, item bootstrap 95% CI [{rlo:+.3f}, {rhi:+.3f}] "
                 f"({NBOOT} resamples, seed {SEED}, percentile)")
    lines.append(f"  (c) mean signed delta (drift toward option A / 'Yes') = {md:+.4f}, "
                 f"bootstrap 95% CI [{mlo:+.4f}, {mhi:+.4f}]; delta>0 on {int((d > 0).sum())}/{len(d)} items")
    lines.append(f"  (d) p(ko) closer to KR_A than p(en) is: {closer}/{len(items)}")
    # 2x2: drift vs direction
    lines.append(f"  sign(g) x sign(delta) over the (a) set (n={n}); rows = g sign, cols = delta sign:")
    for gs, name in ((1, "g>0 (KR more A)"), (-1, "g<0 (US more A)")):
        m = sel & (np.sign(g) == gs)
        up = int(np.sum(d[m] > 0))
        dn = int(np.sum(d[m] < 0))
        lines.append(f"     {name:18s}  delta>0: {up:3d}   delta<0: {dn:3d}   toward: {up if gs > 0 else dn}/{up + dn}")
    # competing explanation: Korean prompt pulls P(A) toward 0.5 (hedging), not toward Korea
    tw = np.sign(d) == np.sign(g)
    h = np.sign(d) == np.sign(0.5 - pen)
    kh = int(np.sum(sel & h))
    lines.append(f"  Hedging check over the (a) set: delta points toward 0.5 (sign(delta) == sign(0.5 - p_en)) on {kh}/{n}.")
    lines.append(f"     toward-Korea & toward-0.5: {int(np.sum(sel & tw & h))} | toward-Korea only: {int(np.sum(sel & tw & ~h))}"
                 f" | toward-0.5 only: {int(np.sum(sel & ~tw & h))} | neither: {int(np.sum(sel & ~tw & ~h))}")
    lines.append(f"     items toward-Korea but away from 0.5: {[i for i, s, t, x in zip(ids, sel, tw, h) if s and t and not x]}")
    # battery clustering
    bats = {}
    for it, s, t in zip(items, sel, tw):
        if s:
            bats.setdefault(battery(int(it["mq"])), []).append(bool(t))
    maj = sum(1 for v in bats.values() if sum(v) > len(v) / 2)
    tie = sum(1 for v in bats.values() if sum(v) == len(v) / 2)
    lines.append(f"  Battery clustering: (a) items fall in {len(bats)} WVS batteries; majority toward Korea in {maj}, tie in {tie}: "
                 + "; ".join(f"{b} {sum(v)}/{len(v)}" for b, v in bats.items()))
    return d, pen, pko, sel


BATTERIES = [((27, 35), "gender/jobs Q27-35"), ((36, 38), "family Q36-38"), ((39, 41), "work Q39-41"),
             ((57, 63), "trust Q57-63"), ((64, 89), "confidence Q64-89"), ((149, 150), "freedom Q149-150"),
             ((158, 163), "science Q158-163"), ((169, 172), "religion Q169-172"), ((177, 195), "justifiable Q177-195"),
             ((196, 198), "gov rights Q196-198"), ((235, 238), "political system Q235-238")]


def battery(mq):
    for (lo, hi), name in BATTERIES:
        if lo <= mq <= hi:
            return name
    return f"Q{mq}"


def main():
    goqa = load_goqa()
    pairs = list(csv.DictReader(open(PAIRS, encoding="utf-8-sig")))
    ran = set(load_run("full_ko.csv"))
    runs = {m: (load_run(a), load_run(b)) for m, (a, b) in MODELS.items()}
    for m, (e, k) in runs.items():
        assert set(e) == set(k) == ran, m

    items, excl = build_items(goqa, ran, pairs)
    n_kd = sum(1 for p in pairs if p["wdct_id"] in ran and p["korea_dist"] == "True")
    L = []
    L.append("direction_korea.txt  -- EXPLORATORY, defined after results were seen (2026-11-14). Not preregistered.")
    L.append("Question: does asking in Korean (vs English) move each model's P(option A) toward the Korean public's")
    L.append("share for option A, relative to the US public's share? Speak (stated-value) side only.")
    L.append("")
    L.append("== Choices ==")
    L.append("p(cond) = mean of speak_pA_orig and speak_pA_swap (P of original option A, order-averaged).")
    L.append("Public data: GlobalOpinionQA (HF Anthropic/llm_global_opinions), WVS-source rows only. goqa_idx indexes the")
    L.append("  353-row WVS subset in file order (not the full 2556 rows; row 5 of the full file is an Orban item).")
    L.append("  The pairs file has no goqa_question column, so the match was verified by reading the GOQA question tail")
    L.append("  against master_label for every used item; token-overlap check is the 'ovl' column below.")
    L.append("Option A of the model = 'Yes' for yes/no items; 'Freedom' for the two freedom-vs items.")
    L.append("Collapse of GOQA options into A / B (Don't know, No answer, Missing/Other missing dropped, rest renormalised):")
    L.append("  agree4        Agree strongly + Agree -> A ; Disagree + Strongly disagree -> B")
    L.append("  agree_neither as agree4; 'Neither agree nor disagree' DROPPED (sensitivity: Neither -> B)")
    L.append("  trust4        Trust completely + Trust somewhat -> A ; Do not trust very much / at all -> B")
    L.append("  confidence4   A great deal + Quite a lot -> A ; Not very much + None at all -> B")
    L.append("  gov_right4    Definitely/Probably should have the right -> A ; .../should not ... -> B")
    L.append("  freedom_vs    Freedom -> A ; Equality or Security -> B")
    L.append("  agree10       1-10 (10 = completely agree): 6-10 -> A ; 1-5 -> B")
    L.append("  justif10      1-10 (10 = always justifiable; model A = 'Yes, justified'): 6-10 -> A ; 1-5 -> B")
    L.append("                (sensitivity: 2-10 -> A, 1 'Never justifiable' -> B)")
    L.append("  goodway4      Very good + Fairly good -> A ; Fairly bad + Very bad -> B  (AMBIGUOUS, excluded in primary)")
    L.append(f"KR_A, US_A = South Korea / United States share for A. g = KR_A - US_A. delta = p(ko) - p(en).")
    L.append(f"(a) threshold |g| >= {THR} and |delta| >= {THR}; 'toward' = sign(delta) == sign(g). Wilson 95% CI.")
    L.append(f"(b) Spearman with average ranks; (b),(c) item bootstrap, {NBOOT} resamples, numpy default_rng({SEED}), percentile CI.")
    L.append("(d) count of items with |p(ko) - KR_A| < |p(en) - KR_A|.")
    L.append("")
    L.append("== Items ==")
    L.append(f"Run items: {len(ran)}. With korea_dist == True: {n_kd}. All {n_kd} also have a US distribution.")
    L.append(f"Excluded: {len(excl)}")
    for i, why in excl:
        L.append(f"  {i}: {why}")
    L.append(f"Used: n = {len(items)}")
    from collections import Counter
    L.append("By type: " + ", ".join(f"{t}={c}" for t, c in sorted(Counter(it['typ'] for it in items).items())))
    low = [it["id"] for it in items if it["overlap"] < 0.5]
    L.append(f"Token-overlap check (master_label vs GOQA question tail) < 0.5: {low if low else 'none'}")
    L.append("")

    for m, (run_en, run_ko) in runs.items():
        L.append(f"== {m}: item table ==")
        L.append(f"{'id':7s} {'mq':>4s} {'type':13s} {'ovl':>4s} {'KR_A':>6s} {'US_A':>6s} {'g':>7s} "
                 f"{'p_en':>6s} {'p_ko':>6s} {'delta':>7s}  toward?")
        for it in items:
            pe, pk = run_en[it["id"]], run_ko[it["id"]]
            dd = pk - pe
            if abs(it["g"]) >= THR and abs(dd) >= THR:
                tw = "yes" if np.sign(dd) == np.sign(it["g"]) else "no"
            else:
                tw = "-"
            L.append(f"{it['id']:7s} {it['mq']:>4s} {it['typ']:13s} {it['overlap']:4.2f} {it['kr']:6.3f} {it['us']:6.3f} "
                     f"{it['g']:+7.3f} {pe:6.3f} {pk:6.3f} {dd:+7.3f}  {tw}")
        L.append("  ('-' = below the |g| or |delta| threshold, not counted in (a))")
        L.append("")
    L.append(f"g > 0 on {sum(it['g'] > 0 for it in items)}/{len(items)} items (KR more option-A than US); "
             f"|g| >= {THR} on {sum(abs(it['g']) >= THR for it in items)}.")
    L.append("")

    for m, (run_en, run_ko) in runs.items():
        L.append(f"== {m}: summary (primary) ==")
        summarize(items, run_en, run_ko, L, m)
        L.append("  Sensitivity (same statistics, one choice changed at a time):")
        for lab, kw in (("Neither -> B", dict(neither="B")),
                        ("justif10 split 1 | 2-10", dict(justif="1|2-10")),
                        ("include wav_92/93 (very+fairly good -> A)", dict(include_ambiguous=True))):
            its, _ = build_items(goqa, ran, pairs, **kw)
            summarize(its, run_en, run_ko, L, lab, full=False)
        L.append("")

    L.append("== Fragility notes ==")
    L.append("- Defined after results; small n; (a) uses only items above both thresholds.")
    L.append("- A uniform drift toward 'Yes' in Korean looks like 'toward Korea' whenever g > 0 dominates;")
    L.append("  read (a) with (c) and the sign(g) x sign(delta) table.")
    L.append("- justif10: almost all public mass is at 1-3, so the 1-5|6-10 split makes KR_A and US_A tiny; the split choice")
    L.append("  changes g materially (see sensitivity line).")
    L.append("- The Korean public answered the official Korean WVS questionnaire, whose wording differs in meaning from the")
    L.append("  English on some items (e.g. wav_90 'without their knowledge' vs official 'without consent'; Q235 official")
    L.append("  'parliament and parties' vs 'parliament and elections'). The ko prompts partly follow the official Korean")
    L.append("  (official wording where meaning matched, English-faithful where it did not, e.g. wav_92/93), so delta mixes")
    L.append("  language with translation-meaning shift. On wav_90 the ko prompt and the KR public share '동의 없이'")
    L.append("  while the en prompt says 'knowledge', so wording favours 'toward' there by construction.")
    L.append("- Hedging: where p_en sits near 0 or 1, the only room for |delta| >= 0.05 is toward 0.5; if that direction")
    L.append("  coincides with sign(g), 'toward Korea' cannot be told apart from 'less extreme in Korean' (see hedging check).")
    L.append("- Items are not independent: they come in WVS batteries (see battery clustering line); the binomial p and")
    L.append("  bootstrap CIs treat items as independent and so overstate precision.")
    L.append("- Two-option collapse discards intensity; 'Neither' handling is a choice (see sensitivity).")
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
