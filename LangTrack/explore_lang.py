"""Exploratory, descriptive only (REPORT_SKELETON.md 탐색 줄 r8, fixed before Korean results).

usage: py explore_lang.py <runs_dir> <prefix>
prints, per model: item-level p̄ (en, en′, ko) for the international-organization group (wav_43-50),
the religion group (wav_62-70) and the five this/your-country items, then every item whose chosen
answer (p̄ > 0.5) differs between English and Korean.
"""
import csv
import os
import sys

GROUPS = {
    "intl_org": [f"wav_{i}" for i in range(43, 51)],
    "religion": [f"wav_{i}" for i in range(62, 71)],
    "country_ref": ["wav_89", "wav_90", "wav_92", "wav_93", "wav_94"],
}


def pbar(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[r["id"]] = {sd: (float(r[f"{sd}_pA_orig"]) + float(r[f"{sd}_pA_swap"])) / 2 for sd in ("speak", "act")}
    return out


def main():
    d, pre = sys.argv[1], sys.argv[2]
    p = {lg: pbar(os.path.join(d, f"{pre}{lg}.csv")) for lg in ("en", "en2", "ko")}
    for g, ids in GROUPS.items():
        print(f"-- {g}  (p̄ = P(option A), A = Yes / first option)")
        for i in ids:
            if i not in p["ko"]:
                continue
            row = " | ".join(f"{sd} en {p['en'][i][sd]:.2f} en′ {p['en2'][i][sd]:.2f} ko {p['ko'][i][sd]:.2f}" for sd in ("speak", "act"))
            print(f"{i:7} {row}")
    for sd in ("speak", "act"):
        flips = [i for i in p["ko"] if (p["en"][i][sd] > 0.5) != (p["ko"][i][sd] > 0.5)]
        print(f"-- answer changed EN→KO ({sd}): {len(flips)}  " + ", ".join(
            f"{i}({p['en'][i][sd]:.2f}→{p['ko'][i][sd]:.2f})" for i in flips))


if __name__ == "__main__":
    main()
