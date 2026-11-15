"""Rebuild the English item file for run_lang.py from the public WDCT data.

usage: python build_items_en.py <WDCT.csv> <out_items_en.csv>
WDCT.csv comes from https://github.com/icip-cas/Word-Deed-Consistency-Test (data/WDCT.csv).
The 90 item ids are in ../data/item_ids.csv. The output has the columns run_lang.py reads
for the English condition (wdct_id, master_q, en_speak, en_act).
The Korean translation and the English paraphrases are not rebuilt here (available on request).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pick(row, names):
    for n in names:
        if n in row:
            return row[n]
    raise KeyError(f"none of {names} in WDCT columns {list(row)}")


def main():
    wdct, out = sys.argv[1], sys.argv[2]
    ids = list(csv.DictReader(open(os.path.join(HERE, "..", "data", "item_ids.csv"), encoding="utf-8")))
    src = {}
    for r in csv.DictReader(open(wdct, encoding="utf-8")):
        src[pick(r, ("id", "ID"))] = (pick(r, ("word question", "word_question", "speak", "word")),
                                     pick(r, ("deed question", "deed_question", "act", "deed")))
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wdct_id", "master_q", "en_speak", "en_act"])
        for r in ids:
            speak, act = src[r["wdct_id"]]
            w.writerow([r["wdct_id"], r["master_q"], speak, act])
    print("wrote", len(ids), "items to", out)


if __name__ == "__main__":
    main()
