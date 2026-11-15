# WDCT language probe

Code and run outputs for the Apart Research AI x Epistemics Sprint (Track 1) report
*Rewording Moves a Small Model's Stated Values Nearly as Much as Switching Language:
A Pre-registered Word–Deed Probe in English and Korean* (November 2026).

Two small open models (Qwen2.5-1.5B-Instruct, EXAONE-4.0-1.2B) answer 90 stated-value / scenario-choice
pairs from the Word–Deed Consistency Test (WDCT) in English, in a meaning-preserving English paraphrase,
and in a reviewed Korean translation. Each item is asked in both option orders and scored by the
next-token probability of "A" vs "B".

## Contents

| Path | What |
|---|---|
| `run_probs.py` | Word–deed harness (written before the sprint): option swap, A/B token ids, next-token P(A) |
| `LangTrack/run_lang.py` | Runs one model on one condition (`en`, `en2` = paraphrase, `ko`) |
| `LangTrack/analyze_lang.py` | Pre-registered analysis (order agreement gate, corrected shift c, H2, H1, bootstrap CIs) |
| `LangTrack/explore_lang.py` | Exploratory tables (religion, "this country" items, answer flips) |
| `LangTrack/direction_korea.py` | Exploratory: shifts toward the Korean public's opinions (GlobalOpinionQA) |
| `LangTrack/make_figures.py`, `make_figure_png.py` | Figure 1 |
| `LangTrack/build_items_en.py` | Rebuilds the English item file from the public WDCT data |
| `LangTrack/runs/` | Run outputs: item id, WVS question number, P(A) in both orders and A/B mass, per model and condition; analysis text files |
| `data/item_ids.csv` | The 90 WDCT item ids and their WVS Wave 7 question numbers |

## Data

Item text is not redistributed here. The WDCT items are from Xu et al. (ICLR 2025),
[icip-cas/Word-Deed-Consistency-Test](https://github.com/icip-cas/Word-Deed-Consistency-Test)
(`data/WDCT.csv`; that repository's README releases its code under CC BY-NC-SA 4.0 for noncommercial use).
The World Values Survey Wave 7 questionnaires are from [worldvaluessurvey.org](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp).

- English items: `python LangTrack/build_items_en.py WDCT.csv items_en.csv`
- Korean translation (reviewed item by item) and English paraphrases: available from the author on request for research use.

## Reproduce

```bash
pip install torch transformers pandas numpy
python LangTrack/run_lang.py Qwen/Qwen2.5-1.5B-Instruct en items_en.csv LangTrack/runs/full_en.csv
python LangTrack/analyze_lang.py LangTrack/runs full_
python LangTrack/make_figures.py LangTrack/runs LangTrack/figures
```

The Korean (`ko`) and paraphrase (`en2`) conditions need the item files available on request. Runs used CPU, float32,
an explicit empty system message, and transformers 5.x (EXAONE 4.0 is supported natively; no remote code).

## Citation

If you use this, cite the report and WDCT:
Xu, R., Lin, H., Han, X., Zheng, J., Zhou, W., Sun, L., & Sun, Y. 2025. Large Language Models Often Say One Thing and Do Another. ICLR 2025.
