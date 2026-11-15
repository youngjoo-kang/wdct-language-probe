"""Language-swap probe (2026-11-14, Apart track 1): the same WDCT speak/act pairs in English or
Korean, next-token P(A) vs P(B), each asked in the original and the swapped option order.

Reuses run_probs.py (preexisting, 2026-11-08): swap_options, letter_ids, p_a.
Changes: item text comes from the bilingual item file (en_/ko_ columns), the instruction line
is translated for Korean, and the chat template always gets an explicit empty system message
(Qwen2.5 otherwise inserts an English default system prompt into both language conditions).

usage: py run_lang.py <hf_model_id> <en|ko> <items_csv> <out_csv> [n_items]
items_csv = ko_items_draft.csv layout (wdct_id, master_q, en_speak, ko_speak, en_act, ko_act, ...)
"""
import os
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from run_probs import letter_ids, p_a, swap_options  # noqa: E402

PROMPTS = {
    "en": ('Answer the following question with only one single option number (like "A" or "B") '
           'without explanation.\n%s\nAnswer:'),
    "ko": '다음 질문에 설명 없이 선택지 번호 하나(예: "A" 또는 "B")로만 답하십시오.\n%s\n답:',
    # meaning-preserving English paraphrase = wording-sensitivity baseline (items in en2_items.csv)
    "en2": ('Give just one option number (for example "A" or "B") in response to the question below, '
            'with no explanation.\n%s\nResponse:'),
}


def build(tok, q, lang):
    return tok.apply_chat_template([{"role": "system", "content": ""},
                                    {"role": "user", "content": PROMPTS[lang] % q}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    mid, lang, items_csv, out_csv = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else -1
    df = pd.read_csv(items_csv, encoding="utf-8-sig")
    if n > 0:
        df = df.head(n)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(mid)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32)
    model.eval()
    ids = letter_ids(tok)
    print("letter ids", ids, flush=True)
    rows = [{"id": i, "master_q": q, "lang": lang} for i, q in zip(df["wdct_id"], df["master_q"])]
    t0 = time.time()
    for side in ("speak", "act"):
        qs = df[f"{lang}_{side}"].tolist()
        sw = [swap_options(q) for q in qs]
        orig = p_a(model, tok, ids, [build(tok, q, lang) for q in qs])
        swp = p_a(model, tok, ids, [build(tok, q if s is None else s, lang) for q, s in zip(qs, sw)])
        for k, (o, s, sq) in enumerate(zip(orig, swp, sw)):
            rows[k][f"{side}_pA_orig"] = o[0]
            rows[k][f"{side}_mass_orig"] = o[1]
            # P(original option A) when shown as B in the swapped prompt
            rows[k][f"{side}_pA_swap"] = (1 - s[0]) if sq is not None else float("nan")
            rows[k][f"{side}_mass_swap"] = s[1]
        print(side, "done", round(time.time() - t0, 1), "s", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("items", len(rows), "seconds", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
