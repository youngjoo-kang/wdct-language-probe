"""Word-deed direction probe: next-token P(A) vs P(B) for WDCT word (speak) and deed (act)
questions, each asked in the original and the swapped option order.

Prompt follows the authors' test_local_models.py. Base models get the raw prompt,
instruct models get the chat template (as in the authors' use_prompt_template switch).

usage: python run_probs.py <hf_model_id> <is_chat 0|1> <out_csv> [n_items] [seed]
"""
import os
import re
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = os.environ.get("WDCT_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "WDCT.csv"))
PROMPT = ('Answer the following question with only one single option number (like "A" or "B") '
          'without explanation.\n%s\nAnswer:')
OPT = re.compile(r"^([AB])\. ", re.M)


def swap_options(q):
    """Swap the texts after 'A. ' and 'B. ' so the same option appears under the other letter."""
    lines = q.split("\n")
    ia = [i for i, l in enumerate(lines) if l.startswith("A. ")]
    ib = [i for i, l in enumerate(lines) if l.startswith("B. ")]
    if len(ia) != 1 or len(ib) != 1:
        return None
    a, b = lines[ia[0]][3:], lines[ib[0]][3:]
    lines[ia[0]], lines[ib[0]] = "A. " + b, "B. " + a
    return "\n".join(lines)


def letter_ids(tok):
    ids = {}
    for L in "AB":
        cand = set()
        for s in (L, " " + L):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1:
                cand.add(t[0])
        ids[L] = sorted(cand)
    return ids


def build(tok, q, is_chat):
    text = PROMPT % q
    if is_chat:
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True)
    return text


@torch.no_grad()
def p_a(model, tok, ids, texts, bs=4):
    out = []
    # chat templates that already start with BOS (Llama 3) must not get a second one
    add_special = not (tok.bos_token and texts[0].startswith(tok.bos_token))
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=add_special)
        logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, -1)
        la = torch.logsumexp(lp[:, ids["A"]], -1)
        lb = torch.logsumexp(lp[:, ids["B"]], -1)
        mass = (la.exp() + lb.exp())
        pa = torch.softmax(torch.stack([la, lb], -1), -1)[:, 0]
        out += list(zip(pa.tolist(), mass.tolist()))
    return out


def main():
    mid, is_chat, out_csv = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
    df = pd.read_csv(DATA)
    if n > 0:
        df = pd.concat([g.sample(min(len(g), max(1, round(n * len(g) / len(df)))), random_state=seed)
                        for _, g in df.groupby("Type")], ignore_index=True)
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
    rows, t0 = [], time.time()
    for side in ("speak", "act"):
        qs = df[side].tolist()
        sw = [swap_options(q) for q in qs]
        orig = p_a(model, tok, ids, [build(tok, q, is_chat) for q in qs])
        swp = p_a(model, tok, ids, [build(tok, q if s is None else s, is_chat) for q, s in zip(qs, sw)])
        for k, (o, s, sq) in enumerate(zip(orig, swp, sw)):
            if len(rows) <= k:
                rows.append({"Type": df["Type"].iloc[k], "id": df["id"].iloc[k],
                             "correct_answer": df["correct_answer"].iloc[k]})
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
