"""
Convert the s1K reasoning dataset (https://github.com/simplescaling/s1) into the
prompt/response .jsonl format consumed by `SFTDataset` in data.py.

Each output line is:  {"prompt": <str>, "response": <str>}

`SFTDataset` later tokenizes prompt/response *separately* and concatenates them:
    token_ids     = encode(prompt) + encode(response)
    response_mask = [0]*len(prompt) + [1]*len(response)
so the prompt carries the (optional) instruction template and the response is the
raw answer text. Keep the prompt<->response join clean by putting the separator
(space / newline) at the START of the response, not the end of the prompt --
GPT-2 BPE merges a leading space into the following token, matching natural text.

The SEDD model here is fixed-length (model.length, default 1024) and uses the
GPT-2 tokenizer (vocab 50257). s1K reasoning traces are far longer than 1024, so
we emit TWO variants:
  * filtered : drop any sample whose prompt+response exceeds --max-length
  * trunc    : keep the prompt, truncate the response (keeping a trailing EOS)
Truncation is verified by re-encoding exactly the way SFTDataset does, so every
written sample is guaranteed to fit in --max-length.

Usage
-----
    python prepare_s1k.py                       # defaults: s1K-1.1, trunc->sft_*.jsonl
    python prepare_s1k.py --max-length 1024 --n-val 50
    python prepare_s1k.py --no-template         # prompt = raw question
    python prepare_s1k.py --no-eos              # do not append EOS to the response
"""

import argparse
import json
import os
import random

from datasets import load_dataset
from transformers import GPT2TokenizerFast


# ---------------------------------------------------------------------------
# building prompt / response strings
# ---------------------------------------------------------------------------

def build_prompt(question: str, use_template: bool) -> str:
    question = (question or "").strip()
    if use_template:
        return f"Question: {question}\nAnswer:"
    return question


def build_response_core(example: dict, use_template: bool) -> str:
    """Raw answer text (no EOS yet). Falls back gemini->solution if empty."""
    answer = (example.get("deepseek_attempt")
              or example.get("gemini_attempt")
              or example.get("solution")
              or "").strip()
    # separator lives on the response side (see module docstring)
    sep = " " if use_template else "\n"
    return sep + answer


# ---------------------------------------------------------------------------
# tokenisation helpers -- must mirror SFTDataset / tokenize_prompt_and_output
# ---------------------------------------------------------------------------

def encode(tokenizer, text: str):
    return tokenizer.encode(text, add_special_tokens=False)


def fits(tokenizer, prompt: str, response: str, max_length: int) -> int:
    """Return total token length of prompt+response as SFTDataset would see it."""
    return len(encode(tokenizer, prompt)) + len(encode(tokenizer, response))


def truncate_response(tokenizer, prompt: str, response: str,
                      max_length: int, eos_str: str):
    """
    Truncate `response` (token-level) so that encode(prompt)+encode(response)
    <= max_length, keeping a trailing EOS. Returns the truncated response text,
    re-verified by re-encoding (decode->encode can drift by a token at the
    boundary, so we trim in a short loop until it genuinely fits).

    Returns None if the prompt alone already does not leave room for a response.
    """
    prompt_len = len(encode(tokenizer, prompt))
    eos_len = len(encode(tokenizer, eos_str)) if eos_str else 0
    # need room for at least one real response token + the EOS
    budget = max_length - prompt_len - eos_len
    if budget < 1:
        return None

    resp_ids = encode(tokenizer, response)
    resp_ids = resp_ids[:budget]
    resp_text = tokenizer.decode(resp_ids) + eos_str

    # re-encode verification loop (handles decode->encode drift)
    while fits(tokenizer, prompt, resp_text, max_length) > max_length and resp_ids:
        resp_ids = resp_ids[:-1]
        resp_text = tokenizer.decode(resp_ids) + eos_str
    if not resp_ids:
        return None
    return resp_text


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):>5} samples -> {path}")


def length_stats(tokenizer, records, max_length):
    lens = [fits(tokenizer, r["prompt"], r["response"], max_length) for r in records]
    if not lens:
        return "  (empty)"
    lens.sort()
    n = len(lens)
    return (f"  tokens  min={lens[0]}  median={lens[n // 2]}  "
            f"max={lens[-1]}  mean={sum(lens) / n:.0f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="simplescaling/s1K-1.1",
                   help="HF dataset id (e.g. simplescaling/s1K-1.1 or simplescaling/s1K)")
    p.add_argument("--split", default="train")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--max-length", type=int, default=1024,
                   help="must match cfg.model.length")
    p.add_argument("--n-val", type=int, default=50,
                   help="number of held-out validation samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-template", dest="template", action="store_false",
                   help="use the raw question as prompt instead of 'Question: ...\\nAnswer:'")
    p.add_argument("--no-eos", dest="eos", action="store_false",
                   help="do not append <|endoftext|> to the response")
    p.add_argument("--default-variant", default="trunc", choices=["trunc", "filtered"],
                   help="which variant is also copied to data/sft_{train,val}.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_str = tokenizer.eos_token if args.eos else ""

    print(f"Loading {args.dataset} [{args.split}] ...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"  {len(ds)} raw examples; template={args.template}, append_eos={args.eos}, "
          f"max_length={args.max_length}")

    filtered, trunc = [], []
    n_truncated = 0
    n_dropped = 0   # prompt alone too long (trunc variant)

    for ex in ds:
        prompt = build_prompt(ex["question"], args.template)
        response = build_response_core(ex, args.template) + eos_str

        total = fits(tokenizer, prompt, response, args.max_length)

        # ---- filtered variant: keep only if it already fits --------------
        if total <= args.max_length:
            filtered.append({"prompt": prompt, "response": response})
            trunc.append({"prompt": prompt, "response": response})
            continue

        # ---- trunc variant: cut the response down to size ----------------
        resp_trunc = truncate_response(tokenizer, prompt,
                                       build_response_core(ex, args.template),
                                       args.max_length, eos_str)
        if resp_trunc is None:
            n_dropped += 1
            continue
        trunc.append({"prompt": prompt, "response": resp_trunc})
        n_truncated += 1

    print(f"\nProcessed: {len(ds)} examples")
    print(f"  fit-as-is (filtered kept): {len(filtered)}")
    print(f"  truncated into trunc:      {n_truncated}")
    print(f"  dropped (prompt too long): {n_dropped}")

    # ---- shuffle + split ---------------------------------------------------
    def split(records):
        recs = list(records)
        random.Random(args.seed).shuffle(recs)
        n_val = min(args.n_val, max(0, len(recs) - 1))
        return recs[n_val:], recs[:n_val]   # train, val

    out = args.out_dir
    print("\n[filtered]  (every sample complete, may be very few)")
    print(length_stats(tokenizer, filtered, args.max_length))
    f_train, f_val = split(filtered)
    write_jsonl(os.path.join(out, "s1k_train_filtered.jsonl"), f_train)
    write_jsonl(os.path.join(out, "s1k_val_filtered.jsonl"), f_val)

    print("\n[trunc]     (all samples kept, long responses cut to max_length)")
    print(length_stats(tokenizer, trunc, args.max_length))
    t_train, t_val = split(trunc)
    write_jsonl(os.path.join(out, "s1k_train_trunc.jsonl"), t_train)
    write_jsonl(os.path.join(out, "s1k_val_trunc.jsonl"), t_val)

    # ---- copy chosen variant to the paths sft_base.yaml expects -----------
    default_train = f_train if args.default_variant == "filtered" else t_train
    default_val = f_val if args.default_variant == "filtered" else t_val
    print(f"\n[default -> {args.default_variant}]  (used by configs/sft_base.yaml)")
    write_jsonl(os.path.join(out, "sft_train.jsonl"), default_train)
    write_jsonl(os.path.join(out, "sft_val.jsonl"), default_val)

    print("\nDone.")


if __name__ == "__main__":
    main()
