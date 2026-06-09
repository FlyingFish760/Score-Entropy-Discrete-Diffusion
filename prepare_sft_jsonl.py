"""
Re-process an EXISTING {"prompt","response"} SFT .jsonl into the fixed-length
variants consumed by SFTDataset (data.py). Use this when the prompt template is
already baked into the data (e.g. the R1-style <think>/<answer> data), so unlike
prepare_s1k.py there is NO dataset loading and NO template building here.

What it does (mirrors prepare_s1k.py's post-processing philosophy):
  * append EOS to COMPLETE responses -> a genuine "answer finished" stop signal
  * enforce --max-length, emitting two variants:
      - filtered : drop samples whose prompt+response(+EOS) exceeds max_length
      - trunc    : truncate the response to fit; a truncated answer is INCOMPLETE
                   (its real ending is cut off), so it gets NO trailing EOS
  * prompt/response are kept verbatim; only EOS + length are handled

Usage
-----
    python prepare_sft_jsonl.py --input path/to/sft_train.jsonl
    python prepare_sft_jsonl.py --input sft_train.jsonl --max-length 1024 --no-eos
    python prepare_sft_jsonl.py --input sft_train.jsonl --n-val 100   # carve a val split
"""

import argparse
import json
import os
import random

from transformers import GPT2TokenizerFast


# ---------------------------------------------------------------------------
# tokenisation helpers -- must mirror SFTDataset / tokenize_prompt_and_output
# ---------------------------------------------------------------------------

def encode(tokenizer, text: str):
    return tokenizer.encode(text, add_special_tokens=False)


def fits(tokenizer, prompt: str, response: str, max_length: int) -> int:
    """Total token length of prompt+response as SFTDataset would see it."""
    return len(encode(tokenizer, prompt)) + len(encode(tokenizer, response))


def truncate_response(tokenizer, prompt: str, response: str, max_length: int):
    """
    Truncate `response` (token-level) so that encode(prompt)+encode(response)
    <= max_length. NO trailing EOS: a truncated sample is an *incomplete* answer
    (the real content continues past max_length), so it must not carry the
    "reasoning finished" stop signal. Full budget goes to content. Re-verified by
    re-encoding (decode->encode can drift by a token at the boundary).

    Returns None if the prompt alone leaves no room for a response.
    """
    prompt_len = len(encode(tokenizer, prompt))
    budget = max_length - prompt_len
    if budget < 1:
        return None

    resp_ids = encode(tokenizer, response)[:budget]
    resp_text = tokenizer.decode(resp_ids)

    while fits(tokenizer, prompt, resp_text, max_length) > max_length and resp_ids:
        resp_ids = resp_ids[:-1]
        resp_text = tokenizer.decode(resp_ids)
    if not resp_ids:
        return None
    return resp_text


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            records.append({"prompt": o["prompt"], "response": o["response"]})
    return records


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
    p.add_argument("--input", required=True,
                   help="existing {prompt,response} jsonl to re-process")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: same dir as --input)")
    p.add_argument("--max-length", type=int, default=1024,
                   help="must match cfg.model.length")
    p.add_argument("--n-val", type=int, default=0,
                   help="held-out validation samples (0 = no split, process file as-is)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-eos", dest="eos", action="store_false",
                   help="do not append <|endoftext|> to complete responses")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_str = tokenizer.eos_token if args.eos else ""

    in_path = args.input
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(in_path))
    stem = os.path.splitext(os.path.basename(in_path))[0]

    print(f"Reading {in_path} ...")
    records = read_jsonl(in_path)
    print(f"  {len(records)} records; append_eos={args.eos}, max_length={args.max_length}")

    filtered, trunc = [], []
    n_truncated = 0
    n_dropped = 0

    for rec in records:
        prompt = rec["prompt"]
        response = rec["response"] + eos_str          # complete -> stop signal

        if fits(tokenizer, prompt, response, args.max_length) <= args.max_length:
            filtered.append({"prompt": prompt, "response": response})
            trunc.append({"prompt": prompt, "response": response})
            continue

        # too long: truncate the ORIGINAL response (no EOS, it is incomplete)
        resp_trunc = truncate_response(tokenizer, prompt, rec["response"], args.max_length)
        if resp_trunc is None:
            n_dropped += 1
            continue
        trunc.append({"prompt": prompt, "response": resp_trunc})
        n_truncated += 1

    print(f"\nProcessed: {len(records)} records")
    print(f"  fit-as-is (filtered kept): {len(filtered)}")
    print(f"  truncated into trunc:      {n_truncated}")
    print(f"  dropped (prompt too long): {n_dropped}")

    def split(recs):
        recs = list(recs)
        random.Random(args.seed).shuffle(recs)
        n_val = min(args.n_val, max(0, len(recs) - 1))
        return recs[n_val:], recs[:n_val]   # train, val

    print("\n[filtered]  (every sample complete, ends with EOS)")
    print(length_stats(tokenizer, filtered, args.max_length))
    print("\n[trunc]     (all samples kept; truncated ones have no EOS)")
    print(length_stats(tokenizer, trunc, args.max_length))

    if args.n_val > 0:
        f_train, f_val = split(filtered)
        t_train, t_val = split(trunc)
        write_jsonl(os.path.join(out_dir, f"{stem}_filtered_train.jsonl"), f_train)
        write_jsonl(os.path.join(out_dir, f"{stem}_filtered_val.jsonl"), f_val)
        write_jsonl(os.path.join(out_dir, f"{stem}_trunc_train.jsonl"), t_train)
        write_jsonl(os.path.join(out_dir, f"{stem}_trunc_val.jsonl"), t_val)
    else:
        write_jsonl(os.path.join(out_dir, f"{stem}_filtered.jsonl"), filtered)
        write_jsonl(os.path.join(out_dir, f"{stem}_trunc.jsonl"), trunc)

    print("\nDone. Point cfg.data.train / cfg.data.valid at the variant you want.")


if __name__ == "__main__":
    main()
