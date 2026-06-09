"""
Build the SFTDataset-ready jsonl from the RAW reasoning data
(problem / reasoning_trace / expected_answer), applying the r1_zero prompt
template, and ADDITIONALLY carrying the ground-truth answer in each record.

Output record:
    {"prompt": <template+question, ends with "Assistant: <think>">,
     "response": <reasoning_trace, +EOS if complete>,
     "answer": <expected_answer, the correct answer for evaluation>}

SFTDataset (data.py) only reads "prompt"/"response", so the extra "answer" field
is harmless for training and available for conditional-generation grading.

Length handling (same philosophy as prepare_s1k.py):
  * append EOS to COMPLETE responses -> a genuine "answer finished" stop signal
  * enforce --max-length, emitting two variants:
      - filtered : drop samples whose prompt+response(+EOS) exceeds max_length
      - trunc    : truncate the response to fit; a truncated answer is INCOMPLETE,
                   so it gets NO trailing EOS
  * if the prompt alone >= max_length the sample is dropped (prompt is never cut)

Usage
-----
    python prepare_sft_jsonl.py --input path/to/sft_gpt-oss-120b_filtered.jsonl
    python prepare_sft_jsonl.py --input raw.jsonl --template prompt_template/r1_zero.prompt
    python prepare_sft_jsonl.py --input raw.jsonl --n-val 100 --no-eos
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
    Truncate `response` (token-level) so prompt+response <= max_length. NO trailing
    EOS: a truncated sample is an *incomplete* answer (content continues past
    max_length), so it must not carry the "reasoning finished" stop signal. Full
    budget goes to content. Re-verified by re-encoding (decode->encode can drift).

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

def load_raw(path):
    """Load a JSON array OR a line-delimited jsonl of raw records."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_template(path):
    """Load the prompt template, normalising CRLF/CR -> LF to match training data."""
    with open(path, "r", encoding="utf-8") as f:
        tmpl = f.read()
    tmpl = tmpl.replace("\r\n", "\n").replace("\r", "\n")
    if "{question}" not in tmpl:
        raise ValueError(f"template {path} has no {{question}} placeholder")
    return tmpl


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
                   help="raw data (JSON array or jsonl) with problem/reasoning_trace/expected_answer")
    p.add_argument("--template", default="prompt_template/r1_zero.prompt",
                   help="prompt template file containing a {question} placeholder")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: same dir as --input)")
    p.add_argument("--question-field", default="problem")
    p.add_argument("--response-field", default="reasoning_trace")
    p.add_argument("--answer-field", default="expected_answer",
                   help="ground-truth answer field carried into the output as 'answer'")
    p.add_argument("--max-length", type=int, default=1024,
                   help="must match cfg.model.length")
    p.add_argument("--n-val", type=int, default=0,
                   help="held-out validation samples (0 = no split)")
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

    template = load_template(args.template)
    print(f"Reading {in_path} ...")
    raw = load_raw(in_path)
    print(f"  {len(raw)} records; append_eos={args.eos}, max_length={args.max_length}")

    filtered, trunc = [], []
    n_truncated = 0
    n_dropped = 0

    for ex in raw:
        question = str(ex[args.question_field])
        reasoning = str(ex[args.response_field])
        answer = ex.get(args.answer_field, "")

        prompt = template.replace("{question}", question)
        response = reasoning + eos_str          # complete -> stop signal

        if fits(tokenizer, prompt, response, args.max_length) <= args.max_length:
            rec = {"prompt": prompt, "response": response, "answer": answer}
            filtered.append(rec)
            trunc.append(rec)
            continue

        # too long: truncate the ORIGINAL reasoning (no EOS, it is incomplete)
        resp_trunc = truncate_response(tokenizer, prompt, reasoning, args.max_length)
        if resp_trunc is None:
            n_dropped += 1
            continue
        trunc.append({"prompt": prompt, "response": resp_trunc, "answer": answer})
        n_truncated += 1

    print(f"\nProcessed: {len(raw)} records")
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
