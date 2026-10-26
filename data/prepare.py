"""
data/prepare.py
Download, tokenize, and split the four text domains for the MoE specialization project.

Domains
  stories  : TinyStories                      (roneneldan/TinyStories)
  code     : Python from CodeParrot           (codeparrot/codeparrot-clean)
  math     : GSM8K + OpenWebMath              (openai/gsm8k, open-web-math/open-web-math)
  wikitext : WikiText-103 (control domain)    (Salesforce/wikitext, wikitext-103-raw-v1)

For each domain this writes two files of GPT-2 token ids (uint16):
  <domain>_train.bin   and   <domain>_test.bin
plus one meta.json describing what was saved.

How the split works
  1. Stream documents until we have enough tokens.
  2. Shuffle the documents (fixed seed), so e.g. GSM8K and OpenWebMath are mixed.
  3. Put whole documents into the test set until it is full, the rest into train.
     So no document appears in both train and test.

Usage
  # quick test on a laptop (~1M train tokens per domain, a few minutes)
  python data/prepare.py --small --out_dir data

  # full MVP size (~20M train tokens per domain)
  python data/prepare.py --out_dir data

  # redo just one domain
  python data/prepare.py --out_dir data --domains math
"""

import argparse
import json
import os
import random
import time

import numpy as np
import tiktoken
from datasets import load_dataset

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token  # 50256, marks the end of each document


# ---------------------------------------------------------------------------
# Document sources: each function yields plain-text documents, one at a time.
# streaming=True means we download only what we read, not the whole dataset.
# ---------------------------------------------------------------------------

def stories_docs():
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for ex in ds:
        yield ex["text"]


def code_docs():
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for ex in ds:
        yield ex["content"]


def gsm8k_docs():
    for split in ("train", "test"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for ex in ds:
            yield f"Question: {ex['question']}\nAnswer: {ex['answer']}"


def openwebmath_docs():
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for ex in ds:
        yield ex["text"]


def math_docs():
    # All of GSM8K first (it is small, ~1-2M tokens), then OpenWebMath to fill
    # the rest. Documents are shuffled later, so both end up in train and test.
    yield from gsm8k_docs()
    yield from openwebmath_docs()


def wikitext_docs():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    for ex in ds:
        text = ex["text"].strip()
        if text:  # WikiText has many empty lines
            yield text


SOURCES = {
    "stories": (stories_docs, "roneneldan/TinyStories"),
    "code": (code_docs, "codeparrot/codeparrot-clean (Python)"),
    "math": (math_docs, "openai/gsm8k (main) + open-web-math/open-web-math"),
    "wikitext": (wikitext_docs, "Salesforce/wikitext (wikitext-103-raw-v1)"),
}


# ---------------------------------------------------------------------------
# Tokenize and split
# ---------------------------------------------------------------------------

def collect_docs(name, doc_iter, target_tokens, max_doc_tokens, num_threads):
    """Tokenize documents until we have at least target_tokens tokens."""
    docs, buf = [], []
    total = 0
    t0 = time.time()

    def flush():
        nonlocal total
        for ids in ENC.encode_ordinary_batch(buf, num_threads=num_threads):
            if not ids:
                continue
            # Cap very long documents (e.g. huge code files) so no single
            # document dominates a domain, then mark the document end.
            ids = ids[:max_doc_tokens] + [EOT]
            docs.append(np.array(ids, dtype=np.uint16))
            total += len(ids)
        buf.clear()

    for text in doc_iter:
        buf.append(text)
        if len(buf) >= 256:
            flush()
            print(f"\r  {name}: {total / 1e6:6.2f}M / {target_tokens / 1e6:.2f}M tokens "
                  f"({time.time() - t0:5.0f}s)", end="", flush=True)
            if total >= target_tokens:
                break
    if buf:
        flush()
    print(f"\r  {name}: {total / 1e6:6.2f}M tokens from {len(docs):,} documents "
          f"({time.time() - t0:.0f}s)          ")

    if total < target_tokens:
        print(f"  WARNING: {name} ran out of data at {total:,} tokens "
              f"(wanted {target_tokens:,}).")
    return docs


def split_docs(docs, train_tokens, test_tokens, seed):
    """Shuffle whole documents, fill test first, the rest becomes train."""
    rng = random.Random(seed)
    rng.shuffle(docs)

    test, n_test, i = [], 0, 0
    while n_test < test_tokens and i < len(docs):
        test.append(docs[i])
        n_test += len(docs[i])
        i += 1
    train = docs[i:]

    test_arr = np.concatenate(test)[:test_tokens]
    train_arr = np.concatenate(train)[:train_tokens] if train else np.zeros(0, np.uint16)
    return train_arr, test_arr


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="data", help="where to write the .bin files")
    p.add_argument("--domains", nargs="+", default=list(SOURCES),
                   choices=list(SOURCES), help="which domains to prepare")
    p.add_argument("--train_tokens", type=int, default=20_000_000)
    p.add_argument("--test_tokens", type=int, default=1_000_000)
    p.add_argument("--small", action="store_true",
                   help="quick test size: 1M train / 100k test tokens per domain")
    p.add_argument("--max_doc_tokens", type=int, default=4096)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if args.small:
        args.train_tokens, args.test_tokens = 1_000_000, 100_000

    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["tokenizer"] = "gpt2 (tiktoken)"
    meta["dtype"] = "uint16"
    meta["eot_token"] = EOT
    meta.setdefault("domains", {})

    print(f"Preparing {args.domains}: {args.train_tokens:,} train + "
          f"{args.test_tokens:,} test tokens each -> {args.out_dir}/")

    for name in args.domains:
        fn, source = SOURCES[name]
        # Collect a little extra so truncation and the test overshoot never
        # leave train short.
        target = args.train_tokens + args.test_tokens + 2 * args.max_doc_tokens
        docs = collect_docs(name, fn(), target, args.max_doc_tokens, args.num_threads)
        train_arr, test_arr = split_docs(docs, args.train_tokens, args.test_tokens, args.seed)

        for split, arr in (("train", train_arr), ("test", test_arr)):
            path = os.path.join(args.out_dir, f"{name}_{split}.bin")
            arr.tofile(path)
            print(f"  wrote {path}  ({len(arr):,} tokens, {arr.nbytes / 1e6:.1f} MB)")

        meta["domains"][name] = {
            "source": source,
            "train_tokens": int(len(train_arr)),
            "test_tokens": int(len(test_arr)),
            "documents_collected": len(docs),
            "max_doc_tokens": args.max_doc_tokens,
            "seed": args.seed,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Done. Summary saved to {meta_path}")


if __name__ == "__main__":
    main()