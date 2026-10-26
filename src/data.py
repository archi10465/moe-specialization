"""
src/data.py
Mixed-domain batch loader for the MoE specialization project.

Training batches contain an EQUAL number of sequences from each domain, and
every sequence comes with a domain label (0 = stories, 1 = code, 2 = math).
That label is what lets us measure expert specialization later.

Evaluation uses its own fixed random generator, so every model is evaluated
on exactly the same test windows (a fair comparison across runs).

Quick sanity check (run after data/prepare.py):
  python src/data.py --data_dir data
"""

import argparse
import os

import numpy as np
import torch

DOMAINS = ["stories", "code", "math"]  # label = index in this list


class MixedDomainLoader:
    def __init__(self, data_dir, domains=DOMAINS, block_size=256,
                 per_domain=16, device="cpu", seed=1337):
        """
        data_dir   : folder with <domain>_train.bin / <domain>_test.bin
        domains    : list of domain names; use ["wikitext"] for the control run
        block_size : tokens per sequence (the model's context length)
        per_domain : sequences per domain in each training batch
                     (batch size = per_domain * len(domains))
        seed       : same seed -> same training data order in every run
        """
        self.domains = list(domains)
        self.block_size = block_size
        self.per_domain = per_domain
        self.device = device
        self.rng = np.random.default_rng(seed)

        # np.memmap reads from disk on demand, so the files never need to fit in RAM
        self.data = {
            split: {
                d: np.memmap(os.path.join(data_dir, f"{d}_{split}.bin"),
                             dtype=np.uint16, mode="r")
                for d in self.domains
            }
            for split in ("train", "test")
        }

    # -- helpers ----------------------------------------------------------

    def _windows(self, arr, starts):
        bs = self.block_size
        x = np.stack([arr[i:i + bs] for i in starts]).astype(np.int64)
        y = np.stack([arr[i + 1:i + 1 + bs] for i in starts]).astype(np.int64)
        return x, y

    def _to_device(self, *arrays):
        out = []
        for a in arrays:
            t = torch.from_numpy(a)
            if self.device.startswith("cuda"):
                t = t.pin_memory().to(self.device, non_blocking=True)
            else:
                t = t.to(self.device)
            out.append(t)
        return out

    # -- training batches -------------------------------------------------

    def get_batch(self, split="train"):
        """
        Returns x, y, domain
          x      : (B, T) input token ids
          y      : (B, T) targets = x shifted left by one token
          domain : (B,)   domain label for each sequence
        B = per_domain * number of domains, equal from each domain.
        """
        xs, ys, labels = [], [], []
        for label, d in enumerate(self.domains):
            arr = self.data[split][d]
            starts = self.rng.integers(0, len(arr) - self.block_size - 1,
                                       size=self.per_domain)
            x, y = self._windows(arr, starts)
            xs.append(x)
            ys.append(y)
            labels.append(np.full(self.per_domain, label, dtype=np.int64))

        x, y, dom = np.concatenate(xs), np.concatenate(ys), np.concatenate(labels)
        perm = self.rng.permutation(len(x))  # interleave domains inside the batch
        return self._to_device(x[perm], y[perm], dom[perm])

    # -- evaluation batches -----------------------------------------------

    def eval_batches(self, domain, n_batches=50, batch_size=32,
                     split="test", seed=2024):
        """
        Yields n_batches of (x, y, domain_label) from ONE domain.
        Uses its own generator, so the same windows are used for every model.
        """
        label = self.domains.index(domain)
        arr = self.data[split][domain]
        rng = np.random.default_rng(seed)
        for _ in range(n_batches):
            starts = rng.integers(0, len(arr) - self.block_size - 1, size=batch_size)
            x, y = self._windows(arr, starts)
            dom = np.full(batch_size, label, dtype=np.int64)
            yield self._to_device(x, y, dom)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tiktoken

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--domains", nargs="+", default=DOMAINS)
    args = p.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    loader = MixedDomainLoader(args.data_dir, domains=args.domains)

    print("Token counts:")
    for d in loader.domains:
        print(f"  {d:9s} train {len(loader.data['train'][d]):>12,}   "
              f"test {len(loader.data['test'][d]):>10,}")

    x, y, dom = loader.get_batch("train")
    print(f"\nBatch shapes: x {tuple(x.shape)}, y {tuple(y.shape)}, domain {tuple(dom.shape)}")
    counts = {d: int((dom == i).sum()) for i, d in enumerate(loader.domains)}
    print(f"Sequences per domain in this batch: {counts}")
    assert torch.equal(x[:, 1:], y[:, :-1]), "y should be x shifted by one token"
    print("Check passed: y is x shifted by one token.")

    print("\nOne sample per domain (first 300 characters):")
    for i, d in enumerate(loader.domains):
        row = (dom == i).nonzero()[0].item()
        text = enc.decode(x[row].tolist())[:300].replace("\n", "\\n ")
        print(f"\n--- {d} ---\n{text}")