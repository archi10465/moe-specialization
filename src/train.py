"""
src/train.py
Train one model (dense or MoE) and save a checkpoint.

Settings come from a YAML config file. Any setting can be overridden on the
command line with --name value, e.g. --max_iters 200 --device mps.

Examples
  # quick test on a Mac
  python src/train.py --config configs/dense.yaml --run_name dense_mac_test \
      --max_iters 200 --per_domain 4 --eval_interval 100 --eval_batches 5 \
      --eval_batch_size 8

  # real run on Colab (data on Google Drive)
  python src/train.py --config configs/dense.yaml \
      --data_dir /content/drive/MyDrive/moe-data \
      --out_dir /content/drive/MyDrive/moe-checkpoints --wandb true

Output: <out_dir>/<run_name>/ckpt.pt and config.json
"""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import fields

import numpy as np
import torch
import yaml

from data import MixedDomainLoader
from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    run_name="run", data_dir="data", out_dir="checkpoints",
    domains=["stories", "code", "math"],
    per_domain=16, max_iters=5000, warmup_iters=200,
    lr=6e-4, min_lr=6e-5, weight_decay=0.1, grad_clip=1.0,
    eval_interval=500, eval_batches=20, eval_batch_size=32, log_interval=10,
    seed=1337, device="auto", wandb=False, wandb_project="moe-specialization",
)


def load_config():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a YAML config file")
    args, extra = p.parse_known_args()

    cfg = dict(DEFAULTS)
    with open(args.config) as f:
        cfg.update(yaml.safe_load(f) or {})

    # Command-line overrides: --key value (value parsed as YAML, so 200 -> int, true -> bool)
    if len(extra) % 2 != 0:
        raise SystemExit(f"Overrides must come in --key value pairs, got: {extra}")
    for key, value in zip(extra[::2], extra[1::2]):
        if not key.startswith("--"):
            raise SystemExit(f"Bad override {key!r}; use --key value")
        cfg[key[2:]] = yaml.safe_load(value)
    return cfg


def model_config(cfg):
    names = {f.name for f in fields(GPTConfig)}
    return GPTConfig(**{k: v for k, v in cfg.items() if k in names})


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def lr_at(it, cfg):
    """Linear warmup, then cosine decay from lr to min_lr."""
    if it < cfg["warmup_iters"]:
        return cfg["lr"] * (it + 1) / cfg["warmup_iters"]
    progress = (it - cfg["warmup_iters"]) / max(1, cfg["max_iters"] - cfg["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


@torch.no_grad()
def evaluate(model, loader, cfg, ctx):
    """Average test loss for each domain separately."""
    model.eval()
    results = {}
    for d in loader.domains:
        losses = []
        for x, y, _ in loader.eval_batches(d, n_batches=cfg["eval_batches"],
                                          batch_size=cfg["eval_batch_size"]):
            with ctx:
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        results[d] = float(np.mean(losses))
    results["mean"] = float(np.mean([results[d] for d in loader.domains]))
    model.train()
    return results


@torch.no_grad()
def sample_text(model, device, ctx, max_new_tokens=80):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    out = {}
    for prompt in ["Once upon a time", "def ", "Question: "]:
        idx = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        with ctx:
            gen = model.generate(idx, max_new_tokens)
        out[prompt] = enc.decode(gen[0].tolist())
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    device = pick_device(cfg["device"])
    device_type = "cuda" if device.startswith("cuda") else device

    # Same seed -> same starting weights and same data order (fairness rule)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Mixed precision on GPU: bfloat16 where supported (A100, L4), else float16 (T4)
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        dtype = torch.float32
        ctx = nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

    loader = MixedDomainLoader(cfg["data_dir"], domains=cfg["domains"],
                               block_size=cfg.get("block_size", 256),
                               per_domain=cfg["per_domain"], device=device, seed=cfg["seed"])

    mcfg = model_config(cfg)
    model = GPT(mcfg).to(device)
    optimizer = model.configure_optimizers(cfg["weight_decay"], cfg["lr"], (0.9, 0.95), device_type)

    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    batch_tokens = cfg["per_domain"] * len(cfg["domains"]) * mcfg.block_size
    print(f"Run: {cfg['run_name']} | device: {device} ({dtype}) | "
          f"params: {model.num_params() / 1e6:.1f}M | "
          f"batch: {batch_tokens:,} tokens | steps: {cfg['max_iters']}")

    if cfg["wandb"]:
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=cfg)

    t0 = time.time()
    history = []
    for it in range(cfg["max_iters"]):
        lr = lr_at(it, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y, _ = loader.get_batch("train")
        with ctx:
            _, loss, aux = model(x, y)
            total = loss + aux
        scaler.scale(total).backward()
        if cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        model.after_optimizer_step()

        if it % cfg["log_interval"] == 0:
            dt = time.time() - t0
            tok_s = batch_tokens * (it + 1) / max(dt, 1e-9)
            log = {"train/loss": loss.item(), "train/aux": aux.item(), "lr": lr}
            util_str = ""
            moe_layers = model.moe_layers()
            if moe_layers:
                # Utilization per layer: 1.0 = perfectly even, near 0 = collapse
                from moe import utilization
                utils = [utilization(layer.last_load) for layer in moe_layers]
                util_str = f" | util {np.mean(utils):.3f} (min {min(utils):.3f})"
                log["train/util_mean"] = float(np.mean(utils))
                for i, u in enumerate(utils):
                    log[f"util/layer{i}"] = u
            print(f"step {it:5d} | loss {loss.item():.4f} | aux {aux.item():.4f} | "
                  f"lr {lr:.2e} | {tok_s / 1e3:.1f}k tok/s{util_str}")
            if cfg["wandb"]:
                wandb.log(log, step=it)

        if (it > 0 and it % cfg["eval_interval"] == 0) or it == cfg["max_iters"] - 1:
            res = evaluate(model, loader, cfg, ctx)
            res["step"] = it
            history.append(res)
            print("  eval | " + " | ".join(f"{k} {v:.4f}" for k, v in res.items() if k != "step"))
            if cfg["wandb"]:
                wandb.log({f"test/{k}": v for k, v in res.items() if k != "step"}, step=it)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed / 60:.1f} min")

    samples = sample_text(model, device, ctx)
    print("\nSample text:")
    for prompt, text in samples.items():
        print(f"\n--- prompt: {prompt!r} ---\n{text}")

    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    torch.save({
        "model": model.state_dict(),
        "model_config": mcfg.__dict__,
        "config": cfg,
        "step": cfg["max_iters"],
        "eval_history": history,
        "train_minutes": elapsed / 60,
    }, ckpt_path)
    with open(os.path.join(run_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\nSaved checkpoint to {ckpt_path}")

    if cfg["wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()