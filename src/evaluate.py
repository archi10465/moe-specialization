"""
src/evaluate.py
Turn a saved checkpoint into all of the project's numbers and per-run plots.

For each checkpoint it:
  1. Runs the SAME test windows for every model (fixed seed) through it,
     one domain at a time, and records which experts every token chose.
  2. Builds a counts table: layer x expert x domain.
  3. Computes, per MoE layer and overall:
       routing purity  : per expert, share of its tokens from its most common
                         domain, averaged over experts (1.0 = fully specialized,
                         ~0.33 = no specialization with 3 domains)
       utilization     : normalized entropy of expert usage (1.0 = even, 0 = collapse)
       normalized MI   : mutual information between expert choice and domain,
                         divided by the domain entropy (0 = expert says nothing
                         about domain, 1 = expert choice reveals the domain)
     plus the test loss for each domain.
  4. Optional --ablation: switches off one expert in one layer at a time and
     measures the increase in each domain's loss (layer x expert x domain).
  5. Saves <out_dir>/<run_name>.json and per-run plots.

Usage
  python src/evaluate.py --ckpt checkpoints/run_D/ckpt.pt --data_dir data
  python src/evaluate.py --ckpt checkpoints/*/ckpt.pt --data_dir data --ablation

Dense checkpoints get per-domain loss only (there is no routing to measure).
"""

import argparse
import glob
import json
import os
from contextlib import nullcontext

import numpy as np
import torch

from data import MixedDomainLoader
from model import GPT, GPTConfig

# Domain colors: first three slots of a colorblind-validated categorical palette
DOMAIN_COLORS = {"stories": "#2a78d6", "code": "#eb6834", "math": "#1baf7a", "wikitext": "#2a78d6"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# Loading and running
# ---------------------------------------------------------------------------

def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def autocast_ctx(device):
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def domain_losses(model, loader, n_batches, batch_size, ctx, counts=None):
    """Mean test loss per domain. If counts is given, also fills layer x expert x domain."""
    layers = model.moe_layers()
    for layer in layers:
        layer.record_routing = counts is not None
    losses = {}
    for d_i, d in enumerate(loader.domains):
        vals = []
        for x, y, _ in loader.eval_batches(d, n_batches=n_batches, batch_size=batch_size):
            with ctx:
                _, loss, _ = model(x, y)
            vals.append(loss.item())
            if counts is not None:
                for l_i, layer in enumerate(layers):
                    idx = layer.last_topk_idx.reshape(-1).cpu()
                    counts[l_i, :, d_i] += torch.bincount(idx, minlength=layer.n_experts).numpy()
        losses[d] = float(np.mean(vals))
    for layer in layers:
        layer.record_routing = False
    return losses


# ---------------------------------------------------------------------------
# Metrics from the counts table (layer x expert x domain)
# ---------------------------------------------------------------------------

def purity(c):
    """c: expert x domain. Average over experts (that got any tokens) of max share."""
    totals = c.sum(axis=1)
    used = totals > 0
    return float((c[used].max(axis=1) / totals[used]).mean()) if used.any() else float("nan")


def utilization(c):
    """c: expert x domain. Normalized entropy of total usage per expert."""
    load = c.sum(axis=1)
    p = load / load.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(c.shape[0]))


def normalized_mi(c):
    """c: expert x domain. I(expert; domain) / H(domain)."""
    joint = c / c.sum()
    pe = joint.sum(axis=1, keepdims=True)
    pd = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / (pe @ pd)[nz])).sum()
    h_d = -(pd[pd > 0] * np.log(pd[pd > 0])).sum()
    return float(max(mi / h_d, 0.0)) if h_d > 0 else float("nan")


# ---------------------------------------------------------------------------
# Plots for one run
# ---------------------------------------------------------------------------

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_usage(counts, domains, run_name, path):
    """One panel per layer: bar = expert, height = share of all routing slots,
    stacked by domain. Collapse = a few tall bars; specialization = one color per bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, E, D = counts.shape
    fig, axes = plt.subplots(1, L, figsize=(2.2 * L, 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for l_i, ax in enumerate(axes):
        share = counts[l_i] / counts[l_i].sum()
        bottom = np.zeros(E)
        for d_i, d in enumerate(domains):
            ax.bar(np.arange(E), share[:, d_i], bottom=bottom, width=0.7,
                   color=DOMAIN_COLORS.get(d, "#888"), edgecolor="white", linewidth=1,
                   label=d if l_i == 0 else None)
            bottom += share[:, d_i]
        ax.axhline(1 / E, color=INK_2, linewidth=0.8, linestyle="--")
        top = max(float((counts[l] / counts[l].sum()).sum(axis=1).max()) for l in range(L))
        ax.set_ylim(0, max(top, 1 / E) * 1.12)
        ax.set_xticks(np.arange(E), [f"E{e + 1}" for e in range(E)])
        ax.set_title(f"Layer {l_i + 1}", fontsize=9, color=INK)
        style_axes(ax)
    axes[0].set_ylabel("Share of routing slots", fontsize=8, color=INK_2)
    fig.legend(loc="upper center", ncol=len(domains), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{run_name}: expert usage by domain (dashed = perfectly even)",
                 fontsize=10, color=INK, y=1.10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(damage, domains, run_name, path):
    """Heatmap per layer: rows = experts, columns = domains, cell = loss increase
    when that expert is switched off. Concentrated dark cells = real specialization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, E, D = damage.shape
    vmax = max(float(np.nanmax(damage)), 1e-6)
    fig, axes = plt.subplots(1, L, figsize=(1.9 * L, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for l_i, ax in enumerate(axes):
        im = ax.imshow(np.clip(damage[l_i], 0, None), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for e in range(E):
            for d_i in range(D):
                v = damage[l_i, e, d_i]
                ax.text(d_i, e, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.6 * vmax else INK)
        ax.set_xticks(range(D), domains, fontsize=7, rotation=45)
        ax.set_yticks(range(E), [f"E{e + 1}" for e in range(E)], fontsize=7)
        ax.set_title(f"Layer {l_i + 1}", fontsize=9, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    cbar.set_label("Loss increase when expert is off", fontsize=8, color=INK_2)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"{run_name}: ablation damage (negative values shown as 0, labeled)",
                 fontsize=10, color=INK, y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_checkpoint(path, args, device):
    model, ckpt = load_model(path, device)
    cfg = ckpt["config"]
    run_name = cfg["run_name"]
    domains = cfg["domains"]
    ctx = autocast_ctx(device)
    loader = MixedDomainLoader(args.data_dir, domains=domains,
                               block_size=model.config.block_size, device=device)
    layers = model.moe_layers()
    print(f"\n=== {run_name} ({path}) | {'MoE' if layers else 'dense'} ===")

    result = {
        "run_name": run_name,
        "checkpoint": path,
        "balance": cfg.get("balance", "dense") if layers else "dense",
        "scope": cfg.get("scope") if layers else None,
        "aux_coef": cfg.get("aux_coef") if layers else None,
        "seed": cfg.get("seed"),
        "max_iters": cfg.get("max_iters"),
        "domains": domains,
        "eval_batches": args.n_batches,
        "eval_batch_size": args.batch_size,
    }

    counts = np.zeros((len(layers), layers[0].n_experts, len(domains))) if layers else None
    base = domain_losses(model, loader, args.n_batches, args.batch_size, ctx, counts)
    result["loss"] = base
    result["loss_mean"] = float(np.mean(list(base.values())))
    print("  loss   | " + " | ".join(f"{d} {v:.4f}" for d, v in base.items()))

    if layers:
        result["counts"] = counts.tolist()
        result["purity_per_layer"] = [purity(counts[l]) for l in range(len(layers))]
        result["util_per_layer"] = [utilization(counts[l]) for l in range(len(layers))]
        result["nmi_per_layer"] = [normalized_mi(counts[l]) for l in range(len(layers))]
        for key in ("purity", "util", "nmi"):
            result[key] = float(np.nanmean(result[f"{key}_per_layer"]))
        print(f"  purity {result['purity']:.3f} | util {result['util']:.3f} | nmi {result['nmi']:.3f}")
        print("  purity per layer: " + " ".join(f"{v:.3f}" for v in result["purity_per_layer"]))
        plot_usage(counts, domains, run_name, os.path.join(args.out_dir, f"{run_name}_usage.png"))

        if args.ablation:
            n_layers, n_experts = counts.shape[0], counts.shape[1]
            damage = np.zeros((n_layers, n_experts, len(domains)))
            ab_base = domain_losses(model, loader, args.ablation_batches, args.batch_size, ctx)
            for l_i, layer in enumerate(layers):
                for e in range(n_experts):
                    layer.disabled_experts = {e}
                    ab = domain_losses(model, loader, args.ablation_batches, args.batch_size, ctx)
                    layer.disabled_experts = set()
                    damage[l_i, e] = [ab[d] - ab_base[d] for d in domains]
                print(f"  ablation layer {l_i + 1}/{n_layers} done")
            result["ablation_damage"] = damage.tolist()
            result["ablation_batches"] = args.ablation_batches
            # How concentrated is the damage? For each (layer, expert), the share of
            # its positive damage that falls on its single most-hurt domain.
            pos = np.clip(damage, 0, None)
            tot = pos.sum(axis=2)
            conc = np.where(tot > 0, pos.max(axis=2) / np.where(tot > 0, tot, 1), np.nan)
            result["ablation_concentration"] = float(np.nanmean(conc))
            print(f"  ablation concentration {result['ablation_concentration']:.3f} "
                  f"(1.0 = each expert's damage hits one domain, ~0.33 = spread evenly)")
            plot_ablation(damage, domains, run_name,
                          os.path.join(args.out_dir, f"{run_name}_ablation.png"))

    out = os.path.join(args.out_dir, f"{run_name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved {out}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="checkpoint path(s); globs allowed")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--n_batches", type=int, default=50, help="test batches per domain")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--ablation", action="store_true", help="also run expert ablation (slower)")
    p.add_argument("--ablation_batches", type=int, default=10,
                   help="test batches per domain for each ablation")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = pick_device(args.device)
    paths = sorted({f for pattern in args.ckpt for f in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No checkpoints matched {args.ckpt}")
    for path in paths:
        evaluate_checkpoint(path, args, device)


if __name__ == "__main__":
    main()