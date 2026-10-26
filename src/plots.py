"""
src/plots.py
Compare runs: reads every results/<run>.json written by evaluate.py and draws
the project's three cross-run plots.

  1. specialization_vs_utilization.png : one dot per run (the trade-off, Q1)
  2. purity_by_layer.png               : purity per layer, one line per run (Q2)
  3. loss_by_domain.png                : test loss per domain for every run
  Also prints a summary table (markdown) you can paste into the writeup.

Usage
  python src/plots.py --results_dir results
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colorblind-validated categorical palette, used in fixed order (never cycled)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DOMAIN_COLORS = {"stories": SERIES[0], "code": SERIES[1], "math": SERIES[2]}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def describe(r):
    if r["balance"] == "dense":
        return "dense"
    if r["balance"] == "none":
        return "no balancing"
    if r["balance"] == "bias":
        return f"bias, {r['scope']}"
    return f"aux {r['aux_coef']}, {r['scope']}"


def load_results(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if "loss" in r:
            runs.append(r)
    return runs


def plot_tradeoff(moe, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for r in moe:
        ax.scatter(r["util"], r["purity"], s=60, color=SERIES[0],
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{r['run_name']} ({describe(r)})", (r["util"], r["purity"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK)
    n_domains = len(moe[0]["domains"])
    ax.axhline(1 / n_domains, color=INK_2, linestyle="--", linewidth=0.8)
    ax.text(0.01, 1 / n_domains + 0.01, "no specialization", fontsize=7, color=INK_2,
            transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1.05)
    ax.set_ylim(1 / n_domains - 0.05, 1.02)
    ax.set_xlabel("Utilization (1.0 = experts used evenly)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("Routing purity (1.0 = fully specialized)", fontsize=8.5, color=INK_2)
    ax.set_title("Specialization vs. utilization, one dot per run", fontsize=10, color=INK, loc="left")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_purity_by_layer(moe, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for i, r in enumerate(moe[:len(SERIES)]):
        y = r["purity_per_layer"]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", label=f"{r['run_name']} ({describe(r)})")
        ax.annotate(r["run_name"], (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=INK, va="center")
    n_domains = len(moe[0]["domains"])
    ax.axhline(1 / n_domains, color=INK_2, linestyle="--", linewidth=0.8)
    ax.set_xticks(np.arange(1, len(moe[0]["purity_per_layer"]) + 1))
    ax.set_xlabel("Layer", fontsize=8.5, color=INK_2)
    ax.set_ylabel("Routing purity", fontsize=8.5, color=INK_2)
    ax.set_title("Which layers specialize? Purity per layer", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.08, 1))
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_losses(runs, path):
    domains = [d for d in runs[0]["domains"]]
    n, width = len(runs), 0.8 / len(domains)
    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * n + 2), 3.8))
    x = np.arange(n)
    for d_i, d in enumerate(domains):
        vals = [r["loss"].get(d, np.nan) for r in runs]
        ax.bar(x + (d_i - (len(domains) - 1) / 2) * width, vals, width=width * 0.92,
               color=DOMAIN_COLORS.get(d, SERIES[d_i]), label=d)
    ax.set_xticks(x, [r["run_name"] for r in runs], fontsize=8)
    ax.set_ylabel("Test loss (lower is better)", fontsize=8.5, color=INK_2)
    ax.set_title("Test loss by domain", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, ncol=len(domains))
    style_axes(ax)
    ax.xaxis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary_table(runs):
    domains = runs[0]["domains"]
    head = ["Run", "Setting", "Purity", "Util", "NMI", *[f"Loss {d}" for d in domains], "Loss mean"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        def fmt(k):
            v = r.get(k)
            return f"{v:.3f}" if isinstance(v, (int, float)) else "—"
        row = [r["run_name"], describe(r), fmt("purity"), fmt("util"), fmt("nmi"),
               *[f"{r['loss'][d]:.3f}" for d in domains], f"{r['loss_mean']:.3f}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    args = p.parse_args()

    runs = load_results(args.results_dir)
    if not runs:
        raise SystemExit(f"No result files in {args.results_dir}; run src/evaluate.py first.")
    order = {"dense": 0}
    runs.sort(key=lambda r: (order.get(r["balance"], 1), r["run_name"]))
    moe = [r for r in runs if r["balance"] != "dense"]

    if moe:
        plot_tradeoff(moe, os.path.join(args.results_dir, "specialization_vs_utilization.png"))
        plot_purity_by_layer(moe, os.path.join(args.results_dir, "purity_by_layer.png"))
    plot_losses(runs, os.path.join(args.results_dir, "loss_by_domain.png"))

    table = summary_table(runs)
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)
    print(f"\nPlots and summary.md saved in {args.results_dir}/")


if __name__ == "__main__":
    main()