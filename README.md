# Do MoE Experts Specialize Under Load Balancing?

NYU Machine Learning, Fall 2026 final project.
Team: Achyuthan Sivasankar, Archi Patel, Ashik M John

## Question
When we force the experts in a Mixture-of-Experts model to share work evenly, do they stop specializing, and does that matter? We reproduce the main finding of Falke et al. (2026), "MoE Routing Testbed" (balancing scope matters more than method), on overlapping domains (stories, code, math), and extend it with per-layer analysis and expert ablation.

## Repo structure
- `data/prepare.py`: download, tokenize, split each domain
- `src/model.py`: transformer (adapted from nanoGPT)
- `src/moe.py`: MoE layer and balancing rules
- `src/data.py`: mixed-domain batch loader
- `src/train.py`: training loop
- `src/evaluate.py`: specialization metrics and plots
- `configs/`: one config per experiment run
- `results/`: metrics and plots

## Setup
pip install -r requirements.txt

## Runs
| Run | Balancing | Scope |
|---|---|---|
| Dense | none (no MoE) | — |
| A | none | — |
| B | aux loss, weak | local |
| D | aux loss, weak | global |
| F | bias-based | global |

## Acknowledgements
Model code adapted from nanoGPT (Karpathy). Balancing references: Switch Transformer (Fedus et al., 2022), Auxiliary-Loss-Free Load Balancing (Wang et al., 2024).
