"""
src/moe.py
Mixture-of-Experts feed-forward layer with three load-balancing modes.

For every token:
  1. The router gives each of the E experts a score (softmax probabilities).
  2. The token goes to its top-k experts (k = 2).
  3. Their outputs are combined, weighted by the router probabilities of the
     chosen experts (renormalized to sum to 1).
There is NO capacity limit and NO token dropping: every token is processed.

Balancing modes (config.balance)
  "none" : nothing. Expected to collapse (Run A).
  "aux"  : Switch-Transformer auxiliary loss, added to the training loss:
              aux = aux_coef * E * sum_i( f_i * P_i )
           f_i = share of routing slots that went to expert i
           P_i = average router probability of expert i
           It equals aux_coef * 1.0 when usage is perfectly even, more when uneven.
           config.scope decides which tokens f and P are computed over:
             "local"  : each sequence separately, then averaged (Runs B, C)
             "global" : the whole batch at once (Runs D, E)
  "bias" : DeepSeek auxiliary-loss-free balancing (Run F). Each expert has a
           bias that is added to its score ONLY when choosing the top-k. The
           mixing weights still use the original probabilities. After every
           optimizer step: overused experts' bias -= bias_update,
           underused experts' bias += bias_update. No gradients involved.

For analysis (src/evaluate.py)
  layer.record_routing = True  -> stores last_topk_idx (B, T, k) and last_probs (B, T, E)
  layer.disabled_experts = {e} -> expert e can never be chosen (expert ablation)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import MLP


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.balance in ("none", "aux", "bias"), f"unknown balance {config.balance!r}"
        assert config.scope in ("local", "global"), f"unknown scope {config.scope!r}"
        assert 1 <= config.top_k <= config.n_experts

        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.balance = config.balance
        self.scope = config.scope
        self.aux_coef = config.aux_coef
        self.bias_update = config.bias_update

        self.router = nn.Linear(config.n_embd, config.n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config) for _ in range(config.n_experts)])

        # Bias-based balancing state. Buffers are saved in the checkpoint but
        # are not trained by the optimizer.
        self.register_buffer("expert_bias", torch.zeros(config.n_experts))
        self.register_buffer("_load_accum", torch.zeros(config.n_experts), persistent=False)

        # Analysis hooks
        self.record_routing = False
        self.disabled_experts = set()
        self.last_topk_idx = None
        self.last_probs = None
        self.last_load = None  # routing slots per expert in the last forward pass

    def forward(self, x):
        B, T, C = x.shape
        E, k = self.n_experts, self.top_k

        # 1. Router probabilities, computed in float32 for stability
        logits = self.router(x).float()          # (B, T, E)
        probs = F.softmax(logits, dim=-1)        # (B, T, E)

        # 2. Choose top-k experts (bias only affects the CHOICE, not the weights)
        select = probs
        if self.balance == "bias":
            select = probs + self.expert_bias
        if self.disabled_experts:
            select = select.clone()
            select[..., list(self.disabled_experts)] = float("-inf")
        topk_idx = select.topk(k, dim=-1).indices                     # (B, T, k)
        weights = probs.gather(-1, topk_idx)                          # (B, T, k)
        weights = weights / weights.sum(dim=-1, keepdim=True)         # sum to 1

        # 3. Send each token through its chosen experts and combine
        xf = x.reshape(-1, C)                                         # (N, C)
        idx_f = topk_idx.reshape(-1, k)                               # (N, k)
        w_f = weights.reshape(-1, k)                                  # (N, k)
        out = torch.zeros_like(xf)
        for e in range(E):
            token_ids, slot = (idx_f == e).nonzero(as_tuple=True)
            if token_ids.numel() == 0:
                continue
            y = self.experts[e](xf[token_ids]) * w_f[token_ids, slot].unsqueeze(-1)
            out.index_add_(0, token_ids, y.to(out.dtype))
        out = out.view(B, T, C)

        # 4. Usage counts: how many routing slots each expert got
        assign = F.one_hot(topk_idx, E).sum(dim=-2).float()           # (B, T, E), 0/1 per expert
        load = assign.sum(dim=(0, 1)).detach()                        # (E,)
        self.last_load = load
        if self.training and self.balance == "bias":
            self._load_accum += load

        if self.record_routing:
            self.last_topk_idx = topk_idx.detach()
            self.last_probs = probs.detach()

        # 5. Auxiliary balancing loss
        if self.balance == "aux":
            if self.scope == "local":
                f = assign.mean(dim=1) / k                            # (B, E) per sequence
                P = probs.mean(dim=1)                                 # (B, E)
                aux = E * (f * P).sum(dim=-1).mean()
            else:
                f = assign.mean(dim=(0, 1)) / k                       # (E,) whole batch
                P = probs.mean(dim=(0, 1))                            # (E,)
                aux = E * (f * P).sum()
            aux = self.aux_coef * aux
        else:
            aux = out.new_zeros((), dtype=torch.float32)

        return out, aux

    @torch.no_grad()
    def after_optimizer_step(self):
        """Bias-based balancing: nudge each expert's bias toward even usage."""
        if self.balance != "bias" or self._load_accum.sum() == 0:
            return
        load = self._load_accum
        self.expert_bias += self.bias_update * torch.sign(load.mean() - load)
        self._load_accum.zero_()


def utilization(load):
    """Normalized entropy of expert usage: 1.0 = perfectly even, 0 = one expert does everything."""
    p = load / load.sum().clamp_min(1e-9)
    ent = -(p * (p.clamp_min(1e-12)).log()).sum()
    return (ent / torch.log(torch.tensor(float(len(load))))).item()


# ---------------------------------------------------------------------------
# Self-test: python src/moe.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from model import GPTConfig

    torch.manual_seed(0)
    x = torch.randn(4, 16, 256)
    for balance, scope in [("none", "global"), ("aux", "local"), ("aux", "global"), ("bias", "global")]:
        cfg = GPTConfig(n_experts=6, top_k=2, balance=balance, scope=scope, aux_coef=0.01)
        layer = MoELayer(cfg)
        out, aux = layer(x)
        assert out.shape == x.shape, "output shape must match input"
        out.sum().backward()
        assert layer.router.weight.grad is not None, "router must receive gradients"
        assert layer.last_load.sum().item() == 4 * 16 * 2, "every token must use exactly 2 slots"
        layer.after_optimizer_step()
        print(f"{balance:4s} / {scope:6s}: output OK | aux {aux.item():.5f} | "
              f"utilization {utilization(layer.last_load):.3f} | bias {layer.expert_bias.tolist()}")

    # Ablation check: a disabled expert must never be chosen
    layer = MoELayer(GPTConfig(n_experts=6, top_k=2))
    layer.disabled_experts = {3}
    layer.record_routing = True
    layer(x)
    assert not (layer.last_topk_idx == 3).any(), "disabled expert was chosen"
    print("ablation  : disabled expert never chosen")
    print("All MoE self-tests passed.")