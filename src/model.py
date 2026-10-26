"""
src/model.py
A small GPT-style transformer, adapted from nanoGPT by Andrej Karpathy
(https://github.com/karpathy/nanoGPT, MIT License).

Changes from nanoGPT:
  - Each block's feed-forward network can be either a normal MLP (dense model)
    or an MoE layer (from src/moe.py), chosen by config.n_experts.
  - forward() also returns the total balancing loss from all MoE layers
    (0 for the dense model and for balance="none" / "bias").
  - after_optimizer_step() lets bias-based balancing update its per-expert
    biases once per training step.
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304   # GPT-2's 50257 padded to a multiple of 64 (faster on GPU)
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    # Feed-forward / MoE settings. n_experts = 0 means a normal dense model.
    expert_hidden: int = 1024  # hidden size of each FFN (dense MLP or one expert)
    n_experts: int = 0
    top_k: int = 2
    balance: str = "none"      # "none" | "aux" | "bias"
    scope: str = "global"      # "local" (per sequence) | "global" (whole batch)
    aux_coef: float = 0.0      # strength of the auxiliary loss
    bias_update: float = 0.001 # step size for bias-based balancing


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias (PyTorch's version always has one)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """A normal feed-forward network. Also used as one expert inside the MoE layer."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.expert_hidden, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.expert_hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.is_moe = config.n_experts > 0
        if self.is_moe:
            from moe import MoELayer  # imported only when needed
            self.mlp = MoELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        if self.is_moe:
            h, aux = self.mlp(self.ln_2(x))
        else:
            h, aux = self.mlp(self.ln_2(x)), x.new_zeros(())
        return x + h, aux


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying, as in GPT-2

        self.apply(self._init_weights)
        # Scaled init for the residual projections, as in GPT-2
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- helpers ------------------------------------------------------------

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def moe_layers(self):
        return [b.mlp for b in self.transformer.h if b.is_moe]

    def after_optimizer_step(self):
        """Called once per training step (used by bias-based balancing)."""
        for layer in self.moe_layers():
            layer.after_optimizer_step()

    # -- forward ------------------------------------------------------------

    def forward(self, idx, targets=None):
        """
        Returns logits, loss, aux_loss
          loss     : next-token prediction loss (None if no targets)
          aux_loss : total balancing loss from all MoE layers (0 if none)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        aux_total = x.new_zeros(())
        for block in self.transformer.h:
            x, aux = block(x)
            aux_total = aux_total + aux
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])  # only the last position is needed
            loss = None
        return logits, loss, aux_total

    # -- optimizer ----------------------------------------------------------

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]    # weight matrices, embeddings
        no_decay = [p for p in params if p.dim() < 2]  # biases, LayerNorm weights
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                 **({"fused": True} if use_fused else {}))

    # -- text generation ----------------------------------------------------

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx