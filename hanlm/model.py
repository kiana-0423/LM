from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    dim: int = 768
    layers: int = 12
    heads: int = 12
    hidden_dim: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0

    def __post_init__(self):
        if min(self.vocab_size, self.dim, self.layers, self.heads,
               self.hidden_dim, self.max_seq_len) <= 0:
            raise ValueError("All model dimensions must be positive")
        if self.dim % self.heads or (self.dim // self.heads) % 2:
            raise ValueError("dim must be divisible by heads, with an even head dimension")


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def rotary(x, offset, theta):
    d = x.shape[-1]
    freq = theta ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    pos = torch.arange(offset, offset + x.shape[-2], device=x.device).float()
    angles = pos[:, None] * freq[None, :]
    c, s = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, past=None, use_cache=False):
        b, t, _ = x.shape
        def split(proj):
            return proj(x).view(b, t, self.cfg.heads, -1).transpose(1, 2)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        offset = 0 if past is None else past[0].shape[-2]
        q, k = rotary(q, offset, self.cfg.rope_theta), rotary(k, offset, self.cfg.rope_theta)
        if past is not None:
            k, v = torch.cat((past[0], k), -2), torch.cat((past[1], v), -2)
        # Cached decoding needs an offset-aware mask for multi-token continuations.
        mask = None
        if offset:
            mask = (torch.arange(k.shape[-2], device=x.device)[None, :]
                    <= torch.arange(offset, offset + t, device=x.device)[:, None])
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=not bool(offset))
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(y), (k, v) if use_cache else None


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.gate_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)

    def forward(self, x, past=None, use_cache=False):
        h, cache = self.attn(self.attn_norm(x), past, use_cache)
        x = x + h
        h = self.ffn_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)), cache


class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)
        self.gradient_checkpointing = False
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, past=None, use_cache=False):
        offset = 0 if past is None else past[0][0].shape[-2]
        if ids.shape[1] + offset > self.cfg.max_seq_len:
            raise ValueError("Context exceeds max_seq_len")
        x, caches = self.embedding(ids), []
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training and not use_cache:
                x = checkpoint(lambda h, layer=block: layer(h)[0], x, use_reentrant=False)
                cache = None
            else:
                x, cache = block(x, None if past is None else past[i], use_cache)
            caches.append(cache)
        # Shared input/output embeddings; no duplicate output-head parameter.
        logits = F.linear(self.norm(x), self.embedding.weight)
        return (logits, caches) if use_cache else logits


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def add_lora(model, rank=8, alpha=16):
    if rank < 1 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    model.requires_grad_(False)
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, alpha))


@torch.no_grad()
def merge_lora(model):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight.add_(child.b @ child.a, alpha=child.scale)
                setattr(module, name, child.base)
    model.requires_grad_(True)
    return model
