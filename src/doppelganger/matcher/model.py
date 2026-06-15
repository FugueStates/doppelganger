"""
Matcher model: target audio -> Operator params.

Three pieces, per the 2026-06-15 decision to split off the discrete Algorithm:
  • AudioEncoder      — target log-magnitude spectrogram -> conditioning embedding z.
  • AlgoClassifier    — z -> 11-way Algorithm logits (the ONE discrete param given its
                        own head; FM topology is categorical and a softmax+CE fits it
                        far better than asking a continuous denoiser to hit a rounded bin).
  • ParamDenoiser     — FiLM-conditioned MLP over the FLAT param vector (NOT a 2-D U-Net:
                        params have no spatial structure). x0-diffusion: predicts the
                        clean normalized params for everything EXCEPT Algorithm.

The Matcher owns the schema-derived bookkeeping that splits the 195-D normalized vector
into {Algorithm column} + {the 194 the denoiser models}, and reassembles a full 195-D
vector (algo slot filled from the classifier) to feed the frozen renderer.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..schema import OperatorSchema

ALGO_NAME = "Algorithm"


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t [B] long -> [B, dim]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
        a = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([a.sin(), a.cos()], dim=-1)


class AudioEncoder(nn.Module):
    """log-mag [B, F, T] -> [B, emb]. Compact strided 2-D CNN + global average pool.
    GroupNorm on the first activation absorbs the wide dynamic range of raw log-mag."""

    def __init__(self, emb: int = 256, ch: int = 32):
        super().__init__()
        c = [1, ch, ch * 2, ch * 4, ch * 8]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                       nn.GroupNorm(8, c[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c[-1], emb)

    def forward(self, logmag: torch.Tensor) -> torch.Tensor:
        h = self.net(logmag.unsqueeze(1))            # [B, C, f, t]
        return self.proj(self.pool(h).flatten(1))    # [B, emb]


class FiLMResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, cond):
        g, b = self.film(cond).chunk(2, dim=-1)
        x = torch.nn.functional.gelu(self.l1(self.norm(h)))
        x = g * x + b
        return h + self.l2(x)


class ParamDenoiser(nn.Module):
    """(noisy params x_t [B,D], t [B], cond [B,cond_dim]) -> x0_hat [B,D] (in [-1,1])."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 512, blocks: int = 6,
                 t_dim: int = 128):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalTimeEmb(t_dim), nn.Linear(t_dim, t_dim), nn.GELU())
        self.in_proj = nn.Linear(dim, hidden)
        full_cond = cond_dim + t_dim
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, full_cond) for _ in range(blocks)])
        self.out = nn.Linear(hidden, dim)

    def forward(self, x_t, t, cond):
        c = torch.cat([cond, self.t_emb(t)], dim=-1)
        h = self.in_proj(x_t)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)


class Matcher(nn.Module):
    def __init__(self, schema: OperatorSchema, emb: int = 256, cond_dim: int = 256,
                 algo_emb: int = 32, hidden: int = 512, blocks: int = 6):
        super().__init__()
        self.schema = schema
        names = [p.name for p in schema.params]
        self.algo_idx = names.index(ALGO_NAME)
        self.n_algo = (schema.params[self.algo_idx].cardinality or 11)
        keep = [i for i in range(len(names)) if i != self.algo_idx]
        self.register_buffer("keep_idx", torch.tensor(keep), persistent=False)
        self.dim = len(keep)                                    # 194

        self.encoder = AudioEncoder(emb)
        self.algo_head = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, self.n_algo))
        self.algo_embedding = nn.Embedding(self.n_algo, algo_emb)
        self.cond_proj = nn.Linear(emb + algo_emb + 2, cond_dim)  # + note_n + vel_n
        self.denoiser = ParamDenoiser(self.dim, cond_dim, hidden, blocks)

    # --- normalized-vector <-> (algo, denoiser-subset) bookkeeping --------------
    def true_algo(self, params01: torch.Tensor) -> torch.Tensor:
        """Recover the integer Algorithm index from a normalized param matrix [B,195]."""
        return torch.round(params01[:, self.algo_idx] * (self.n_algo - 1)).long()

    def subset(self, params01: torch.Tensor) -> torch.Tensor:
        """[B,195] normalized -> [B,194] denoiser target in [-1,1] (Algorithm removed)."""
        return 2.0 * params01[:, self.keep_idx] - 1.0

    def assemble(self, x0: torch.Tensor, algo: torch.Tensor) -> torch.Tensor:
        """denoiser x0 [B,194] in [-1,1] + algo index [B] -> full [B,195] normalized [0,1]."""
        B = x0.shape[0]
        full = torch.zeros(B, len(self.schema.params), device=x0.device, dtype=x0.dtype)
        full[:, self.keep_idx] = ((x0.clamp(-1, 1) + 1) / 2)
        full[:, self.algo_idx] = algo.to(full.dtype) / max(self.n_algo - 1, 1)
        return full

    # --- forward pieces ---------------------------------------------------------
    def encode(self, logmag: torch.Tensor) -> torch.Tensor:
        return self.encoder(logmag)

    def cond(self, z, algo, note, velocity) -> torch.Tensor:
        a = self.algo_embedding(algo)
        nv = torch.stack([note / 127.0, velocity / 127.0], dim=-1)
        return self.cond_proj(torch.cat([z, a, nv], dim=-1))

    def x0(self, x_t, t, cond) -> torch.Tensor:
        return self.denoiser(x_t, t, cond)
