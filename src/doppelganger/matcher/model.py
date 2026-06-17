"""
Matcher model: target audio -> Operator params, DIRECT one-shot prediction.

Deterministic (v2, 2026-06-16). The earlier diffusion denoiser is gone: it was
demonstrably vestigial here (its x0-MSE railed instead of falling, no multimodality
materialized), and a deterministic predictor is the most direct realization of the
product — `audio -> params` in a single forward pass, deployable in the extension via
ONNX on CPU, no synth in the loop. See docs/matcher-design.md.

Three pieces:
  • AudioEncoder   — target log-magnitude spectrogram -> embedding z.
  • AlgoClassifier — z -> 11-way Algorithm logits (the one discrete param; softmax+CE).
  • ParamHead      — z (+ algo embed, note, velocity) -> a position in [0,1] for each of
                     the other 194 params, mapped AFFINELY into the dataset MANIFOLD box.

The manifold box (per-param observed [min,max] over the training set) is the fix for the
railing/silence that the audio-loss-driven version produced: the predictor can only emit
presets inside the distribution the data lives in — which is audible by construction
(the collector's SAMPLING_RULES guarantee Volume in [0.45,1], Transpose in ±12, device
on, etc.). A sigmoid output + affine map keeps the prediction smoothly inside that box
(gradient everywhere), so the audio loss can match WITHIN the manifold but can never
escape it to chase the renderer's slack (Volume -> 0 silence, Transpose -> +48, ...).
"""

from __future__ import annotations

import torch
from torch import nn

from ..schema import OperatorSchema

ALGO_NAME = "Algorithm"


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


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(x)


class ParamHead(nn.Module):
    """conditioning vector -> sigmoid position-in-manifold [B, dim] (one forward pass)."""

    def __init__(self, dim: int, cond_dim: int, hidden: int = 512, blocks: int = 4):
        super().__init__()
        self.inp = nn.Linear(cond_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.out = nn.Linear(hidden, dim)

    def forward(self, cond):
        h = self.inp(cond)
        for blk in self.blocks:
            h = blk(h)
        return torch.sigmoid(self.out(h))            # [B, dim] in [0,1]


class Matcher(nn.Module):
    def __init__(self, schema: OperatorSchema, emb: int = 256, cond_dim: int = 256,
                 algo_emb: int = 32, hidden: int = 512, blocks: int = 4):
        super().__init__()
        self.schema = schema
        names = [p.name for p in schema.params]
        self.algo_idx = names.index(ALGO_NAME)
        self.n_algo = schema.params[self.algo_idx].cardinality or 11
        keep = [i for i in range(len(names)) if i != self.algo_idx]
        self.register_buffer("keep_idx", torch.tensor(keep), persistent=False)
        self.dim = len(keep)                                    # 194

        self.encoder = AudioEncoder(emb)
        self.algo_head = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, self.n_algo))
        self.algo_embedding = nn.Embedding(self.n_algo, algo_emb)
        self.cond_proj = nn.Linear(emb + algo_emb + 2, cond_dim)  # + note_n + vel_n
        self.param_head = ParamHead(self.dim, cond_dim, hidden, blocks)

        # Manifold bounds in normalized-param space over the KEPT dims. Persistent buffers
        # (saved in the checkpoint, loaded at inference). Default [0,1]; set from data.
        self.register_buffer("man_lo", torch.zeros(self.dim))
        self.register_buffer("man_hi", torch.ones(self.dim))

    def set_manifold(self, full_lo: torch.Tensor, full_hi: torch.Tensor) -> None:
        self.man_lo = full_lo[self.keep_idx].to(self.man_lo)
        self.man_hi = full_hi[self.keep_idx].to(self.man_hi)

    # --- vector bookkeeping -----------------------------------------------------
    def true_algo(self, params01: torch.Tensor) -> torch.Tensor:
        return torch.round(params01[:, self.algo_idx] * (self.n_algo - 1)).long()

    def target_pos(self, params01: torch.Tensor) -> torch.Tensor:
        """true normalized params -> position-in-manifold [0,1] over kept dims (the
        param-head's supervision target / anchor)."""
        keep = params01[:, self.keep_idx]
        span = (self.man_hi - self.man_lo).clamp_min(1e-6)
        return ((keep - self.man_lo) / span).clamp(0, 1)

    def assemble(self, pos: torch.Tensor, algo: torch.Tensor) -> torch.Tensor:
        """pos [B,dim] in [0,1] -> full [B,195] normalized, mapped into the manifold box."""
        B = pos.shape[0]
        full = torch.zeros(B, len(self.schema.params), device=pos.device, dtype=pos.dtype)
        full[:, self.keep_idx] = self.man_lo + pos * (self.man_hi - self.man_lo)
        full[:, self.algo_idx] = algo.to(full.dtype) / max(self.n_algo - 1, 1)
        return full

    # --- forward pieces ---------------------------------------------------------
    def encode(self, logmag: torch.Tensor) -> torch.Tensor:
        return self.encoder(logmag)

    def cond(self, z, algo, note, velocity) -> torch.Tensor:
        a = self.algo_embedding(algo)
        nv = torch.stack([note / 127.0, velocity / 127.0], dim=-1)
        return self.cond_proj(torch.cat([z, a, nv], dim=-1))

    def predict(self, z, algo, note, velocity) -> torch.Tensor:
        """z + algo + pitch -> position-in-manifold [B, dim] (deterministic)."""
        return self.param_head(self.cond(z, algo, note, velocity))
