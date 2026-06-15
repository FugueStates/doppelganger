"""
Diffusion schedule + sampler for the param matcher.

x0-PARAMETERIZATION (the denoiser predicts the clean param vector, not the noise): this
is what lets us attach the audio-loss-through-renderer term cleanly — at any noise level
the model emits a concrete param estimate we can render and score. The data is the
normalized param vector mapped to [-1, 1].

Cosine noise schedule (Nichol & Dhariwal 2021) — gentler at high SNR than linear, which
matters for a low-dimensional, structured target like a 194-D preset (vs a megapixel
image). Reverse sampling is DDIM (deterministic by default) so a handful of steps yields
candidates fast enough to render+rank a whole population at inference.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class Diffusion(nn.Module):
    def __init__(self, n_steps: int = 1000, s: float = 0.008):
        super().__init__()
        self.n_steps = n_steps
        t = torch.linspace(0, n_steps, n_steps + 1)
        f = torch.cos(((t / n_steps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - betas, dim=0))  # [n_steps]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward-noise x0 to level t. x0/noise [B, D]; t [B] long."""
        ab = self.alpha_bar[t].unsqueeze(-1)            # [B,1]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(self, x0_fn, shape, cond, device, steps: int = 50,
                    eta: float = 0.0) -> torch.Tensor:
        """Reverse DDIM. `x0_fn(x_t, t[B] long, cond) -> x0_hat`. Returns x0 in [-1,1]."""
        x = torch.randn(shape, device=device)
        ts = torch.linspace(self.n_steps - 1, 0, steps, device=device).round().long()
        for i in range(steps):
            t = ts[i]
            t_b = t.expand(shape[0])
            x0 = x0_fn(x, t_b, cond).clamp(-1, 1)
            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i < steps - 1 else torch.ones((), device=device)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).clamp_min(1e-8).sqrt()
            if eta > 0 and i < steps - 1:
                sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp_min(0).sqrt()
                x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp_min(0).sqrt() * eps \
                    + sigma * torch.randn_like(x)
            else:
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).clamp_min(0).sqrt() * eps
        return x
