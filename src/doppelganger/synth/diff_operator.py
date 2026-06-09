"""
DiffOperator — a DIFFERENTIABLE FM synth that approximates Ableton Operator.

This is the DDSP/"physics-informed" forward model: a torch reimplementation of
Operator's FM topology (the *known physics* of phase-modulation synthesis), used to
provide gradients/consistency losses the real (non-differentiable) Operator can't.
Final params are still applied to the REAL Operator; this is a training aid + the
clone experiment.

v1 scope (calibrate, measure fidelity, then extend):
  • 4 sine operators (A,B,C,D), phase modulation, the 11 algorithm routings.
  • Per-op ADSR amplitude envelopes with note-on/off (so release is modeled).
  • Free CALIBRATION constants (FM depth scale, envelope time mapping) to fit to real
    Operator. NOT yet modeled: additive/non-sine waveforms, filter, LFO, self-feedback,
    pitch/velocity scaling. Add after the core FM calibrates.

Operators are indexed 0=A,1=B,2=C,3=D. Edges are (modulator -> carrier).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

# --- The 11 algorithm routings (CONFIRMED from Operator's UI icons) ----------
# (modulator, carrier) edges + which ops are summed to output. A=0,B=1,C=2,D=3.
ALGORITHMS: list[dict] = [
    {"edges": [(3, 2), (2, 1), (1, 0)], "carriers": [0]},               # 1  D>C>B>A
    {"edges": [(3, 1), (2, 1), (1, 0)], "carriers": [0]},               # 2  D>B, C>B, B>A
    {"edges": [(3, 0), (2, 1), (1, 0)], "carriers": [0]},               # 3  D>A, C>B>A
    {"edges": [(3, 2), (3, 1), (2, 0), (1, 0)], "carriers": [0]},       # 4  D>C>A, D>B>A
    {"edges": [(3, 2), (2, 1), (2, 0)], "carriers": [0, 1]},            # 5  D>C, C>B, C>A
    {"edges": [(3, 2), (2, 1)], "carriers": [0, 1]},                    # 6  D>C>B, A alone
    {"edges": [(3, 0), (2, 0), (1, 0)], "carriers": [0]},               # 7  D>A, C>A, B>A
    {"edges": [(3, 2), (1, 0)], "carriers": [0, 2]},                    # 8  D>C, B>A
    {"edges": [(3, 1)], "carriers": [0, 1, 2]},                         # 9  A alone, D>B, C alone
    {"edges": [(3, 2)], "carriers": [0, 1, 2]},                         # 10 A alone, B alone, D>C
    {"edges": [], "carriers": [0, 1, 2, 3]},                            # 11 additive
]
N_OPS = 4


def _topo_order(edges: list[tuple[int, int]]) -> list[int]:
    """Order ops so every modulator is computed before the carriers it feeds."""
    order: list[int] = []
    remaining = set(range(N_OPS))
    while remaining:
        for op in sorted(remaining):
            # ready if no remaining op modulates `op` (its modulator inputs are done)
            if not any(mod in remaining and car == op for mod, car in edges):
                order.append(op)
                remaining.discard(op)
                break
    return order


@dataclass
class CalibConstants:
    fm_scale: float = 6.0       # maps op level -> radians of phase deviation (fit this)
    max_attack: float = 2.0     # seconds at param=1 (fit)
    max_decay: float = 3.0
    max_release: float = 4.0
    env_curve: float = 4.0      # exponential curvature of envelope segments (Operator is curved)


def adsr(
    a: torch.Tensor, d: torch.Tensor, s: torch.Tensor, r: torch.Tensor,
    t: torch.Tensor, note_off: float, cc: CalibConstants,
) -> torch.Tensor:
    """Differentiable ADSR with EXPONENTIAL (curved) segments, like Operator.
    a/d/s/r are [B] in [0,1]; t is [T] seconds. Returns [B, T].
    Each segment v0->v1 over time T: v(p) = v1 + (v0-v1)*exp(-k*p), p in [0,1]."""
    k = cc.env_curve
    a_t = (a * cc.max_attack).clamp(min=1e-3).unsqueeze(1)   # [B,1]
    d_t = (d * cc.max_decay).clamp(min=1e-3).unsqueeze(1)
    r_t = (r * cc.max_release).clamp(min=1e-3).unsqueeze(1)
    s = s.unsqueeze(1)
    tt = t.unsqueeze(0)  # [1,T]
    off = torch.as_tensor(note_off, device=t.device)

    def seg(v0, v1, prog):  # curved interpolation v0 -> v1
        return v1 + (v0 - v1) * torch.exp(-k * prog.clamp(0, 1))

    attack = seg(torch.zeros_like(s), torch.ones_like(s), tt / a_t)        # 0 -> 1
    decay = seg(torch.ones_like(s), s, (tt - a_t) / d_t)                    # 1 -> s
    pre_off = torch.where(tt < a_t, attack, decay)                         # envelope while held
    held_at_off = torch.where(
        off < a_t,
        seg(torch.zeros_like(s), torch.ones_like(s), off / a_t),
        seg(torch.ones_like(s), s, (off - a_t) / d_t),
    )
    release = seg(held_at_off, torch.zeros_like(s), (tt - off) / r_t)      # held -> 0
    return torch.where(tt < off, pre_off, release).clamp(min=0.0)


class DiffOperator(nn.Module):
    """Renders [B, n_samples] mono audio from per-op FM controls (one algorithm)."""

    def __init__(self, sample_rate: int = 16000, n_samples: int = 48000,
                 note_off: float = 1.5, calib: CalibConstants | None = None):
        super().__init__()
        self.sr = sample_rate
        self.n = n_samples
        self.note_off = note_off
        self.cc = calib or CalibConstants()
        self.register_buffer("t", torch.arange(n_samples).float() / sample_rate)

    def render(
        self,
        coarse: torch.Tensor,   # [B,4] frequency multipliers
        level: torch.Tensor,    # [B,4] op output levels in [0,1]
        adsr_params: torch.Tensor,  # [B,4,4] = (a,d,s,r) per op, each in [0,1]
        algo: int,              # algorithm index 0..10 (hard-selected)
        f0: float = 261.63,     # Ableton MIDI note 60 ("C3" label) = 261.6 Hz
        volume: torch.Tensor | None = None,  # [B] in [0,1]
        calib: CalibConstants | None = None,  # learnable constants (defaults to self.cc)
    ) -> torch.Tensor:
        cc = calib if calib is not None else self.cc
        B = coarse.shape[0]
        edges = ALGORITHMS[algo]["edges"]
        carriers = ALGORITHMS[algo]["carriers"]
        order = _topo_order(edges)

        # base (unmodulated) phase per op: 2*pi*f0*coarse*t
        phase_base = 2 * math.pi * f0 * coarse.unsqueeze(-1) * self.t.view(1, 1, -1)  # [B,4,T]
        env = torch.stack(
            [adsr(adsr_params[:, i, 0], adsr_params[:, i, 1], adsr_params[:, i, 2],
                  adsr_params[:, i, 3], self.t, self.note_off, cc) for i in range(N_OPS)],
            dim=1,
        )  # [B,4,T]

        out = [None] * N_OPS  # each op's env-shaped unit oscillator
        mods_of = {c: [m for m, car in edges if car == c] for c in range(N_OPS)}
        for op in order:
            mod = torch.zeros(B, self.n, device=coarse.device)
            for m in mods_of[op]:
                mod = mod + cc.fm_scale * level[:, m].unsqueeze(-1) * out[m]
            out[op] = env[:, op] * torch.sin(phase_base[:, op] + mod)

        audio = sum(level[:, c].unsqueeze(-1) * out[c] for c in carriers)
        if volume is not None:
            audio = audio * volume.unsqueeze(-1)
        # normalize headroom (avoid clipping; relative timbre is what matters)
        peak = audio.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        return audio / peak


if __name__ == "__main__":
    # sanity: render a 2-op FM tone (algo 1 reduced), check sound + differentiability
    synth = DiffOperator()
    B = 2
    coarse = torch.tensor([[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], requires_grad=True)
    level = torch.tensor([[1.0, 0.6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    adsr_p = torch.zeros(B, 4, 4)
    adsr_p[..., 0] = 0.01  # fast attack
    adsr_p[..., 2] = 0.8   # sustain
    adsr_p[..., 3] = 0.2   # release
    audio = synth.render(coarse, level, adsr_p, algo=0)
    loss = audio.pow(2).mean()
    loss.backward()
    print("audio", tuple(audio.shape), "peak", float(audio.abs().max()),
          "rms", float(audio.pow(2).mean().sqrt()))
    print("grad to level:", level.grad is not None and bool(level.grad.abs().sum() > 0))
