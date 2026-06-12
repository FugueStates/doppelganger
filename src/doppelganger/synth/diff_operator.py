"""
DiffOperator — a DIFFERENTIABLE FM synth that approximates Ableton Operator.

This is the DDSP/"physics-informed" forward model: a torch reimplementation of
Operator's FM topology (the *known physics* of phase-modulation synthesis). Inside the
HybridRenderer it is the gray-box PRIOR — it places the FM sidebands; the neural residual
corrects the remainder. Final params are still applied to the REAL Operator.

v3 (the Batch-5 refactor — see docs/batch5-refactor.md for the full old->new reasoning):
  • 4 sine operators (A,B,C,D), phase modulation, the 11 algorithm routings.
  • FULLY VECTORIZED across algorithms: every routing edge goes (higher idx -> lower idx),
    so D,C,B,A is a valid topological order for ALL 11 algorithms — one batched pass with
    per-sample adjacency masks replaced the old per-algorithm Python grouping (which split
    a batch of 16 into ~10 tiny sub-batches and throttled the GPU on kernel-launch
    overhead). This also unblocks torch.compile (no data-dependent Python control flow)
    and pays again in Phase B, where the matcher renders changing params every step.
  • PER-SAMPLE f0: `f0` may be a [B] tensor (pitch conditioning, Batch 5) or a float
    (legacy single-pitch calibrate/search paths).
  • fp64 PHASE: at 3 s × ~12 kHz the phase reaches ~2.4e5 rad where fp32 spacing is
    ~0.03 rad — audible broadband noise in the prior's tail. Phase *cycles* are computed
    in fp64, wrapped with remainder(·,1), then cast to fp32 (elementwise fp64 is
    bandwidth-bound — negligible cost, even on a 3090's weak fp64 units).
  • Per-op ADSR with note-on/off and PER-SEGMENT learnable curvature.
  • ANTI-ALIASED rendering: oversample -> windowed-sinc low-pass -> decimate. The
    decimating conv now uses stride=os (computes only the kept samples — 2x cheaper,
    bit-identical to the old full-conv-then-slice).
  • BOUNDED LEARNABLE calibration (fm depth, envelope times + curvature): trained jointly
    with the residual via the spectral loss. Bounded (sigmoid x ceiling) to avoid the
    degeneracy a free least-squares fit showed.
  • `volume` was REMOVED: it was applied before peak-normalization, which cancels a
    global scale exactly — output and gradient were identically unaffected (a dead knob).
  Still not modeled: non-sine waveforms, filter, LFO, self-feedback, fixed-frequency
  oscillators and Fine detune (pending the freq_map verification sweeps), velocity.

Operators are indexed 0=A,1=B,2=C,3=D. Edges are (modulator -> carrier).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
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

# The vectorization invariant: every modulator index > its carrier index, so computing
# ops in the fixed order D(3) -> C(2) -> B(1) -> A(0) is a valid topological order for
# EVERY algorithm. Verified at import so a future routing edit can't silently break it.
assert all(m > c for a in ALGORITHMS for m, c in a["edges"]), \
    "ALGORITHMS invariant broken: an edge modulates upward — fixed-order rendering invalid"


def _routing_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """ALGORITHMS table -> (adj [11,4,4], carriers [11,4]) mask tensors.
    adj[a, m, c] = 1 if op m modulates op c under algorithm a."""
    adj = torch.zeros(len(ALGORITHMS), N_OPS, N_OPS)
    car = torch.zeros(len(ALGORITHMS), N_OPS)
    for a, spec in enumerate(ALGORITHMS):
        for m, c in spec["edges"]:
            adj[a, m, c] = 1.0
        for c in spec["carriers"]:
            car[a, c] = 1.0
    return adj, car


@dataclass
class CalibConstants:
    """Fixed calibration (legacy path: calibrate.py / search.py pass this explicitly).
    Inside the renderer these are LEARNED instead — see DiffOperator's bounded params."""
    fm_scale: float = 6.0       # maps op level -> radians of phase deviation
    max_attack: float = 2.0     # seconds at param=1
    max_decay: float = 3.0
    max_release: float = 4.0
    env_curve: float = 4.0      # exponential curvature of envelope segments


def adsr(
    a: torch.Tensor, d: torch.Tensor, s: torch.Tensor, r: torch.Tensor,
    t: torch.Tensor, note_off: float,
    max_a, max_d, max_r, curve_a, curve_d, curve_r,
) -> torch.Tensor:
    """Differentiable ADSR with EXPONENTIAL (curved) segments, like Operator.
    a/d/s/r are [B] in [0,1]; t is [T] seconds. max_*/curve_* are scalars (float or
    learnable tensor). Each segment v0->v1 over time T: v(p)=v1+(v0-v1)*exp(-k*p)."""
    a_t = (a * max_a).clamp(min=1e-3).unsqueeze(1)   # [B,1]
    d_t = (d * max_d).clamp(min=1e-3).unsqueeze(1)
    r_t = (r * max_r).clamp(min=1e-3).unsqueeze(1)
    s = s.unsqueeze(1)
    tt = t.unsqueeze(0)  # [1,T]
    off = torch.as_tensor(note_off, device=t.device)
    z, one = torch.zeros_like(s), torch.ones_like(s)

    def seg(v0, v1, prog, k):  # curved interpolation v0 -> v1
        return v1 + (v0 - v1) * torch.exp(-k * prog.clamp(0, 1))

    attack = seg(z, one, tt / a_t, curve_a)                                 # 0 -> 1
    decay = seg(one, s, (tt - a_t) / d_t, curve_d)                          # 1 -> s
    pre_off = torch.where(tt < a_t, attack, decay)                         # held
    held_at_off = torch.where(
        off < a_t,
        seg(z, one, off / a_t, curve_a),
        seg(one, s, (off - a_t) / d_t, curve_d),
    )
    release = seg(held_at_off, z, (tt - off) / r_t, curve_r)               # held -> 0
    return torch.where(tt < off, pre_off, release).clamp(min=0.0)


def _inv_sigmoid(v: float, ceil: float) -> float:
    """Raw value whose sigmoid*ceil == v (to init a bounded learnable param)."""
    x = min(max(v / ceil, 1e-4), 1 - 1e-4)
    return math.log(x / (1 - x))


class DiffOperator(nn.Module):
    """Renders [B, n_samples] mono audio from per-op FM controls. The batch may mix
    algorithms freely (per-sample `algo` tensor) — see the vectorization note above."""

    # ceilings for the bounded learnable calibration (sigmoid * ceiling)
    _CEIL = {"fm": 30.0, "a": 4.0, "d": 8.0, "r": 10.0, "curve": 12.0}

    def __init__(self, sample_rate: int = 16000, n_samples: int = 48000,
                 note_off: float = 1.5, calib: CalibConstants | None = None,
                 oversample: int = 2, learn_calib: bool = True):
        super().__init__()
        self.sr = sample_rate
        self.n = n_samples
        self.note_off = note_off
        self.os = max(1, oversample)
        # time grids: base rate (legacy/external) + oversampled (rendering)
        self.register_buffer("t", torch.arange(n_samples).float() / sample_rate)
        self.register_buffer("t_os", torch.arange(n_samples * self.os).float()
                             / (sample_rate * self.os))
        if self.os > 1:
            self.register_buffer("aa_kernel", self._make_aa_kernel(self.os))
        # Routing masks. persistent=False: they're constants derived from ALGORITHMS,
        # keeping them out of the state dict preserves old-checkpoint compatibility.
        adj, car = _routing_tensors()
        self.register_buffer("adj", adj, persistent=False)        # [11,4,4]
        self.register_buffer("carriers", car, persistent=False)   # [11,4]

        init = calib or CalibConstants()
        c = self._CEIL
        self.raw_fm = nn.Parameter(torch.tensor(_inv_sigmoid(init.fm_scale, c["fm"])))
        self.raw_a = nn.Parameter(torch.tensor(_inv_sigmoid(init.max_attack, c["a"])))
        self.raw_d = nn.Parameter(torch.tensor(_inv_sigmoid(init.max_decay, c["d"])))
        self.raw_r = nn.Parameter(torch.tensor(_inv_sigmoid(init.max_release, c["r"])))
        self.raw_ca = nn.Parameter(torch.tensor(_inv_sigmoid(init.env_curve, c["curve"])))
        self.raw_cd = nn.Parameter(torch.tensor(_inv_sigmoid(init.env_curve, c["curve"])))
        self.raw_cr = nn.Parameter(torch.tensor(_inv_sigmoid(init.env_curve, c["curve"])))
        if not learn_calib:
            for p in (self.raw_fm, self.raw_a, self.raw_d, self.raw_r,
                      self.raw_ca, self.raw_cd, self.raw_cr):
                p.requires_grad_(False)

    # --- bounded learnable calibration (sigmoid * ceiling) --------------------
    @property
    def fm_scale(self):
        return torch.sigmoid(self.raw_fm) * self._CEIL["fm"]

    @property
    def max_attack(self):
        return torch.sigmoid(self.raw_a) * self._CEIL["a"]

    @property
    def max_decay(self):
        return torch.sigmoid(self.raw_d) * self._CEIL["d"]

    @property
    def max_release(self):
        return torch.sigmoid(self.raw_r) * self._CEIL["r"]

    @property
    def curves(self):
        c = self._CEIL["curve"]
        return (torch.sigmoid(self.raw_ca) * c, torch.sigmoid(self.raw_cd) * c,
                torch.sigmoid(self.raw_cr) * c)

    def _make_aa_kernel(self, os: int, zeros: int = 16) -> torch.Tensor:
        """Hann-windowed sinc low-pass at the original Nyquist, for OS-rate decimation."""
        K = 2 * zeros * os + 1
        n = torch.arange(K).float() - (K - 1) / 2
        fc = 0.5 / os  # cutoff in cycles/sample at the oversampled rate
        kernel = 2 * fc * torch.sinc(2 * fc * n)
        kernel = kernel * torch.hann_window(K, periodic=False)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, K)

    def _decimate(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n*os] -> [B, n] anti-alias low-pass + subsample. stride=os makes the conv
        compute only the samples we keep (identical output, ~os-times cheaper than the
        old full-resolution conv followed by [::os] slicing)."""
        if self.os == 1:
            return x[:, :self.n]
        pad = self.aa_kernel.shape[-1] // 2
        x = F.conv1d(x.unsqueeze(1), self.aa_kernel, padding=pad, stride=self.os).squeeze(1)
        return x[:, :self.n]

    def render(
        self,
        coarse: torch.Tensor,   # [B,4] frequency multipliers (ignored if freq_hz given)
        level: torch.Tensor,    # [B,4] op output levels in [0,1]
        adsr_params: torch.Tensor,  # [B,4,4] = (a,d,s,r) per op, each in [0,1]
        algo: int | torch.Tensor,   # algorithm index 0..10 — int or per-sample [B] long
        f0: float | torch.Tensor = 261.63,  # fundamental Hz — float or per-sample [B]
        calib: CalibConstants | None = None,  # fixed override (legacy); else learned params
        freq_hz: torch.Tensor | None = None,  # [B,4] absolute per-op Hz — overrides
        # f0*coarse. The renderer uses this for the MEASURED frequency laws (floored
        # ratio x fine multiplier, note-independent fixed oscillators); the legacy
        # coarse/f0 path remains for calibrate.py and quick tests.
    ) -> torch.Tensor:
        B = level.shape[0]
        device = level.device
        t = self.t_os                       # render at the oversampled rate
        if calib is not None:               # legacy fixed-constant path
            fm_scale = calib.fm_scale
            max_a, max_d, max_r = calib.max_attack, calib.max_decay, calib.max_release
            cv_a = cv_d = cv_r = calib.env_curve
        else:                               # learned, bounded calibration
            fm_scale = self.fm_scale
            max_a, max_d, max_r = self.max_attack, self.max_decay, self.max_release
            cv_a, cv_d, cv_r = self.curves

        if not torch.is_tensor(algo):
            algo = torch.full((B,), int(algo), dtype=torch.long, device=device)
        adj = self.adj[algo]                # [B,4,4] per-sample routing
        car = self.carriers[algo]           # [B,4]

        if freq_hz is None:                 # legacy path: per-op Hz = f0 x coarse ratio
            if not torch.is_tensor(f0):
                f0 = torch.full((B,), float(f0), device=device)
            freq_hz = f0.view(B, 1) * coarse
        # Phase in CYCLES, computed in fp64 then wrapped to [0,1): at fp32 the raw phase
        # (~2.4e5 rad at 3 s, ratio 48) has ~0.03 rad quantization — a noise floor the
        # residual would have to clean up. remainder() keeps the value small so the fp32
        # sin sees full precision. Differentiable (d remainder/dx = 1 a.e.).
        cycles = freq_hz.double().unsqueeze(-1) * t.double()
        phase_base = (2 * math.pi) * torch.remainder(cycles, 1.0).float()  # [B,4,T*os]
        env = torch.stack(
            [adsr(adsr_params[:, i, 0], adsr_params[:, i, 1], adsr_params[:, i, 2],
                  adsr_params[:, i, 3], t, self.note_off,
                  max_a, max_d, max_r, cv_a, cv_d, cv_r) for i in range(N_OPS)],
            dim=1,
        )  # [B,4,T*os]

        # Fixed topological order D->C->B->A (valid for all algorithms — see invariant).
        # Each op's modulation input = adjacency-masked sum of already-computed ops.
        out = [torch.zeros(B, t.shape[0], device=device)] * N_OPS
        for op in (3, 2, 1, 0):
            mod = torch.zeros(B, t.shape[0], device=device)
            for m in range(op + 1, N_OPS):
                w = adj[:, m, op] * level[:, m]                  # [B] gate x level
                if w.requires_grad or bool((w != 0).any()):
                    mod = mod + (fm_scale * w).unsqueeze(-1) * out[m]
            out[op] = env[:, op] * torch.sin(phase_base[:, op] + mod)

        audio = sum((car[:, c] * level[:, c]).unsqueeze(-1) * out[c] for c in range(N_OPS))
        audio = self._decimate(audio)       # anti-alias + back to base rate
        # normalize headroom (avoid clipping; relative timbre is what matters).
        # NOTE: this is exactly why a post-hoc global volume control was a no-op.
        peak = audio.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        return audio / peak


if __name__ == "__main__":
    # sanity: render a mixed-algorithm batch, check sound + differentiability
    synth = DiffOperator()
    B = 3
    coarse = torch.tensor([[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 3.0, 2.0, 1.0]],
                          requires_grad=True)
    level = torch.tensor([[1.0, 0.6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.8, 0.5, 0.4, 0.2]],
                         requires_grad=True)
    adsr_p = torch.zeros(B, 4, 4)
    adsr_p[..., 0] = 0.01  # fast attack
    adsr_p[..., 2] = 0.8   # sustain
    adsr_p[..., 3] = 0.2   # release
    algo = torch.tensor([0, 10, 6])                       # mixed algorithms in ONE call
    f0 = torch.tensor([261.63, 261.63, 523.25])           # mixed pitches in ONE call
    audio = synth.render(coarse, level, adsr_p, algo, f0=f0)
    loss = audio.pow(2).mean()
    loss.backward()
    print("audio", tuple(audio.shape), "peak", float(audio.abs().max()),
          "rms", float(audio.pow(2).mean().sqrt()))
    print("grad to level:", level.grad is not None and bool(level.grad.abs().sum() > 0))
    print("grad to coarse:", coarse.grad is not None and bool(coarse.grad.abs().sum() > 0))
    print("grad to fm_scale:", bool(synth.raw_fm.grad is not None and synth.raw_fm.grad.abs() > 0))
    print("calib:", f"fm={float(synth.fm_scale):.2f} a={float(synth.max_attack):.2f} "
          f"d={float(synth.max_decay):.2f} r={float(synth.max_release):.2f}")
    # legacy scalar path (calibrate.py): int algo + float f0 + fixed constants
    legacy = synth.render(coarse.detach(), level.detach(), adsr_p, algo=0, f0=261.63,
                          calib=CalibConstants())
    print("legacy path ok:", tuple(legacy.shape))
