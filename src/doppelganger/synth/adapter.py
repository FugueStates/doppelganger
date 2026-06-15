"""
Map Operator parameters -> DiffOperator control tensors.

Two paths (see docs/batch5-refactor.md):

1. `ControlMap` (NEW, the training/matcher path) — a torch module that takes the
   NORMALIZED param matrix [B, n_params] (every param in [0,1], the same encoding the
   conditioning encoder eats) and derives the physics controls with differentiable
   slicing + affine denormalization. This replaced the old per-step Python double loop
   (`to_controls` over dicts), which (a) was slow, and (b) SEVERED THE AUTOGRAD GRAPH at
   the renderer's input — `float(p.get(...))` cannot backprop, so the Phase-B matcher
   could never receive gradients through the renderer. With ControlMap, gradients flow
   from the spectral loss through the physics to a predicted param TENSOR.

2. `to_controls` (legacy dict path) — kept for calibrate.py and other offline tools that
   work with raw param dicts and don't need gradients w.r.t. params.

Also new vs the old version:
- `Transpose` is now mapped (global semitone offset, ±48): the old adapter ignored it,
  so the physics rendered at the wrong fundamental for almost every dataset sample
  (collection randomizes Transpose ±12). One line: f0_eff = f0 * 2^(semitones/12).
- `volume` was dropped (it was a no-op — cancelled exactly by peak normalization).
- `Osc-X On` gates level with a hard >=0.5 mask (it's a binary param; gradients flow
  through the LEVEL, which is what the matcher needs).

FREQUENCY LAWS — MEASURED on the real Operator (freq_map sweeps, 2026-06-12, zero
residual; see dataset/freq_maps_report.json + docs/batch5-refactor.md §8):
- Coarse:   ratio = max(floor(raw), 0.5)      (raw 3.81 -> 3, NOT round; raw 0 -> 0.5.
            The old code fed raw 3.81 in as the literal ratio — up to a whole ratio
            unit of systematic sideband misplacement on every oscillator.)
- Fine:     MULTIPLICATIVE — freq *= (1 + fine_raw/1000)  (slope doubled at Coarse 2).
- Fixed:    note-INDEPENDENT — Hz = 10^(floor(mul) - 3) * 200^(fixfreq_raw), exact
            exponential per Mul decade (floor CONFIRMED by the fmap_fixedmul sweep:
            step transitions measured at integer raws, like Coarse).
- Quantize: NO effect on the coarse->ratio mapping (measured identical) — unmodeled.
The quantizers (floor) are applied with a STRAIGHT-THROUGH estimator: forward uses the
true quantized value, backward passes the gradient through unchanged — so the Phase-B
matcher still receives a useful gradient on Coarse / Fix Freq Mul.

v1 reads the params the sine-FM core uses; filter/LFO/waveform/feedback params are
ignored until those are modeled.

v1 reads the params the sine-FM core uses; filter/LFO/waveform/feedback params are
ignored until those are modeled.
"""

from __future__ import annotations

import torch
from torch import nn

from ..schema import OperatorSchema

OSCS = ["A", "B", "C", "D"]
ENV_PREFIX = ["Ae", "Be", "Ce", "De"]


def midi_to_hz(note: torch.Tensor) -> torch.Tensor:
    """MIDI note number -> Hz (equal temperament, A4=440). Ableton's 'C3' = MIDI 60
    = 261.63 Hz (the octave-naming gotcha that once cost us an octave)."""
    return 440.0 * torch.pow(2.0, (note - 69.0) / 12.0)


def normalize_params(param_dicts: list[dict], schema: OperatorSchema) -> torch.Tensor:
    """Raw param dicts -> [B, n_params] matrix, every param in [0,1].

    Continuous: (v - min) / span. Quantized: index / (cardinality - 1).
    This is THE shared param encoding: the conditioning encoder consumes it, ControlMap
    denormalizes the physics subset from it, precompute caches it, and the Phase-B
    matcher will predict it."""
    out = torch.zeros(len(param_dicts), len(schema.params))
    for b, p in enumerate(param_dicts):
        for i, pa in enumerate(schema.params):
            v = float(p.get(pa.name, pa.default))
            if pa.is_quantized:
                out[b, i] = v / max((pa.cardinality or 1) - 1, 1)
            else:
                span = pa.max - pa.min
                out[b, i] = (v - pa.min) / span if span else 0.0
    return out.clamp(0, 1)


def denormalize_params(vec, schema: OperatorSchema) -> dict:
    """Inverse of normalize_params for ONE preset: normalized [P] in [0,1] -> {name: raw}.

    Mirrors the adapter convention exactly (quantized raw index = round(norm*(card-1));
    continuous = min + norm*span). This is deliberately NOT schema.Param.denormalize —
    that treats a quantized `norm` as ALREADY being the index, whereas the whole
    matcher/renderer pipeline stores quantized params as index/(card-1) in [0,1]. Using
    the wrong one silently corrupts every multi-option categorical (Algorithm, waveforms,
    filter type, envelope modes) when the predicted preset is applied to the real device."""
    out = {}
    for i, pa in enumerate(schema.params):
        v = float(vec[i])
        if pa.is_quantized:
            card = pa.cardinality or 1
            out[pa.name] = float(min(card - 1, max(0, round(v * max(card - 1, 1)))))
        else:
            out[pa.name] = pa.min + min(1.0, max(0.0, v)) * (pa.max - pa.min)
    return out


class ControlMap(nn.Module):
    """Differentiable normalized-params [B,P] -> DiffOperator controls.

    All schema-derived constants are non-persistent buffers (move with .to(device),
    stay out of the state dict so old checkpoints keep loading)."""

    def __init__(self, schema: OperatorSchema):
        super().__init__()
        by_name = {p.name: (i, p) for i, p in enumerate(schema.params)}

        def idx(name: str) -> int:
            return by_name[name][0]

        def span(name: str) -> tuple[float, float]:
            p = by_name[name][1]
            return p.min, p.max - p.min

        self.i_algo = idx("Algorithm")
        self.n_algo = (by_name["Algorithm"][1].cardinality or 11) - 1  # 10
        self.i_transpose = idx("Transpose")
        t_min, t_span = span("Transpose")
        self.t_min, self.t_span = float(t_min), float(t_span)

        cols: dict[str, list] = {k: [] for k in
                                 ("coarse", "level", "on", "fine", "fix_on", "fix_freq", "fix_mul")}
        coarse_aff, mul_aff, adsr_idx = [], [], []
        for X, e in zip(OSCS, ENV_PREFIX):
            cols["coarse"].append(idx(f"{X} Coarse"))
            coarse_aff.append(span(f"{X} Coarse"))
            cols["level"].append(idx(f"Osc-{X} Level"))
            cols["on"].append(idx(f"Osc-{X} On"))
            cols["fine"].append(idx(f"{X} Fine"))            # normalized == raw/1000
            cols["fix_on"].append(idx(f"{X} Fix On "))       # (trailing space is real)
            cols["fix_freq"].append(idx(f"{X} Fix Freq"))    # raw [0,1] -> normalized == raw
            cols["fix_mul"].append(idx(f"{X} Fix Freq Mul"))
            mul_aff.append(span(f"{X} Fix Freq Mul"))
            # schema confirms all envelope A/D/S/R are raw [0,1] -> normalized == raw
            adsr_idx.append([idx(f"{e} {seg}") for seg in ("Attack", "Decay", "Sustain", "Release")])
        for k, v in cols.items():
            self.register_buffer(f"{k}_idx", torch.tensor(v), persistent=False)
        self.register_buffer("coarse_min", torch.tensor([a[0] for a in coarse_aff]), persistent=False)
        self.register_buffer("coarse_span", torch.tensor([a[1] for a in coarse_aff]), persistent=False)
        self.register_buffer("mul_min", torch.tensor([a[0] for a in mul_aff]), persistent=False)
        self.register_buffer("mul_span", torch.tensor([a[1] for a in mul_aff]), persistent=False)
        self.register_buffer("adsr_idx", torch.tensor(adsr_idx), persistent=False)  # [4,4]

    @staticmethod
    def _ste_floor(x: torch.Tensor) -> torch.Tensor:
        """floor with a straight-through gradient (forward quantized, backward identity)."""
        return x + (x.floor() - x).detach()

    def forward(self, vec: torch.Tensor) -> dict:
        """vec [B, P] normalized params -> control tensors. Differentiable except the
        discrete algo index and the binary gates; quantized frequency laws use a
        straight-through estimator (see module docstring)."""
        coarse_raw = self.coarse_min + vec[:, self.coarse_idx] * self.coarse_span  # [B,4]
        # MEASURED: ratio = max(floor(raw), 0.5); Fine multiplies by (1 + fine/1000)
        ratio = self._ste_floor(coarse_raw).clamp(min=0.5) * (1.0 + vec[:, self.fine_idx])
        # MEASURED: fixed mode Hz = 10^(mul_step - 3) * 200^fixfreq (note-independent)
        mul_raw = self.mul_min + vec[:, self.fix_mul_idx] * self.mul_span          # [B,4]
        fixed_hz = torch.pow(10.0, self._ste_floor(mul_raw) - 3.0) \
            * torch.pow(200.0, vec[:, self.fix_freq_idx])
        fix_on = (vec[:, self.fix_on_idx] >= 0.5).float()                          # hard gate
        on = (vec[:, self.on_idx] >= 0.5).float()                                  # hard gate
        level = vec[:, self.level_idx] * on                                        # [B,4]
        B = vec.shape[0]
        adsr = vec[:, self.adsr_idx.reshape(-1)].reshape(B, 4, 4)                  # [B,4,4]
        semitones = self.t_min + vec[:, self.i_transpose] * self.t_span            # [B]
        algo = torch.round(vec[:, self.i_algo] * self.n_algo).long().clamp(0, self.n_algo)
        return {"ratio": ratio, "fix_on": fix_on, "fixed_hz": fixed_hz,
                "level": level, "adsr": adsr, "semitones": semitones, "algo": algo}


def to_controls(params_list: list[dict], device: str = "cpu") -> dict:
    """LEGACY dict path (calibrate.py etc.): list of {param_name: raw_value} ->
    batched control tensors + per-sample algorithm index list. Not differentiable
    w.r.t. params — training/matching use ControlMap instead."""
    B = len(params_list)
    coarse = torch.zeros(B, 4)
    level = torch.zeros(B, 4)
    adsr = torch.zeros(B, 4, 4)  # (attack, decay, sustain, release)
    semitones = torch.zeros(B)
    algo = []

    for b, p in enumerate(params_list):
        for i, X in enumerate(OSCS):
            on = 1.0 if p.get(f"Osc-{X} On", 1.0) >= 0.5 else 0.0
            coarse[b, i] = float(p.get(f"{X} Coarse", 1.0))
            level[b, i] = float(p.get(f"Osc-{X} Level", 0.0)) * on
            e = ENV_PREFIX[i]
            adsr[b, i, 0] = float(p.get(f"{e} Attack", 0.0))
            adsr[b, i, 1] = float(p.get(f"{e} Decay", 0.0))
            adsr[b, i, 2] = float(p.get(f"{e} Sustain", 1.0))
            adsr[b, i, 3] = float(p.get(f"{e} Release", 0.0))
        semitones[b] = float(p.get("Transpose", 0.0))
        algo.append(int(round(p.get("Algorithm", 0.0))))

    return {
        "coarse": coarse.to(device),
        "level": level.to(device),
        "adsr": adsr.to(device),
        "semitones": semitones.to(device),
        "algo": algo,
    }
