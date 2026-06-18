"""
Parameter codec (v3) — converts Operator's 195 raw params <-> the model's grouped,
normalized targets, computes gating masks, and handles the special encodings.

Output head kinds:
  • continuous : K-bin classification, target = Gaussian SOFT label around the true bin
                 (InverSynth/Sound2Synth trick — fixes regression-to-mean)
  • binary     : one logit, BCE, target 0/1
  • categorical: one softmax head each (own cardinality), cross-entropy
  • RATIO      : the per-oscillator Coarse params, treated as CATEGORICAL over integer
                 ratios (ratio = floor(Coarse); freq-map finding). This is the correct
                 representation for FM's primary timbre control and avoids the
                 binned-continuous edge-decode bias that put the carrier an octave off.
  • FROZEN     : out-of-scope params pinned to a fixed value — NOT predicted (no head, no
                 loss), set at decode. Used for params whose binned-continuous decode would
                 be audibly wrong at a range edge yet aren't gated inaudible (Fine detune,
                 oscillator feedback). Keeps the predicted preset clean.

GATING: an oscillator's params are inaudible when it's OFF; `encode` returns a 0/1 mask per
param so the loss/metrics ignore masked params (also dampens the many-to-one problem).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..schema import OperatorSchema, Param

# Coarse params -> categorical integer ratio. ratio = floor(raw Coarse), raw in [0,48].
RATIO_PARAMS = ("A Coarse", "B Coarse", "C Coarse", "D Coarse")
N_RATIO = 49  # classes 0..48

# Out-of-scope params pinned to a fixed value (not predicted), because their
# binned-continuous decode is biased (off a range edge or just imprecise) and the error is
# AUDIBLE — they aren't gated inaudible. Especially the pitch params: the carrier must
# track the played note exactly, so any drift here detunes the whole sound.
#   • Fine detune + oscillator feedback (edge-pinned at 0)
#   • Transpose + per-osc Freq<Vel — global/velocity pitch offsets, pinned to 0 so pitch is
#     exactly the played note (a stray -0.05 semitone Transpose was audibly flattening it)
FROZEN_DEFAULT: dict[str, float] = {
    **{f"{X} Fine": 0.0 for X in "ABCD"},
    **{f"Osc-{X} Feedb": 0.0 for X in "ABCD"},
    "Transpose": 0.0,
    **{f"{X} Freq<Vel": 0.0 for X in "ABCD"},
}


def _oscillator_of(name: str) -> str | None:
    for o in "ABCD":
        if name.startswith(f"Osc-{o}") or name.startswith(f"{o} ") or name.startswith(f"{o}e "):
            return o
    return None


def _gate_of(name: str) -> str | None:
    o = _oscillator_of(name)
    if o is None:
        return None
    gate = f"Osc-{o} On"
    return None if name == gate else gate


@dataclass
class EncodedTargets:
    cont: np.ndarray
    binary: np.ndarray
    cat: np.ndarray        # int64 [n_cat] class indices (incl. ratio params at the end)
    cont_mask: np.ndarray
    binary_mask: np.ndarray
    cat_mask: np.ndarray


class ParamCodec:
    def __init__(self, schema: OperatorSchema, n_bins: int = 32,
                 frozen: dict[str, float] | None = None,
                 ratio_params: tuple[str, ...] = RATIO_PARAMS, n_ratio: int = N_RATIO):
        self.schema = schema
        self.n_bins = n_bins
        self.n_ratio = n_ratio
        self.frozen = dict(FROZEN_DEFAULT if frozen is None else frozen)
        by = {p.name: p for p in schema.params}
        froz = set(self.frozen)
        ratio_set = {n for n in ratio_params if n in by}
        self.ratio_set = ratio_set

        self.cont = [p for p in schema.continuous if p.name not in ratio_set and p.name not in froz]
        self.binary = [p for p in schema.binary if p.name not in froz]
        base_cat = [p for p in schema.categorical if p.name not in froz]
        ratio_ps = [by[n] for n in ratio_params if n in by and n not in froz]
        self.categorical = base_cat + ratio_ps               # ratio params appended
        self.cat_is_ratio = [False] * len(base_cat) + [True] * len(ratio_ps)

        self.cont_gates = [_gate_of(p.name) for p in self.cont]
        self.binary_gates = [_gate_of(p.name) for p in self.binary]
        self.cat_gates = [_gate_of(p.name) for p in self.categorical]

    @property
    def n_cont(self) -> int:
        return len(self.cont)

    @property
    def n_binary(self) -> int:
        return len(self.binary)

    @property
    def cat_cardinalities(self) -> list[int]:
        return [self.n_ratio if r else (p.cardinality or 1)
                for p, r in zip(self.categorical, self.cat_is_ratio)]

    @staticmethod
    def _mask(gates, params) -> np.ndarray:
        return np.array([0.0 if (g is not None and params.get(g, 1.0) < 0.5) else 1.0
                         for g in gates], dtype=np.float32)

    def _cat_class(self, p: Param, is_ratio: bool, params: dict) -> int:
        v = params.get(p.name, p.default)
        if is_ratio:
            return int(min(self.n_ratio - 1, max(0, math.floor(v))))   # ratio = floor(Coarse)
        return int(p.normalize(v))                                     # categorical index

    def encode(self, params: dict) -> EncodedTargets:
        cont = np.array([p.normalize(params.get(p.name, p.default)) for p in self.cont], dtype=np.float32)
        binary = np.array([p.normalize(params.get(p.name, p.default)) for p in self.binary], dtype=np.float32)
        cat = np.array([self._cat_class(p, r, params)
                        for p, r in zip(self.categorical, self.cat_is_ratio)], dtype=np.int64)
        return EncodedTargets(cont, binary, cat,
                              self._mask(self.cont_gates, params),
                              self._mask(self.binary_gates, params),
                              self._mask(self.cat_gates, params))

    def decode(self, cont, binary, cat_indices) -> dict:
        out: dict[str, float] = {}
        for p, v in zip(self.cont, cont):
            out[p.name] = p.denormalize(float(v))
        for p, v in zip(self.binary, binary):
            out[p.name] = p.denormalize(1.0 if v >= 0.5 else 0.0)
        for p, is_ratio, idx in zip(self.categorical, self.cat_is_ratio, cat_indices):
            out[p.name] = float(int(idx)) if is_ratio else p.denormalize(int(idx))
        for name, val in self.frozen.items():
            out[name] = float(val)                                     # pinned, not predicted
        return out

    # -- binned-classification helpers (continuous params) --------------------
    def soft_bins(self, values01, sigma: float = 1.5):
        import torch
        K = self.n_bins
        centers = (torch.arange(K, device=values01.device) + 0.5) / K
        d = (values01.unsqueeze(-1) - centers) * K
        return torch.softmax(-0.5 * (d / sigma) ** 2, dim=-1)

    def expected_value(self, bin_probs):
        import torch
        K = self.n_bins
        centers = (torch.arange(K, device=bin_probs.device) + 0.5) / K
        return (bin_probs * centers).sum(-1)
