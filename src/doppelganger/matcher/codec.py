"""
Parameter codec (v3) — converts Operator's 195 raw params <-> the model's grouped,
normalized targets, computes gating masks, and handles the binned-classification encoding
for continuous params.

Output head kinds (schema.ParamKind):
  • continuous (142): K-bin classification, target = Gaussian SOFT label around the true
                      bin (the InverSynth/Sound2Synth trick that fixes regression-to-mean)
  • binary      (34): one logit, BCE, target 0/1
  • categorical (19): one softmax head each (own cardinality), cross-entropy

GATING: an oscillator's ~30 params (level, tuning, waveform, its amp envelope) are
inaudible when that oscillator is OFF; supervising them injects noise (and is a source of
the many-to-one problem). `encode` returns a 0/1 mask per param; the loss/metrics ignore
masked params. A param is masked when its controlling "Osc-X On" toggle is Off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import OperatorSchema, Param


def _oscillator_of(name: str) -> str | None:
    for o in "ABCD":
        if name.startswith(f"Osc-{o}") or name.startswith(f"{o} ") or name.startswith(f"{o}e "):
            return o
    return None


def _gate_of(name: str) -> str | None:
    """The 'Osc-X On' toggle gating this param's audibility (None if ungated). The toggle
    itself is never gated."""
    o = _oscillator_of(name)
    if o is None:
        return None
    gate = f"Osc-{o} On"
    return None if name == gate else gate


@dataclass
class EncodedTargets:
    cont: np.ndarray       # float32 [n_cont] in [0,1]
    binary: np.ndarray     # float32 [n_binary] 0/1
    cat: np.ndarray        # int64   [n_cat] class indices
    cont_mask: np.ndarray  # float32 [n_cont] 1=active 0=ignore
    binary_mask: np.ndarray
    cat_mask: np.ndarray


class ParamCodec:
    def __init__(self, schema: OperatorSchema, n_bins: int = 32):
        self.schema = schema
        self.n_bins = n_bins
        self.cont: list[Param] = schema.continuous
        self.binary: list[Param] = schema.binary
        self.categorical: list[Param] = schema.categorical
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
        return [p.cardinality or 1 for p in self.categorical]

    @staticmethod
    def _mask(gates, params) -> np.ndarray:
        return np.array([0.0 if (g is not None and params.get(g, 1.0) < 0.5) else 1.0
                         for g in gates], dtype=np.float32)

    def encode(self, params: dict) -> EncodedTargets:
        cont = np.array([p.normalize(params.get(p.name, p.default)) for p in self.cont], dtype=np.float32)
        binary = np.array([p.normalize(params.get(p.name, p.default)) for p in self.binary], dtype=np.float32)
        cat = np.array([int(p.normalize(params.get(p.name, p.default))) for p in self.categorical], dtype=np.int64)
        return EncodedTargets(cont, binary, cat,
                              self._mask(self.cont_gates, params),
                              self._mask(self.binary_gates, params),
                              self._mask(self.cat_gates, params))

    def decode(self, cont, binary, cat_indices) -> dict:
        """cont [n_cont] in [0,1], binary [n_binary] prob/0-1, cat_indices [n_cat] -> raw dict."""
        out: dict[str, float] = {}
        for p, v in zip(self.cont, cont):
            out[p.name] = p.denormalize(float(v))
        for p, v in zip(self.binary, binary):
            out[p.name] = p.denormalize(1.0 if v >= 0.5 else 0.0)
        for p, idx in zip(self.categorical, cat_indices):
            out[p.name] = p.denormalize(int(idx))
        return out

    # -- binned-classification helpers (continuous params) --------------------
    def soft_bins(self, values01, sigma: float = 1.5):
        """values01 [..., n_cont] in [0,1] -> Gaussian soft labels [..., n_cont, K].
        A peak at the true bin smeared by sigma bins — soft cross-entropy against these
        beats hard regression (no collapse-to-mean) and is robust to bin quantization."""
        import torch
        K = self.n_bins
        centers = (torch.arange(K, device=values01.device) + 0.5) / K   # [K] bin centers in [0,1]
        d = (values01.unsqueeze(-1) - centers) * K                       # distance in bins
        logits = -0.5 * (d / sigma) ** 2
        return torch.softmax(logits, dim=-1)

    def expected_value(self, bin_probs):
        """bin_probs [..., K] -> expected value in [0,1] (decode a binned prediction)."""
        import torch
        K = self.n_bins
        centers = (torch.arange(K, device=bin_probs.device) + 0.5) / K
        return (bin_probs * centers).sum(-1)
