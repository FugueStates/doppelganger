"""
Parameter codec — converts between Operator's 195 raw parameters and the model's
grouped, normalized targets/outputs, and computes GATING MASKS.

The model has three kinds of output head (see schema.ParamKind):
  • continuous (142): one sigmoid/binned value each, target in [0, 1]   -> binned CE
  • binary      (34): one logit each, target 0/1                        -> BCE
  • categorical (19): one softmax head each (its own cardinality)       -> cross-entropy

GATING: many params are inaudible in a given preset — most importantly, an
oscillator's ~30 params (level, tuning, waveform, its amp envelope, etc.) when that
oscillator is OFF. Supervising them just injects noise into the loss. So `encode`
also returns a 0/1 mask per param group; the loss and metrics ignore masked params.
A param is masked when its controlling oscillator's "Osc-X On" toggle is Off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import OperatorSchema, Param, ParamKind


def _oscillator_of(name: str) -> str | None:
    """Which oscillator (A/B/C/D) a parameter belongs to, if any."""
    for o in "ABCD":
        if name.startswith(f"Osc-{o}") or name.startswith(f"{o} ") or name.startswith(f"{o}e "):
            return o
    return None


def _gate_of(name: str) -> str | None:
    """The 'Osc-X On' toggle that gates this param's audibility (None if ungated).

    The toggle itself is never gated (we always supervise whether an osc is on)."""
    o = _oscillator_of(name)
    if o is None:
        return None
    gate = f"Osc-{o} On"
    return None if name == gate else gate


@dataclass
class EncodedTargets:
    cont: np.ndarray  # float32 [n_cont], in [0,1]
    binary: np.ndarray  # float32 [n_binary], 0/1
    cat: np.ndarray  # int64 [n_cat], class indices
    cont_mask: np.ndarray  # float32 [n_cont], 1=active 0=ignore
    binary_mask: np.ndarray  # float32 [n_binary]
    cat_mask: np.ndarray  # float32 [n_cat]


class ParamCodec:
    def __init__(self, schema: OperatorSchema):
        self.schema = schema
        self.cont: list[Param] = schema.continuous
        self.binary: list[Param] = schema.binary
        self.categorical: list[Param] = schema.categorical
        # the gating toggle name for each param in each group (or None)
        self.cont_gates = [_gate_of(p.name) for p in self.cont]
        self.binary_gates = [_gate_of(p.name) for p in self.binary]
        self.cat_gates = [_gate_of(p.name) for p in self.categorical]

    # -- sizes the model needs --------------------------------------------
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
    def _mask(gates: list[str | None], params: dict[str, float]) -> np.ndarray:
        # active unless the gating toggle exists and is Off (value < 0.5)
        return np.array(
            [0.0 if (g is not None and params.get(g, 1.0) < 0.5) else 1.0 for g in gates],
            dtype=np.float32,
        )

    # -- encode: raw params dict -> training targets + masks --------------
    def encode(self, params: dict[str, float]) -> EncodedTargets:
        cont = np.array([p.normalize(params[p.name]) for p in self.cont], dtype=np.float32)
        binary = np.array([p.normalize(params[p.name]) for p in self.binary], dtype=np.float32)
        cat = np.array([int(p.normalize(params[p.name])) for p in self.categorical], dtype=np.int64)
        return EncodedTargets(
            cont=cont,
            binary=binary,
            cat=cat,
            cont_mask=self._mask(self.cont_gates, params),
            binary_mask=self._mask(self.binary_gates, params),
            cat_mask=self._mask(self.cat_gates, params),
        )

    # -- decode: model outputs -> raw params dict (for inference) ----------
    def decode(
        self,
        cont: np.ndarray,  # [n_cont] in [0,1]
        binary: np.ndarray,  # [n_binary] probabilities or 0/1
        cat_indices: np.ndarray,  # [n_cat] argmax class indices
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for p, v in zip(self.cont, cont):
            out[p.name] = p.denormalize(float(v))
        for p, v in zip(self.binary, binary):
            out[p.name] = p.denormalize(1.0 if v >= 0.5 else 0.0)
        for p, idx in zip(self.categorical, cat_indices):
            out[p.name] = p.denormalize(int(idx))
        return out
