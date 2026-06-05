"""
Parameter codec — converts between Operator's 195 raw parameters and the model's
grouped, normalized targets/outputs.

The model has three kinds of output head (see schema.ParamKind):
  • continuous (142): one sigmoid value each, target in [0, 1]      -> MSE loss
  • binary      (34): one logit each, target 0/1                    -> BCE loss
  • categorical (19): one softmax head each (its own cardinality)   -> cross-entropy

This module fixes the ORDER of params within each group (so the model's output
columns are stable) and handles encode (params -> targets) and decode
(model outputs -> raw params to send back to Operator at inference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import OperatorSchema, Param, ParamKind


@dataclass
class EncodedTargets:
    cont: np.ndarray  # float32 [n_cont], in [0,1]
    binary: np.ndarray  # float32 [n_binary], 0/1
    cat: np.ndarray  # int64 [n_cat], class indices


class ParamCodec:
    def __init__(self, schema: OperatorSchema):
        self.schema = schema
        self.cont: list[Param] = schema.continuous
        self.binary: list[Param] = schema.binary
        self.categorical: list[Param] = schema.categorical

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

    # -- encode: raw params dict -> training targets ----------------------
    def encode(self, params: dict[str, float]) -> EncodedTargets:
        cont = np.array(
            [p.normalize(params[p.name]) for p in self.cont], dtype=np.float32
        )
        binary = np.array(
            [p.normalize(params[p.name]) for p in self.binary], dtype=np.float32
        )
        cat = np.array(
            [int(p.normalize(params[p.name])) for p in self.categorical], dtype=np.int64
        )
        return EncodedTargets(cont=cont, binary=binary, cat=cat)

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
