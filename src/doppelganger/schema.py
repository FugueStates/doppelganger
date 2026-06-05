"""
Operator parameter schema — the shared contract between data collection, training,
and inference.

The schema JSON is produced by the data-collection extension (it reads every
``DeviceParameter`` of a fresh Operator). Each parameter is one of three kinds,
which map directly onto the model's output heads:

* ``continuous``  — a real-valued knob (e.g. ``Filter Freq``). Normalized to [0, 1]
                    using its own ``min``/``max``.
* ``binary``      — a quantized parameter with exactly 2 options (an on/off toggle).
                    Treated as a single 0/1 class.
* ``categorical`` — a quantized parameter with >2 options (e.g. ``Algorithm`` (11),
                    ``Osc-A Wave`` (23)). Treated as a softmax class index.

Why per-parameter normalization is mandatory: Operator's continuous ranges vary
wildly (0..1, -100..100, -48..48 semitones, 0..1000 fine, ...), so a single global
scale would be meaningless. The schema carries each range so train and inference
normalize identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ParamKind(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class Param:
    index: int
    name: str
    min: float
    max: float
    default: float
    value: float
    is_quantized: bool
    value_items: list[str] | None  # option names, for quantized params

    @property
    def kind(self) -> ParamKind:
        if not self.is_quantized:
            return ParamKind.CONTINUOUS
        return ParamKind.BINARY if self.cardinality == 2 else ParamKind.CATEGORICAL

    @property
    def cardinality(self) -> int | None:
        """Number of discrete options (None for continuous params)."""
        return len(self.value_items) if (self.is_quantized and self.value_items) else None

    def normalize(self, raw: float) -> float:
        """Raw Live value -> training target.

        Continuous -> float in [0, 1]. Quantized -> integer class index (as float).
        """
        if self.kind is ParamKind.CONTINUOUS:
            span = self.max - self.min
            if span == 0:
                return 0.0
            return (raw - self.min) / span
        # Quantized parameters: the Live value *is* the option index.
        return float(round(raw))

    def denormalize(self, norm: float) -> float:
        """Training/prediction value -> raw Live value to feed back into setValue.

        Continuous: [0, 1] -> [min, max] (clamped). Quantized: class index -> value.
        """
        if self.kind is ParamKind.CONTINUOUS:
            clamped = min(1.0, max(0.0, norm))
            return self.min + clamped * (self.max - self.min)
        card = self.cardinality or 1
        return float(min(card - 1, max(0, round(norm))))


@dataclass(frozen=True)
class OperatorSchema:
    device: str
    params: list[Param]

    @classmethod
    def load(cls, path: str | Path) -> "OperatorSchema":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        params = [
            Param(
                index=p["index"],
                name=p["name"],
                min=p["min"],
                max=p["max"],
                default=p["defaultValue"],
                value=p["value"],
                is_quantized=p["isQuantized"],
                value_items=(
                    [v["name"] for v in p["valueItems"]] if p.get("valueItems") else None
                ),
            )
            for p in data["parameters"]
        ]
        return cls(device=data["device"], params=params)

    # -- convenience views ---------------------------------------------------
    def by_kind(self, kind: ParamKind) -> list[Param]:
        return [p for p in self.params if p.kind is kind]

    @property
    def continuous(self) -> list[Param]:
        return self.by_kind(ParamKind.CONTINUOUS)

    @property
    def binary(self) -> list[Param]:
        return self.by_kind(ParamKind.BINARY)

    @property
    def categorical(self) -> list[Param]:
        return self.by_kind(ParamKind.CATEGORICAL)

    def summary(self) -> str:
        return (
            f"{self.device}: {len(self.params)} params "
            f"({len(self.continuous)} continuous, {len(self.binary)} binary, "
            f"{len(self.categorical)} categorical)"
        )
