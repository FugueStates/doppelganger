"""Shared audio/feature config — used by the dataset, the model, and ONNX export."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    clip_seconds: float = 2.0
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    fmin: float = 30.0
    fmax: float | None = None

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)
