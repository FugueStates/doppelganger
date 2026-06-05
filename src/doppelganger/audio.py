"""Audio I/O helpers (thin wrapper over soundfile)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Clip:
    samples: np.ndarray  # shape (n_frames, n_channels), float32 in [-1, 1]
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.samples.shape[0] / self.sample_rate

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    def mono(self) -> np.ndarray:
        """Average channels down to a 1-D mono signal."""
        return self.samples.mean(axis=1)

    @property
    def peak(self) -> float:
        return float(np.abs(self.samples).max()) if self.samples.size else 0.0

    @property
    def rms(self) -> float:
        """Root-mean-square level — a better "is this actually audible" measure than peak.

        (A click has a high peak but is perceptually quiet; RMS reflects loudness.)
        """
        return float(np.sqrt(np.mean(self.samples**2))) if self.samples.size else 0.0


def load_wav(path: str | Path) -> Clip:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return Clip(samples=data, sample_rate=sr)
