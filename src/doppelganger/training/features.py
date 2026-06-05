"""
Log-mel spectrogram as a torch Module — runs INSIDE the model.

Putting feature extraction in the graph means (a) train and inference use the exact
same DSP, and (b) when we export to ONNX, Tool 3 just feeds raw audio — no need to
reimplement mel-spectrograms in JavaScript.

We build the mel filterbank ourselves (HTK formula) so it's a plain constant matmul
that exports cleanly, and use torch.stft for the spectrogram.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular mel filterbank, shape [n_mels, n_fft//2 + 1]."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2, n_freqs)

    mel_pts = np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[m] = np.clip(np.minimum(left, right), 0.0, None)
    return fb


class LogMel(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop: int = 256,
        n_mels: int = 128,
        fmin: float = 30.0,
        fmax: float | None = None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        fmax = fmax or sample_rate / 2
        fb = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
        self.register_buffer("mel_fb", torch.from_numpy(fb))  # [n_mels, n_freqs]
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio [B, T] -> log-mel [B, 1, n_mels, frames]."""
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )  # [B, n_freqs, frames]
        power = spec.abs() ** 2
        mel = torch.matmul(self.mel_fb, power)  # [B, n_mels, frames]
        log_mel = torch.log(mel + 1e-6)
        return log_mel.unsqueeze(1)  # add channel dim
