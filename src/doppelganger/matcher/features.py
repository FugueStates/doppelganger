"""
Log-mel spectrogram as a torch Module — runs INSIDE the v3 matcher.

In-graph feature extraction means (a) train and inference use identical DSP, and (b) the
ONNX export feeds raw audio — Tool 3 needs no JS mel reimplementation. The mel filterbank
is a plain constant matmul (HTK formula) that exports cleanly; torch.stft does the STFT.
(Adapted from the deprecated training/features.py — the formulation the research validates.)
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
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024, hop: int = 256,
                 n_mels: int = 128, fmin: float = 30.0, fmax: float | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        fmax = fmax or sample_rate / 2
        fb = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
        self.register_buffer("mel_fb", torch.from_numpy(fb))   # [n_mels, n_freqs]
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio [B, T] -> log-mel [B, 1, n_mels, frames] (per-sample standardized)."""
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop, window=self.window,
                          center=True, return_complex=True)        # [B, n_freqs, frames]
        mel = torch.matmul(self.mel_fb, spec.abs() ** 2)            # [B, n_mels, frames]
        log_mel = torch.log(mel + 1e-6)
        # per-sample standardize (stabilizes the CNN across the wide log-mag range)
        m = log_mel.mean(dim=(-2, -1), keepdim=True)
        s = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return ((log_mel - m) / s).unsqueeze(1)
