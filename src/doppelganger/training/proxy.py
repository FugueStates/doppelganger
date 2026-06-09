"""
Neural proxy — a *differentiable* surrogate for "what Operator sounds like".

Operator can't be backpropagated through, so we can't train the matcher with an
audio loss directly. The proxy bridges that: a small MLP `params -> audio embedding`
trained on our offline (params, rendered-audio) pairs to imitate the synth. Once
trained and frozen, it lets the matcher optimize for *sound* — the matcher's
predicted params go through the proxy to a predicted embedding, pulled toward the
target audio's embedding (see train.py --proxy).  (Combes et al. 2025.)

Embedding target: time-pooled log-mel statistics (mean + std over frames) = a
timbre fingerprint. Self-contained (no pretrained audio model); can be upgraded to
a CLAP/PaSST/EfficientAT embedding later.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import AudioConfig
from .features import LogMel


class MelEmbedding(nn.Module):
    """Raw audio -> [B, 2*n_mels] pooled log-mel (mean & std over time)."""

    def __init__(self, audio: AudioConfig):
        super().__init__()
        self.logmel = LogMel(audio.sample_rate, audio.n_fft, audio.hop, audio.n_mels, audio.fmin, audio.fmax)
        self.dim = 2 * audio.n_mels

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.logmel(audio)[:, 0]  # [B, n_mels, T]
        return torch.cat([x.mean(-1), x.std(-1)], dim=-1)  # [B, 2*n_mels]


def gt_param_vector(
    cont: torch.Tensor, binary: torch.Tensor, cat: torch.Tensor, cat_cardinalities: list[int]
) -> torch.Tensor:
    """Ground-truth params -> proxy input vector (cont + binary + one-hot categoricals)."""
    one_hots = [F.one_hot(cat[:, i], c).float() for i, c in enumerate(cat_cardinalities)]
    return torch.cat([cont, binary, *one_hots], dim=-1)


def pred_param_vector(
    cont_values: torch.Tensor, binary_probs: torch.Tensor, cat_softmaxes: list[torch.Tensor]
) -> torch.Tensor:
    """Matcher predictions -> proxy input vector (differentiable: soft values/probs)."""
    return torch.cat([cont_values, binary_probs, *cat_softmaxes], dim=-1)


class SynthProxy(nn.Module):
    def __init__(
        self,
        n_cont: int,
        n_binary: int,
        cat_cardinalities: list[int],
        embed_dim: int,
        hidden: int = 512,
    ):
        super().__init__()
        in_dim = n_cont + n_binary + sum(cat_cardinalities)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, param_vec: torch.Tensor) -> torch.Tensor:
        return self.net(param_vec)
