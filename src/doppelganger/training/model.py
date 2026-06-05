"""
SoundMatcher: raw audio -> log-mel -> ResNet -> 195 Operator parameter heads.

Loss design follows the synth-matching literature (InverSynth, Sound2Synth):
  • continuous  : each knob is DISCRETIZED into N bins and predicted by classification
                  with GAUSSIAN SOFT LABELS (adjacent bins share probability mass).
                  This beats plain MSE regression on this many-to-one mapping — MSE
                  collapses multimodal targets to an (audibly wrong) average, while a
                  softmax can place mass on multiple plausible values and pick a mode.
  • binary      : logits + BCE.
  • categorical : softmax + cross-entropy with light label smoothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import AudioConfig
from .features import LogMel


class BasicBlock(nn.Module):
    """A standard ResNet basic block (two 3x3 convs + identity/projection skip)."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short: nn.Module = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


def _make_layer(cin: int, cout: int, blocks: int, stride: int) -> nn.Sequential:
    layers = [BasicBlock(cin, cout, stride)]
    layers += [BasicBlock(cout, cout, 1) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


class SoundMatcher(nn.Module):
    def __init__(
        self,
        audio: AudioConfig,
        n_cont: int,
        n_binary: int,
        cat_cardinalities: list[int],
        n_bins: int = 64,
        widths: tuple[int, int, int, int] = (64, 128, 256, 512),
        blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
        embedding: int = 768,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.logmel = LogMel(
            sample_rate=audio.sample_rate,
            n_fft=audio.n_fft,
            hop=audio.hop,
            n_mels=audio.n_mels,
            fmin=audio.fmin,
            fmax=audio.fmax,
        )
        w0, w1, w2, w3 = widths
        b0, b1, b2, b3 = blocks
        self.stem = nn.Sequential(
            nn.Conv2d(1, w0, 3, 1, 1, bias=False), nn.BatchNorm2d(w0), nn.ReLU(inplace=True)
        )
        self.backbone = nn.Sequential(
            _make_layer(w0, w0, b0, stride=1),
            _make_layer(w0, w1, b1, stride=2),
            _make_layer(w1, w2, b2, stride=2),
            _make_layer(w2, w3, b3, stride=2),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.embed = nn.Sequential(
            nn.Linear(w3 * 4, embedding), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        # continuous params are predicted as n_bins-way classification each
        self.cont_head = nn.Linear(embedding, n_cont * n_bins)
        self.n_cont = n_cont
        self.binary_head = nn.Linear(embedding, n_binary)
        self.cat_heads = nn.ModuleList(nn.Linear(embedding, c) for c in cat_cardinalities)

    def forward(self, audio: torch.Tensor) -> dict:
        x = self.logmel(audio)
        # per-sample spectrogram standardization (stable input distribution)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-5
        feats = self.embed(self.backbone(self.stem((x - mean) / std)))
        return {
            "cont_logits": self.cont_head(feats).view(-1, self.n_cont, self.n_bins),
            "binary_logits": self.binary_head(feats),
            "cat_logits": [head(feats) for head in self.cat_heads],
        }

    def cont_values(self, out: dict) -> torch.Tensor:
        """Decode continuous predictions to [0,1] values (argmax bin -> bin center)."""
        return out["cont_logits"].argmax(-1).float() / (self.n_bins - 1)


def _gaussian_soft_labels(values01: torch.Tensor, n_bins: int, sigma: float) -> torch.Tensor:
    """values01 [B, n_cont] in [0,1] -> soft targets [B, n_cont, n_bins] (Gaussian on bins)."""
    bins = torch.arange(n_bins, device=values01.device).float()
    centers = (values01 * (n_bins - 1)).unsqueeze(-1)  # [B, n_cont, 1]
    soft = torch.exp(-0.5 * ((bins - centers) / sigma) ** 2)
    return soft / soft.sum(-1, keepdim=True)


def compute_loss(
    outputs: dict,
    batch: dict,
    n_bins: int,
    sigma: float = 1.5,
    cat_smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    # continuous: cross-entropy against Gaussian soft labels
    soft = _gaussian_soft_labels(batch["cont"], n_bins, sigma)
    logp = F.log_softmax(outputs["cont_logits"], dim=-1)
    cont = -(soft * logp).sum(-1).mean()

    bce = F.binary_cross_entropy_with_logits(outputs["binary_logits"], batch["binary"])

    cat_logits = outputs["cat_logits"]
    cat_t = batch["cat"]
    ce = (
        torch.stack(
            [
                F.cross_entropy(cat_logits[i], cat_t[:, i], label_smoothing=cat_smoothing)
                for i in range(len(cat_logits))
            ]
        ).mean()
        if cat_logits
        else torch.zeros((), device=cont.device)
    )

    total = cont + bce + ce
    return total, {"cont": cont.item(), "bce": bce.item(), "ce": ce.item()}
