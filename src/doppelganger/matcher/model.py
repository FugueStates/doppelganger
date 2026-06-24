"""
Matcher model (v3): target audio -> Operator params, DIRECT one-shot, classification heads.

This is the research-validated recipe (deep-research 2026-06-17): LOSS/formulation, not
encoder, is the lever. A modest CNN over a log-mel spectrogram (a transformer was shown
NOT to beat a CNN here) feeds per-parameter heads:
  • continuous -> K-bin classification (decoded by expected value)   [binned + soft labels]
  • binary     -> one logit (BCE)
  • categorical-> one softmax head per param (cross-entropy)
No differentiable renderer anywhere — the perceptual awareness comes from the loss
(per-param weighting, added when we scale past the sniff test), not a synth in the loop.
Pure feed-forward → exports to ONNX, runs on CPU in the extension.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..schema import OperatorSchema
from .codec import ParamCodec
from .features import LogMel


@dataclass(frozen=True)
class MatcherConfig:
    sample_rate: int = 16000
    n_samples: int = 48000          # 3 s
    n_fft: int = 1024
    hop: int = 256
    n_mels: int = 128
    n_bins: int = 64                # K bins per continuous param (64 → finer envelope
                                    # precision + less binned-decode edge bias than 32)
    ch: int = 32                    # base CNN width
    emb: int = 512
    # Pool to a freq x TIME grid, NOT a single vector: collapsing time (global avg pool)
    # discards exactly the temporal information the ADSR envelope lives in, so the model
    # could learn the (spectral) waveform but not the (temporal) envelope. Keeping a coarse
    # time axis lets the param heads read attack/decay/sustain/release.
    # (A 4x-finer time grid was tried for fast filter sweeps — no effect, so reverted.)
    pool_f: int = 4
    pool_t: int = 16


class ConvEncoder(nn.Module):
    """log-mel [B,1,M,T] -> [B, emb]. Strided 2-D CNN, then pool to a freq x TIME grid
    (keeping the time axis so the envelope is readable) and project."""

    def __init__(self, ch: int, emb: int, pool_f: int, pool_t: int):
        super().__init__()
        c = [1, ch, ch * 2, ch * 4, ch * 8]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                       nn.GroupNorm(8, c[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((pool_f, pool_t))   # collapse freq coarsely, KEEP time
        self.proj = nn.Sequential(nn.Linear(c[-1] * pool_f * pool_t, emb), nn.GELU())

    def forward(self, x):
        return self.proj(self.pool(self.net(x)).flatten(1))


class ParamMatcher(nn.Module):
    def __init__(self, schema: OperatorSchema, cfg: MatcherConfig | None = None):
        super().__init__()
        self.cfg = cfg or MatcherConfig()
        c = self.cfg
        self.codec = ParamCodec(schema, c.n_bins)
        self.logmel = LogMel(c.sample_rate, c.n_fft, c.hop, c.n_mels)
        self.encoder = ConvEncoder(c.ch, c.emb, c.pool_f, c.pool_t)

        self.n_cont = self.codec.n_cont
        self.n_binary = self.codec.n_binary
        self.cat_cards = self.codec.cat_cardinalities
        # heads off the shared embedding
        self.cont_head = nn.Linear(c.emb, self.n_cont * c.n_bins)
        self.binary_head = nn.Linear(c.emb, self.n_binary)
        self.cat_head = nn.Linear(c.emb, sum(self.cat_cards))
        # offsets to slice the concatenated categorical logits per param
        offs, a = [], 0
        for card in self.cat_cards:
            offs.append((a, a + card)); a += card
        self.cat_offsets = offs

    def forward(self, audio: torch.Tensor) -> dict:
        """audio [B,T] -> {cont_logits [B,n_cont,K], binary_logits [B,n_binary],
        cat_logits [B, sum_card]}."""
        z = self.encoder(self.logmel(audio))
        return {
            "cont_logits": self.cont_head(z).view(-1, self.n_cont, self.cfg.n_bins),
            "binary_logits": self.binary_head(z),
            "cat_logits": self.cat_head(z),
        }

    def cat_slices(self, cat_logits: torch.Tensor) -> list[torch.Tensor]:
        return [cat_logits[:, a:b] for a, b in self.cat_offsets]

    # -- decode predictions -> normalized values (for metrics + inference) ------
    @torch.no_grad()
    def decode(self, out: dict):
        cont = self.codec.expected_value(out["cont_logits"].softmax(-1))   # [B,n_cont] in [0,1]
        binary = out["binary_logits"].sigmoid()                            # [B,n_binary]
        cat = torch.stack([s.argmax(-1) for s in self.cat_slices(out["cat_logits"])], dim=1)
        return cont, binary, cat
