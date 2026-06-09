"""
HybridRenderer — the differentiable Operator surrogate (gray-box).

params -> [physics FM core -> log-STFT] + [FiLM-conditioned neural residual] -> predicted
log-magnitude spectrogram(s). The physics core gives the harmonic skeleton; the residual
learns Operator's undocumented remainder. Frozen after training, it provides the audio
loss for the one-shot matcher. See docs/renderer-design.md.

v2: MULTI-RESOLUTION linear-STFT log-magnitude output (n_fft in {512,1024,2048} — captures
sharp FM peaks AND transients) + Snake-activated residual CNN + frequency positional
encoding + FiLM from a normalized-param MLP. One residual backbone is SHARED across
resolutions (convs are size-agnostic; FiLM is per-channel), so it learns a resolution-
invariant correction and stays parameter-efficient. `forward` returns {n_fft: log-mag}.
(Transformer/preset-tokenizer encoder, adversarial polish, learnable envelope splines,
anti-aliased physics core = the remaining documented follow-ons.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..schema import OperatorSchema
from .adapter import to_controls
from .diff_operator import DiffOperator


@dataclass(frozen=True)
class RendererConfig:
    sample_rate: int = 16000
    n_samples: int = 48000              # 3 s
    # Multi-resolution linear STFT (preserves FM sidebands; mel would blur them). Small FFT
    # = sharp time/transients, large FFT = sharp frequency/sideband separation.
    n_ffts: tuple = (512, 1024, 2048)
    canon_fft: int = 1024               # resolution used for headline metrics / mean baseline
    note_off: float = 1.5
    f0: float = 261.63                  # Ableton MIDI 60
    oversample: int = 2                 # physics-core anti-aliasing (render at os x, LPF, decimate)

    @property
    def hops(self) -> tuple:
        return tuple(n // 4 for n in self.n_ffts)


def log_mag(wave: torch.Tensor, n_fft: int, hop: int, eps: float = 1e-5) -> torch.Tensor:
    """[B, T] waveform -> [B, F, frames] log-magnitude linear STFT."""
    win = torch.hann_window(n_fft, device=wave.device)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
    return torch.log(spec.abs() + eps)


class Snake(nn.Module):
    """Periodic activation x + (1/a) sin^2(a x) — periodic inductive bias for FM/sinusoids."""

    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x + (1.0 / (self.a + 1e-6)) * torch.sin(self.a * x) ** 2


class FiLMBlock(nn.Module):
    def __init__(self, cin: int, cout: int, code_dim: int):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(cin, cout, 3, padding=1))
        self.film = nn.Linear(code_dim, 2 * cout)
        self.act = Snake(cout)

    def forward(self, x, code):
        h = self.conv(x)
        gamma, beta = self.film(code).chunk(2, dim=-1)
        h = gamma[:, :, None, None] * h + beta[:, :, None, None]
        return self.act(h)


def _param_vector(param_dicts: list[dict], schema: OperatorSchema) -> torch.Tensor:
    """Normalize all params to [0,1] for the conditioning encoder."""
    out = torch.zeros(len(param_dicts), len(schema.params))
    for b, p in enumerate(param_dicts):
        for i, pa in enumerate(schema.params):
            v = float(p.get(pa.name, pa.default))
            if pa.is_quantized:
                out[b, i] = v / max((pa.cardinality or 1) - 1, 1)
            else:
                span = pa.max - pa.min
                out[b, i] = (v - pa.min) / span if span else 0.0
    return out.clamp(0, 1)


class HybridRenderer(nn.Module):
    def __init__(self, schema: OperatorSchema, cfg: RendererConfig | None = None,
                 code_dim: int = 256, ch: int = 48, dropout: float = 0.2):
        super().__init__()
        self.schema = schema
        self.cfg = cfg or RendererConfig()
        self.physics = DiffOperator(self.cfg.sample_rate, self.cfg.n_samples,
                                    self.cfg.note_off, oversample=self.cfg.oversample)
        n_params = len(schema.params)
        # Dropout on the conditioning code is the key anti-memorization regularizer: it stops
        # the residual from using the (near-unique) param vector as a per-sample lookup key.
        self.encoder = nn.Sequential(
            nn.Linear(n_params, code_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(code_dim, code_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        # Frequency positional encoding: a FiLM-CNN is translation-equivariant in frequency
        # and can't otherwise place a peak at a preset-specific absolute bin. These extra
        # channels (linear ramp + sinusoids) tell every conv WHERE in frequency it is.
        self.n_freqenc = 5  # 1 linear + 4 sinusoidal
        self.in_block = FiLMBlock(1 + self.n_freqenc, ch, code_dim)
        self.mid = nn.ModuleList([FiLMBlock(ch, ch, code_dim) for _ in range(3)])
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def _freq_encoding(self, n_freq: int, n_time: int, batch: int, device) -> torch.Tensor:
        """[B, n_freqenc, F, T] absolute-frequency coordinate channels."""
        f = torch.linspace(0.0, 1.0, n_freq, device=device)
        feats = [f] + [torch.sin((2 ** k) * torch.pi * f) for k in range(4)]
        enc = torch.stack(feats, dim=0)  # [n_freqenc, F]
        return enc[None, :, :, None].expand(batch, -1, -1, n_time)

    def _render_physics(self, controls: dict) -> torch.Tensor:
        """Render the FM core waveform, grouping the batch by algorithm."""
        algos = controls["algo"]
        B = controls["coarse"].shape[0]
        wave = torch.zeros(B, self.cfg.n_samples, device=controls["coarse"].device)
        for a in sorted(set(algos)):
            idx = [i for i, x in enumerate(algos) if x == a]
            sub = self.physics.render(
                controls["coarse"][idx], controls["level"][idx], controls["adsr"][idx],
                a, f0=self.cfg.f0, volume=controls["volume"][idx])
            wave[idx] = sub
        return wave

    def _residual(self, phys: torch.Tensor, code: torch.Tensor, device) -> torch.Tensor:
        """phys [B,1,F,T] log-mag -> corrected log-mag [B,F,T] (shared across resolutions)."""
        B, _, n_freq, n_time = phys.shape
        x = torch.cat([phys, self._freq_encoding(n_freq, n_time, B, device)], dim=1)
        h = self.in_block(x, code)
        for blk in self.mid:
            h = blk(h, code)
        return (phys + self.out_conv(h)).squeeze(1)

    def forward(self, param_dicts: list[dict], device: str = "cpu",
                apply_residual: bool = True) -> dict:
        """Returns {n_fft: predicted log-mag [B, F, T]} for each STFT resolution."""
        controls = to_controls(param_dicts, device)
        wave = self._render_physics(controls)
        code = None if not apply_residual else \
            self.encoder(_param_vector(param_dicts, self.schema).to(device))
        out = {}
        for n_fft, hop in zip(self.cfg.n_ffts, self.cfg.hops):
            phys = log_mag(wave, n_fft, hop).unsqueeze(1)
            out[n_fft] = phys.squeeze(1) if code is None else self._residual(phys, code, device)
        return out

    def target_logmag(self, wave: torch.Tensor) -> dict:
        """Real audio -> {n_fft: log-mag [B, F, T]} matching forward()'s resolutions."""
        return {n_fft: log_mag(wave, n_fft, hop)
                for n_fft, hop in zip(self.cfg.n_ffts, self.cfg.hops)}


if __name__ == "__main__":
    # smoke test: forward + backward on synthetic params (no dataset needed)
    from .sensitivity import rich_base
    from .spectral import renderer_loss
    sch = OperatorSchema.load("schemas/operator.json")
    base = rich_base(sch)
    variant = dict(base); variant["A Coarse"] = 4.0
    model = HybridRenderer(sch)
    preds = model([base, variant])
    target = torch.randn(2, model.cfg.n_samples)
    loss = renderer_loss(preds, model.target_logmag(target))
    loss.backward()
    g = sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    shapes = {k: tuple(v.shape) for k, v in preds.items()}
    print("pred specs", shapes, "loss", round(loss.item(), 3), "encoder grad>0:", bool(g > 0))
