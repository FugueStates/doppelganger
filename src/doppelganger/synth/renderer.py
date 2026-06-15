"""
HybridRenderer — the differentiable Operator surrogate (gray-box).

params -> [physics FM core -> log-STFT] + [FiLM-conditioned neural residual] -> predicted
log-magnitude spectrogram(s). The physics core gives the harmonic skeleton; the residual
learns Operator's undocumented remainder. Frozen after training, it provides the audio
loss for the one-shot matcher. See docs/renderer-design.md.

v3 (the Batch-5 refactor — full old->new reasoning in docs/batch5-refactor.md):
- TENSOR-IN FORWARD: `forward` accepts the normalized param matrix [B, n_params]
  directly (list-of-dicts still works for offline tools). The old dict-only interface
  rebuilt tensors with Python loops every step (slow) and SEVERED AUTOGRAD at the input —
  the Phase-B matcher needs d(spectral loss)/d(params), which now flows end-to-end
  through ControlMap + the physics. The discrete bits (algorithm, osc on/off) stay hard
  selections — the matcher must handle those separately (classify / straight-through).
- PITCH/VELOCITY CONDITIONING (Batch 5): per-sample `note` drives the physics f0
  (together with the previously-ignored Transpose param) and, with `velocity`, joins the
  conditioning encoder as two extra tokens (cfg.condition_pitch). Operator timbre is not
  pitch-invariant, so the pitch-diverse dataset requires this; velocity reaches only the
  residual (its level/FM-index mapping is Operator secret sauce, not modeled physics).
- ONE physics render per forward: `return_physics=True` returns (hybrid, physics)
  predictions from the SAME waveform — evaluation used to render the physics twice.
- Snake activation rewritten via the identity sin^2(ax) = (1-cos(2ax))/2 — same math,
  same parameters, fewer large fp32 intermediates under bf16 autocast (sin/cos upcast),
  which is what forced gradient checkpointing at batch 16 on a 3090.

v2 (unchanged): MULTI-RESOLUTION linear-STFT log-magnitude output (512/1024/2048),
Snake-activated residual CNN shared across resolutions, frequency positional encoding,
FiLM conditioning, preset-tokenizer Transformer encoder (the validated +50.1% lever).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..schema import OperatorSchema
from .adapter import ControlMap, midi_to_hz, normalize_params
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
    f0: float = 261.63                  # legacy field (pre-Batch-5 ckpts); f0 now comes from `note`
    oversample: int = 2                 # physics-core anti-aliasing (render at os x, LPF, decimate)
    # --- conditioning + residual capacity (Batch 3/4) ---
    encoder: str = "transformer"        # "transformer" (preset-tokenizer) or "mlp" (legacy)
    code_dim: int = 256                 # FiLM conditioning vector width
    ch: int = 64                        # residual CNN width (Batch 4 capacity: 48 -> 64)
    n_blocks: int = 4                   # residual CNN mid-blocks (Batch 4 capacity: 3 -> 4)
    dropout: float = 0.2
    tf_dim: int = 160                   # preset-tokenizer Transformer token width
    tf_layers: int = 3
    tf_heads: int = 8
    grad_checkpoint: bool = True        # recompute residual blocks in backward (saves VRAM)
    # Batch 5: condition on the played note + velocity (extra encoder tokens + physics f0).
    # False reproduces the pre-Batch-5 architecture (and loads its checkpoints).
    condition_pitch: bool = True

    @property
    def hops(self) -> tuple:
        return tuple(n // 4 for n in self.n_ffts)


_WIN_CACHE: dict = {}


def _hann(n_fft: int, device) -> torch.Tensor:
    """Cached Hann window — avoids reallocating it on every STFT (6x per training step)."""
    key = (n_fft, str(device))
    w = _WIN_CACHE.get(key)
    if w is None:
        w = torch.hann_window(n_fft, device=device)
        _WIN_CACHE[key] = w
    return w


def log_mag(wave: torch.Tensor, n_fft: int, hop: int, eps: float = 1e-5) -> torch.Tensor:
    """[B, T] waveform -> [B, F, frames] log-magnitude linear STFT."""
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, window=_hann(n_fft, wave.device),
                      center=True, return_complex=True)
    return torch.log(spec.abs() + eps)


class Snake(nn.Module):
    """Periodic activation x + (1/a) sin^2(a x) — periodic inductive bias for FM/sinusoids.
    Computed as x + (1 - cos(2ax)) / (2a): identical function & parameters, but one
    upcast trig intermediate instead of two (sin then square) under bf16 autocast."""

    def __init__(self, channels: int):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        return x + (1.0 - torch.cos(2.0 * self.a * x)) / (2.0 * (self.a + 1e-6))


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


class MlpConditioner(nn.Module):
    """Legacy conditioner: normalized-param vector (+ note/vel when conditioned) -> MLP ->
    code. Dropout stops the residual from using the (near-unique) param vector as a
    per-sample lookup key."""

    def __init__(self, n_params: int, code_dim: int, dropout: float, condition_pitch: bool):
        super().__init__()
        self.condition_pitch = condition_pitch
        self.net = nn.Sequential(
            nn.Linear(n_params + (2 if condition_pitch else 0), code_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(code_dim, code_dim), nn.ReLU(), nn.Dropout(dropout),
        )

    def forward(self, vec, note_n=None, vel_n=None):  # vec [B, n_params] in [0,1]
        if self.condition_pitch:
            vec = torch.cat([vec, note_n.unsqueeze(-1), vel_n.unsqueeze(-1)], dim=-1)
        return self.net(vec)


class PresetTransformerConditioner(nn.Module):
    """Preset-tokenizer + Transformer conditioner (best on Dexed FM in the Neural-Proxies
    work). Each parameter becomes a token = (which-param embedding) + (its value, lifted to
    Fourier features so non-linear value->timbre effects are easy to model). A [CLS] token
    summarizes the preset via self-attention over all params -> the FiLM code. Self-attention
    lets the model reason about param INTERACTIONS (algorithm x ratio x level) the MLP can't.
    Batch 5: note + velocity join as two extra tokens (own ID embeddings, shared Fourier
    value lift) — exactly the slot this design reserved for them."""

    def __init__(self, n_params: int, code_dim: int, d: int, layers: int, heads: int,
                 dropout: float, condition_pitch: bool, n_fourier: int = 6):
        super().__init__()
        self.condition_pitch = condition_pitch
        self.param_id = nn.Embedding(n_params, d)            # which param this token is
        self.register_buffer("freqs", (2.0 ** torch.arange(n_fourier)) * torch.pi)
        self.value_proj = nn.Linear(2 * n_fourier, d)        # Fourier(value) -> token
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if condition_pitch:  # created conditionally so pre-Batch-5 checkpoints still load
            self.extra_id = nn.Parameter(torch.randn(2, d) * 0.02)  # note, velocity token IDs
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, code_dim)
        self.register_buffer("ids", torch.arange(n_params))

    def _lift(self, v: torch.Tensor) -> torch.Tensor:
        """values in [0,1], any shape -> Fourier features [..., 2*n_fourier]."""
        ff = v.unsqueeze(-1) * self.freqs
        return torch.cat([ff.sin(), ff.cos()], dim=-1)

    def forward(self, vec, note_n=None, vel_n=None):  # vec [B, n_params] in [0,1]
        B, _ = vec.shape
        tokens = self.value_proj(self._lift(vec)) + self.param_id(self.ids)[None]  # [B,P,d]
        parts = [self.cls.expand(B, -1, -1), tokens]
        if self.condition_pitch:
            extra = torch.stack([note_n, vel_n], dim=-1)                       # [B,2]
            parts.append(self.value_proj(self._lift(extra)) + self.extra_id[None])
        x = torch.cat(parts, dim=1)
        return self.out(self.encoder(x)[:, 0])               # CLS -> code [B, code_dim]


class HybridRenderer(nn.Module):
    def __init__(self, schema: OperatorSchema, cfg: RendererConfig | None = None):
        super().__init__()
        self.schema = schema
        self.cfg = cfg or RendererConfig()
        c = self.cfg
        self.physics = DiffOperator(c.sample_rate, c.n_samples, c.note_off, oversample=c.oversample)
        self.control_map = ControlMap(schema)
        n_params = len(schema.params)
        if c.encoder == "transformer":
            self.encoder = PresetTransformerConditioner(
                n_params, c.code_dim, c.tf_dim, c.tf_layers, c.tf_heads, c.dropout,
                c.condition_pitch)
        else:
            self.encoder = MlpConditioner(n_params, c.code_dim, c.dropout, c.condition_pitch)
        # Frequency positional encoding: a FiLM-CNN is translation-equivariant in frequency
        # and can't otherwise place a peak at a preset-specific absolute bin. These extra
        # channels (linear ramp + sinusoids) tell every conv WHERE in frequency it is.
        self.n_freqenc = 5  # 1 linear + 4 sinusoidal
        self.in_block = FiLMBlock(1 + self.n_freqenc, c.ch, c.code_dim)
        self.mid = nn.ModuleList([FiLMBlock(c.ch, c.ch, c.code_dim) for _ in range(c.n_blocks)])
        self.out_conv = nn.Conv2d(c.ch, 1, 3, padding=1)

    def _freq_encoding(self, n_freq: int, n_time: int, batch: int, device) -> torch.Tensor:
        """[B, n_freqenc, F, T] absolute-frequency coordinate channels."""
        f = torch.linspace(0.0, 1.0, n_freq, device=device)
        feats = [f] + [torch.sin((2 ** k) * torch.pi * f) for k in range(4)]
        enc = torch.stack(feats, dim=0)  # [n_freqenc, F]
        return enc[None, :, :, None].expand(batch, -1, -1, n_time)

    def _residual(self, phys: torch.Tensor, code: torch.Tensor, device) -> torch.Tensor:
        """phys [B,1,F,T] log-mag -> corrected log-mag [B,F,T] (shared across resolutions)."""
        B, _, n_freq, n_time = phys.shape
        x = torch.cat([phys, self._freq_encoding(n_freq, n_time, B, device)], dim=1)
        h = self.in_block(x, code)
        # Gate checkpointing on GRAD-ENABLED, not .training: checkpointing only matters for
        # the backward pass, and the matcher backprops its audio loss through this FROZEN
        # (eval-mode) renderer to the input params — without checkpointing here the residual
        # over 3 STFT resolutions OOMs a 24 GB card. Under no_grad (the renderer's own eval /
        # inspect / the matcher's DDIM sampling) it's a no-op, so nothing else changes.
        ckpt = self.cfg.grad_checkpoint and torch.is_grad_enabled()
        for blk in self.mid:
            # checkpointing stores far fewer activations (recomputes them in backward) —
            # the residual runs over 3 resolutions so this is the main VRAM lever.
            h = checkpoint(blk, h, code, use_reentrant=False) if ckpt else blk(h, code)
        return (phys + self.out_conv(h)).squeeze(1)

    def _prepare(self, params, device, note, velocity):
        """params (list[dict] | [B,P] normalized tensor) -> (vec, note, velocity) tensors."""
        if torch.is_tensor(params):
            vec = params.to(device)
        else:  # offline/legacy dict path (search, smoke tests) — not differentiable
            vec = normalize_params(params, self.schema).to(device)
        B = vec.shape[0]
        if note is None:  # legacy single-pitch default: Ableton C3 (MIDI 60)
            note = torch.full((B,), 60.0, device=device)
        else:
            note = note.to(device).float()
        if velocity is None:
            velocity = torch.full((B,), 100.0, device=device)
        else:
            velocity = velocity.to(device).float()
        return vec, note, velocity

    def forward(self, params, device: str = "cpu", apply_residual: bool = True,
                note: torch.Tensor | None = None, velocity: torch.Tensor | None = None,
                return_physics: bool = False):
        """params: [B, n_params] NORMALIZED tensor (differentiable, the training/matcher
        path) or list[dict] of raw values (offline tools). note/velocity: [B] tensors
        (MIDI number / 0-127), default C3/100 for legacy single-pitch data.

        Returns {n_fft: predicted log-mag [B, F, T]}. With return_physics=True returns
        (hybrid, physics) dicts sharing ONE physics render (the old evaluate() called
        forward twice and rendered the physics twice)."""
        vec, note, velocity = self._prepare(params, device, note, velocity)
        controls = self.control_map(vec)
        # f0 = played note shifted by the (previously ignored) global Transpose param
        f0 = midi_to_hz(note) * torch.pow(2.0, controls["semitones"] / 12.0)
        # per-op Hz from the MEASURED frequency laws (see adapter.py): ratio-tracking
        # ops follow f0 x (floored Coarse x Fine multiplier); fixed ops ignore the note
        freq_hz = torch.where(controls["fix_on"] > 0.5, controls["fixed_hz"],
                              f0.unsqueeze(1) * controls["ratio"])
        wave = self.physics.render(controls["ratio"], controls["level"], controls["adsr"],
                                   controls["algo"], freq_hz=freq_hz)
        code = None if not apply_residual else \
            self.encoder(vec, note / 127.0, velocity / 127.0)
        out, phys_out = {}, {}
        for n_fft, hop in zip(self.cfg.n_ffts, self.cfg.hops):
            phys = log_mag(wave, n_fft, hop).unsqueeze(1)
            phys_out[n_fft] = phys.squeeze(1)
            out[n_fft] = phys_out[n_fft] if code is None else self._residual(phys, code, device)
        if return_physics:
            return out, phys_out
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
    target = torch.randn(2, model.cfg.n_samples)

    # 1) legacy dict path (offline tools)
    preds = model([base, variant])
    loss = renderer_loss(preds, model.target_logmag(target))
    loss.backward()
    g = sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    print("dict path:", {k: tuple(v.shape) for k, v in preds.items()},
          "loss", round(loss.item(), 3), "encoder grad>0:", bool(g > 0))

    # 2) tensor path — THE MATCHER REQUIREMENT: gradients must reach the param tensor
    model.zero_grad()
    vec = normalize_params([base, variant], sch).requires_grad_(True)
    note = torch.tensor([60.0, 72.0])
    vel = torch.tensor([100.0, 80.0])
    hyb, phys = model(vec, note=note, velocity=vel, return_physics=True)
    loss = renderer_loss(hyb, model.target_logmag(target))
    loss.backward()
    cm = model.control_map
    phys_cols = torch.cat([cm.coarse_idx, cm.level_idx, cm.adsr_idx.reshape(-1),
                           torch.tensor([cm.i_transpose])])
    print("tensor path: loss", round(loss.item(), 3),
          "| grad to params:", bool(vec.grad is not None and vec.grad.abs().sum() > 0),
          "| grad reaches physics cols:", bool(vec.grad[:, phys_cols].abs().sum() > 0),
          "| physics output shapes:", {k: tuple(v.shape) for k, v in phys.items()})
