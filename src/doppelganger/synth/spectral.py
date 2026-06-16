"""
Spectral losses for the renderer (and later the matcher).

Two families:
- `multiscale_stft_loss` operates on WAVEFORMS (used by the CMA-ES search).
- The `*_logmag` losses + `renderer_loss` operate on LOG-MAGNITUDE spectrograms, since
  the HybridRenderer predicts log-mag directly. They are the renderer's training objective
  and will also be the matcher's audio loss (backprop params -> renderer -> these).

Why each term exists (FM-specific):
- energy-weighted L1: silent bins vastly outnumber loud ones; plain L1 lets the optimizer
  hedge the loud, preset-distinguishing FM sidebands toward the mean. Weighting by real
  magnitude forces it to fit where the sound actually is.
- spectral convergence: a magnitude-relative term (Frobenius ratio) that emphasizes getting
  the overall spectral shape right, scale-invariant to loudness.
- frequency transport: plain spectral L1 is "blind to frequency" — a sideband one bin off
  gets the same penalty as one octave off, and zero gradient pointing it home. The 1-D
  Wasserstein distance along frequency (L1 of the per-frame magnitude CDFs) gives a smooth
  gradient that MOVES energy to the correct frequency. Critical for FM sideband placement.
"""

from __future__ import annotations

import torch

_FFTS = (2048, 1024, 512, 256, 128)


def multiscale_stft_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """a, b: [B, n_samples] mono. Sum over FFT sizes of linear + log magnitude L1."""
    total = a.new_zeros(())
    for n_fft in _FFTS:
        win = torch.hann_window(n_fft, device=a.device)
        kw = dict(n_fft=n_fft, hop_length=n_fft // 4, window=win, return_complex=True, center=True)
        A = torch.stft(a, **kw).abs()
        B = torch.stft(b, **kw).abs()
        total = total + (A - B).abs().mean()
        total = total + (torch.log(A + eps) - torch.log(B + eps)).abs().mean()
    return total


def energy_weighted_logmag_l1(pred_lm: torch.Tensor, tgt_lm: torch.Tensor,
                              alpha: float = 10.0) -> torch.Tensor:
    """[B, F, T] log-mags. L1 up-weighted by real magnitude so loud bins dominate."""
    w = tgt_lm.exp()  # linear magnitude
    w = w / w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-9)  # per-sample, mean 1
    weight = 1.0 + alpha * w
    return (weight * (pred_lm - tgt_lm).abs()).sum() / weight.sum()


def spectral_convergence(pred_lm: torch.Tensor, tgt_lm: torch.Tensor,
                         eps: float = 1e-9) -> torch.Tensor:
    """Frobenius-norm ratio on linear magnitude — scale-relative spectral-shape error."""
    P, T = pred_lm.exp(), tgt_lm.exp()
    B = T.shape[0]
    num = torch.linalg.vector_norm((T - P).reshape(B, -1), dim=1)
    den = torch.linalg.vector_norm(T.reshape(B, -1), dim=1).clamp_min(eps)
    return (num / den).mean()


def freq_transport_l1(pred_lm: torch.Tensor, tgt_lm: torch.Tensor,
                      eps: float = 1e-8) -> torch.Tensor:
    """1-D Wasserstein along frequency (per time frame): L1 between the magnitude CDFs.
    Gives a gradient that MOVES misplaced energy to the right frequency bin."""
    P, T = pred_lm.exp(), tgt_lm.exp()
    Pn = P / P.sum(dim=-2, keepdim=True).clamp_min(eps)  # normalize over frequency
    Tn = T / T.sum(dim=-2, keepdim=True).clamp_min(eps)
    return (Pn.cumsum(dim=-2) - Tn.cumsum(dim=-2)).abs().mean()


def renderer_loss(preds: dict[int, torch.Tensor], targets: dict[int, torch.Tensor],
                  alpha: float = 10.0, w_sc: float = 1.0, w_ot: float = 1.0) -> torch.Tensor:
    """Multi-resolution renderer/matcher objective: averaged over STFT resolutions."""
    total = preds[next(iter(preds))].new_zeros(())
    for k in preds:
        p, t = preds[k], targets[k]
        total = total + energy_weighted_logmag_l1(p, t, alpha) \
            + w_sc * spectral_convergence(p, t) + w_ot * freq_transport_l1(p, t)
    return total / len(preds)


def renderer_loss_per_sample(preds: dict[int, torch.Tensor], targets: dict[int, torch.Tensor],
                             alpha: float = 10.0, w_sc: float = 1.0, w_ot: float = 1.0) -> torch.Tensor:
    """Per-sample [B] version of renderer_loss — same three terms, reduced over (F,T) only,
    averaged over resolutions. Used to SNR-weight the matcher's audio loss per timestep
    (the scalar renderer_loss can't be weighted per sample)."""
    total = None
    for k in preds:
        p, t = preds[k], targets[k]
        B = t.shape[0]
        w = t.exp()
        w = w / w.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-9)
        weight = 1.0 + alpha * w
        ew = (weight * (p - t).abs()).sum(dim=(-2, -1)) / weight.sum(dim=(-2, -1))   # [B]
        P, T = p.exp(), t.exp()
        num = torch.linalg.vector_norm((T - P).reshape(B, -1), dim=1)
        den = torch.linalg.vector_norm(T.reshape(B, -1), dim=1).clamp_min(1e-9)
        sc = num / den                                                                # [B]
        Pn = P / P.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        Tn = T / T.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        ot = (Pn.cumsum(dim=-2) - Tn.cumsum(dim=-2)).abs().mean(dim=(-2, -1))         # [B]
        term = ew + w_sc * sc + w_ot * ot
        total = term if total is None else total + term
    return total / len(preds)
