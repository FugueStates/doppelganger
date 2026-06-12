"""
Diagnose renderer fidelity BEYOND the misleading aggregate log-mag L1.

The full spectrogram is dominated by near-silent bins (high freqs, decay tails) that
are ~constant across presets, so predict-the-mean scores deceptively well and the
hybrid's edge looks tiny. These metrics ask the question that actually matters for the
matcher: does the renderer get the LOUD, preset-distinguishing bins right?

    uv run python -m doppelganger.synth.inspect_renderer            # CPU, 256 val samples
    uv run python -m doppelganger.synth.inspect_renderer --n 512 --device cuda

Reports, for hybrid / physics-only / predict-mean:
  - overall L1                 (matches the training log)
  - energy-weighted L1         (each bin weighted by the real magnitude -> ignores silence)
  - loud-bin L1                (only the top-10%-energy bins per sample)
  - win-rate                   (fraction of val samples where hybrid beats the mean)

Batch-5 refactor: the loud-bin gap is now reported at EVERY STFT resolution, not just
the canonical 1024 — sideband PLACEMENT (what the freq-transport loss exists for) is
sharpest at 2048, envelope/transient shape at 512, so the per-resolution split shows
WHERE fidelity gains come from. Uses the same hash split + tensor batches as training
(note the one-time discontinuity vs pre-refactor random-split numbers).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..schema import OperatorSchema
from .renderer import HybridRenderer, RendererConfig, log_mag
from .train_renderer import RendererDataset, collate, hash_split, mean_spectrogram, _repo_root

# Config fields newer than a checkpoint default to the OLD architecture so the saved
# weights still load (a pre-Batch-5 ckpt has no pitch tokens and expects fixed f0).
_LEGACY_CFG_DEFAULTS = {"condition_pitch": False}


def cfg_from_checkpoint(ck: dict) -> RendererConfig:
    """Rebuild the EXACT config the checkpoint was trained with (encoder, oversample,
    ch, ...), not the current defaults — else the architecture won't match the weights."""
    fields = {f.name for f in dataclasses.fields(RendererConfig)}
    merged = {**_LEGACY_CFG_DEFAULTS, **ck.get("cfg", {})}
    return RendererConfig(**{k: v for k, v in merged.items() if k in fields})


def _l1_per_sample(pred, target):  # [B,F,T] -> [B]
    return (pred - target).abs().mean(dim=(1, 2))


def _energy_weighted_l1(pred, target):
    """L1 weighted by real linear magnitude (= exp(log-mag)); silence contributes ~0."""
    w = target.exp()  # recover magnitude from log-mag
    return ((pred - target).abs() * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1e-9)


def _loud_bin_l1(pred, target, frac=0.10):
    """L1 over only the top-`frac` loudest bins per sample (where the sound lives)."""
    B = target.shape[0]
    err = (pred - target).abs().reshape(B, -1)
    mag = target.reshape(B, -1)
    k = max(1, int(frac * mag.shape[1]))
    idx = mag.topk(k, dim=1).indices
    return err.gather(1, idx).mean(dim=1)


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "dataset" / "operator"))
    ap.add_argument("--ckpt", default=str(root / "models" / "operator" / "renderer.pt"))
    ap.add_argument("--n", type=int, default=256, help="number of held-out val samples to score")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = cfg_from_checkpoint(ck)

    ds = RendererDataset(Path(args.data), cfg, schema)
    if not cfg.condition_pitch:
        ds.restrict_to_note(60)  # same pitch guard as training
    _, va = hash_split(ds)
    va_dl = DataLoader(va, batch_size=16, collate_fn=collate)

    model = HybridRenderer(schema, cfg).to(args.device)
    model.load_state_dict(ck["state_dict"])  # EMA weights (what we deploy)
    model.eval()
    canon = ck.get("canon_fft", cfg.canon_fft)
    mean_specs = {}  # mean baseline per resolution (ckpt only stores the canonical one)
    print(f"loaded {args.ckpt} (best_val={ck.get('best_val', float('nan')):.4f}, "
          f"epoch={ck.get('epoch', '?')}, canon_fft={canon}, "
          f"condition_pitch={cfg.condition_pitch}) | "
          f"scoring {min(args.n, len(va))} val samples on {args.device}")

    # mean baselines: reuse the checkpoint's canonical one; others from the val loader
    # (the baseline only needs to be CONSISTENT for comparisons, not train-derived).
    for n_fft, hop in zip(cfg.n_ffts, cfg.hops):
        if n_fft == canon and "mean_spec" in ck:
            mean_specs[n_fft] = ck["mean_spec"].to(args.device)
            continue
        s, n = None, 0
        with torch.no_grad():
            for _, waves, _, _ in va_dl:
                t = log_mag(waves.to(args.device).float(), n_fft, hop)
                s = t.sum(0) if s is None else s + t.sum(0)
                n += t.shape[0]
        mean_specs[n_fft] = s / n

    # accumulators: {n_fft: {metric: {variant: [per-sample values]}}}
    metrics = {n_fft: {m: {v: [] for v in ("hybrid", "physics", "mean")}
                       for m in ("overall", "energy_wt", "loud_bins")}
               for n_fft in cfg.n_ffts}
    seen = 0
    with torch.no_grad():
        for params, waves, notes, vels in va_dl:
            if seen >= args.n:
                break
            waves = waves.to(args.device).float()
            targets = model.target_logmag(waves)
            hyb, phys = model(params.to(args.device), args.device,
                              note=notes, velocity=vels, return_physics=True)
            for n_fft in cfg.n_ffts:
                target = targets[n_fft]
                preds = {"hybrid": hyb[n_fft], "physics": phys[n_fft],
                         "mean": mean_specs[n_fft].expand_as(target)}
                for v, p in preds.items():
                    metrics[n_fft]["overall"][v].append(_l1_per_sample(p, target))
                    metrics[n_fft]["energy_wt"][v].append(_energy_weighted_l1(p, target))
                    metrics[n_fft]["loud_bins"][v].append(_loud_bin_l1(p, target))
            seen += params.shape[0]

    for n_fft in cfg.n_ffts:
        cat = {m: {v: torch.cat(vals) for v, vals in metrics[n_fft][m].items()}
               for m in metrics[n_fft]}
        tag = " (canonical)" if n_fft == canon else ""
        print(f"\n--- n_fft={n_fft}{tag} ---")
        print(f"{'metric':<12}{'hybrid':>10}{'physics':>10}{'mean':>10}{'hyb<mean %':>12}")
        for m in ("overall", "energy_wt", "loud_bins"):
            h, p, mn = (cat[m][v].mean().item() for v in ("hybrid", "physics", "mean"))
            win = (cat[m]["hybrid"] < cat[m]["mean"]).float().mean().item() * 100
            print(f"{m:<12}{h:>10.4f}{p:>10.4f}{mn:>10.4f}{win:>11.1f}%")
        mn, h = cat["loud_bins"]["mean"].mean().item(), cat["loud_bins"]["hybrid"].mean().item()
        print(f"loud-bin gap closed vs mean: {100 * (mn - h) / mn:.1f}%")

    print("\n(the canonical-resolution loud-bin gap is the headline number; 2048 shows "
          "sideband placement, 512 shows envelope/transient shape)")


if __name__ == "__main__":
    main()
