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
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ..schema import OperatorSchema
from .renderer import HybridRenderer, RendererConfig
from .train_renderer import RendererDataset, collate, mean_spectrogram, _repo_root


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

    cfg = RendererConfig()
    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    ds = RendererDataset(Path(args.data), cfg)
    n_val = max(1, int(0.1 * len(ds)))
    _, va = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    va_dl = DataLoader(va, batch_size=16, collate_fn=collate)

    model = HybridRenderer(schema, cfg).to(args.device)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model.load_state_dict(ck["state_dict"])  # EMA weights (what we deploy)
    model.eval()
    canon = ck.get("canon_fft", cfg.canon_fft)
    mean_spec = ck["mean_spec"].to(args.device) if "mean_spec" in ck \
        else mean_spectrogram(model, va_dl, args.device, canon)
    print(f"loaded {args.ckpt} (best_val={ck.get('best_val', float('nan')):.4f}, "
          f"epoch={ck.get('epoch', '?')}, canon_fft={canon}) | "
          f"scoring {min(args.n, len(va))} val samples on {args.device}")

    # accumulators: {metric: {variant: [per-sample values]}}
    metrics = {"overall": {}, "energy_wt": {}, "loud_bins": {}}
    for m in metrics:
        for v in ("hybrid", "physics", "mean"):
            metrics[m][v] = []
    seen = 0
    with torch.no_grad():
        for dicts, waves in va_dl:
            if seen >= args.n:
                break
            target = model.target_logmag(waves.to(args.device))[canon]
            preds = {
                "hybrid": model(dicts, args.device, apply_residual=True)[canon],
                "physics": model(dicts, args.device, apply_residual=False)[canon],
                "mean": mean_spec.expand_as(target),
            }
            for v, p in preds.items():
                metrics["overall"][v].append(_l1_per_sample(p, target))
                metrics["energy_wt"][v].append(_energy_weighted_l1(p, target))
                metrics["loud_bins"][v].append(_loud_bin_l1(p, target))
            seen += target.shape[0]

    cat = {m: {v: torch.cat(metrics[m][v]) for v in metrics[m]} for m in metrics}
    print(f"\n{'metric':<12}{'hybrid':>10}{'physics':>10}{'mean':>10}{'hyb<mean %':>12}")
    for m in ("overall", "energy_wt", "loud_bins"):
        h, p, mn = (cat[m][v].mean().item() for v in ("hybrid", "physics", "mean"))
        win = (cat[m]["hybrid"] < cat[m]["mean"]).float().mean().item() * 100
        print(f"{m:<12}{h:>10.4f}{p:>10.4f}{mn:>10.4f}{win:>11.1f}%")

    # how much of the mean->perfect gap does the hybrid close, on the loud bins?
    mn, h = cat["loud_bins"]["mean"].mean().item(), cat["loud_bins"]["hybrid"].mean().item()
    print(f"\nloud-bin gap closed vs mean: {100 * (mn - h) / mn:.1f}%  "
          f"(this is the number that matters for the matcher's gradient)")


if __name__ == "__main__":
    main()
