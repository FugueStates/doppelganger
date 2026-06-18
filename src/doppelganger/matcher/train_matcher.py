"""
Train the v3 matcher (audio -> Operator params, classification heads, NO renderer).

    uv run python -m doppelganger.matcher.train_matcher --data dataset/sniff --epochs 80

Loss = soft-CE over binned continuous params (Gaussian soft labels) + BCE binary + CE
categorical, each masked by oscillator gating. (Per-parameter perceptual weighting from
sensitivity.py is added when we scale past the sniff test.)

THE SNIFF-TEST GATE: on the ~512 single-oscillator {Sin,Saw,Square,Triangle}+ADSR set,
the eval prints **waveform accuracy** (Osc-A Wave, should approach ~1.0) and **ADSR MAE**
(Ae A/D/S/R, should be small). If the model can't learn those, the formulation is wrong.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..schema import OperatorSchema
from ..synth.train_renderer import _repo_root
from .data import MatcherDataset, hash_split
from .model import MatcherConfig, ParamMatcher


def _masked_mean(per_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (per_elem * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss(model, out, batch, sigma):
    _, cont, binary, cat, cont_m, bin_m, cat_m = batch
    # continuous: soft cross-entropy over K bins
    soft = model.codec.soft_bins(cont, sigma)                       # [B,n_cont,K]
    logp = F.log_softmax(out["cont_logits"], dim=-1)
    cont_ce = -(soft * logp).sum(-1)                                # [B,n_cont]
    l_cont = _masked_mean(cont_ce, cont_m)
    # binary: BCE
    bce = F.binary_cross_entropy_with_logits(out["binary_logits"], binary, reduction="none")
    l_bin = _masked_mean(bce, bin_m)
    # categorical: per-head CE
    cat_ce = torch.stack([F.cross_entropy(s, cat[:, j], reduction="none")
                          for j, s in enumerate(model.cat_slices(out["cat_logits"]))], dim=1)
    l_cat = _masked_mean(cat_ce, cat_m)
    return l_cont + l_bin + l_cat, (l_cont.item(), l_bin.item(), l_cat.item())


@torch.no_grad()
def evaluate(model, loader, device, sniff_idx):
    model.eval()
    wave_i, adsr_i = sniff_idx
    cont_ae = cont_n = cat_hit = cat_n = bin_hit = bin_n = 0.0
    wave_hit = wave_n = 0.0
    adsr_ae = adsr_n = 0.0
    for batch in loader:
        audio, cont, binary, cat, cont_m, bin_m, cat_m = [b.to(device) for b in batch]
        out = model(audio)
        pc, pb, pcat = model.decode(out)
        cont_ae += ((pc - cont).abs() * cont_m).sum().item(); cont_n += cont_m.sum().item()
        cat_hit += (((pcat == cat).float()) * cat_m).sum().item(); cat_n += cat_m.sum().item()
        bin_hit += ((((pb >= 0.5).float() == binary).float()) * bin_m).sum().item(); bin_n += bin_m.sum().item()
        # sniff metrics
        wave_hit += (pcat[:, wave_i] == cat[:, wave_i]).float().sum().item(); wave_n += audio.shape[0]
        adsr_ae += (pc[:, adsr_i] - cont[:, adsr_i]).abs().sum().item(); adsr_n += audio.shape[0] * len(adsr_i)
    return dict(wave_acc=wave_hit / max(wave_n, 1), adsr_mae=adsr_ae / max(adsr_n, 1),
                cont_mae=cont_ae / max(cont_n, 1), cat_acc=cat_hit / max(cat_n, 1),
                bin_acc=bin_hit / max(bin_n, 1))


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "dataset" / "sniff"))
    ap.add_argument("--out", default=str(root / "models" / "operator" / "matcher_v3.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian soft-label width (bins)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    cfg = MatcherConfig(n_bins=args.bins)
    model = ParamMatcher(schema, cfg).to(device)
    ds = MatcherDataset(args.data, model.codec, cfg)
    if args.limit:
        ds.ids = ds.ids[:args.limit]
    tr, va = hash_split(ds)
    dl_kw = dict(num_workers=args.workers, pin_memory=device.startswith("cuda"))
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=len(tr) > args.batch_size, **dl_kw)
    va_dl = DataLoader(va, batch_size=args.batch_size, **dl_kw)

    # indices for the sniff metrics
    cont_names = [p.name for p in model.codec.cont]
    cat_names = [p.name for p in model.codec.categorical]
    wave_i = cat_names.index("Osc-A Wave")
    adsr_i = [cont_names.index(n) for n in ("Ae Attack", "Ae Decay", "Ae Sustain", "Ae Release")]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{schema.summary()} | train={len(tr)} val={len(va)} device={device} | "
          f"bins={args.bins} sigma={args.sigma}")

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = [0.0, 0.0, 0.0]; nb = 0
        for batch in tr_dl:
            batch = [b.to(device) for b in batch]
            out_ = model(batch[0])
            loss, parts = compute_loss(model, out_, batch, args.sigma)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in range(3): agg[k] += parts[k]
            nb += 1
        sched.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(model, va_dl, device, (wave_i, adsr_i))
            score = m["wave_acc"] - m["adsr_mae"]  # the sniff gate, higher is better
            improved = score > best
            print(f"epoch {epoch:3} | train cont={agg[0]/nb:.3f} bin={agg[1]/nb:.3f} cat={agg[2]/nb:.3f} "
                  f"| WAVE_ACC={m['wave_acc']:.3f} ADSR_MAE={m['adsr_mae']:.3f} "
                  f"| cont_mae={m['cont_mae']:.3f} cat_acc={m['cat_acc']:.3f} bin_acc={m['bin_acc']:.3f}"
                  f"{'  <- best' if improved else ''}")
            if improved:
                best = score
                torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                            "epoch": epoch, "metrics": m}, out)
        else:
            print(f"epoch {epoch:3} | train cont={agg[0]/nb:.3f} bin={agg[1]/nb:.3f} cat={agg[2]/nb:.3f}")

    print(f"Done. Best sniff score={best:.3f}. Saved -> {out}")


if __name__ == "__main__":
    main()
