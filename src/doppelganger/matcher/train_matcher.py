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
def evaluate(model, loader, device, watch):
    """`watch` = list of (label, 'cat'|'cont', index-or-index-list) — per-param probes
    (cat -> accuracy, cont -> MAE) on top of the overall masked cont_mae/cat_acc/bin_acc."""
    model.eval()
    cont_ae = cont_n = cat_hit = cat_n = bin_hit = bin_n = 0.0
    w_acc = {w[0]: [0.0, 0.0] for w in watch if w[1] == "cat"}
    w_mae = {w[0]: [0.0, 0.0] for w in watch if w[1] == "cont"}
    cat_idx = {w[0]: w[2] for w in watch if w[1] == "cat"}
    cont_idx = {w[0]: (w[2] if isinstance(w[2], list) else [w[2]]) for w in watch if w[1] == "cont"}
    for batch in loader:
        audio, cont, binary, cat, cont_m, bin_m, cat_m = [b.to(device) for b in batch]
        pc, pb, pcat = model.decode(model(audio))
        cont_ae += ((pc - cont).abs() * cont_m).sum().item(); cont_n += cont_m.sum().item()
        cat_hit += (((pcat == cat).float()) * cat_m).sum().item(); cat_n += cat_m.sum().item()
        bin_hit += ((((pb >= 0.5).float() == binary).float()) * bin_m).sum().item(); bin_n += bin_m.sum().item()
        # weight each probe by its gate weight (= controlling osc level) so the metric
        # reflects audible cases — e.g. RATIO_ACC counts ratios where the modulator is heard,
        # not the low-index samples where the ratio is unobservable.
        for lbl, i in cat_idx.items():
            wgt = cat_m[:, i]
            w_acc[lbl][0] += ((pcat[:, i] == cat[:, i]).float() * wgt).sum().item()
            w_acc[lbl][1] += wgt.sum().item()
        for lbl, idxs in cont_idx.items():
            wgt = cont_m[:, idxs]
            w_mae[lbl][0] += ((pc[:, idxs] - cont[:, idxs]).abs() * wgt).sum().item()
            w_mae[lbl][1] += wgt.sum().item()
    m = {lbl: h / max(n, 1) for lbl, (h, n) in w_acc.items()}
    m.update({lbl: ae / max(n, 1) for lbl, (ae, n) in w_mae.items()})
    m.update(cont_mae=cont_ae / max(cont_n, 1), cat_acc=cat_hit / max(cat_n, 1), bin_acc=bin_hit / max(bin_n, 1))
    return m


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "dataset" / "sniff"))
    ap.add_argument("--out", default=str(root / "models" / "operator" / "matcher_v3.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--bins", type=int, default=64, help="bins per continuous param (finer = better envelope precision)")
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

    # per-param probes (whichever exist in this stage's predicted set)
    cont_names = [p.name for p in model.codec.cont]
    cat_names = [p.name for p in model.codec.categorical]
    watch = [("WAVE_ACC", "cat", cat_names.index("Osc-A Wave")),
             ("ADSR_MAE", "cont", [cont_names.index(n) for n in
                                   ("Ae Attack", "Ae Decay", "Ae Sustain", "Ae Release")])]
    for X in "BCD":                    # each active modulator's ratio + level (index-weighted)
        if f"{X} Coarse" in cat_names:
            watch.append((f"RATIO_{X}", "cat", cat_names.index(f"{X} Coarse")))
        if f"Osc-{X} Level" in cont_names:
            watch.append((f"IDX_{X}", "cont", [cont_names.index(f"Osc-{X} Level")]))

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
            m = evaluate(model, va_dl, device, watch)
            # gate score: reward the watched accuracies, penalize the watched errors
            score = (sum(m[w[0]] for w in watch if w[1] == "cat")
                     - sum(m[w[0]] for w in watch if w[1] == "cont"))
            improved = score > best
            probes = " ".join(f"{w[0]}={m[w[0]]:.3f}" for w in watch)
            print(f"epoch {epoch:3} | train cont={agg[0]/nb:.3f} bin={agg[1]/nb:.3f} cat={agg[2]/nb:.3f} "
                  f"| {probes} | cont_mae={m['cont_mae']:.3f} cat_acc={m['cat_acc']:.3f} "
                  f"bin_acc={m['bin_acc']:.3f}{'  <- best' if improved else ''}")
            if improved:
                best = score
                torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                            "epoch": epoch, "metrics": m}, out)
        else:
            print(f"epoch {epoch:3} | train cont={agg[0]/nb:.3f} bin={agg[1]/nb:.3f} cat={agg[2]/nb:.3f}")

    print(f"Done. Best sniff score={best:.3f}. Saved -> {out}")


if __name__ == "__main__":
    main()
