"""
Train the v3 matcher (audio -> Operator params, classification heads, NO renderer).

    uv run python -m doppelganger.matcher.train_matcher --data dataset/sniff --epochs 80

Loss = soft-CE over binned continuous params (Gaussian soft labels) + BCE binary + CE
categorical (each masked/weighted by oscillator level) + an ENVELOPE-SHAPE loss.

The envelope-shape loss addresses param-loss != perceptual-loss for the ADSR: a "moderate"
decay-param error can flip a slow swell into a percussive pluck (see docs/matcher-v3.md
Finding #9 / sample 0008262). We render each oscillator's predicted vs true ADSR amplitude
CURVE (differentiable, via diff_operator.adsr) and L1 them — so the envelope error is
weighted by its actual perceptual (curve) impact, not uniformly in param space. Cheap (a
1-D curve, no FM render). Level-weighted so quiet oscillators' envelopes don't dominate.

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
from ..synth.diff_operator import CalibConstants, adsr
from ..synth.train_renderer import _repo_root
from .data import MatcherDataset, hash_split
from .model import MatcherConfig, ParamMatcher


def _masked_mean(per_elem: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (per_elem * mask).sum() / mask.sum().clamp_min(1.0)


def build_env_spec(cont_names: list[str], cfg: MatcherConfig, device, n_t: int = 200,
                   note_off: float = 1.5):
    """Per-oscillator ADSR cont-indices + level-index, a time grid, and calib — for the
    envelope-shape loss. note_off = the collector's note duration (NOTE_BEATS=3 -> 1.5s).
    Only oscillators whose full ADSR + level are predicted (continuous)."""
    cal = CalibConstants()
    t = torch.linspace(0.0, cfg.n_samples / cfg.sample_rate, n_t, device=device)
    osc = {}
    for X, e in zip("ABCD", ("Ae", "Be", "Ce", "De")):
        try:
            ai = [cont_names.index(f"{e} {s}") for s in ("Attack", "Decay", "Sustain", "Release")]
            li = cont_names.index(f"Osc-{X} Level")
            osc[X] = (ai, li)
        except ValueError:
            pass
    return {"osc": osc, "t": t, "note_off": note_off, "calib": cal}


def envelope_l1(cont_pred, cont_true, env_spec):
    """Mean L1 between predicted and true ADSR amplitude CURVES, level-weighted per
    oscillator. cont_pred/cont_true: [B, n_cont] in [0,1] (differentiable in cont_pred)."""
    t, no, cal = env_spec["t"], env_spec["note_off"], env_spec["calib"]
    ck = cal.env_curve
    tot = cont_pred.new_zeros(()); wsum = cont_pred.new_zeros(())
    for X, (ai, li) in env_spec["osc"].items():
        def curve(p):
            return adsr(p[:, ai[0]], p[:, ai[1]], p[:, ai[2]], p[:, ai[3]], t, no,
                        cal.max_attack, cal.max_decay, cal.max_release, ck, ck, ck)
        l = (curve(cont_pred) - curve(cont_true)).abs().mean(-1)     # [B]
        w = cont_true[:, li].clamp(0, 1)
        tot = tot + (l * w).sum(); wsum = wsum + w.sum()
    return tot / wsum.clamp_min(1.0)


def compute_loss(model, out, batch, sigma, env_spec, w_env):
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
    # envelope-shape loss: render pred (soft-decoded) vs true ADSR curves, L1
    cont_pred = model.codec.expected_value(out["cont_logits"].softmax(-1))
    l_env = envelope_l1(cont_pred, cont, env_spec)
    total = l_cont + l_bin + l_cat + w_env * l_env
    return total, (l_cont.item(), l_bin.item(), l_cat.item(), l_env.item())


@torch.no_grad()
def evaluate(model, loader, device, watch, env_spec):
    """`watch` = list of (label, 'cat'|'cont', index-or-index-list) — per-param probes
    (cat -> accuracy, cont -> MAE) on top of the overall masked cont_mae/cat_acc/bin_acc.
    Also reports ENV_MAE: the perceptual envelope-curve L1 (the metric the env loss targets)."""
    model.eval()
    cont_ae = cont_n = cat_hit = cat_n = bin_hit = bin_n = 0.0
    env_sum = env_n = 0.0
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
        env_sum += envelope_l1(pc, cont, env_spec).item() * audio.shape[0]; env_n += audio.shape[0]
    m = {lbl: h / max(n, 1) for lbl, (h, n) in w_acc.items()}
    m.update({lbl: ae / max(n, 1) for lbl, (ae, n) in w_mae.items()})
    m.update(cont_mae=cont_ae / max(cont_n, 1), cat_acc=cat_hit / max(cat_n, 1),
             bin_acc=bin_hit / max(bin_n, 1), ENV_MAE=env_sum / max(env_n, 1))
    return m


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(root / "dataset" / "sniff")],
                    help="one or more dataset dirs (mixed with folder-namespaced ids)")
    ap.add_argument("--out", default=str(root / "models" / "operator" / "matcher_v3.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--bins", type=int, default=64, help="bins per continuous param (finer = better envelope precision)")
    ap.add_argument("--sigma", type=float, default=1.5, help="Gaussian soft-label width (bins)")
    ap.add_argument("--w-env", type=float, default=1.0, help="weight of the envelope-shape loss")
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
    # filter probes (Stage 3+): appear once the filter section is in the predicted set.
    # (Filter Slope is a binary head — folded into the overall bin_acc, no dedicated probe.)
    if "Filter Type" in cat_names:
        watch.append(("FTYPE_ACC", "cat", cat_names.index("Filter Type")))
    if "Filter Freq" in cont_names:
        watch.append(("CUTOFF_MAE", "cont", [cont_names.index("Filter Freq")]))
    if "Filter Res" in cont_names:
        watch.append(("RES_MAE", "cont", [cont_names.index("Filter Res")]))
    env_spec = build_env_spec(cont_names, cfg, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{schema.summary()} | train={len(tr)} val={len(va)} device={device} | "
          f"bins={args.bins} sigma={args.sigma}")

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        agg = [0.0, 0.0, 0.0, 0.0]; nb = 0
        for batch in tr_dl:
            batch = [b.to(device) for b in batch]
            out_ = model(batch[0])
            loss, parts = compute_loss(model, out_, batch, args.sigma, env_spec, args.w_env)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in range(4): agg[k] += parts[k]
            nb += 1
        sched.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate(model, va_dl, device, watch, env_spec)
            # gate score: reward watched accuracies, penalize watched errors + envelope error
            score = (sum(m[w[0]] for w in watch if w[1] == "cat")
                     - sum(m[w[0]] for w in watch if w[1] == "cont") - m["ENV_MAE"])
            improved = score > best
            probes = " ".join(f"{w[0]}={m[w[0]]:.3f}" for w in watch)
            print(f"epoch {epoch:3} | train cont={agg[0]/nb:.3f} bin={agg[1]/nb:.3f} cat={agg[2]/nb:.3f} "
                  f"env={agg[3]/nb:.3f} | {probes} ENV_MAE={m['ENV_MAE']:.3f} | "
                  f"cont_mae={m['cont_mae']:.3f} cat_acc={m['cat_acc']:.3f} "
                  f"bin_acc={m['bin_acc']:.3f}{'  <- best' if improved else ''}")
            if improved:
                best = score
                torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__,
                            "epoch": epoch, "metrics": m}, out)
        else:
            print(f"epoch {epoch:3} | train cont={agg[0]/nb:.3f} bin={agg[1]/nb:.3f} "
                  f"cat={agg[2]/nb:.3f} env={agg[3]/nb:.3f}")

    print(f"Done. Best sniff score={best:.3f}. Saved -> {out}")


if __name__ == "__main__":
    main()
