"""
Train the one-shot matcher (DIRECT prediction) against the FROZEN neural renderer.

    uv run python -m doppelganger.matcher.train_matcher --epochs 60

Pipeline per step (target audio = a real-Operator render from dataset/operator):
  target log-mag -> AudioEncoder -> z
  z              -> AlgoClassifier -> Algorithm logits          (CE vs true algo)
  z (+ algo,pitch) -> ParamHead -> position in the dataset MANIFOLD for the other 194 params
  assemble[pos + TRUE algo] -> FROZEN renderer(note,vel) -> magnitude-weighted spectral loss

Total = w_audio * audio  +  w_param * MSE(pos, true position)  +  w_algo * CE.
The audio loss is the perceptual driver; the param term anchors predictions to real
presets (and breaks the many-to-one ties toward plausible ones). Because the ParamHead's
sigmoid output is mapped affinely into the per-param dataset manifold, predictions stay
inside the audible distribution the renderer was trained on — the audio loss can match
WITHIN it but can't escape to chase the renderer's slack (which is what railed the old
audio-driven version to silent / +48-pitched presets). v2: deterministic (no diffusion).

Requires a trained renderer (models/operator/renderer.pt) + the dataset cache.
"""

from __future__ import annotations

import argparse
import dataclasses
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..schema import OperatorSchema
from ..synth.inspect_renderer import cfg_from_checkpoint
from ..synth.renderer import HybridRenderer
from ..synth.spectral import renderer_loss, renderer_loss_per_sample
from ..synth.train_renderer import (Ema, RendererDataset, collate, ema_weights,
                                    hash_split, _repo_root)
from .model import Matcher


def load_frozen_renderer(ckpt_path: Path, schema: OperatorSchema, device: str):
    """Rebuild the renderer with its training-time config, load EMA weights, freeze."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cfg_from_checkpoint(ck)
    # Force gradient checkpointing on: the matcher backprops through this renderer and the
    # residual won't fit otherwise (grad_checkpoint doesn't affect architecture, so this is
    # safe for load_state_dict even if the renderer was trained with --no-grad-ckpt).
    cfg = dataclasses.replace(cfg, grad_checkpoint=True)
    renderer = HybridRenderer(schema, cfg).to(device)
    renderer.load_state_dict(ck["state_dict"])
    renderer.eval()
    for p in renderer.parameters():
        p.requires_grad_(False)
    return renderer, cfg, ck.get("canon_fft", cfg.canon_fft)


@torch.no_grad()
def data_stats(loader, n_params, device):
    """One pass over the train set: mean normalized params (predict-the-mean baseline
    preset) + per-param [min,max] (the manifold box the predictor is confined to)."""
    s = torch.zeros(n_params, device=device)
    lo = torch.full((n_params,), float("inf"), device=device)
    hi = torch.full((n_params,), float("-inf"), device=device)
    n = 0
    for params, _, _, _ in loader:
        p = params.to(device)
        s += p.sum(0)
        lo = torch.minimum(lo, p.amin(0))
        hi = torch.maximum(hi, p.amax(0))
        n += p.shape[0]
    return s / max(n, 1), lo, hi


@torch.no_grad()
def evaluate(matcher, renderer, loader, device, canon, mean_p, alpha, algo_topk, max_batches=0):
    """Mirrors inference: predict params for the classifier's top-k algorithms, render each
    through the frozen renderer, keep the best per sample, compare to rendering the
    mean-params preset. Also algo top-1 accuracy and the param-anchor MSE."""
    matcher.eval()
    match_l = base_l = algo_correct = param_se = n = 0.0
    k = min(algo_topk, matcher.n_algo)
    for bi, (params, waves, notes, vels) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        waves = waves.to(device).float(); params = params.to(device)
        notes = notes.to(device); vels = vels.to(device)
        B = params.shape[0]
        target = renderer.target_logmag(waves)
        z = matcher.encode(target[canon])
        logits = matcher.algo_head(z)
        true_algo = matcher.true_algo(params)
        algo_correct += (logits.argmax(-1) == true_algo).float().sum().item()
        topk = logits.topk(k, dim=-1).indices                       # [B,k]
        param_se += F.mse_loss(matcher.predict(z, topk[:, 0], notes, vels),
                               matcher.target_pos(params)).item() * B
        best = torch.full((B,), float("inf"), device=device)
        for j in range(k):
            pos = matcher.predict(z, topk[:, j], notes, vels)
            full = matcher.assemble(pos, topk[:, j])
            best = torch.minimum(best, renderer_loss_per_sample(
                renderer(full, device, note=notes, velocity=vels), target, alpha))
        match_l += best.sum().item()
        base_full = mean_p.unsqueeze(0).expand_as(params)
        base_l += renderer_loss_per_sample(
            renderer(base_full, device, note=notes, velocity=vels), target, alpha).sum().item()
        n += B
    return match_l / n, base_l / n, algo_correct / n, param_se / n


def _parse_args():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "dataset" / "operator"))
    ap.add_argument("--renderer", default=str(root / "models" / "operator" / "renderer.pt"))
    ap.add_argument("--out", default=str(root / "models" / "operator" / "matcher.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="the renderer's residual (3 STFT res) is the VRAM driver; 16 fits a 3090")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=10.0, help="loud-bin emphasis in the audio loss")
    ap.add_argument("--w-audio", type=float, default=1.0, help="audio-loss weight (perceptual driver)")
    ap.add_argument("--w-param", type=float, default=1.0, help="param-anchor MSE weight (stay on real presets)")
    ap.add_argument("--w-algo", type=float, default=1.0, help="algorithm cross-entropy weight")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--algo-topk", type=int, default=3, help="top-k algorithms to render+rank at eval")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=16, help="val batches per eval (0 = all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = _parse_args()
    root = _repo_root()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = args.device
    use_amp = bool(args.amp) and device.startswith("cuda")

    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    renderer, rcfg, canon = load_frozen_renderer(Path(args.renderer), schema, device)
    ds = RendererDataset(Path(args.data), rcfg, schema)
    if not rcfg.condition_pitch:
        ds.restrict_to_note(60)
    if args.limit:
        ds.limit(args.limit)
    tr, va = hash_split(ds)
    dl_kw = dict(collate_fn=collate, num_workers=args.workers,
                 pin_memory=device.startswith("cuda"), persistent_workers=args.workers > 0)
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True, **dl_kw)
    va_dl = DataLoader(va, batch_size=args.batch_size, **dl_kw)

    matcher = Matcher(schema).to(device)
    mean_p, man_lo, man_hi = data_stats(DataLoader(tr, batch_size=args.batch_size, **dl_kw),
                                        len(schema.params), device)
    matcher.set_manifold(man_lo, man_hi)  # confine predictions to the audible data manifold

    opt = torch.optim.AdamW(matcher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(0, min(args.warmup_epochs, args.epochs - 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warmup))],
        milestones=[warmup]) if warmup else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = Ema(matcher, args.ema_decay)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    nfree = int((man_hi[matcher.keep_idx] - man_lo[matcher.keep_idx] > 1e-6).sum())
    print(f"{schema.summary()} | train={len(tr)} val={len(va)} device={device} "
          f"renderer canon_fft={canon} condition_pitch={rcfg.condition_pitch}")
    print(f"  deterministic predictor | dim={matcher.dim} (Algorithm -> classifier); "
          f"{nfree}/{matcher.dim} params vary in the manifold, the rest pinned by the data")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        matcher.train()
        run_a = run_p = run_c = nb = 0.0
        for params, waves, notes, vels in tr_dl:
            waves = waves.to(device, non_blocking=True).float()
            params = params.to(device, non_blocking=True)
            notes = notes.to(device); vels = vels.to(device)
            target = renderer.target_logmag(waves)
            true_algo = matcher.true_algo(params)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()):
                z = matcher.encode(target[canon])
                logits = matcher.algo_head(z)
                pos = matcher.predict(z, true_algo, notes, vels)     # teacher-force true algo
                full = matcher.assemble(pos, true_algo)
                preds = renderer(full, device, note=notes, velocity=vels)
            audio = renderer_loss({k: v.float() for k, v in preds.items()}, target, args.alpha)
            param = F.mse_loss(pos.float(), matcher.target_pos(params))
            ce = F.cross_entropy(logits.float(), true_algo)
            loss = args.w_audio * audio + args.w_param * param + args.w_algo * ce
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(matcher.parameters(), args.clip)
            opt.step()
            ema.update(matcher)
            run_a += audio.item(); run_p += param.item(); run_c += ce.item(); nb += 1
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            with ema_weights(matcher, ema):
                match_l, base_l, algo_acc, param_mse = evaluate(
                    matcher, renderer, va_dl, device, canon, mean_p, args.alpha,
                    args.algo_topk, args.eval_batches)
            gain = 100 * (base_l - match_l) / base_l
            improved = match_l < best - 1e-4
            print(f"epoch {epoch:3} | train audio={run_a/max(nb,1):.4f} param={run_p/max(nb,1):.4f} "
                  f"ce={run_c/max(nb,1):.4f} | val match={match_l:.4f} mean-base={base_l:.4f} "
                  f"(gain {gain:+.1f}%) algo_acc={algo_acc:.3f}{'  <- best' if improved else ''}")
            if improved:
                best = match_l
                deploy_sd = {k: ema.shadow[k].cpu() if k in ema.shadow else v.cpu()
                             for k, v in matcher.state_dict().items()}
                torch.save({"state_dict": deploy_sd, "raw_state_dict": matcher.state_dict(),
                            "ema": ema.shadow, "epoch": epoch, "best_match": best,
                            "algo_topk": args.algo_topk,
                            "renderer_ckpt": str(Path(args.renderer).resolve())}, out)
        else:
            print(f"epoch {epoch:3} | train audio={run_a/max(nb,1):.4f} param={run_p/max(nb,1):.4f} "
                  f"ce={run_c/max(nb,1):.4f}")

    print(f"Done. Best val match audio loss={best:.4f}. Saved matcher -> {out}")


if __name__ == "__main__":
    main()
