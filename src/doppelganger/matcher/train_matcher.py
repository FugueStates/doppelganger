"""
Train the one-shot matcher against the FROZEN neural renderer.

    uv run python -m doppelganger.matcher.train_matcher --epochs 60

Pipeline per step (target audio = a real-Operator render from dataset/operator):
  target log-mag  -> AudioEncoder -> z
  z               -> AlgoClassifier -> Algorithm logits   (CE vs true algo)
  diffusion       -> denoiser predicts clean params x0 (everything but Algorithm)
  audio loss      -> assemble [x0 + TRUE algo] -> FROZEN renderer(note,vel) ->
                     magnitude-weighted spectral loss vs target  (THE DRIVER)

Total = w_audio * audio  +  w_diff * x0-MSE (diffusion / param nudge)  +  w_algo * CE.
The renderer's weights are frozen; gradients flow THROUGH it into the predicted params
(the differentiable tensor-in forward built in the Batch-5 refactor makes this possible).
We teacher-force the TRUE algorithm into both the denoiser conditioning and the render so
a misclassification can't poison the continuous matcher; the classifier learns alone.

Requires a trained renderer (models/operator/renderer.pt) + the dataset cache.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..schema import OperatorSchema
from ..synth.inspect_renderer import cfg_from_checkpoint
from ..synth.renderer import HybridRenderer
from ..synth.spectral import renderer_loss
from ..synth.train_renderer import RendererDataset, collate, hash_split, _repo_root
from .diffusion import Diffusion
from .model import Matcher


def load_frozen_renderer(ckpt_path: Path, schema: OperatorSchema, device: str):
    """Rebuild the renderer with its training-time config, load EMA weights, freeze."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cfg_from_checkpoint(ck)
    renderer = HybridRenderer(schema, cfg).to(device)
    renderer.load_state_dict(ck["state_dict"])
    renderer.eval()
    for p in renderer.parameters():
        p.requires_grad_(False)
    return renderer, cfg, ck.get("canon_fft", cfg.canon_fft)


@torch.no_grad()
def mean_params(loader, n_params, device):
    """Mean normalized param vector over the train set — the predict-the-mean baseline
    preset (render it at each target's pitch; the matcher must beat its audio loss)."""
    s, n = torch.zeros(n_params, device=device), 0
    for params, _, _, _ in loader:
        s += params.to(device).sum(0)
        n += params.shape[0]
    return s / max(n, 1)


@torch.no_grad()
def evaluate(matcher, diffusion, renderer, loader, device, canon, mean_p, steps, alpha,
             max_batches=0):
    """Real matcher metric: DDIM-sample params from the audio, render, score vs target —
    compared against rendering the mean-params preset. Also algo top-1 accuracy."""
    matcher.eval()
    match_l = base_l = algo_correct = n = 0.0
    for bi, (params, waves, notes, vels) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        waves = waves.to(device).float(); params = params.to(device)
        notes = notes.to(device); vels = vels.to(device)
        target = renderer.target_logmag(waves)
        z = matcher.encode(target[canon])
        algo_pred = matcher.algo_head(z).argmax(-1)
        true_algo = matcher.true_algo(params)
        algo_correct += (algo_pred == true_algo).float().sum().item()
        cond = matcher.cond(z, algo_pred, notes, vels)
        x0 = diffusion.ddim_sample(matcher.x0, (params.shape[0], matcher.dim), cond, device, steps)
        full = matcher.assemble(x0, algo_pred)
        match_l += renderer_loss(renderer(full, device, note=notes, velocity=vels), target, alpha).item() * params.shape[0]
        base_full = mean_p.unsqueeze(0).expand_as(params)
        base_l += renderer_loss(renderer(base_full, device, note=notes, velocity=vels), target, alpha).item() * params.shape[0]
        n += params.shape[0]
    return match_l / n, base_l / n, algo_correct / n


def main():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "dataset" / "operator"))
    ap.add_argument("--renderer", default=str(root / "models" / "operator" / "renderer.pt"))
    ap.add_argument("--out", default=str(root / "models" / "operator" / "matcher.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=10.0, help="loud-bin emphasis in the audio loss")
    ap.add_argument("--w-audio", type=float, default=1.0, help="audio-loss weight (the driver)")
    ap.add_argument("--w-diff", type=float, default=1.0, help="diffusion x0-MSE / param-nudge weight")
    ap.add_argument("--w-algo", type=float, default=1.0, help="algorithm cross-entropy weight")
    ap.add_argument("--steps", type=int, default=1000, help="diffusion train timesteps")
    ap.add_argument("--eval-steps", type=int, default=25, help="DDIM steps at eval")
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=16, help="val batches per eval (0 = all)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

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
    diffusion = Diffusion(args.steps).to(device)
    opt = torch.optim.AdamW(matcher.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(0, min(args.warmup_epochs, args.epochs - 1))
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
              torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - warmup))],
        milestones=[warmup]) if warmup else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    mean_p = mean_params(DataLoader(tr, batch_size=args.batch_size, **dl_kw),
                         len(schema.params), device)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{schema.summary()} | train={len(tr)} val={len(va)} device={device} "
          f"renderer canon_fft={canon} condition_pitch={rcfg.condition_pitch} "
          f"| denoiser dim={matcher.dim} (Algorithm split to classifier)")

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        matcher.train()
        run_a = run_d = run_c = nb = 0.0
        for params, waves, notes, vels in tr_dl:
            waves = waves.to(device, non_blocking=True).float()
            params = params.to(device, non_blocking=True)
            notes = notes.to(device); vels = vels.to(device)
            target = renderer.target_logmag(waves)
            true_algo = matcher.true_algo(params)
            x0_true = matcher.subset(params)                       # [B,194] in [-1,1]
            t = torch.randint(0, diffusion.n_steps, (params.shape[0],), device=device)
            noise = torch.randn_like(x0_true)
            x_t = diffusion.q_sample(x0_true, t, noise)
            with (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()):
                z = matcher.encode(target[canon])
                algo_logits = matcher.algo_head(z)
                cond = matcher.cond(z, true_algo, notes, vels)     # teacher-force true algo
                x0_hat = matcher.x0(x_t, t, cond)
                full = matcher.assemble(x0_hat, true_algo)
                preds = renderer(full, device, note=notes, velocity=vels)
            audio = renderer_loss({k: v.float() for k, v in preds.items()}, target, args.alpha)
            diff = F.mse_loss(x0_hat.float(), x0_true)
            ce = F.cross_entropy(algo_logits.float(), true_algo)
            loss = args.w_audio * audio + args.w_diff * diff + args.w_algo * ce
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(matcher.parameters(), args.clip)
            opt.step()
            run_a += audio.item(); run_d += diff.item(); run_c += ce.item(); nb += 1
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            match_l, base_l, algo_acc = evaluate(
                matcher, diffusion, renderer, va_dl, device, canon, mean_p,
                args.eval_steps, args.alpha, args.eval_batches)
            gain = 100 * (base_l - match_l) / base_l
            improved = match_l < best - 1e-4
            print(f"epoch {epoch:3} | train audio={run_a/max(nb,1):.4f} diff={run_d/max(nb,1):.4f} "
                  f"ce={run_c/max(nb,1):.4f} | val match={match_l:.4f} mean-base={base_l:.4f} "
                  f"(gain {gain:+.1f}%) algo_acc={algo_acc:.3f}{'  <- best' if improved else ''}")
            if improved:
                best = match_l
                torch.save({"state_dict": matcher.state_dict(), "epoch": epoch,
                            "best_match": best, "n_steps": args.steps,
                            "renderer_ckpt": str(Path(args.renderer).resolve())}, out)
        else:
            print(f"epoch {epoch:3} | train audio={run_a/max(nb,1):.4f} diff={run_d/max(nb,1):.4f} "
                  f"ce={run_c/max(nb,1):.4f}")

    print(f"Done. Best val match audio loss={best:.4f}. Saved matcher -> {out}")


if __name__ == "__main__":
    main()
