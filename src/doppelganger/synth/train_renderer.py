"""
Train + validate the HybridRenderer (params -> Operator spectrogram).

    uv run python -m doppelganger.synth.train_renderer --epochs 60

Validation reports the held-out log-mag L1 for the full hybrid vs a physics-only
baseline — if the hybrid is clearly lower, the neural residual is earning its keep
(and the renderer is faithful enough to provide the matcher's audio loss).
Requires dataset/operator (regenerate via Auto Collect first).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler

from ..schema import OperatorSchema
from ..training.data import _load_audio
from .renderer import HybridRenderer, RendererConfig
from .spectral import renderer_loss


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class RendererDataset(Dataset):
    """Yields (raw param dict, waveform) from dataset/operator (params + wav)."""

    def __init__(self, root: Path, cfg: RendererConfig):
        self.root = root
        self.cfg = cfg
        self.ids = sorted(p.stem for p in (root / "params").glob("*.json"))
        if not self.ids:
            raise RuntimeError(f"No params under {root/'params'} — regenerate the dataset.")
        self._wav: dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        params = json.loads((self.root / "params" / f"{sid}.json").read_text())["params"]
        if i not in self._wav:
            self._wav[i] = torch.from_numpy(
                _load_audio(self.root / "wav" / f"{sid}.wav", self.cfg.sample_rate, self.cfg.n_samples))
        return params, self._wav[i]


def collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


class Ema:
    """Exponential moving average of weights — the SoTA-standard stabilizer. The averaged
    weights generalize better and are what we deploy (saved as the checkpoint's state_dict)."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def load(self, shadow):
        for k in self.shadow:
            if k in shadow:
                self.shadow[k].copy_(shadow[k].to(self.shadow[k].device))


@contextmanager
def ema_weights(model, ema: Ema):
    """Temporarily swap the EMA weights into the live model (for eval / saving)."""
    msd = model.state_dict()
    backup = {k: msd[k].detach().clone() for k in ema.shadow}
    for k in ema.shadow:
        msd[k].copy_(ema.shadow[k])
    try:
        yield
    finally:
        for k, v in backup.items():
            msd[k].copy_(v)


@torch.no_grad()
def mean_spectrogram(model, loader, device, canon: int):
    """The trivial predict-the-mean baseline: average target log-mag (at the canonical
    resolution) over the train set. If the hybrid can't clearly beat THIS, it's learning
    nothing useful (cf. the old proxy that lost to predict-mean and was worthless)."""
    s, n = None, 0
    for _, waves in loader:
        t = model.target_logmag(waves.to(device))[canon]
        s = t.sum(0) if s is None else s + t.sum(0)
        n += t.shape[0]
    return s / n  # [F, T]


@torch.no_grad()
def evaluate(model, loader, device, mean_spec, canon: int):
    """At the canonical resolution: plain L1 (hyb/phys/mean) for log continuity, plus the
    energy-weighted L1 for hybrid and mean — the latter gates the renderer (the matcher
    needs the loud bins right; hyb_e must drop clearly below mean_e)."""
    model.eval()
    hyb = phys = mean = n = 0.0
    e_num_h = e_num_m = e_den = 0.0
    for dicts, waves in loader:
        t = model.target_logmag(waves.to(device))[canon]
        ph = model(dicts, device, apply_residual=True)[canon]
        pp = model(dicts, device, apply_residual=False)[canon]
        pm = mean_spec.expand_as(t)
        bs = len(dicts)
        hyb += F.l1_loss(ph, t).item() * bs
        phys += F.l1_loss(pp, t).item() * bs
        mean += F.l1_loss(pm, t).item() * bs
        w = t.exp()
        e_num_h += ((ph - t).abs() * w).sum().item()
        e_num_m += ((pm - t).abs() * w).sum().item()
        e_den += w.sum().item()
        n += bs
    return hyb / n, phys / n, mean / n, e_num_h / e_den, e_num_m / e_den


def _ddp_backend() -> str:
    """NCCL on Linux (uses NVLink for all-reduce); gloo elsewhere (e.g. Windows)."""
    if platform.system() != "Windows" and dist.is_nccl_available():
        return "nccl"
    return "gloo"


def _parse_args():
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "dataset" / "operator"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16, help="PER-GPU batch size")
    ap.add_argument("--limit", type=int, default=0, help="cap dataset size (0 = all) for quick dev runs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=10.0, help="loud-bin emphasis in the loss")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--patience", type=int, default=10, help="early-stop after N epochs without val gain")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs to use (0 = auto-detect all)")
    ap.add_argument("--port", type=int, default=29500, help="DDP rendezvous port")
    ap.add_argument("--resume", action="store_true", help="continue from the saved checkpoint")
    ap.add_argument("--out", default=None, help="checkpoint path (default models/operator/renderer.pt)")
    return ap.parse_args()


def worker(rank: int, world_size: int, args):
    """Training body for one process. world_size>1 => DDP (one process per GPU)."""
    distributed = world_size > 1
    is_main = rank == 0
    root = _repo_root()
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(args.port))
        dist.init_process_group(_ddp_backend(), rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = args.device

    torch.backends.cudnn.benchmark = True
    cfg = RendererConfig()
    canon = cfg.canon_fft
    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    ds = RendererDataset(Path(args.data), cfg)
    if args.limit:
        ds.ids = ds.ids[:args.limit]
    n_val = max(1, int(0.1 * len(ds)))
    # identical split on every rank (same seed) so the val set is consistent
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(0))

    if distributed:
        tr_sampler = DistributedSampler(tr, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        tr_dl = DataLoader(tr, batch_size=args.batch_size, sampler=tr_sampler, collate_fn=collate)
    else:
        tr_sampler = None
        tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = HybridRenderer(schema, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(args.out) if args.out else root / "models" / "operator" / "renderer.pt"

    start_epoch, best_val, since_best = 1, float("inf"), 0
    if args.resume and out.exists():
        # ALL ranks restore weights + optimizer state, so they stay in lockstep after resume.
        ck = torch.load(out, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("raw_state_dict", ck["state_dict"]))
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start_epoch = ck["epoch"] + 1
            best_val = ck.get("best_val", float("inf"))
        if is_main:
            print(f"resumed from epoch {start_epoch - 1} (best_val={best_val:.4f})")

    # DDP wraps AFTER any resume-load; its constructor broadcasts rank-0 weights to all ranks.
    net = DDP(model, device_ids=[rank]) if distributed else model

    # EMA + the predict-the-mean baseline + eval all live on rank 0 only.
    ema = mean_spec = va_dl = None
    if is_main:
        out.parent.mkdir(parents=True, exist_ok=True)
        ema = Ema(model, args.ema_decay)
        if args.resume and out.exists() and "ema" in ck:
            ema.load(ck["ema"])
        va_dl = DataLoader(va, batch_size=args.batch_size, collate_fn=collate)
        mean_spec = mean_spectrogram(model, DataLoader(tr, batch_size=args.batch_size,
                                                       collate_fn=collate), device, canon)
        print(f"{schema.summary()} | train={len(tr)} val={len(va)} device={device} "
              f"world_size={world_size} backend={_ddp_backend() if distributed else 'single'} "
              f"| effective batch={args.batch_size * world_size}")

    for epoch in range(start_epoch, args.epochs + 1):
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)  # reshuffle shards each epoch
        net.train()
        run = nb = 0.0
        for dicts, waves in tr_dl:
            target = model.target_logmag(waves.to(device))
            loss = renderer_loss(net(dicts, device), target, args.alpha)
            # NaN guard must agree across ranks (skipping on one rank only would deadlock DDP)
            bad = torch.tensor([0.0 if torch.isfinite(loss) else 1.0], device=device)
            if distributed:
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item() > 0:
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            if is_main:
                ema.update(model)
            run += loss.item(); nb += 1
        sched.step()

        stop = torch.zeros(1, device=device)
        if is_main:
            with ema_weights(model, ema):  # evaluate (and deploy) the averaged weights
                hyb, phys, mean, hyb_e, mean_e = evaluate(model, va_dl, device, mean_spec, canon)
            improved = hyb_e < best_val - 1e-4  # select on the energy-weighted (loud-bin) metric
            print(f"epoch {epoch:3} train={run/max(nb,1):.4f} | L1 hyb={hyb:.4f} phys={phys:.4f} "
                  f"mean={mean:.4f} | energy-wt hyb={hyb_e:.4f} mean={mean_e:.4f}"
                  f"{'  <- best' if improved else ''}")
            if improved:
                best_val, since_best = hyb_e, 0
                # deploy = EMA weights overlaid on the full state dict (keeps non-float buffers).
                deploy_sd = {k: (ema.shadow[k] if k in ema.shadow else v).cpu()
                             for k, v in model.state_dict().items()}
                torch.save({"state_dict": deploy_sd, "raw_state_dict": model.state_dict(),
                            "ema": ema.shadow, "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "epoch": epoch, "best_val": best_val, "mean_spec": mean_spec.cpu(),
                            "canon_fft": canon, "cfg": cfg.__dict__}, out)
            else:
                since_best += 1
                if since_best >= args.patience:
                    print(f"Early stop: no val gain for {args.patience} epochs.")
                    stop[0] = 1.0
        if distributed:  # all ranks must learn rank-0's early-stop decision in lockstep
            dist.broadcast(stop, src=0)
        if stop.item() > 0:
            break

    if is_main:
        print(f"Done. Best val energy-wt hybrid={best_val:.4f}. Saved renderer -> {out}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def main():
    args = _parse_args()
    # auto-detect GPUs; one DDP process per GPU. Single-GPU/CPU falls through to one process.
    n_gpu = args.gpus or (torch.cuda.device_count() if args.device.startswith("cuda") else 0)
    if n_gpu > 1:
        print(f"Launching DDP across {n_gpu} GPUs ({_ddp_backend()})...")
        mp.spawn(worker, args=(n_gpu, args), nprocs=n_gpu)
    else:
        worker(0, 1, args)


if __name__ == "__main__":
    main()
