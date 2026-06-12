"""
Train + validate the HybridRenderer (params -> Operator spectrogram).

    uv run python -m doppelganger.synth.train_renderer --epochs 60

Validation reports the held-out log-mag L1 for the full hybrid vs a physics-only
baseline — if the hybrid is clearly lower, the neural residual is earning its keep
(and the renderer is faithful enough to provide the matcher's audio loss).
Requires dataset/operator (regenerate via Auto Collect first).

Batch-5 refactor notes (full reasoning in docs/batch5-refactor.md):
- Batches are TENSORS end-to-end (normalized param matrix from the v2 cache + note +
  velocity) — the per-step Python dict loops are gone, and the same code path is the
  differentiable one the matcher will use.
- Train/val membership is a PER-ID HASH (crc32(id) % 10 == 0 -> val), not a seeded
  random_split of the current length: adding new data no longer reshuffles membership
  (old val samples were silently becoming train samples), so fidelity numbers stay
  comparable as the dataset grows. One-time discontinuity vs the old split.
- Pitch guard: with --no-condition-pitch (or an old cfg) the dataset FILTERS to note==60
  — training a fixed-f0 renderer on pitch-diverse targets would be silent label noise.
- mean_spec is reused from the checkpoint on --resume (it was recomputed over the full
  train set at every launch); eval renders the physics ONCE per batch (return_physics).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import zlib
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from ..schema import OperatorSchema
from ..training.data import _load_audio
from .adapter import normalize_params
from .precompute import CACHE_VERSION
from .renderer import HybridRenderer, RendererConfig
from .spectral import renderer_loss


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class RendererDataset(Dataset):
    """Yields (normalized params [P] fp32, waveform [T] fp16, note, velocity).

    Fast path: the v2 packed cache (synth.precompute) — audio memmapped fp16 (one copy
    shared across DDP ranks via the OS page cache), params as a single fp32 matrix.
    Falls back to live JSON/WAV loading if the cache is absent or stale (version /
    count / n_samples mismatch).

    Waveforms stay fp16 until they reach the GPU (halves host->device traffic); the
    trainer casts to fp32 there."""

    def __init__(self, root: Path, cfg: RendererConfig, schema: OperatorSchema):
        self.root = root
        self.cfg = cfg
        self.schema = schema
        self.audio = self.params = self.notes = self.vels = None
        ids_on_disk = sorted(p.stem for p in (root / "params").glob("*.json"))
        cache = root / "cache"
        meta_p = cache / "meta.json"
        if meta_p.exists() and (cache / "audio.npy").exists():
            meta = json.loads(meta_p.read_text())
            fresh = (meta.get("version") == CACHE_VERSION
                     and len(meta["ids"]) == len(ids_on_disk)
                     and meta.get("n_samples") == cfg.n_samples)
            if fresh:
                self.ids = meta["ids"]
                self.audio = np.load(cache / "audio.npy", mmap_mode="r")
                self.params = np.load(cache / "params.npy")
                self.notes = np.load(cache / "notes.npy")
                self.vels = np.load(cache / "vels.npy")
            else:
                print("[dataset] cache stale (version / sample count / n_samples changed) — "
                      "live loading; rerun `python -m doppelganger.synth.precompute`")
        if self.audio is None:  # live fallback
            self.ids = ids_on_disk
            self._mem: dict[int, tuple] = {}
        if not self.ids:
            raise RuntimeError(f"No params under {root/'params'} — regenerate the dataset.")
        self.positions = list(range(len(self.ids)))

    def restrict_to_note(self, note: int = 60) -> None:
        """Keep only samples played at `note`. REQUIRED when the model isn't pitch-
        conditioned: a fixed-f0 renderer trained on pitch-diverse targets just learns
        noise for the off-pitch samples (they'd silently poison the run)."""
        before = len(self.positions)
        self.positions = [i for i in self.positions if self._note(i) == note]
        dropped = before - len(self.positions)
        if dropped:
            print(f"[dataset] pitch guard: model is not pitch-conditioned -> kept "
                  f"{len(self.positions)} note={note} samples, dropped {dropped} pitch-diverse")

    def _note(self, i: int) -> int:
        if self.notes is not None:
            return int(self.notes[i])
        d = json.loads((self.root / "params" / f"{self.ids[i]}.json").read_text())
        return int(d.get("note", 60))

    def limit(self, n: int) -> None:
        self.positions = self.positions[:n]

    def id_at(self, idx: int) -> str:
        return self.ids[self.positions[idx]]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i = self.positions[idx]
        if self.audio is not None:  # cached fast path — pure array indexing
            return (torch.from_numpy(self.params[i]),
                    torch.from_numpy(self.audio[i].copy()),            # fp16; copy off the
                    float(self.notes[i]), float(self.vels[i]))         # read-only memmap
        if i not in self._mem:  # live fallback: parse once, keep in RAM (fp16 wave)
            sid = self.ids[i]
            d = json.loads((self.root / "params" / f"{sid}.json").read_text())
            vec = normalize_params([d["params"]], self.schema)[0]
            wave = torch.from_numpy(
                _load_audio(self.root / "wav" / f"{sid}.wav",
                            self.cfg.sample_rate, self.cfg.n_samples)).half()
            self._mem[i] = (vec, wave, float(d.get("note", 60)), float(d.get("velocity", 100)))
        return self._mem[i]


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.tensor([b[2] for b in batch]),
            torch.tensor([b[3] for b in batch]))


def hash_split(ds: RendererDataset, val_mod: int = 10):
    """Deterministic per-id split: crc32(id) % val_mod == 0 -> val (~10%).

    Unlike the old seeded random_split (which re-dealt membership whenever the dataset
    LENGTH changed), an id's membership never changes when new data is added — held-out
    samples stay held-out forever, keeping fidelity metrics comparable across dataset
    versions."""
    val_idx = [k for k in range(len(ds)) if zlib.crc32(ds.id_at(k).encode()) % val_mod == 0]
    val_set = set(val_idx)
    tr_idx = [k for k in range(len(ds)) if k not in val_set]
    return Subset(ds, tr_idx), Subset(ds, val_idx)


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
    for _, waves, _, _ in loader:
        t = model.target_logmag(waves.to(device).float())[canon]
        s = t.sum(0) if s is None else s + t.sum(0)
        n += t.shape[0]
    return s / n  # [F, T]


@torch.no_grad()
def evaluate(model, loader, device, mean_spec, canon: int, max_samples: int = 0):
    """At the canonical resolution: plain L1 (hyb/phys/mean) for log continuity, plus the
    energy-weighted L1 for hybrid and mean — the latter gates the renderer (the matcher
    needs the loud bins right; hyb_e must drop clearly below mean_e). max_samples>0 caps it.
    One forward per batch: hybrid + physics predictions share a single physics render."""
    model.eval()
    hyb = phys = mean = n = 0.0
    e_num_h = e_num_m = e_den = 0.0
    for params, waves, notes, vels in loader:
        if max_samples and n >= max_samples:
            break
        t = model.target_logmag(waves.to(device).float())[canon]
        ph, pp = model(params.to(device), device, note=notes, velocity=vels,
                       return_physics=True)
        ph, pp = ph[canon], pp[canon]
        pm = mean_spec.expand_as(t)
        bs = params.shape[0]
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
    ap.add_argument("--workers", type=int, default=4, help="DataLoader worker processes per rank")
    ap.add_argument("--limit", type=int, default=0, help="cap dataset size (0 = all) for quick dev runs")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip (0 = off); stabilizes bf16 + Transformer")
    ap.add_argument("--warmup-epochs", type=int, default=5, help="linear LR warmup before cosine (Transformers need it)")
    ap.add_argument("--alpha", type=float, default=10.0, help="loud-bin emphasis in the loss")
    ap.add_argument("--encoder", choices=["transformer", "mlp"], default=None,
                    help="conditioning encoder (default: RendererConfig's 'transformer'); 'mlp' for ablation")
    ap.add_argument("--oversample", type=int, default=None, help="physics anti-alias factor (override cfg; 1 = off)")
    ap.add_argument("--condition-pitch", action=argparse.BooleanOptionalAction, default=True,
                    help="condition on note+velocity (Batch 5). --no-condition-pitch filters to C3 data")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                    help="bf16 mixed precision on CUDA (--no-amp to disable)")
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True,
                    help="gradient checkpointing (saves VRAM; --no-grad-ckpt is faster if it fits)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (experimental)")
    ap.add_argument("--eval-every", type=int, default=2, help="run validation every N epochs")
    ap.add_argument("--eval-samples", type=int, default=0, help="cap val samples per eval (0 = full)")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--patience", type=int, default=10, help="early-stop after N evals without val gain")
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
    torch.backends.cuda.matmul.allow_tf32 = True   # free Ampere speedup for fp32 matmuls
    torch.backends.cudnn.allow_tf32 = True
    use_amp = bool(args.amp) and device.startswith("cuda")  # bf16 autocast
    cfg = RendererConfig()
    if args.encoder:
        cfg = dataclasses.replace(cfg, encoder=args.encoder)
    if args.oversample is not None:
        cfg = dataclasses.replace(cfg, oversample=args.oversample)
    cfg = dataclasses.replace(cfg, grad_checkpoint=args.grad_ckpt,
                              condition_pitch=args.condition_pitch)
    canon = cfg.canon_fft
    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    ds = RendererDataset(Path(args.data), cfg, schema)
    if args.limit:  # limit BEFORE the guard — the guard reads every sample's note
        ds.limit(args.limit)
    if not cfg.condition_pitch:
        ds.restrict_to_note(60)  # pitch guard — see RendererDataset.restrict_to_note
    # stable per-id hash split (same on every rank; survives dataset growth)
    tr, va = hash_split(ds)

    dl_kw = dict(collate_fn=collate, num_workers=args.workers,
                 pin_memory=device.startswith("cuda"), persistent_workers=args.workers > 0)
    if distributed:
        tr_sampler = DistributedSampler(tr, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        tr_dl = DataLoader(tr, batch_size=args.batch_size, sampler=tr_sampler, **dl_kw)
    else:
        tr_sampler = None
        tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, **dl_kw)

    model = HybridRenderer(schema, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(0, min(args.warmup_epochs, args.epochs - 1))
    if warmup:  # linear warmup -> cosine: the Transformer encoder diverges without it
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - warmup)],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(args.out) if args.out else root / "models" / "operator" / "renderer.pt"

    start_epoch, best_val, since_best = 1, float("inf"), 0
    ck = None
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
    if args.compile:  # experimental: fuses the residual/encoder; eval still uses uncompiled `model`
        net = torch.compile(net)

    # EMA + the predict-the-mean baseline + eval all live on rank 0 only.
    ema = mean_spec = va_dl = None
    if is_main:
        out.parent.mkdir(parents=True, exist_ok=True)
        ema = Ema(model, args.ema_decay)
        if ck is not None and "ema" in ck:
            ema.load(ck["ema"])
        va_dl = DataLoader(va, batch_size=args.batch_size, **dl_kw)
        if ck is not None and "mean_spec" in ck and ck.get("canon_fft") == canon:
            mean_spec = ck["mean_spec"].to(device)  # reuse: a full train-set pass per launch
        else:
            mean_spec = mean_spectrogram(model, DataLoader(tr, batch_size=args.batch_size, **dl_kw),
                                         device, canon)
        print(f"{schema.summary()} | train={len(tr)} val={len(va)} device={device} "
              f"world_size={world_size} backend={_ddp_backend() if distributed else 'single'} "
              f"| effective batch={args.batch_size * world_size}")
        print(f"  encoder={cfg.encoder} ch={cfg.ch} blocks={cfg.n_blocks} oversample={cfg.oversample} "
              f"condition_pitch={cfg.condition_pitch} amp={'bf16' if use_amp else 'off'} "
              f"grad_ckpt={cfg.grad_checkpoint} tf32=on")

    for epoch in range(start_epoch, args.epochs + 1):
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)  # reshuffle shards each epoch
        net.train()
        run = nb = 0.0
        for params, waves, notes, vels in tr_dl:
            waves = waves.to(device, non_blocking=True).float()  # fp16 on the wire, fp32 on GPU
            params = params.to(device, non_blocking=True)
            target = model.target_logmag(waves)                  # fp32 reference
            with (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()):
                preds = net(params, device, note=notes, velocity=vels)  # heavy ops in bf16
            loss = renderer_loss({k: v.float() for k, v in preds.items()}, target, args.alpha)  # loss in fp32
            # NaN guard must agree across ranks (skipping on one rank only would deadlock DDP)
            bad = torch.tensor([0.0 if torch.isfinite(loss) else 1.0], device=device)
            if distributed:
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item() > 0:
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            if is_main:
                ema.update(model)
            run += loss.item(); nb += 1
        sched.step()

        # eval only every N epochs (always on the last) — eval is single-GPU and idles the
        # other rank, so this is a real wall-clock win
        do_eval = is_main and (epoch % args.eval_every == 0 or epoch == args.epochs)
        stop = torch.zeros(1, device=device)
        if do_eval:
            with ema_weights(model, ema):  # evaluate (and deploy) the averaged weights
                hyb, phys, mean, hyb_e, mean_e = evaluate(
                    model, va_dl, device, mean_spec, canon, args.eval_samples)
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
                    print(f"Early stop: no val gain for {args.patience} eval(s).")
                    stop[0] = 1.0
        elif is_main:
            print(f"epoch {epoch:3} train={run/max(nb,1):.4f} | (no eval; every {args.eval_every})")
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
