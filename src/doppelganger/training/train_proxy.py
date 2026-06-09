"""
Train the neural proxy: params -> audio embedding (imitate Operator).

    uv run python -m doppelganger.training.train_proxy --epochs 60

Unlike the matcher, this direction is a deterministic function (params fully
determine the sound), so it should show a normal, healthy training curve. The
trained proxy (models/operator/proxy.pt) is then frozen and used to give the
matcher a perceptual loss (train.py --proxy).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from ..schema import OperatorSchema
from .codec import ParamCodec
from .config import AudioConfig
from .data import OperatorDataset
from .proxy import MelEmbedding, SynthProxy, gt_param_vector


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@torch.no_grad()
def _eval(proxy, mel, loader, cards, device) -> float:
    proxy.eval()
    err = n = 0.0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        pv = gt_param_vector(b["cont"], b["binary"], b["cat"], cards)
        pred = proxy(pv)
        target = mel(b["audio"])
        err += F.l1_loss(pred, target, reduction="sum").item()
        n += target.numel()
    return err / max(n, 1)


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Train the synth proxy (params -> audio embedding).")
    ap.add_argument("--data", default=str(root / "dataset" / "operator"))
    ap.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    ap.add_argument("--out", default=str(root / "models" / "operator" / "proxy.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    audio = AudioConfig()
    codec = ParamCodec(OperatorSchema.load(args.schema))
    cards = codec.cat_cardinalities

    full = OperatorDataset(args.data, codec, audio)
    print(f"caching {len(full)} clips…")
    full.precompute()
    n_val = max(1, int(0.1 * len(full)))
    tr, va = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    pin = args.device == "cuda"
    tr_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    va_dl = DataLoader(va, batch_size=args.batch_size, pin_memory=pin)

    mel = MelEmbedding(audio).to(args.device).eval()
    proxy = SynthProxy(codec.n_cont, codec.n_binary, cards, embed_dim=mel.dim).to(args.device)
    opt = torch.optim.Adam(proxy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    writer = SummaryWriter(str(root / "runs" / "proxy"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        proxy.train()
        running = 0.0
        for b in tr_dl:
            b = {k: v.to(args.device) for k, v in b.items()}
            pv = gt_param_vector(b["cont"], b["binary"], b["cat"], cards)
            with torch.no_grad():
                target = mel(b["audio"])
            loss = F.l1_loss(proxy(pv), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        val = _eval(proxy, mel, va_dl, cards, args.device)
        train = running / len(tr_dl)
        writer.add_scalar("proxy/train_l1", train, epoch)
        writer.add_scalar("proxy/val_l1", val, epoch)
        print(f"epoch {epoch:3} train_l1={train:.4f} val_l1={val:.4f}")
        torch.save(
            {
                "state_dict": proxy.state_dict(),
                "n_cont": codec.n_cont,
                "n_binary": codec.n_binary,
                "cat_cardinalities": cards,
                "embed_dim": mel.dim,
                "audio": asdict(audio),
            },
            out,
        )
    writer.close()
    print(f"Done. Saved proxy to {out}")


if __name__ == "__main__":
    main()
