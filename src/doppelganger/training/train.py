"""
Train the SoundMatcher.

    uv run python -m doppelganger.training.train --epochs 30
    uv run python -m doppelganger.training.train --overfit 64   # sanity: should ~memorize

Saves the best checkpoint to models/operator/model.pt (+ the info ONNX export needs).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from ..schema import OperatorSchema
from .codec import ParamCodec
from .config import AudioConfig
from .data import OperatorDataset
from .model import SoundMatcher, compute_loss


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@torch.no_grad()
def evaluate(model: SoundMatcher, loader: DataLoader, device: str) -> dict:
    model.eval()
    n = 0
    cont_mae = bin_correct = bin_total = 0.0
    cat_correct = cat_total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["audio"])
        bs = batch["audio"].shape[0]
        n += bs
        cont_mae += (model.cont_values(out) - batch["cont"]).abs().mean().item() * bs
        bin_pred = (out["binary_logits"] > 0).float()
        bin_correct += (bin_pred == batch["binary"]).float().sum().item()
        bin_total += batch["binary"].numel()
        for i, logits in enumerate(out["cat_logits"]):
            cat_correct += (logits.argmax(1) == batch["cat"][:, i]).sum().item()
            cat_total += bs
    return {
        "cont_mae": cont_mae / n,
        "binary_acc": bin_correct / max(bin_total, 1),
        "cat_acc": cat_correct / max(cat_total, 1),
    }


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Train the Operator sound-matcher.")
    ap.add_argument("--data", default=str(root / "dataset" / "operator"))
    ap.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    ap.add_argument("--out", default=str(root / "models" / "operator"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--overfit", type=int, default=0, help="train+eval on first N samples (sanity)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--run-name", default="operator", help="TensorBoard run subfolder")
    args = ap.parse_args()

    audio = AudioConfig()
    schema = OperatorSchema.load(args.schema)
    codec = ParamCodec(schema)
    print(schema.summary(), "| device:", args.device)

    full = OperatorDataset(args.data, codec, audio)
    if args.overfit:
        train_ds = val_ds = Subset(full, list(range(min(args.overfit, len(full)))))
        print(f"OVERFIT mode on {len(train_ds)} samples")
    else:
        n_val = max(1, int(0.1 * len(full)))
        train_ds, val_ds = random_split(
            full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0)
        )
        print(f"train={len(train_ds)} val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
    # A fixed slice of train, evaluated each epoch, to watch the train-vs-val gap live.
    train_eval_dl = DataLoader(
        Subset(train_ds, list(range(min(512, len(train_ds))))), batch_size=args.batch_size
    )
    writer = SummaryWriter(str(_repo_root() / "runs" / args.run_name))

    model = SoundMatcher(
        audio, codec.n_cont, codec.n_binary, codec.cat_cardinalities
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(batch["audio"])
            loss, parts = compute_loss(out, batch, model.n_bins)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        val_m = evaluate(model, val_dl, args.device)
        tr_m = evaluate(model, train_eval_dl, args.device)
        avg = running / len(train_dl)

        writer.add_scalar("loss/train", avg, epoch)
        writer.add_scalar("lr", sched.get_last_lr()[0], epoch)
        for k in ("cont_mae", "binary_acc", "cat_acc"):
            writer.add_scalar(f"val/{k}", val_m[k], epoch)
            writer.add_scalar(f"train/{k}", tr_m[k], epoch)

        print(
            f"epoch {epoch:3} loss={avg:.4f} | "
            f"train cat_acc={tr_m['cat_acc']:.3f} | "
            f"val cont_mae={val_m['cont_mae']:.4f} bin_acc={val_m['binary_acc']:.3f} "
            f"cat_acc={val_m['cat_acc']:.3f}"
        )

        metrics = val_m
        score = metrics["cont_mae"]
        if score < best:
            best = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "n_cont": codec.n_cont,
                    "n_binary": codec.n_binary,
                    "cat_cardinalities": codec.cat_cardinalities,
                    "n_bins": model.n_bins,
                    "audio": asdict(audio),
                },
                out_dir / "model.pt",
            )
    writer.close()
    print(f"Done. Best val cont_mae={best:.4f}. Saved to {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
