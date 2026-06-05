"""
Per-parameter diagnostic: is the model learning the *audible* params (waveform,
filter, algorithm) or nothing? Reports train vs val accuracy per categorical head,
next to the random baseline, so we can tell "needs more data" (train>>val on
learnable heads) from "something's broken" (even train near random on audible heads).

    uv run python -m doppelganger.training.diagnose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from ..schema import OperatorSchema
from .codec import ParamCodec
from .config import AudioConfig
from .data import OperatorDataset
from .model import SoundMatcher


@torch.no_grad()
def cat_head_accuracy(model, loader, device, n_heads):
    model.eval()
    correct = torch.zeros(n_heads)
    total = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        out = model(audio)
        for i, logits in enumerate(out["cat_logits"]):
            correct[i] += (logits.argmax(1).cpu() == batch["cat"][:, i]).sum().item()
        total += audio.shape[0]
    return (correct / total).tolist()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(root / "dataset" / "operator"))
    ap.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    ap.add_argument("--ckpt", default=str(root / "models" / "operator" / "model.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    audio = AudioConfig(**ck["audio"])
    codec = ParamCodec(OperatorSchema.load(args.schema))

    full = OperatorDataset(args.data, codec, audio)
    n_val = max(1, int(0.1 * len(full)))
    train_ds, val_ds = random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0)
    )

    model = SoundMatcher(
        audio, ck["n_cont"], ck["n_binary"], ck["cat_cardinalities"], n_bins=ck.get("n_bins", 64)
    ).to(args.device)
    model.load_state_dict(ck["state_dict"])

    tr = cat_head_accuracy(model, DataLoader(train_ds, batch_size=128), args.device, len(codec.categorical))
    va = cat_head_accuracy(model, DataLoader(val_ds, batch_size=128), args.device, len(codec.categorical))

    print(f"{'param':24} {'card':>4} {'rand':>6} {'train':>7} {'val':>7}")
    rows = []
    for i, p in enumerate(codec.categorical):
        card = p.cardinality or 1
        rows.append((p.name, card, 1.0 / card, tr[i], va[i]))
    for name, card, rand, t, v in sorted(rows, key=lambda r: -r[4]):
        print(f"{name:24} {card:>4} {rand:>6.3f} {t:>7.3f} {v:>7.3f}")


if __name__ == "__main__":
    main()
