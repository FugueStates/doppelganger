"""
Evaluate a saved checkpoint on the same train/val split used in training, reporting
the train-vs-val gap (so we can tell "needs more data" from "needs more capacity").

    uv run python -m doppelganger.training.eval
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
from .train import evaluate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = _repo_root()
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

    for name, ds in (("train", train_ds), ("val", val_ds)):
        dl = DataLoader(ds, batch_size=64)
        m = evaluate(model, dl, args.device)
        print(
            f"{name:5}: cont_mae={m['cont_mae']:.4f}  bin_acc={m['binary_acc']:.3f}  "
            f"cat_acc={m['cat_acc']:.3f}  (n={len(ds)})"
        )

    # Context: how good is "just predict the average preset"?
    cont_stack = torch.stack([full[i]["cont"] for i in range(min(1000, len(full)))])
    mean_mae = (cont_stack - cont_stack.mean(0)).abs().mean().item()
    print(f"baseline cont_mae if always predicting the mean: {mean_mae:.4f}")


if __name__ == "__main__":
    main()
