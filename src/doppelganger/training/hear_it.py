"""
"Hear it" evaluation — the metric that actually matters.

Parameter accuracy is misleading (many param settings make near-identical sounds).
This re-renders the model's PREDICTED params in the real Operator and measures how
close the *audio* is to the target.

Workflow:
  1. uv run python -m doppelganger.training.hear_it predict --k 64
       -> predicts params for K held-out samples, writes dataset/predict/manifest.json
          and copies the target audio to dataset/predict/targets/.
  2. In Live: "doppelganger: Apply Predicted Batch", then export All Individual Tracks
     into dataset/predict/render/ (prefix e.g. "predict").  (or use export_ableton)
  3. uv run python -m doppelganger.training.hear_it compare
       -> log-mel distance (target vs predicted render), against a shuffled baseline.

A matched distance well below the shuffled baseline means the model is capturing the
sound — even if raw parameter accuracy looks low.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from ..schema import OperatorSchema
from .codec import ParamCodec
from .config import AudioConfig
from .data import OperatorDataset, _load_audio
from .features import LogMel
from .model import SoundMatcher


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_model(ckpt: str, device: str):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    audio = AudioConfig(**ck["audio"])
    model = SoundMatcher(
        audio, ck["n_cont"], ck["n_binary"], ck["cat_cardinalities"], n_bins=ck.get("n_bins", 64)
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, audio


@torch.no_grad()
def predict(args) -> None:
    root = _repo_root()
    device = args.device
    model, audio = _load_model(args.ckpt, device)
    codec = ParamCodec(OperatorSchema.load(args.schema))

    full = OperatorDataset(root / "dataset" / "operator", codec, audio)
    n_val = max(1, int(0.1 * len(full)))
    from torch.utils.data import random_split

    _, val_ds = random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0)
    )

    predict_dir = root / "dataset" / "predict"
    targets = predict_dir / "targets"
    targets.mkdir(parents=True, exist_ok=True)

    k = min(args.k, len(val_ds))
    tracks, mapping = {}, {}
    for j in range(k):
        item = val_ds[j]
        src_id = full.ids[val_ds.indices[j]]
        out = model(item["audio"].unsqueeze(0).to(device))
        cont = model.cont_values(out)[0].cpu().numpy()
        binp = torch.sigmoid(out["binary_logits"][0]).cpu().numpy()
        cat = np.array([logits[0].argmax().item() for logits in out["cat_logits"]])
        tname = f"op_{j:04d}"
        tracks[tname] = codec.decode(cont, binp, cat)
        mapping[tname] = src_id
        shutil.copy2(full.root / "wav" / f"{src_id}.wav", targets / f"{tname}.wav")

    (predict_dir / "manifest.json").write_text(
        json.dumps({"device": "Operator", "prefix": "predict", "tracks": tracks, "mapping": mapping}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote predictions for {k} samples -> {predict_dir/'manifest.json'}")
    print('Next: in Live run "Apply Predicted Batch", export into dataset/predict/render/, then `compare`.')


def _logmel(path: Path, logmel: LogMel, audio: AudioConfig) -> torch.Tensor:
    mono = _load_audio(path, audio.sample_rate, audio.n_samples)
    return logmel(torch.from_numpy(mono).unsqueeze(0))[0]  # [1, n_mels, frames]


def compare(args) -> None:
    root = _repo_root()
    predict_dir = root / "dataset" / "predict"
    manifest = json.loads((predict_dir / "manifest.json").read_text(encoding="utf-8"))
    mapping = manifest["mapping"]

    render_dir = predict_dir / "render"
    prefix = _detect_prefix(render_dir)
    if prefix is None:
        print(f"No 'op_*' renders found in {render_dir}. Export the predicted batch there first.")
        return

    audio = AudioConfig()
    logmel = LogMel(audio.sample_rate, audio.n_fft, audio.hop, audio.n_mels, audio.fmin, audio.fmax).eval()

    tgt_mels, pred_mels = {}, {}
    for tname in mapping:
        tgt = predict_dir / "targets" / f"{tname}.wav"
        pred = render_dir / f"{prefix} {tname}.wav"
        if tgt.exists() and pred.exists():
            tgt_mels[tname] = _logmel(tgt, logmel, audio)
            pred_mels[tname] = _logmel(pred, logmel, audio)

    names = list(tgt_mels)
    if not names:
        print("No matching target/render pairs found.")
        return

    matched = [float((tgt_mels[n] - pred_mels[n]).abs().mean()) for n in names]
    # shuffled baseline: target_i vs a different render
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(names))
    shuffled = [
        float((tgt_mels[names[i]] - pred_mels[names[perm[i]]]).abs().mean())
        for i in range(len(names))
    ]
    print(f"pairs={len(names)}")
    print(f"  matched   log-mel L1: {np.mean(matched):.4f}")
    print(f"  shuffled  log-mel L1: {np.mean(shuffled):.4f}  (baseline; higher = model captures sound)")
    print(f"  ratio matched/shuffled: {np.mean(matched)/np.mean(shuffled):.3f}  (<1 is good)")


def _detect_prefix(folder: Path) -> str | None:
    import re

    pat = re.compile(r"^(.*) op_\d+\.wav$")
    counts: dict[str, int] = {}
    if not folder.exists():
        return None
    for wav in folder.glob("*.wav"):
        m = pat.match(wav.name)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return max(counts, key=counts.get) if counts else None


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(root / "models" / "operator" / "model.pt"))
    ap.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--k", type=int, default=64)
    p.set_defaults(func=predict)
    c = sub.add_parser("compare")
    c.set_defaults(func=compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
