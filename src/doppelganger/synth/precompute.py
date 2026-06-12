"""
Pack dataset/operator into a fast training cache (v2) so the renderer trainer avoids
re-reading ~100k WAVs/JSONs every epoch.

    uv run python -m doppelganger.synth.precompute            # full dataset
    uv run python -m doppelganger.synth.precompute --limit 64 # quick test cache

Writes dataset/operator/cache/:
  - audio.npy  : [N, n_samples] float16, peak-normalized audio (memmapped at train time —
                 halves RAM vs fp32 and the OS page cache shares ONE copy across DDP ranks)
  - params.npy : [N, n_params] float32, NORMALIZED param matrix (the shared encoding:
                 conditioning-encoder input, ControlMap input, future matcher target)
  - notes.npy  : [N] int16 MIDI note per sample (legacy C3 batches -> 60)
  - vels.npy   : [N] int16 velocity per sample (legacy -> 100)
  - meta.json  : {ids, n_samples, sample_rate, n_params, version}

WHY v2 (was: params as raw dicts inside meta.json): at ~100k samples that meta.json
would be ~400+ MB of JSON parsed into Python dicts and held by EVERY process (2 DDP
ranks x (main + N workers) on Windows spawn = ~10 copies, gigabytes of RAM), and the
trainer then re-ran a Python double-loop (~B x 195 lookups) on every step to normalize
them. The fp32 matrix is 195 floats/sample (~45 MB at 58k), loads instantly, and the
per-step Python normalization is gone — batches go straight to the GPU as tensors.

RERUN after adding data — the trainer falls back to slow live-loading if the cache
count/version doesn't match (so it never trains on a stale cache silently).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..schema import OperatorSchema
from ..training.data import _load_audio
from .adapter import normalize_params
from .renderer import RendererConfig

CACHE_VERSION = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_cache(root: Path, cfg: RendererConfig, schema: OperatorSchema, limit: int = 0) -> int:
    params_dir, wav_dir = root / "params", root / "wav"
    ids = sorted(p.stem for p in params_dir.glob("*.json"))
    if not ids:
        raise RuntimeError(f"No params under {params_dir} — regenerate the dataset.")
    if limit:
        ids = ids[:limit]
    n, n_samples = len(ids), cfg.n_samples

    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    audio = np.lib.format.open_memmap(
        cache / "audio.npy", mode="w+", dtype=np.float16, shape=(n, n_samples))
    params = np.zeros((n, len(schema.params)), dtype=np.float32)
    notes = np.zeros(n, dtype=np.int16)
    vels = np.zeros(n, dtype=np.int16)
    for i, sid in enumerate(ids):
        d = json.loads((params_dir / f"{sid}.json").read_text())
        params[i] = normalize_params([d["params"]], schema)[0].numpy()
        notes[i] = int(d.get("note", 60))        # legacy C3 samples have no note -> 60
        vels[i] = int(d.get("velocity", 100))
        audio[i] = _load_audio(wav_dir / f"{sid}.wav", cfg.sample_rate, n_samples).astype(np.float16)
        if (i + 1) % 2000 == 0:
            print(f"  packed {i + 1}/{n}")
    audio.flush()
    np.save(cache / "params.npy", params)
    np.save(cache / "notes.npy", notes)
    np.save(cache / "vels.npy", vels)
    (cache / "meta.json").write_text(json.dumps(
        {"version": CACHE_VERSION, "ids": ids, "n_samples": n_samples,
         "sample_rate": cfg.sample_rate, "n_params": len(schema.params)}))
    n60 = int((notes == 60).sum())
    print(f"Wrote cache v{CACHE_VERSION} for {n} samples -> {cache} "
          f"({n60} at C3, {n - n60} pitch-diverse)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(_repo_root() / "dataset" / "operator"))
    ap.add_argument("--schema", default=str(_repo_root() / "schemas" / "operator.json"))
    ap.add_argument("--limit", type=int, default=0, help="cap samples (for a quick test cache)")
    args = ap.parse_args()
    build_cache(Path(args.data), RendererConfig(), OperatorSchema.load(args.schema), args.limit)


if __name__ == "__main__":
    main()
