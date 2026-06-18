"""
Dataset for the v3 matcher: (audio waveform, codec-encoded param targets + gating masks).
Small datasets (the sniff test) are cached in RAM. Train/val split is a stable per-id hash
so adding data never reshuffles membership.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from ..training.data import _load_audio
from .codec import ParamCodec
from .model import MatcherConfig


class MatcherDataset(Dataset):
    def __init__(self, root: str | Path, codec: ParamCodec, cfg: MatcherConfig):
        self.root = Path(root)
        self.codec = codec
        self.cfg = cfg
        self.ids = sorted(p.stem for p in (self.root / "params").glob("*.json"))
        if not self.ids:
            raise RuntimeError(f"No params under {self.root/'params'} — collect data first.")
        self._cache: dict[str, tuple] = {}

    def __len__(self):
        return len(self.ids)

    def id_at(self, i: int) -> str:
        return self.ids[i]

    def __getitem__(self, i: int):
        sid = self.ids[i]
        if sid not in self._cache:
            params = json.loads((self.root / "params" / f"{sid}.json").read_text())["params"]
            audio = _load_audio(self.root / "wav" / f"{sid}.wav",
                                self.cfg.sample_rate, self.cfg.n_samples)
            e = self.codec.encode(params)
            self._cache[sid] = (
                torch.from_numpy(audio), torch.from_numpy(e.cont), torch.from_numpy(e.binary),
                torch.from_numpy(e.cat), torch.from_numpy(e.cont_mask),
                torch.from_numpy(e.binary_mask), torch.from_numpy(e.cat_mask))
        return self._cache[sid]


def hash_split(ds: MatcherDataset, val_mod: int = 10):
    """crc32(id) % val_mod == 0 -> val (~10%); membership stable as data grows."""
    val = [k for k in range(len(ds)) if zlib.crc32(ds.id_at(k).encode()) % val_mod == 0]
    vs = set(val)
    tr = [k for k in range(len(ds)) if k not in vs]
    return Subset(ds, tr), Subset(ds, val)
