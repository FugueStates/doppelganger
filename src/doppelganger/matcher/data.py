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
    def __init__(self, root: str | Path | list, codec: ParamCodec, cfg: MatcherConfig):
        roots = [root] if isinstance(root, (str, Path)) else list(root)
        self.roots = [Path(r) for r in roots]
        self.codec = codec
        self.cfg = cfg
        multi = len(self.roots) > 1   # namespace ids by folder when mixing sets (avoid stem collisions)
        self.ids: list[str] = []
        self._loc: dict[str, tuple[Path, str]] = {}
        for r in self.roots:
            for p in sorted((r / "params").glob("*.json")):
                sid = f"{r.name}/{p.stem}" if multi else p.stem
                self.ids.append(sid)
                self._loc[sid] = (r, p.stem)
        self.ids.sort()
        if not self.ids:
            roots_str = ", ".join(str(r / "params") for r in self.roots)
            raise RuntimeError(f"No params under {roots_str} — collect data first.")
        self._cache: dict[str, tuple] = {}

    def __len__(self):
        return len(self.ids)

    def id_at(self, i: int) -> str:
        return self.ids[i]

    def __getitem__(self, i: int):
        sid = self.ids[i]
        if sid not in self._cache:
            r, stem = self._loc[sid]
            params = json.loads((r / "params" / f"{stem}.json").read_text())["params"]
            audio = _load_audio(r / "wav" / f"{stem}.wav",
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
