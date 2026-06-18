"""
Dataset: (rendered Operator audio, 195 parameter targets).

Reads the paired files written by data collection:
    dataset/operator/wav/<id>.wav      (44.1 kHz stereo, ~2 s)
    dataset/operator/params/<id>.json   ({"params": {name: raw_value, ...}})

Audio is downmixed to mono, resampled to the model's sample rate, and padded/
truncated to a fixed length. Params are encoded to grouped targets by ParamCodec.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from .codec import ParamCodec
from .config import AudioConfig


def _load_audio(path: Path, target_sr: int, n_samples: int, loop: bool = False) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        mono = resample_poly(mono, target_sr, sr).astype(np.float32)
    # Peak-normalize so the model matches *timbre*, not loudness (level is also a
    # predicted param, but absolute level isn't recoverable/important for matching).
    peak = np.abs(mono).max()
    if peak > 1e-6:
        mono = mono / peak
    # fixed length: truncate, or fill the window. Training uses zero-pad (note + release
    # then silence, matching the rendered targets). Inference on SHORT clips (loop=True)
    # TILES instead, so a 0.3 s stab fills the 3 s window with sustained content the model
    # was trained on, rather than 2.7 s of silence that throws the encoder out of domain.
    if len(mono) >= n_samples:
        mono = mono[:n_samples]
    elif loop and len(mono) > 0:
        mono = np.tile(mono, int(np.ceil(n_samples / len(mono))))[:n_samples]
    else:
        mono = np.pad(mono, (0, n_samples - len(mono)))
    return mono.astype(np.float32)


class OperatorDataset(Dataset):
    def __init__(
        self, root: str | Path, codec: ParamCodec, audio: AudioConfig, cache_audio: bool = True
    ):
        self.root = Path(root)
        self.codec = codec
        self.audio = audio
        wav_dir = self.root / "wav"
        self.ids = sorted(p.stem for p in wav_dir.glob("*.wav"))
        if not self.ids:
            raise RuntimeError(f"No wavs found under {wav_dir}")
        # Decode + resample is the real bottleneck; cache the result in RAM so it
        # happens once, not every epoch. (~20k * 2s @ 16k mono ≈ 2.5 GB.)
        self.cache_audio = cache_audio
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def precompute(self, workers: int = 8) -> None:
        """Decode+resample all clips up front, in parallel threads (soundfile/scipy
        release the GIL), so epoch 1 isn't single-threaded-slow."""
        if not self.cache_audio:
            return
        from concurrent.futures import ThreadPoolExecutor

        def load(i: int):
            sid = self.ids[i]
            return i, _load_audio(
                self.root / "wav" / f"{sid}.wav", self.audio.sample_rate, self.audio.n_samples
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, a in ex.map(load, range(len(self.ids))):
                self._cache[i] = a

    def _audio(self, i: int) -> np.ndarray:
        if self.cache_audio and i in self._cache:
            return self._cache[i]
        sid = self.ids[i]
        a = _load_audio(
            self.root / "wav" / f"{sid}.wav", self.audio.sample_rate, self.audio.n_samples
        )
        if self.cache_audio:
            self._cache[i] = a
        return a

    def __getitem__(self, i: int):
        sid = self.ids[i]
        audio = self._audio(i)
        params = json.loads(
            (self.root / "params" / f"{sid}.json").read_text(encoding="utf-8")
        )["params"]
        t = self.codec.encode(params)
        return {
            "audio": torch.from_numpy(audio),
            "cont": torch.from_numpy(t.cont),
            "binary": torch.from_numpy(t.binary),
            "cat": torch.from_numpy(t.cat),
            "cont_mask": torch.from_numpy(t.cont_mask),
            "binary_mask": torch.from_numpy(t.binary_mask),
            "cat_mask": torch.from_numpy(t.cat_mask),
        }
