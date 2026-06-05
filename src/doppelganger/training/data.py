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


def _load_audio(path: Path, target_sr: int, n_samples: int) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        mono = resample_poly(mono, target_sr, sr).astype(np.float32)
    # Peak-normalize so the model matches *timbre*, not loudness (level is also a
    # predicted param, but absolute level isn't recoverable/important for matching).
    peak = np.abs(mono).max()
    if peak > 1e-6:
        mono = mono / peak
    # fixed length: truncate or zero-pad at the end
    if len(mono) >= n_samples:
        mono = mono[:n_samples]
    else:
        mono = np.pad(mono, (0, n_samples - len(mono)))
    return mono.astype(np.float32)


class OperatorDataset(Dataset):
    def __init__(self, root: str | Path, codec: ParamCodec, audio: AudioConfig):
        self.root = Path(root)
        self.codec = codec
        self.audio = audio
        wav_dir = self.root / "wav"
        self.ids = sorted(p.stem for p in wav_dir.glob("*.wav"))
        if not self.ids:
            raise RuntimeError(f"No wavs found under {wav_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        sid = self.ids[i]
        audio = _load_audio(
            self.root / "wav" / f"{sid}.wav", self.audio.sample_rate, self.audio.n_samples
        )
        params = json.loads(
            (self.root / "params" / f"{sid}.json").read_text(encoding="utf-8")
        )["params"]
        t = self.codec.encode(params)
        return {
            "audio": torch.from_numpy(audio),
            "cont": torch.from_numpy(t.cont),
            "binary": torch.from_numpy(t.binary),
            "cat": torch.from_numpy(t.cat),
        }
