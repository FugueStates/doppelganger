"""
Calibrate DiffOperator to the real Operator using the designed sweeps, then report
spectral fidelity. Fits the global constants (fm_scale, envelope time mappings) by
gradient descent on a multi-scale STFT loss between clone renders and real audio.

    uv run python -m doppelganger.synth.calibrate

Reports per-sweep clone-vs-real loss after fitting, against a shuffled baseline
(real vs a mismatched real) — ratio << 1 means the clone tracks the relationship.
Saves fitted constants to models/operator/calib.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn

from ..training.data import _load_audio
from ..datagen.split_export import detect_prefix
from .adapter import to_controls
from .diff_operator import CalibConstants, DiffOperator
from .spectral import multiscale_stft_loss

SR = 16000
N_SAMPLES = 48000  # 3 s
NOTE_OFF = 1.5
F0 = 261.63  # Ableton MIDI note 60 ("C3" label) = 261.6 Hz (NOT 130.8)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_sweep(sweep_dir: Path, device: str):
    manifest = json.loads((sweep_dir / "manifest.json").read_text(encoding="utf-8"))
    prefix = detect_prefix(sweep_dir)
    names = sorted(manifest["tracks"])
    params, audio = [], []
    for n in names:
        wav = sweep_dir / f"{prefix} {n}.wav"
        if not wav.exists():
            continue
        params.append(manifest["tracks"][n])
        audio.append(torch.from_numpy(_load_audio(wav, SR, N_SAMPLES)))
    ctrl = to_controls(params, device)
    real = torch.stack(audio).to(device)
    return ctrl, real, ctrl["algo"][0]  # algo fixed within a sweep


def main() -> None:
    root = _repo_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    synth = DiffOperator(SR, N_SAMPLES, NOTE_OFF).to(device)

    sweeps_dir = root / "dataset" / "sweeps"
    sweeps = []
    for d in sorted(p for p in sweeps_dir.iterdir() if (p / "manifest.json").exists()):
        meta = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if meta.get("kind") == "freq_map":
            continue  # frequency-mapping sweeps (freq_map.py) are not calibration data
        ctrl, real, algo = load_sweep(d, device)
        sweeps.append((d.name, ctrl, real, algo))
        print(f"loaded {d.name}: {real.shape[0]} clips, algo {algo + 1}")

    # learnable constants (positivity via exp on the time mappings)
    fm_scale = nn.Parameter(torch.tensor(6.0, device=device))
    log_t = nn.Parameter(torch.log(torch.tensor([2.0, 3.0, 4.0], device=device)))  # a,d,r
    opt = torch.optim.Adam([fm_scale, log_t], lr=0.05)

    def constants():
        t = log_t.exp()
        return CalibConstants(fm_scale=fm_scale, max_attack=t[0], max_decay=t[1], max_release=t[2])

    for step in range(250):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for _, ctrl, real, algo in sweeps:
            clone = synth.render(ctrl["coarse"], ctrl["level"], ctrl["adsr"], algo,
                                 f0=F0, calib=constants())
            loss = loss + multiscale_stft_loss(clone, real)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 249:
            print(f"  step {step:3} total_loss={loss.item():.3f} "
                  f"fm_scale={fm_scale.item():.2f} times={log_t.exp().tolist()}")

    # fidelity report: per-sweep clone-vs-real vs a shuffled (mismatched) baseline
    print("\n=== fidelity (lower = closer; ratio < 1 = clone tracks it) ===")
    with torch.no_grad():
        for name, ctrl, real, algo in sweeps:
            clone = synth.render(ctrl["coarse"], ctrl["level"], ctrl["adsr"], algo,
                                 f0=F0, calib=constants())
            matched = multiscale_stft_loss(clone, real).item()
            shuffled = multiscale_stft_loss(real[torch.randperm(real.shape[0])], real).item()
            print(f"  {name:14} matched={matched:.3f}  shuffled={shuffled:.3f}  "
                  f"ratio={matched / max(shuffled, 1e-6):.3f}")

    out = root / "models" / "operator" / "calib.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    t = log_t.exp().tolist()
    out.write_text(json.dumps(
        {"fm_scale": fm_scale.item(), "max_attack": t[0], "max_decay": t[1], "max_release": t[2]},
        indent=2), encoding="utf-8")
    print(f"\nSaved calibrated constants -> {out}")


if __name__ == "__main__":
    main()
