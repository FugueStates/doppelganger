"""
Phase B: in-the-loop CMA-ES search against the REAL Operator (the audio-objective
solver). For a target sound, optimize a subset of timbre parameters to minimize a
multi-scale-STFT distance, evaluating candidates by RENDERING them in Ableton.

A CMA-ES population of 64 = one export. Per generation: propose 64 presets -> apply
(SDK "Run Search") -> export 64 -> score vs target -> CMA-ES update -> repeat.

Usage:
  uv run python -m doppelganger.synth.search --target "dataset/operator/wav/0000000.wav" --gens 40
  #   then in Live: right-click -> "doppelganger: Run Search"

Requires `uv sync --extra train --extra datagen` (cma + torch + pywinauto).
Saves the best-matching preset to dataset/search/best.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cma
import numpy as np
import torch

from ..schema import OperatorSchema
from ..datagen.export_ableton import export_batch
from ..datagen.split_export import detect_prefix
from ..training.data import _load_audio
from .sensitivity import rich_base
from .spectral import multiscale_stft_loss

SR, N_SAMPLES = 16000, 48000

# Searched (continuous) timbre params — chosen from the audio-sensitivity ranking.
SEARCH_PARAMS = [
    "Osc-A Level", "Osc-B Level", "Osc-C Level", "Osc-D Level",
    "A Coarse", "B Coarse", "C Coarse", "D Coarse",
    "Filter Freq", "Filter Res", "Filter Drive",
    "Ae Attack", "Ae Decay", "Ae Sustain", "Ae Release",
    "Tone", "Osc-A Feedb", "Fe Amount",
]
# Fixed structure (kept out of the search for the first proof).
FORCED = {
    "Device On": 1.0, "Volume": 0.7, "Algorithm": 0.0,
    **{f"Osc-{X} On": 1.0 for X in "ABCD"},
    **{f"{X} Fix On ": 0.0 for X in "ABCD"},
    **{f"Osc-{X} Wave": 0.0 for X in "ABCD"},          # sine
    **{f"{X}e Mode": 0.0 for X in "ABCD"},             # standard ADSR
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load(path: Path) -> torch.Tensor:
    return torch.from_numpy(_load_audio(path, SR, N_SAMPLES)).unsqueeze(0)


def build_full(schema_by_name: dict, base: dict, x: np.ndarray) -> dict:
    p = dict(base)
    p.update(FORCED)
    for name, xi in zip(SEARCH_PARAMS, x):
        pa = schema_by_name[name]
        p[name] = pa.min + float(min(1.0, max(0.0, xi))) * (pa.max - pa.min)
    return p


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="In-the-loop CMA-ES match against real Operator.")
    ap.add_argument("--target", default=None, help="target wav (default: first dataset render)")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--apply-timeout", type=float, default=120.0)
    args = ap.parse_args()

    schema = OperatorSchema.load(root / "schemas" / "operator.json")
    by_name = {p.name: p for p in schema.params}
    base = rich_base(schema)

    if args.target:
        target_path = Path(args.target)
    else:
        # robust default: any available real-Operator render
        cands = [
            *sorted(root.glob("dataset/operator/wav/*.wav")),
            *sorted(root.glob("dataset/sweeps/*/* op_*.wav")),
        ]
        if not cands:
            raise SystemExit("No target wav found — pass --target <path>")
        target_path = cands[0]
    if not target_path.exists():
        raise SystemExit(f"Target not found: {target_path}")
    target = _load(target_path)
    print(f"target: {target_path.name}")

    sdir = root / "dataset" / "search"
    render = sdir / "render"
    render.mkdir(parents=True, exist_ok=True)
    for f in ("STOP", "gen.txt", "applied.txt"):
        (sdir / f).unlink(missing_ok=True)

    # seed CMA-ES at the base preset (normalized)
    x0 = [(base[n] - by_name[n].min) / max(by_name[n].max - by_name[n].min, 1e-9) for n in SEARCH_PARAMS]
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"popsize": 64, "bounds": [0, 1], "verbose": -9})

    print('Now in Live: right-click a track -> "doppelganger: Run Search".')
    best = (float("inf"), None)
    for gen in range(args.gens):
        if es.stop():
            break
        sols = es.ask()
        tracks = {f"op_{i:04d}": build_full(by_name, base, sols[i]) for i in range(len(sols))}
        (sdir / "manifest.json").write_text(
            json.dumps({"device": "Operator", "tracks": tracks}, indent=2), encoding="utf-8")
        for w in render.glob("*"):  # clear so the export never prompts to overwrite
            if w.is_file():
                w.unlink()
        (sdir / "gen.txt").write_text(str(gen), encoding="utf-8")

        # wait for the SDK to apply this generation
        deadline = time.time() + args.apply_timeout
        while time.time() < deadline:
            try:
                if int((sdir / "applied.txt").read_text()) == gen:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            print('SDK did not apply — is "Run Search" running in Live?')
            break

        export_batch(render, expected_count=len(sols))
        prefix = detect_prefix(render)
        losses = []
        for i in range(len(sols)):
            wav = render / f"{prefix} op_{i:04d}.wav"
            losses.append(float(multiscale_stft_loss(_load(wav), target)) if wav.exists() else 1e6)
        es.tell(sols, losses)

        mi = int(np.argmin(losses))
        if losses[mi] < best[0]:
            best = (losses[mi], sols[mi])
        print(f"gen {gen:3} best={min(losses):.3f} mean={np.mean(losses):.3f} overall_best={best[0]:.3f}")

    (sdir / "STOP").write_text("", encoding="utf-8")
    if best[1] is not None:
        (sdir / "best.json").write_text(
            json.dumps({"target": str(target_path), "loss": best[0],
                        "params": build_full(by_name, base, best[1])}, indent=2), encoding="utf-8")
        print(f"\nDone. Best loss {best[0]:.3f}. Saved preset -> {sdir / 'best.json'}")


if __name__ == "__main__":
    main()
