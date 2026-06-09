"""
Phase A: rank Operator parameters by AUDIO variability.

Perturb each parameter one-at-a-time from a rich base preset, render on the real
Operator, and measure the spectral change vs the base. High change = the parameter
strongly shapes the sound -> include it in the CMA-ES search subset (Phase B).
Expand the subset later by adding the next-most-impactful params.

Reuses the Run-Sweeps automation: this writes perturbation batches as sweep folders,
you click "Run Sweeps" in Live, it auto-exports, then `rank` scores them.

    uv run python -m doppelganger.synth.sensitivity gen      # generate + watch/export
    #   (in Live: "doppelganger: Run Sweeps")
    uv run python -m doppelganger.synth.sensitivity rank     # compute the ranking
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..schema import OperatorSchema, Param
from ..datagen.export_ableton import export_batch
from ..datagen.split_export import detect_prefix
from ..training.data import _load_audio
from .spectral import multiscale_stft_loss

SR, N_SAMPLES = 16000, 48000
TRACKS = 64  # track 0 = base reference; tracks 1..63 = one perturbed param each


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def rich_base(schema: OperatorSchema) -> dict:
    """A preset where most sections are active, so perturbing any param has an audible
    effect (a single sine would hide most params)."""
    base = {p.name: float(p.value) for p in schema.params}
    base.update({"Device On": 1.0, "Volume": 0.7, "Algorithm": 0.0, "Filter Freq": 0.8})
    coarse = {"A": 1.0, "B": 2.0, "C": 1.0, "D": 3.0}
    for X in "ABCD":
        base[f"Osc-{X} On"] = 1.0
        base[f"Osc-{X} Level"] = 0.7
        base[f"{X} Coarse"] = coarse[X]
        base[f"{X} Fix On "] = 0.0
        e = f"{X}e"
        base[f"{e} Mode"] = 0.0
        base[f"{e} Attack"] = 0.05
        base[f"{e} Decay"] = 0.4
        base[f"{e} Sustain"] = 0.7
        base[f"{e} Release"] = 0.3
    return base


def perturbed(p: Param, base_val: float) -> float:
    """A value far from base, to probe the parameter's audio impact."""
    if p.is_quantized:
        n = p.cardinality or 2
        return float((int(round(base_val)) + max(1, n // 2)) % n)
    mid = (p.min + p.max) / 2.0
    return p.min if base_val >= mid else p.max


def gen(args) -> None:
    root = _repo_root()
    schema = OperatorSchema.load(args.schema)
    base = rich_base(schema)
    params = schema.params
    sweeps_root = root / "dataset" / "sweeps"

    # chunk params into batches of (TRACKS-1) perturbations (track 0 is the base)
    per_batch = TRACKS - 1
    dirs = []
    for ci in range(0, len(params), per_batch):
        chunk = params[ci : ci + per_batch]
        d = sweeps_root / f"sens_{ci // per_batch:02d}"
        d.mkdir(parents=True, exist_ok=True)
        tracks = {"op_0000": dict(base)}
        perturbs = {}
        for j, p in enumerate(chunk, start=1):
            cfg = dict(base)
            cfg[p.name] = perturbed(p, base[p.name])
            name = f"op_{j:04d}"
            tracks[name] = cfg
            perturbs[name] = p.name
        (d / "manifest.json").write_text(
            json.dumps({"device": "Operator", "prefix": d.name, "tracks": tracks,
                        "perturbs": perturbs}, indent=2), encoding="utf-8")
        for m in ("READY", "DONE"):
            (d / m).unlink(missing_ok=True)
        dirs.append(d)

    print(f"Generated {len(dirs)} sensitivity batches ({len(params)} params).")
    print('Now in Live: "doppelganger: Run Sweeps". Watching for exports…')
    _watch(dirs, args.idle_timeout)


def _watch(dirs: list[Path], idle_timeout: float) -> None:
    last = time.time()
    while True:
        pending = [d for d in dirs if (d / "READY").exists() and not (d / "DONE").exists()]
        if pending:
            d = pending[0]
            print(f"  exporting {d.name}…")
            try:
                export_batch(d, expected_count=TRACKS)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {d.name} failed: {e}")
            finally:
                (d / "DONE").write_text("", encoding="utf-8")
            last = time.time()
        elif all((d / "DONE").exists() for d in dirs):
            print("All sensitivity batches exported. Run: ... sensitivity rank")
            break
        elif time.time() - last > idle_timeout:
            print("Idle timeout — did you run 'Run Sweeps' in Live?")
            break
        else:
            time.sleep(0.5)


def _load(path: Path) -> torch.Tensor:
    return torch.from_numpy(_load_audio(path, SR, N_SAMPLES)).unsqueeze(0)


def rank(args) -> None:
    root = _repo_root()
    sweeps_root = root / "dataset" / "sweeps"
    scores: dict[str, float] = {}
    for d in sorted(sweeps_root.glob("sens_*")):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        prefix = detect_prefix(d)
        if prefix is None:
            continue
        base_wav = d / f"{prefix} op_0000.wav"
        if not base_wav.exists():
            continue
        base = _load(base_wav)
        for track, param in manifest["perturbs"].items():
            wav = d / f"{prefix} {track}.wav"
            if wav.exists():
                scores[param] = float(multiscale_stft_loss(_load(wav), base))

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    out = root / "dataset" / "sensitivity.json"
    out.write_text(json.dumps([{"param": p, "score": round(s, 4)} for p, s in ranked], indent=2),
                   encoding="utf-8")
    print(f"Ranked {len(ranked)} params by audio impact -> {out}\nTop 25:")
    for p, s in ranked[:25]:
        print(f"  {s:8.3f}  {p}")


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Rank Operator params by audio variability.")
    ap.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--idle-timeout", type=float, default=180.0)
    g.set_defaults(func=gen)
    sub.add_parser("rank").set_defaults(func=rank)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
