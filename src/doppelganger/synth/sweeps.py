"""
Designed parameter sweeps for system-identification of the DiffOperator clone.

Instead of random presets, each experiment isolates ONE relationship by fixing a
clean base config and sweeping a single parameter across the 64 rack tracks. One
export then yields a clean 1-D sweep we can fit a calibration constant to (e.g.
modulator level -> FM index via Bessel sidebands; envelope value -> seconds).

Flow (reuses the existing pipeline):
  uv run python -m doppelganger.synth.sweeps gen fm_index   # writes the apply manifest
  # In Live: "doppelganger: Apply Predicted Batch"  (applies the sweep)
  uv run python -m doppelganger.datagen.export_ableton export "dataset\\sweeps\\fm_index" --expected 64
  uv run python -m doppelganger.datagen.split_export process "dataset\\sweeps\\fm_index" --manifest dataset/predict/manifest.json
  # ...repeat per experiment, then (next) calibrate.py fits DiffOperator to these.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..schema import OperatorSchema


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class Experiment:
    name: str
    swept: str                      # parameter name to sweep
    lo: float
    hi: float
    overrides: dict[str, float] = field(default_factory=dict)  # fixed params for a clean isolation
    note: str = ""


# A clean base = the factory default (single sine on Osc A). Experiments override.
# Common override to make a clean 2-op B->A FM pair (algorithm 1, only A & B on):
_TWO_OP = {
    "Algorithm": 0.0,      # Algo 1 (D>C>B>A); with C,D off this is B->A
    "Osc-A On": 1.0, "Osc-A Level": 1.0,
    "Osc-B On": 1.0,
    "Osc-C On": 0.0, "Osc-D On": 0.0,
    "Ae Attack": 0.0, "Ae Sustain": 1.0,  # steady A so spectrum is clean
    "Be Attack": 0.0, "Be Sustain": 1.0,
}

EXPERIMENTS: dict[str, Experiment] = {
    # modulator level -> FM modulation index (Bessel sideband structure)
    "fm_index": Experiment(
        "fm_index", swept="Osc-B Level", lo=0.0, hi=1.0,
        overrides={**_TWO_OP, "A Coarse": 1.0, "B Coarse": 1.0},
        note="2-op B->A, ratio 1:1; sweep modulator level -> sideband amplitudes",
    ),
    # modulator coarse -> frequency ratio (sideband spacing)
    "coarse_ratio": Experiment(
        "coarse_ratio", swept="B Coarse", lo=1.0, hi=12.0,
        overrides={**_TWO_OP, "A Coarse": 1.0, "Osc-B Level": 0.6},
        note="2-op B->A; sweep modulator coarse -> harmonic spacing",
    ),
    # amp-envelope attack value -> attack time (onset slope)
    "env_attack": Experiment(
        "env_attack", swept="Ae Attack", lo=0.0, hi=1.0,
        overrides={"Osc-A On": 1.0, "Osc-A Level": 1.0, "Osc-B On": 0.0,
                   "Osc-C On": 0.0, "Osc-D On": 0.0, "Ae Sustain": 1.0, "Ae Decay": 0.0},
        note="single sine; sweep attack -> measured onset time",
    ),
    # amp-envelope decay value -> decay time
    "env_decay": Experiment(
        "env_decay", swept="Ae Decay", lo=0.0, hi=1.0,
        overrides={"Osc-A On": 1.0, "Osc-A Level": 1.0, "Osc-B On": 0.0,
                   "Osc-C On": 0.0, "Osc-D On": 0.0, "Ae Attack": 0.0, "Ae Sustain": 0.0},
        note="single sine, sustain 0; sweep decay -> measured decay time",
    ),
}


def build_manifest(exp: Experiment, schema: OperatorSchema, n_tracks: int = 64) -> dict:
    base = {p.name: float(p.value) for p in schema.params}  # factory-default preset
    base.update({k: float(v) for k, v in exp.overrides.items()})
    swept = next(p for p in schema.params if p.name == exp.swept)

    tracks: dict[str, dict] = {}
    for i in range(n_tracks):
        frac = i / (n_tracks - 1) if n_tracks > 1 else 0.0
        params = dict(base)
        val = exp.lo + frac * (exp.hi - exp.lo)
        params[exp.swept] = float(round(val)) if swept.is_quantized else float(val)
        tracks[f"op_{i:04d}"] = params
    return {
        "device": "Operator",
        "prefix": exp.name,
        "experiment": exp.name,
        "swept": exp.swept,
        "range": [exp.lo, exp.hi],
        "note": exp.note,
        "tracks": tracks,
    }


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Generate designed sweep manifests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list experiments")
    g = sub.add_parser("gen", help="write a sweep's manifest to the apply path")
    g.add_argument("experiment", choices=list(EXPERIMENTS))
    g.add_argument("--apply-path", default=str(root / "dataset" / "predict" / "manifest.json"))
    g.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    args = ap.parse_args()

    if args.cmd == "list":
        for e in EXPERIMENTS.values():
            print(f"  {e.name:14} sweep '{e.swept}' [{e.lo},{e.hi}] — {e.note}")
        return

    exp = EXPERIMENTS[args.experiment]
    schema = OperatorSchema.load(args.schema)
    manifest = build_manifest(exp, schema)
    payload = json.dumps(manifest, indent=2)

    # (a) the apply path the SDK "Apply Predicted Batch" reads
    apply_out = Path(args.apply_path)
    apply_out.parent.mkdir(parents=True, exist_ok=True)
    apply_out.write_text(payload, encoding="utf-8")
    # (b) a copy in the sweep's own data folder so calibration reads it self-contained
    sweep_dir = root / "dataset" / "sweeps" / exp.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "manifest.json").write_text(payload, encoding="utf-8")

    print(f"Wrote '{exp.name}' sweep ({len(manifest['tracks'])} tracks).")
    print('Next: in Live run "Apply Predicted Batch", then:')
    print(f'  export (All Individual Tracks, 3s) into:  {sweep_dir}')


if __name__ == "__main__":
    main()
