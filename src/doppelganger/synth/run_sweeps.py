"""
One-command sweep collector (the Python half; pairs with the SDK "Run Sweeps").

Generates every designed-sweep manifest, then watches for the extension to apply
each one and auto-exports it via pywinauto — fully unattended, like Auto Collect.

Usage:
  1. uv run python -m doppelganger.synth.run_sweeps     # generates manifests + watches
  2. In Live (rack built, export length set to ~3s): right-click -> "doppelganger: Run Sweeps"

The extension applies each sweep's params -> writes READY; this watcher exports into
that sweep's folder -> writes DONE; the extension moves to the next. Result:
dataset/sweeps/<exp>/ each contains manifest.json + the 64 exported WAVs (self-contained
for calibration). Requires `uv sync --extra datagen`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..schema import OperatorSchema
from ..datagen.export_ableton import export_batch
from .sweeps import EXPERIMENTS, build_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _generate(sweeps_dir: Path, schema: OperatorSchema) -> list[Path]:
    dirs = []
    for exp in EXPERIMENTS.values():
        d = sweeps_dir / exp.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(
            json.dumps(build_manifest(exp, schema), indent=2), encoding="utf-8"
        )
        for marker in ("READY", "DONE"):  # clear any stale handshake markers
            (d / marker).unlink(missing_ok=True)
        dirs.append(d)
    print(f"Generated {len(dirs)} sweep manifests under {sweeps_dir}")
    return dirs


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Generate + auto-export all designed sweeps.")
    ap.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    ap.add_argument("--num-tracks", type=int, default=64)
    ap.add_argument("--idle-timeout", type=float, default=180.0)
    args = ap.parse_args()

    sweeps_dir = root / "dataset" / "sweeps"
    dirs = _generate(sweeps_dir, OperatorSchema.load(args.schema))

    print('Now in Live: right-click a MIDI track -> "doppelganger: Run Sweeps".')
    print("(Rack must be built; set the export length to ~3s once.)")
    last_activity = time.time()
    while True:
        pending = [
            d for d in dirs
            if (d / "READY").exists() and not (d / "DONE").exists()
        ]
        if pending:
            d = pending[0]
            print(f"\n=== {d.name}: exporting ===")
            try:
                export_batch(d, expected_count=args.num_tracks)
            except Exception as e:  # noqa: BLE001
                print(f"!! {d.name} export failed: {e}")
            finally:
                (d / "DONE").write_text("", encoding="utf-8")
            last_activity = time.time()
        elif all((d / "DONE").exists() for d in dirs):
            print("\nAll sweeps exported. Done.")
            break
        elif time.time() - last_activity > args.idle_timeout:
            done = sum((d / "DONE").exists() for d in dirs)
            print(f"\nIdle {args.idle_timeout:.0f}s — exiting ({done}/{len(dirs)} done). "
                  'Did you run "Run Sweeps" in Live?')
            break
        else:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
