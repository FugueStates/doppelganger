"""
Unattended data-collection watcher (the Python half of Auto Collect).

Pairs with the data-collection extension's "Auto Collect" command via a file
handshake in ``dataset/pending/<batchId>/``:

    extension: randomize params -> write manifest.json + READY
    watcher:   sees READY -> export (pywinauto) -> process -> write DONE
    extension: sees DONE -> randomize next batch -> ...

Run THIS first, then trigger "doppelganger: Auto Collect" in Live:

    uv run python -m doppelganger.datagen.run_loop

Stop with Ctrl+C (or create dataset/STOP to stop the extension side). Requires
`uv sync --extra datagen`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .export_ableton import export_batch
from .split_export import process_pending


def _next_ready_batch(pending: Path) -> Path | None:
    """The oldest batch that is READY but not yet DONE."""
    if not pending.exists():
        return None
    for d in sorted(pending.iterdir()):
        if d.is_dir() and (d / "READY").exists() and not (d / "DONE").exists():
            return d
    return None


def run(
    pending_dir: str | Path,
    num_tracks: int,
    idle_timeout: float,
    min_rms: float,
    keep_source: bool = False,
    out_dir: str | Path | None = None,
) -> None:
    pending = Path(pending_dir)
    total_written = 0
    last_activity = time.time()
    dest = Path(out_dir) if out_dir else "dataset/operator (default)"
    print(f"Watching {pending} … -> {dest}  (Ctrl+C to stop)")

    while True:
        batch = _next_ready_batch(pending)
        if batch is None:
            if time.time() - last_activity > idle_timeout:
                print(f"No new batches for {idle_timeout:.0f}s — exiting.")
                break
            time.sleep(0.5)
            continue

        print(f"\n=== {batch.name}: exporting ===")
        try:
            export_batch(batch, expected_count=num_tracks)
            written, skipped = process_pending(
                batch, out_dir=out_dir, min_rms=min_rms, delete_source=not keep_source
            )
            total_written += written
            print(f"=== {batch.name}: +{written} (skipped {skipped}); total={total_written} ===")
        except Exception as e:  # noqa: BLE001
            print(f"!! {batch.name} failed: {e}")
        finally:
            # Always release the extension's wait, even on failure, so it can continue.
            (batch / "DONE").write_text("", encoding="utf-8")
            last_activity = time.time()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Unattended Ableton data-collection watcher.")
    ap.add_argument("--pending", default=str(repo_root / "dataset" / "pending"),
                    help="handshake dir the extension writes batches to (matches config.ts PENDING_DIR)")
    ap.add_argument("--out", default=None,
                    help="output dataset dir for processed (audio,params) pairs "
                         "(default: dataset/operator). Use e.g. dataset/sniff to keep a set separate.")
    ap.add_argument("--num-tracks", type=int, default=64, help="files to wait for per export")
    ap.add_argument("--idle-timeout", type=float, default=90.0, help="exit after this idle gap (s)")
    ap.add_argument("--min-rms", type=float, default=0.01)
    ap.add_argument(
        "--keep-source",
        action="store_true",
        help="keep raw exported WAVs in pending/ (default: delete after copying to dataset)",
    )
    args = ap.parse_args()
    run(args.pending, args.num_tracks, args.idle_timeout, args.min_rms, args.keep_source, args.out)


if __name__ == "__main__":
    main()
