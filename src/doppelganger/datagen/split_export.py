"""
Turn one Ableton "All Individual Tracks" export into dataset entries.

Ableton names each exported file ``{prefix} {track name}.wav`` (plus a ``.asd``
sidecar we ignore). It also exports return tracks and the master, which we filter
out by requiring our data-collection track-name prefix (default ``op_``).

The data-collection extension writes a per-batch ``manifest.json`` mapping each
track name to the raw Operator parameter values it set. We pair each WAV with its
track's params to produce ``(audio, params)`` examples.

Usage (inspect a folder without a manifest):
    uv run python -m doppelganger.datagen.split_export "<export_dir>" --prefix TESTEST --all
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..audio import load_wav


@dataclass(frozen=True)
class ExportedTrack:
    track_name: str
    wav_path: Path


def discover_export(
    export_dir: str | Path,
    prefix: str,
    track_prefix: str | None = "op_",
) -> list[ExportedTrack]:
    """Find per-track WAVs in an export folder.

    Files are named ``{prefix} {track_name}.wav``. The master file is just
    ``{prefix}.wav`` (no track name) and is skipped. If ``track_prefix`` is given,
    only tracks whose name starts with it are returned (filters out return tracks).
    """
    export_dir = Path(export_dir)
    lead = f"{prefix} "
    found: list[ExportedTrack] = []

    for wav in sorted(export_dir.glob(f"{prefix}*.wav")):
        stem = wav.name[:-4]  # drop ".wav"
        if not stem.startswith(lead):
            continue  # the master file "{prefix}.wav" has no trailing track name
        track_name = stem[len(lead):]
        if track_prefix and not track_name.startswith(track_prefix):
            continue
        found.append(ExportedTrack(track_name=track_name, wav_path=wav))

    return found


def detect_prefix(export_dir: str | Path) -> str | None:
    """Infer the export name prefix from the files actually present.

    Robust to the manifest's recorded prefix not matching the folder (e.g. when the
    batch id was typed manually). Returns the most common ``{prefix}`` seen in
    ``{prefix} op_<n>.wav`` filenames, or None.
    """
    pat = re.compile(r"^(.*) op_\d+\.wav$")
    counts: dict[str, int] = {}
    for wav in Path(export_dir).glob("*.wav"):
        m = pat.match(wav.name)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return max(counts, key=counts.get) if counts else None


def process_batch(
    export_dir: str | Path,
    prefix: str,
    manifest: dict,
    out_wav_dir: str | Path,
    out_param_dir: str | Path,
    start_index: int,
    track_prefix: str = "op_",
    delete_source: bool = False,
) -> int:
    """Pair each exported track with its params and write dataset entries.

    ``manifest`` is ``{"tracks": {track_name: {param_name: raw_value, ...}}}``.
    Returns the number of examples written. Files are written as zero-padded
    indices: ``<start_index>.wav`` / ``<start_index>.json``.
    """
    out_wav_dir = Path(out_wav_dir)
    out_param_dir = Path(out_param_dir)
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_param_dir.mkdir(parents=True, exist_ok=True)

    tracks_params: dict = manifest.get("tracks", {})
    written = 0
    idx = start_index

    for track in discover_export(export_dir, prefix, track_prefix):
        params = tracks_params.get(track.track_name)
        if params is None:
            # A track with no logged params is a bug in the batch; skip loudly.
            print(f"  ! no manifest params for {track.track_name}, skipping")
            continue

        name = f"{idx:07d}"
        dest_wav = out_wav_dir / f"{name}.wav"
        shutil.copy2(track.wav_path, dest_wav)
        (out_param_dir / f"{name}.json").write_text(
            json.dumps({"track": track.track_name, "params": params}, indent=2),
            encoding="utf-8",
        )
        if delete_source:
            track.wav_path.unlink(missing_ok=True)
            track.wav_path.with_suffix(".wav.asd").unlink(missing_ok=True)

        written += 1
        idx += 1

    return written


def _repo_root() -> Path:
    # .../src/doppelganger/datagen/split_export.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _next_index(state_file: Path) -> int:
    try:
        return int(json.loads(state_file.read_text(encoding="utf-8"))["next"])
    except Exception:
        return 0


def process_pending(
    batch_dir: str | Path,
    out_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    min_rms: float = 0.01,
    delete_source: bool = False,
) -> tuple[int, int]:
    """Pair a batch's exported WAVs with its manifest and append to the dataset.

    The export prefix is auto-detected from the files present (robust to manifest /
    folder prefix mismatches). Per-track params come from the manifest (defaults to
    ``<batch_dir>/manifest.json``; override with ``manifest_path``). Renders quieter
    than ``min_rms`` (perceptually inaudible) are skipped. Returns (written, skipped).
    """
    batch_dir = Path(batch_dir)
    mpath = Path(manifest_path) if manifest_path else batch_dir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))

    prefix = detect_prefix(batch_dir) or manifest.get("prefix")
    if prefix is None:
        raise RuntimeError(f"No 'op_*' export files found in {batch_dir}")
    print(f"Using prefix '{prefix}', manifest '{mpath.name}'")

    out_dir = Path(out_dir) if out_dir else _repo_root() / "dataset" / "operator"
    out_wav = out_dir / "wav"
    out_param = out_dir / "params"
    out_wav.mkdir(parents=True, exist_ok=True)
    out_param.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "_state.json"

    idx = _next_index(state_file)
    tracks_params = manifest.get("tracks", {})
    written = skipped_quiet = skipped_missing = 0

    for track in discover_export(batch_dir, prefix, track_prefix="op_"):
        params = tracks_params.get(track.track_name)
        if params is None:
            skipped_missing += 1
            continue

        clip = load_wav(track.wav_path)
        if clip.rms < min_rms:
            skipped_quiet += 1
            continue

        name = f"{idx:07d}"
        shutil.copy2(track.wav_path, out_wav / f"{name}.wav")
        (out_param / f"{name}.json").write_text(
            json.dumps(
                {"track": track.track_name, "batch": prefix, "params": params}, indent=2
            ),
            encoding="utf-8",
        )
        if delete_source:
            track.wav_path.unlink(missing_ok=True)
            Path(str(track.wav_path) + ".asd").unlink(missing_ok=True)
        idx += 1
        written += 1

    state_file.write_text(json.dumps({"next": idx}, indent=2), encoding="utf-8")
    if skipped_missing:
        print(f"  ! {skipped_missing} track(s) had no manifest params (prefix mismatch?)")
    return written, skipped_quiet + skipped_missing


def _inspect(args: argparse.Namespace) -> None:
    track_prefix = None if args.all else args.track_prefix
    tracks = discover_export(args.export_dir, args.prefix, track_prefix)
    print(f"Found {len(tracks)} track file(s) with prefix '{args.prefix}':")
    for t in tracks:
        clip = load_wav(t.wav_path)
        print(
            f"  {t.track_name:32}  {clip.duration:.3f}s  "
            f"{clip.sample_rate}Hz  {clip.channels}ch  peak={clip.peak:.4f}"
        )


def _process(args: argparse.Namespace) -> None:
    written, skipped = process_pending(
        args.batch_dir,
        manifest_path=args.manifest,
        min_rms=args.min_rms,
        delete_source=args.delete_source,
    )
    print(f"Wrote {written} example(s), skipped {skipped} (quiet/missing).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Process Ableton multitrack exports.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    insp = sub.add_parser("inspect", help="list tracks in an export folder")
    insp.add_argument("export_dir")
    insp.add_argument("--prefix", required=True)
    insp.add_argument("--track-prefix", default="op_")
    insp.add_argument("--all", action="store_true")
    insp.set_defaults(func=_inspect)

    proc = sub.add_parser("process", help="pair a batch's WAVs with its manifest into the dataset")
    proc.add_argument("batch_dir")
    proc.add_argument("--manifest", default=None, help="manifest.json path (default: <batch_dir>/manifest.json)")
    proc.add_argument("--min-rms", type=float, default=0.01, help="skip renders quieter than this RMS")
    proc.add_argument("--delete-source", action="store_true")
    proc.set_defaults(func=_process)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
