"""
Frequency-mapping verification sweeps: measure how Operator's RAW parameter values map
to actual rendered FREQUENCIES, using the real device.

WHY: the DiffOperator physics prior feeds raw param values straight into the math —
`A Coarse` 3.187 is used as the literal FM ratio, `Fine` / `Fix On` / `Fix Freq` are
ignored, and `A Quantize` is unmodeled. If the device actually snaps Coarse to discrete
ratios, or Fine adds ratio/1000, every prior sideband is systematically misplaced (the
old coarse_ratio calibration scoring a poor 1.37 hints at exactly this). These sweeps
nail the mappings down empirically so they can be coded into the prior with confidence.

Flow (mirrors run_sweeps.py; fully unattended):
  1. uv run python -m doppelganger.synth.freq_map gen     # writes manifests + watches
  2. In Live (rack built, export length ~3s): right-click ->
       "doppelganger: Collect Freq Maps"
     (the extension PINS each track's clip to the manifest's note — the rack's clips
      hold random pitches after Auto Collect, which would ruin the measurement — then
      applies params, writes READY, and this watcher exports + writes DONE)
  3. uv run python -m doppelganger.synth.freq_map analyze # tables + fits + verdicts

Experiments (one 64-track export each, single pure-sine Osc A, steady envelope):
  fmap_coarse / fmap_coarse_n72 : sweep A Coarse 0..48 at C3 and +1 octave
       -> is raw Coarse the literal ratio? quantized? note-proportional (ratio model)?
  fmap_fine_c1 / fmap_fine_c2   : sweep A Fine 0..1000 at Coarse 1 and 2
       -> additive (ratio += fine/1000) vs multiplicative (slope doubles at Coarse 2)?
  fmap_fixed / fmap_fixed_n72   : Fix On; 4 Fix-Freq-Mul values x 16 Fix-Freq steps
       -> raw->Hz law (linear/exponential), and is it truly note-independent?
  fmap_quant                    : sweep A Coarse 0..48 with A Quantize ON
       -> what Quantize actually snaps to.

Results land in dataset/sweeps/fmap_*/ (self-contained: manifest + WAVs);
`analyze` writes dataset/freq_maps_report.json. Analysis reads the WAVs at their
NATIVE sample rate (no 16 kHz resample) so high ratios stay below Nyquist.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..schema import OperatorSchema
from ..datagen.split_export import detect_prefix

N_TRACKS = 64
NOTE_C3, NOTE_C4 = 60, 72       # Ableton "C3" = MIDI 60 = 261.63 Hz
VELOCITY = 100
MEASURE_WINDOW = (0.30, 1.30)   # steady sustain region (note off at 1.5 s)
SILENCE_RMS = 1e-3
MAX_MEASURE_HZ = 20000.0        # above this the render aliases (Live exports at 44.1k) —
                                # e.g. ratio 45 @ C4 folded to 39.28, exactly 44100-f


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def midi_hz(note: float) -> float:
    return 440.0 * 2.0 ** ((note - 69.0) / 12.0)


# --- experiment definitions ---------------------------------------------------------

def _clean_base(schema: OperatorSchema) -> dict:
    """Factory defaults + everything pinned so the ONLY thing sounding is a steady pure
    sine on Osc A — any frequency we measure is attributable to the swept params."""
    base = {p.name: float(p.value) for p in schema.params}
    base.update({
        "Device On": 1.0, "Volume": 0.8, "Algorithm": 0.0, "Transpose": 0.0,
        "Spread": 0.0, "Glide On": 0.0, "Panorama": 0.0, "Pan < Key": 0.0, "Pan < Rnd": 0.0,
        "Osc-A On": 1.0, "Osc-A Level": 1.0, "Osc-A Wave": 0.0, "Osc-A Feedb": 0.0,
        "Osc-A Phase": 0.0, "A Coarse": 1.0, "A Fine": 0.0, "A Fix On ": 0.0,
        "A Quantize": 0.0, "A Freq<Vel": 0.0, "Osc-A Lev < Vel": 0.0, "Osc-A Lev < Key": 0.0,
        "Osc-B On": 0.0, "Osc-C On": 0.0, "Osc-D On": 0.0,
        "Ae Mode": 0.0, "Ae Attack": 0.0, "Ae Decay": 0.0, "Ae Sustain": 1.0,
        "Ae Release": 0.1, "Ae Init": 0.0, "Ae Peak": 1.0, "Ae Loop": 0.0,
        "Pe On": 0.0, "LFO On": 0.0, "Filter On": 0.0,
        "Shaper Mix": 0.0, "Shaper Drive": 0.0,
    })
    return base


@dataclass
class FreqMapExp:
    name: str
    note: int
    overrides: dict
    sweep: dict[str, list[float]]   # param -> per-track values (parallel lists, len 64)
    purpose: str


def _grid(lo: float, hi: float, n: int) -> list[float]:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def build_experiments(schema: OperatorSchema) -> list[FreqMapExp]:
    by_name = {p.name: p for p in schema.params}
    coarse_vals = _grid(by_name["A Coarse"].min, by_name["A Coarse"].max, N_TRACKS)
    fine_vals = _grid(by_name["A Fine"].min, by_name["A Fine"].max, N_TRACKS)
    # fixed-frequency grid: 4 Mul values x 16 Freq steps = 64 tracks
    mul_p, freq_p = by_name["A Fix Freq Mul"], by_name["A Fix Freq"]
    muls = _grid(mul_p.min, mul_p.max, 4)
    freqs = _grid(freq_p.min, freq_p.max, 16)
    fixed_sweep = {
        "A Fix Freq Mul": [m for m in muls for _ in freqs],
        "A Fix Freq": [f for _ in muls for f in freqs],
    }
    return [
        FreqMapExp("fmap_coarse", NOTE_C3, {}, {"A Coarse": coarse_vals},
                   "raw Coarse -> ratio (identity? quantized?)"),
        FreqMapExp("fmap_coarse_n72", NOTE_C4, {}, {"A Coarse": coarse_vals},
                   "Coarse at +1 octave (ratio model check: Hz should double)"),
        FreqMapExp("fmap_fine_c1", NOTE_C3, {"A Coarse": 1.0}, {"A Fine": fine_vals},
                   "Fine at Coarse 1 (slope of ratio vs fine)"),
        FreqMapExp("fmap_fine_c2", NOTE_C3, {"A Coarse": 2.0}, {"A Fine": fine_vals},
                   "Fine at Coarse 2 (additive vs multiplicative)"),
        FreqMapExp("fmap_fixed", NOTE_C3, {"A Fix On ": 1.0}, fixed_sweep,
                   "Fix Freq x Mul -> Hz law"),
        FreqMapExp("fmap_fixed_n72", NOTE_C4, {"A Fix On ": 1.0}, fixed_sweep,
                   "fixed at +1 octave (must be note-independent)"),
        FreqMapExp("fmap_quant", NOTE_C3, {"A Quantize": 1.0}, {"A Coarse": coarse_vals},
                   "Coarse with Quantize ON (what it snaps to)"),
        # Disambiguates Mul's quantizer: with FixFreq=0.9 each Mul step s gives a clean
        # 10x-separated Hz (~119 / 1189 / 11885 for s=3/4/5). Where the measured step
        # JUMPS along the raw sweep reveals the boundary: at integers -> floor (like
        # Coarse), at x.5 -> round. The first collection couldn't tell (grid points fit
        # both), and the dataset randomizes Mul continuously, so the boundary matters.
        FreqMapExp("fmap_fixedmul", NOTE_C3,
                   {"A Fix On ": 1.0, "A Fix Freq": 0.9},
                   {"A Fix Freq Mul": _grid(2.0, 5.0, N_TRACKS)},
                   "Mul quantizer boundaries (floor vs round)"),
    ]


def build_manifest(exp: FreqMapExp, schema: OperatorSchema) -> dict:
    base = _clean_base(schema)
    base.update(exp.overrides)
    tracks, swept = {}, {}
    for i in range(N_TRACKS):
        params = dict(base)
        vals = {}
        for pname, series in exp.sweep.items():
            params[pname] = float(series[i])
            vals[pname] = float(series[i])
        name = f"op_{i:04d}"
        tracks[name] = params
        swept[name] = vals
    return {
        "device": "Operator",
        "kind": "freq_map",        # routes to "Collect Freq Maps"; Run Sweeps skips these
        "prefix": exp.name,
        "experiment": exp.name,
        "note": exp.note,          # the extension pins every clip to this pitch/velocity
        "velocity": VELOCITY,
        "purpose": exp.purpose,
        "swept": swept,
        "tracks": tracks,
    }


# --- gen + watch ----------------------------------------------------------------------

def generate(sweeps_dir: Path, schema: OperatorSchema, only: str | None = None) -> list[Path]:
    """Write the sweep manifests. `only` limits to one experiment — already-DONE folders
    keep their marker, so "Collect Freq Maps" re-exports just the new one."""
    dirs = []
    for exp in build_experiments(schema):
        if only and exp.name != only:
            continue
        d = sweeps_dir / exp.name
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(
            json.dumps(build_manifest(exp, schema), indent=2), encoding="utf-8")
        for marker in ("READY", "DONE"):
            (d / marker).unlink(missing_ok=True)
        dirs.append(d)
    print(f"Generated {len(dirs)} freq-map manifests under {sweeps_dir}")
    return dirs


def watch(dirs: list[Path], idle_timeout: float) -> None:
    from ..datagen.export_ableton import export_batch  # needs --extra datagen
    print('Now in Live: right-click a MIDI track -> "doppelganger: Collect Freq Maps".')
    print("(Rack must be built; set the export length to ~3s once.)")
    last = time.time()
    while True:
        pending = [d for d in dirs if (d / "READY").exists() and not (d / "DONE").exists()]
        if pending:
            d = pending[0]
            # Clear stale WAVs from a previous attempt FIRST: unlike Auto Collect's
            # always-fresh batch folders, these folders have fixed names, and existing
            # files would make Live's Save dialog raise an overwrite prompt that blocks
            # the keystroke automation.
            for w in d.glob("*.wav"):
                w.unlink()
            print(f"\n=== {d.name}: exporting ===", flush=True)
            try:
                n = export_batch(d, expected_count=N_TRACKS)
                print(f"=== {d.name}: {n} wav(s) exported ===", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"!! {d.name} export failed: {e}")
            finally:
                (d / "DONE").write_text("", encoding="utf-8")
            last = time.time()
        elif all((d / "DONE").exists() for d in dirs):
            print("\nAll freq-map sweeps exported. Run: "
                  "uv run python -m doppelganger.synth.freq_map analyze")
            break
        elif time.time() - last > idle_timeout:
            done = sum((d / "DONE").exists() for d in dirs)
            print(f"\nIdle {idle_timeout:.0f}s — exiting ({done}/{len(dirs)} done). "
                  'Did you run "Collect Freq Maps" in Live?')
            break
        else:
            time.sleep(0.5)


# --- analysis -------------------------------------------------------------------------

def measure_hz(wav: Path) -> float | None:
    """Dominant frequency of the steady portion, at NATIVE sample rate.
    Hann window + 8x zero-padded rFFT + parabolic peak interpolation (~sub-0.1 Hz)."""
    data, sr = sf.read(str(wav), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    a, b = int(MEASURE_WINDOW[0] * sr), int(MEASURE_WINDOW[1] * sr)
    seg = mono[a:b]
    if len(seg) < sr // 4 or float(np.sqrt(np.mean(seg ** 2))) < SILENCE_RMS:
        return None  # too short / silent
    seg = seg * np.hanning(len(seg))
    n = 8 * len(seg)
    mag = np.abs(np.fft.rfft(seg, n=n))
    lo = int(20.0 * n / sr)  # ignore DC/rumble
    k = lo + int(np.argmax(mag[lo:]))
    if 0 < k < len(mag) - 1:  # parabolic interpolation on log magnitude
        l, c, r = (math.log(max(mag[k + o], 1e-12)) for o in (-1, 0, 1))
        k = k + 0.5 * (l - r) / (l - 2 * c + r)
    return float(k * sr / n)


def cents(f_a: float, f_b: float) -> float:
    return 1200.0 * math.log2(f_a / f_b)


def _load_measurements(d: Path) -> list[dict] | None:
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    prefix = detect_prefix(d)
    if prefix is None:
        return None
    rows = []
    for track in sorted(manifest["swept"]):
        wav = d / f"{prefix} {track}.wav"
        if not wav.exists():
            continue
        hz = measure_hz(wav)
        rows.append({"track": track, **manifest["swept"][track], "hz": hz,
                     "note": manifest["note"]})
    return rows


def _fit_coarse(rows: list[dict], label: str) -> dict:
    """Score candidate raw->ratio models by MEAN |cents| error.

    Mean, not median: with a uniform raw grid ~half the points have fractional part
    < 0.5 where floor == round, so a median can score a wrong model as perfect (the
    first collection's 'round: 0.0 cents' was exactly that fluke). Points whose true
    frequency exceeds MAX_MEASURE_HZ are excluded — they alias in the render."""
    note_hz = midi_hz(rows[0]["note"])
    pts = [(r["A Coarse"], r["hz"] / note_hz) for r in rows if r["hz"] and r["hz"] > 20]
    kept = [(raw, ratio) for raw, ratio in pts
            if max(math.floor(raw), 0.5) * note_hz <= MAX_MEASURE_HZ]
    snap_set = [0.25, 0.5] + list(range(1, 49))
    models = {
        "identity (ratio = raw)": lambda x: max(x, 1e-6),
        "floor (ratio = max(floor,0.5))": lambda x: max(math.floor(x), 0.5),
        "round (ratio = round(raw))": lambda x: max(round(x), 1e-6),
        "snap {0.25,0.5,1..48}": lambda x: min(snap_set, key=lambda s: abs(s - x)),
    }
    scored = {}
    for mname, fn in models.items():
        errs = [abs(cents(ratio, fn(raw))) for raw, ratio in kept if fn(raw) > 0]
        scored[mname] = round(float(np.mean(errs)), 1) if errs else None
    best = min((m for m in scored if scored[m] is not None), key=lambda m: scored[m])
    print(f"\n[{label}] {len(pts)} measurable tracks "
          f"({len(pts) - len(kept)} excluded as aliased > {MAX_MEASURE_HZ:.0f} Hz)")
    print(f"  {'raw':>8} {'measured ratio':>15}")
    for raw, ratio in kept[:: max(1, len(kept) // 12)]:  # a readable subsample
        print(f"  {raw:8.3f} {ratio:15.4f}")
    for m, e in scored.items():
        mark = "  <- best" if m == best else ""
        print(f"  model: {m:34} mean err = {e} cents{mark}")
    return {"points": pts, "model_errors_cents": scored, "best_model": best}


def _fit_fine(rows: list[dict], coarse: float, label: str) -> dict:
    note_hz = midi_hz(rows[0]["note"])
    pts = [(r["A Fine"], r["hz"] / note_hz) for r in rows if r["hz"] and r["hz"] > 20]
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    slope, intercept = np.polyfit(x, y, 1)
    print(f"\n[{label}] ratio = {intercept:.4f} + {slope * 1000:.4f} * (fine/1000)   "
          f"(base coarse {coarse}; additive predicts slope*1000 = 1.0 at ANY coarse, "
          f"multiplicative predicts {coarse:.1f})")
    return {"coarse": coarse, "intercept": float(intercept),
            "ratio_per_1000_fine": float(slope * 1000)}


def _fit_fixed(rows: list[dict], label: str) -> dict:
    """Per Mul group, fit log10(Hz) = a + b*fixfreq over the measurable (>22 Hz) points.
    The verified law is Hz = 10^(mul_step - 3) * 200^fixfreq, i.e. base 10^(a) should
    land on a decade and the factor 10^b on ~200."""
    by_mul: dict[float, list] = {}
    for r in rows:
        if r["hz"] and r["hz"] > 1:
            by_mul.setdefault(r["A Fix Freq Mul"], []).append((r["A Fix Freq"], r["hz"]))
    out = {}
    print(f"\n[{label}]  (law: Hz = 10^(mul_step - 3) * 200^fixfreq)")
    for mul, pts in sorted(by_mul.items()):
        usable = [(a, b) for a, b in pts if b > 22]  # sub-22 Hz = below measurement floor
        if len(usable) < 4:
            print(f"  mul={mul:8.3f}: sub-audio across the knob (expected for low steps)")
            out[str(mul)] = {"points": [(float(a), float(b)) for a, b in pts], "fit": None}
            continue
        x = np.array([p[0] for p in usable]); y = np.log10([p[1] for p in usable])
        b, a = np.polyfit(x, y, 1)
        resid = float(np.abs(y - (a + b * x)).max())
        step = a + 3.0  # implied mul step if the law holds
        print(f"  mul={mul:8.3f}: base={10**a:10.4f} Hz  factor={10**b:7.2f}  "
              f"max|resid|={resid:.4f} dec  -> implied step {step:.2f}")
        out[str(mul)] = {"points": [(float(a2), float(b2)) for a2, b2 in pts],
                         "log10_base": float(a), "log10_slope": float(b),
                         "implied_step": float(step)}
    return out


def _fit_fixedmul(rows: list[dict]) -> dict:
    """Locate Mul's quantizer boundaries: with FixFreq=0.9 each step s is a clean decade
    (~119/1189/11885 Hz for s=3/4/5). Boundaries at integers -> floor; at x.5 -> round."""
    pts = [(r["A Fix Freq Mul"], r["hz"]) for r in rows if r["hz"] and r["hz"] > 22]
    steps = [(raw, round(math.log10(hz / (200.0 ** 0.9)) + 3.0)) for raw, hz in pts]
    bounds = [(steps[i][0] + steps[i + 1][0]) / 2.0
              for i in range(len(steps) - 1) if steps[i][1] != steps[i + 1][1]]
    print(f"\n[Mul quantizer] {len(pts)} measurable tracks; step transitions at raw ≈ "
          f"{[round(b, 3) for b in bounds]}")
    if bounds:
        d_int = float(np.mean([abs(b - round(b)) for b in bounds]))
        d_half = float(np.mean([abs(b - (math.floor(b) + 0.5)) for b in bounds]))
        verdict = "FLOOR (boundaries at integers, like Coarse)" if d_int < d_half \
            else "ROUND (boundaries at x.5)"
        print(f"  mean distance to integer {d_int:.3f} vs to half {d_half:.3f} -> {verdict}")
    else:
        verdict = "no transitions measured"
    return {"points": pts, "boundaries": bounds, "verdict": verdict}


def analyze(sweeps_dir: Path, report_path: Path) -> None:
    data: dict[str, list[dict]] = {}
    for name in ("fmap_coarse", "fmap_coarse_n72", "fmap_fine_c1", "fmap_fine_c2",
                 "fmap_fixed", "fmap_fixed_n72", "fmap_quant", "fmap_fixedmul"):
        d = sweeps_dir / name
        if (d / "manifest.json").exists():
            rows = _load_measurements(d)
            if rows:
                data[name] = rows
    if not data:
        raise SystemExit("No exported freq-map sweeps found — run `gen` + Collect Freq Maps first.")

    report: dict = {}
    if "fmap_coarse" in data:
        report["coarse"] = _fit_coarse(data["fmap_coarse"], "Coarse @ C3")
    if "fmap_coarse_n72" in data:
        report["coarse_n72"] = _fit_coarse(data["fmap_coarse_n72"], "Coarse @ C4")
        if "fmap_coarse" in data:  # ratio model: same raw -> same ratio at both notes
            a = {round(r["A Coarse"], 3): r["hz"] for r in data["fmap_coarse"] if r["hz"]}
            b = {round(r["A Coarse"], 3): r["hz"] for r in data["fmap_coarse_n72"] if r["hz"]}
            diffs = [abs(cents(b[k], a[k]) - 1200.0) for k in a if k in b and a[k] > 20 and b[k] > 20]
            med = float(np.median(diffs)) if diffs else None
            print(f"\n[octave check] median |cents(C4/C3) - 1200| = {med:.1f} cents "
                  f"-> Coarse {'IS' if med is not None and med < 10 else 'is NOT cleanly'} "
                  f"a note-proportional ratio")
            report["coarse_octave_check_cents"] = med
    if "fmap_fine_c1" in data:
        report["fine_c1"] = _fit_fine(data["fmap_fine_c1"], 1.0, "Fine @ Coarse 1")
    if "fmap_fine_c2" in data:
        report["fine_c2"] = _fit_fine(data["fmap_fine_c2"], 2.0, "Fine @ Coarse 2")
        if "fine_c1" in report:
            s1, s2 = report["fine_c1"]["ratio_per_1000_fine"], report["fine_c2"]["ratio_per_1000_fine"]
            verdict = "ADDITIVE (ratio += fine/1000 * k)" if abs(s2 - s1) < 0.25 * abs(s1) \
                else "MULTIPLICATIVE (scales with coarse)"
            print(f"\n[fine verdict] slopes {s1:.3f} vs {s2:.3f} -> {verdict}")
            report["fine_verdict"] = verdict
    if "fmap_fixed" in data:
        report["fixed"] = _fit_fixed(data["fmap_fixed"], "Fixed Freq @ C3")
    if "fmap_fixed_n72" in data:
        report["fixed_n72"] = _fit_fixed(data["fmap_fixed_n72"], "Fixed Freq @ C4")
        if "fmap_fixed" in data:
            key = lambda r: (round(r["A Fix Freq Mul"], 4), round(r["A Fix Freq"], 4))  # noqa: E731
            a = {key(r): r["hz"] for r in data["fmap_fixed"] if r["hz"] and r["hz"] > 1}
            b = {key(r): r["hz"] for r in data["fmap_fixed_n72"] if r["hz"] and r["hz"] > 1}
            diffs = [abs(cents(b[k], a[k])) for k in a if k in b]
            med = float(np.median(diffs)) if diffs else None
            print(f"\n[fixed note-independence] median |cents(C4 vs C3)| = {med:.1f} "
                  f"-> {'note-independent (confirmed)' if med is not None and med < 10 else 'NOTE-DEPENDENT?!'}")
            report["fixed_note_independence_cents"] = med
    if "fmap_fixedmul" in data:
        report["fixedmul"] = _fit_fixedmul(data["fmap_fixedmul"])
    if "fmap_quant" in data:
        note_hz = midi_hz(data["fmap_quant"][0]["note"])
        pts = [(r["A Coarse"], r["hz"] / note_hz) for r in data["fmap_quant"]
               if r["hz"] and r["hz"] > 20]
        print(f"\n[Quantize ON] raw -> measured ratio ({len(pts)} tracks):")
        for raw, ratio in pts[:: max(1, len(pts) // 16)]:
            print(f"  {raw:8.3f} -> {ratio:.4f}")
        report["quantize"] = {"points": pts}

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report -> {report_path}")


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Operator frequency-mapping sweeps.")
    ap.add_argument("--schema", default=str(root / "schemas" / "operator.json"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="write manifests, then watch + auto-export")
    g.add_argument("--idle-timeout", type=float, default=180.0)
    g.add_argument("--no-watch", action="store_true", help="only write the manifests")
    g.add_argument("--only", default=None, help="generate a single experiment (e.g. fmap_fixedmul)")
    sub.add_parser("analyze", help="measure frequencies + fit the mappings")
    args = ap.parse_args()

    sweeps_dir = root / "dataset" / "sweeps"
    if args.cmd == "gen":
        dirs = generate(sweeps_dir, OperatorSchema.load(args.schema), args.only)
        if not args.no_watch:
            watch(dirs, args.idle_timeout)
    else:
        analyze(sweeps_dir, root / "dataset" / "freq_maps_report.json")


if __name__ == "__main__":
    main()
