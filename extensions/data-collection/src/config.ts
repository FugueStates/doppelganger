/**
 * Shared configuration for the data-collection extension.
 *
 * Tune NUM_TRACKS to what your machine handles comfortably (more tracks = more
 * samples per export = better throughput, until CPU/RAM complains). The render
 * length is set in Ableton's Export dialog, not here — keep it consistent
 * (RENDER_SECONDS) so every sample has the same duration.
 */

export type ApiVersion = "1.0.0";

export const INSTRUMENT_NAME = "Operator";

/** Track-name prefix so exported files are identifiable and return tracks are ignored. */
export const TRACK_PREFIX = "op_";

/** How many Operator tracks the rack contains (= samples per export). */
export const NUM_TRACKS = 64;

/** The MIDI note every track plays (fixed for v1, like the sound-matching literature). */
export const NOTE_PITCH = 60; // C3 in Live's convention
export const NOTE_VELOCITY = 100;
export const NOTE_BEATS = 4; // note length in beats (2.0 s @ 120 BPM)

/** Suggested export length — set this as your export selection/loop in Live. */
export const RENDER_SECONDS = 3.0; // note (2 s) + ~1 s release tail

/**
 * Sampling rules — the fix for "random params = silence".
 *
 * Fully uniform-random parameters produce mostly silent/inaudible presets (random
 * long attack, zero osc level, closed filter, low volume, oscillators off). So we
 * pin the few *audibility-gating* params to musical ranges and let everything
 * *timbral* (algorithm, waveforms, tuning, FM feedback, LFO, pitch env, filter
 * type/res/drive, shaper, spread, pan, glide, …) stay fully random for diversity.
 *
 * Strategy: guarantee audibility through oscillator A only (Operator's primary
 * carrier) — force A on with a sane amp envelope and solid level, plus open the
 * global filter/volume. Oscillators B/C/D stay FULLY random (on/off, level,
 * envelope, tuning), so we still get clean single-oscillator tones AND rich FM.
 *
 * (Assumes A is a carrier in every algorithm — if the next batch still has silent
 * renders, that assumption is wrong and we widen the forced set.)
 */
export type Rule =
  | { kind: "forceMax" } // quantized toggle -> "On"
  | { kind: "forceIndex"; index: number } // quantized -> a specific option
  | { kind: "fracRange"; lo: number; hi: number }; // sample within a fraction of [min,max]

const forceMax: Rule = { kind: "forceMax" };
const forceIndex = (index: number): Rule => ({ kind: "forceIndex", index });
const frac = (lo: number, hi: number): Rule => ({ kind: "fracRange", lo, hi });

function buildSamplingRules(): Record<string, Rule> {
  return {
    "Device On": forceMax,
    Volume: frac(0.45, 1.0),
    "Filter Freq": frac(0.35, 1.0),
    // Oscillator A = guaranteed audible carrier:
    "Osc-A On": forceMax,
    "Osc-A Level": frac(0.5, 1.0),
    "Ae Mode": forceIndex(0), // "None" -> standard ADSR (predictable sustain)
    "Ae Attack": frac(0.0, 0.1), // short enough to sound within the note
    "Ae Peak": frac(0.6, 1.0),
    "Ae Sustain": frac(0.5, 1.0),
    "Ae Init": frac(0.0, 0.5),
    // Keep the CARRIER's pitch in an audible range (the "outside hearing range"
    // fix). Modulators B/C/D keep wild Coarse ratios -> that's FM timbre, not pitch.
    "A Fix On ": forceIndex(0), // (trailing space is the real param name) off -> A tracks the note
    "A Coarse": frac(1 / 48, 4 / 48), // multiplier ~1..4 (unison to ~2 octaves up)
    Transpose: frac(0.375, 0.625), // -12..+12 semitones (range is -48..48)
    // Oscillators B/C/D and everything else stay fully random (no rule).
  };
}

export const SAMPLING_RULES: Record<string, Rule> = buildSamplingRules();

// --- Output paths (absolute on this machine) -------------------------------
export const REPO_ROOT = "D:/AbletonExtensions/doppelganger";
export const PENDING_DIR = `${REPO_ROOT}/dataset/pending`;
export const STATE_FILE = `${REPO_ROOT}/dataset/state.json`;

// --- Unattended auto-collect loop ------------------------------------------
/** How many batches "Auto Collect" runs before stopping (NUM_TRACKS samples each).
 *  1000 × 64 ≈ 64k renders (~48k usable after the RMS filter) in ~3–4 hours. */
export const MAX_BATCHES = 1000;
/** How long to wait for the Python watcher to export+process one batch (ms). */
export const BATCH_TIMEOUT_MS = 180_000;
/** Create this file to stop the loop early. */
export const STOP_FILE = `${REPO_ROOT}/dataset/STOP`;

/** Manifest of model-predicted params to apply for the "hear it" evaluation. */
export const PREDICT_MANIFEST = `${REPO_ROOT}/dataset/predict/manifest.json`;
