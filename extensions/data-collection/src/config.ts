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

/**
 * Played note + velocity.
 *
 * We now VARY the note and velocity per sample (each was previously fixed at C3/100),
 * so the renderer can be conditioned on pitch — Operator's timbre is NOT pitch-invariant
 * (Fixed-frequency operators and the absolute-Hz filter change the sound with the note),
 * and velocity modulates level/FM index. The chosen note+velocity are recorded per sample
 * in the manifest -> dataset JSON so training (Batch 5) can condition on them.
 *
 * Set RANDOMIZE_NOTE = false to fall back to the old fixed C3/100 behavior.
 */
export const NOTE_PITCH = 60; // C3 — fallback / initial-clip note (Live's convention)
export const NOTE_VELOCITY = 100;
export const RANDOMIZE_NOTE = true;
/** MIDI note range to sample from. 36..84 = C1..C5 — most synth-usable pitches; the very
 *  top is where 16 kHz capture starts losing bright FM partials above the 8 kHz Nyquist. */
export const NOTE_PITCH_MIN = 36;
export const NOTE_PITCH_MAX = 84;
/** Velocity range. Kept moderately high so velocity->level scaling doesn't push presets
 *  below the RMS audibility filter (which would waste renders). */
export const VELOCITY_MIN = 70;
export const VELOCITY_MAX = 127;
/** Note length: SHORT relative to the render so the release tail is captured.
 *  3 beats = 1.5 s note, then note-off + release within a 3 s render. Without this
 *  the held note filled the whole render and decay/release params were unhearable. */
export const NOTE_BEATS = 3;

/** Export length — set this as your export selection/loop in Live (note + release tail). */
export const RENDER_SECONDS = 3.0;

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
  | { kind: "forceValue"; value: number } // absolute raw value (clamped to [min,max])
  | { kind: "randomChoice"; indices: number[] } // quantized -> random pick from a set
  | { kind: "fracRange"; lo: number; hi: number }; // sample within a fraction of [min,max]

const forceMax: Rule = { kind: "forceMax" };
const forceIndex = (index: number): Rule => ({ kind: "forceIndex", index });
const forceValue = (value: number): Rule => ({ kind: "forceValue", value });
const randomChoice = (indices: number[]): Rule => ({ kind: "randomChoice", indices });
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

/**
 * SNIFF-TEST MODE (matcher-v3 branch). When true, "Randomize Batch" / "Auto Collect"
 * generate the minimal validation set: a SINGLE oscillator (A) playing one of four basic
 * waveforms with a random ADSR envelope, EVERYTHING else neutral/off. This is the gate for
 * the rebuilt matcher — if it can't learn waveform + envelope from these, the method is
 * wrong. Collect ~512 (≈8 batches of 64), then train `doppelganger.matcher.train_matcher`.
 *
 * SET FALSE before any full-timbre data collection. In sniff mode the note is pinned to C3.
 */
export const SNIFF_MODE = true;
/** Osc-A Wave indices for the 4 basic waveforms (from the schema valueItems):
 *  0 = Sine, 9 = Saw 64, 17 = Square 64, 19 = Triangle. */
export const SNIFF_WAVEFORMS = [0, 9, 17, 19];

function buildSniffRules(): Record<string, Rule> {
  return {
    // global: audible, neutral, deterministic
    "Device On": forceMax,
    Volume: forceValue(0.8),
    Algorithm: forceIndex(0),
    Transpose: forceValue(0), // 0 semitones
    Spread: forceValue(0),
    "Glide On": forceIndex(0),
    Panorama: forceValue(0),
    // oscillator A = the only voice; sweep its waveform + amp envelope
    "Osc-A On": forceMax,
    "Osc-A Level": forceMax,
    "Osc-A Wave": randomChoice(SNIFF_WAVEFORMS),
    "Osc-A Feedb": forceValue(0), // no feedback (keeps the waveform clean)
    "A Fix On ": forceIndex(0), // tracks the note
    "A Coarse": forceValue(1), // ratio 1
    "A Fine": forceValue(0),
    "Ae Mode": forceIndex(0), // None = standard ADSR
    "Ae Init": forceValue(0),
    "Ae Peak": forceMax,
    "Ae Attack": frac(0, 1), // <-- the four things we actually vary + learn
    "Ae Decay": frac(0, 1),
    "Ae Sustain": frac(0, 1),
    "Ae Release": frac(0, 1),
    // silence the other oscillators and all timbre-coloring sections
    "Osc-B On": forceIndex(0),
    "Osc-C On": forceIndex(0),
    "Osc-D On": forceIndex(0),
    "Filter On": forceIndex(0),
    "LFO On": forceIndex(0),
    "Pe On": forceIndex(0),
    "Shaper Mix": forceValue(0),
    "Shaper Drive": forceValue(0),
  };
}

/** Active rule set when SNIFF_MODE; unruled params fall back to their neutral default. */
export const SNIFF_RULES: Record<string, Rule> = buildSniffRules();

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

/** Root of the designed-sweep folders (each subdir has a manifest.json). */
export const SWEEPS_DIR = `${REPO_ROOT}/dataset/sweeps`;

/** In-the-loop CMA-ES search handshake directory. */
export const SEARCH_DIR = `${REPO_ROOT}/dataset/search`;
