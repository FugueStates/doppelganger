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
  | { kind: "randomInt"; lo: number; hi: number } // random integer raw value in [lo,hi]
  | { kind: "fracRange"; lo: number; hi: number }; // sample within a fraction of [min,max]

const forceMax: Rule = { kind: "forceMax" };
const forceIndex = (index: number): Rule => ({ kind: "forceIndex", index });
const forceValue = (value: number): Rule => ({ kind: "forceValue", value });
const randomChoice = (indices: number[]): Rule => ({ kind: "randomChoice", indices });
const randomInt = (lo: number, hi: number): Rule => ({ kind: "randomInt", lo, hi });
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

/**
 * Staged dataset expansion (matcher-v3). Each stage adds capability; pick which the
 * collector generates. Stage 0 (passed): single-osc waveform+ADSR. Stage 1: 2-operator
 * FM (B->A) — adds the modulator (Osc-B) ratio + level, the core FM controls.
 */
export const COLLECTION_STAGE: number = 3;

/** Stage 1 — 2-operator FM. Carrier A (waveform + amp env, pitch = note) modulated by B
 *  (sine, varying ratio + level = modulation index, steady envelope). C/D off; the linear
 *  A<-B chain is algorithm 0 with C/D disabled. Everything else neutral/off. */
function buildStage1Rules(): Record<string, Rule> {
  return {
    "Device On": forceMax,
    Volume: forceValue(0.8),
    Algorithm: forceIndex(0), // D>C>B>A chain; with C,D off this is B->A
    Transpose: forceValue(0),
    Spread: forceValue(0),
    "Glide On": forceIndex(0),
    Panorama: forceValue(0),
    // carrier A: pitch = note, varying waveform + amp envelope (kept from Stage 0)
    "Osc-A On": forceMax,
    "Osc-A Level": forceMax,
    "Osc-A Wave": randomChoice(SNIFF_WAVEFORMS),
    "Osc-A Feedb": forceValue(0),
    "A Fix On ": forceIndex(0),
    "A Coarse": forceValue(1), // ratio 1 (carrier tracks the note)
    "A Fine": forceValue(0),
    "Ae Mode": forceIndex(0),
    "Ae Init": forceValue(0),
    "Ae Peak": forceMax,
    "Ae Attack": frac(0, 1),
    "Ae Decay": frac(0, 1),
    "Ae Sustain": frac(0, 1),
    "Ae Release": frac(0, 1),
    // modulator B -> A: sine, varying RATIO (Coarse) + LEVEL (modulation index), steady env
    "Osc-B On": forceMax,
    "Osc-B Wave": forceIndex(0), // sine (classic FM modulator)
    "Osc-B Level": frac(0, 1), // modulation index (the key FM brightness control)
    "B Coarse": randomInt(1, 16), // modulator ratio (categorical integer ratio)
    "B Fine": forceValue(0),
    "Osc-B Feedb": forceValue(0),
    "B Fix On ": forceIndex(0),
    "Be Mode": forceIndex(0),
    "Be Init": forceValue(0),
    "Be Peak": forceMax,
    "Be Attack": forceValue(0), // steady modulator (constant FM over the note) — Stage 1
    "Be Decay": forceValue(0),
    "Be Sustain": forceMax,
    "Be Release": forceValue(0.1),
    // C, D off; all timbre-coloring sections off
    "Osc-C On": forceIndex(0),
    "Osc-D On": forceIndex(0),
    "Filter On": forceIndex(0),
    "LFO On": forceIndex(0),
    "Pe On": forceIndex(0),
    "Shaper Mix": forceValue(0),
    "Shaper Drive": forceValue(0),
  };
}

/** Stage 2 — 3-operator FM, PARALLEL modulators (B→A and C→A). All oscillators ON; PRESENCE
 *  is controlled by LEVEL (level 0 = silent), not the On toggle. Carrier A (waveform + amp
 *  env, pitch = note); modulators B and C (sine, varying ratio + level, steady env) each
 *  modulate the carrier DIRECTLY — algorithm 7 (D>A, C>A, B>A) with D's level = 0.
 *  (Pivoted from the serial chain C→B→A / algo 0: a deep serial chain entangles the
 *  operators and is hard to invert — see docs/matcher-v3.md Finding #6. Parallel modulators
 *  put each operator's sidebands directly on the carrier → far more separable, and each
 *  operator's effect depends only on its own level, so the own-level weighting is exact.)
 *  Mixing C's level toward 0 makes a sample effectively 2-op, so the set spans 2–3 ops. */
function buildStage2Rules(): Record<string, Rule> {
  const carrierAndGlobals: Record<string, Rule> = {
    "Device On": forceMax,
    Volume: forceValue(0.8),
    Algorithm: forceIndex(6), // algo 7: D>A, C>A, B>A (parallel modulators on carrier A)
    Transpose: forceValue(0),
    Spread: forceValue(0),
    "Glide On": forceIndex(0),
    Panorama: forceValue(0),
    // every oscillator ON — presence is set by level, not the toggle
    "Osc-A On": forceMax,
    "Osc-B On": forceMax,
    "Osc-C On": forceMax,
    "Osc-D On": forceMax,
    // carrier A
    "Osc-A Level": forceMax,
    "Osc-A Wave": randomChoice(SNIFF_WAVEFORMS),
    "Osc-A Feedb": forceValue(0),
    "A Fix On ": forceIndex(0),
    "A Coarse": forceValue(1),
    "A Fine": forceValue(0),
    "Ae Mode": forceIndex(0),
    "Ae Init": forceValue(0),
    "Ae Peak": forceMax,
    "Ae Attack": frac(0, 1),
    "Ae Decay": frac(0, 1),
    "Ae Sustain": frac(0, 1),
    "Ae Release": frac(0, 1),
    "Osc-D Level": forceValue(0), // D disabled via level (so the chain is C→B→A)
    "Filter On": forceIndex(0),
    "LFO On": forceIndex(0),
    "Pe On": forceIndex(0),
    "Shaper Mix": forceValue(0),
    "Shaper Drive": forceValue(0),
  };
  // a sine modulator with varying ratio + level (= FM index) and a steady envelope
  const modulator = (X: string, e: string): Record<string, Rule> => ({
    [`Osc-${X} Wave`]: forceIndex(0), // sine
    [`Osc-${X} Level`]: frac(0, 1), // modulation index (0 ⇒ this operator is effectively off)
    [`${X} Coarse`]: randomInt(1, 16), // ratio (categorical)
    [`${X} Fine`]: forceValue(0),
    [`Osc-${X} Feedb`]: forceValue(0),
    [`${X} Fix On `]: forceIndex(0),
    [`${e} Mode`]: forceIndex(0),
    [`${e} Init`]: forceValue(0),
    [`${e} Peak`]: forceMax,
    [`${e} Attack`]: forceValue(0),
    [`${e} Decay`]: forceValue(0),
    [`${e} Sustain`]: forceMax,
    [`${e} Release`]: forceValue(0.1),
  });
  return { ...carrierAndGlobals, ...modulator("B", "Be"), ...modulator("C", "Ce") };
}

/** Stage 3 — STATIC FILTER sniff. A SINGLE harmonically-rich carrier (saw / square, so the
 *  filter has partials to act on) + amp ADSR, with the global filter ENGAGED and swept:
 *  Type (LP / HP / BP), Freq (cutoff), Res, Slope. The filter ENVELOPE is OFF (Fe Amount 0 →
 *  STATIC filter); the sweep is Stage 4. No FM (B/C/D silent) so filter learning is ISOLATED
 *  from the modulator-ratio identifiability ceiling. Note is pinned to C3 by SNIFF_MODE —
 *  ideal here, since the cutoff is an absolute frequency and varying pitch would couple.
 *  The codec freezes the rest of the filter section to match (circuits/drive/morph/vel-key/
 *  LFO + the Fe envelope) — see matcher/codec.py FILTER_FROZEN. */
function buildStage3Rules(): Record<string, Rule> {
  return {
    "Device On": forceMax,
    Volume: forceValue(0.8),
    Algorithm: forceIndex(0),
    Transpose: forceValue(0),
    Spread: forceValue(0),
    "Glide On": forceIndex(0),
    Panorama: forceValue(0),
    // carrier A = the only voice: rich waveform (saw/square) + amp envelope, pitch = note
    "Osc-A On": forceMax,
    "Osc-A Level": forceMax,
    "Osc-A Wave": randomChoice([9, 17]), // Saw 64 / Square 64 — rich harmonics for the filter
    "Osc-A Feedb": forceValue(0),
    "A Fix On ": forceIndex(0),
    "A Coarse": forceValue(1),
    "A Fine": forceValue(0),
    "Ae Mode": forceIndex(0),
    "Ae Init": forceValue(0),
    "Ae Peak": forceMax,
    "Ae Attack": frac(0, 1),
    "Ae Decay": frac(0, 1),
    "Ae Sustain": frac(0, 1),
    "Ae Release": frac(0, 1),
    // B/C/D silent — no FM, isolate the filter
    "Osc-B On": forceIndex(0),
    "Osc-C On": forceIndex(0),
    "Osc-D On": forceIndex(0),
    // THE GLOBAL FILTER — engaged + swept (the Stage-3 learning target)
    "Filter On": forceMax,
    "Filter Type": randomChoice([0, 1, 2]), // Lowpass / Highpass / Bandpass
    "Filter Freq": frac(0.15, 0.9), // cutoff (avoid extremes that filter to near-silence)
    "Filter Res": frac(0.0, 0.6), // resonance (max raw 1.25; stay below the self-osc squeal)
    "Filter Slope": randomChoice([0, 1]), // 12 / 24 dB
    "Filter Morph": forceValue(0),
    "Filter Drive": forceValue(0),
    "Filter Circuit - LP/HP": forceIndex(0), // Clean
    "Filter Circuit - BP/NO/Morph": forceIndex(0), // Clean
    "Filt < Vel": forceValue(0),
    "Filt < Key": forceValue(0),
    "Filt < LFO": forceIndex(0),
    "Fe Amount": forceValue(0), // filter ENVELOPE off → static filter (Stage 4 unfreezes)
    "Fe Mode": forceIndex(0),
    // other timbre-coloring sections off
    "LFO On": forceIndex(0),
    "Pe On": forceIndex(0),
    "Shaper Mix": forceValue(0),
    "Shaper Drive": forceValue(0),
  };
}

/** Active rule set when SNIFF_MODE; unruled params fall back to their neutral default. */
export const SNIFF_RULES: Record<string, Rule> =
  COLLECTION_STAGE >= 3 ? buildStage3Rules()
    : COLLECTION_STAGE === 2 ? buildStage2Rules()
    : COLLECTION_STAGE === 1 ? buildStage1Rules()
    : buildSniffRules();

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
