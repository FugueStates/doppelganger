/**
 * Operator rack management + parameter randomization.
 *
 * The rack is a set of MIDI tracks named op_0000, op_0001, … each holding an
 * Operator and a fixed MIDI note. We build it ONCE, then re-randomize the same
 * tracks every batch (so exported filenames stay stable and map back to tracks).
 */

import {
  type initialize,
  MidiTrack,
  type Device,
  type DeviceParameter,
} from "@ableton-extensions/sdk";

import {
  type ApiVersion,
  INSTRUMENT_NAME,
  TRACK_PREFIX,
  NOTE_PITCH,
  NOTE_VELOCITY,
  NOTE_BEATS,
  RANDOMIZE_NOTE,
  NOTE_PITCH_MIN,
  NOTE_PITCH_MAX,
  VELOCITY_MIN,
  VELOCITY_MAX,
  SAMPLING_RULES,
  SNIFF_MODE,
  SNIFF_RULES,
} from "./config.js";

type Ctx = ReturnType<typeof initialize>;

export function trackName(index: number): string {
  return `${TRACK_PREFIX}${String(index).padStart(4, "0")}`;
}

/** All MIDI tracks that are part of our Operator rack (named op_* with Operator on them). */
export function findOperatorTracks(context: Ctx): MidiTrack<ApiVersion>[] {
  const song = context.application.song;
  if (!song) return [];
  return song.tracks.filter(
    (t): t is MidiTrack<ApiVersion> =>
      t instanceof MidiTrack &&
      t.name.startsWith(TRACK_PREFIX) &&
      t.devices.length > 0 &&
      t.devices[0]!.name === INSTRUMENT_NAME,
  );
}

/** Ensures the rack has `count` Operator tracks; creates any that are missing. Returns total. */
export async function buildRack(context: Ctx, count: number): Promise<number> {
  const song = context.application.song;
  if (!song) throw new Error("No active song.");

  const existing = findOperatorTracks(context).length;
  for (let i = existing; i < count; i++) {
    const track = await song.createMidiTrack();
    track.name = trackName(i);
    await track.insertDevice(INSTRUMENT_NAME, 0);

    const clip = await track.createMidiClip(0, NOTE_BEATS);
    clip.notes = [
      { pitch: NOTE_PITCH, startTime: 0, duration: NOTE_BEATS, velocity: NOTE_VELOCITY },
    ];
  }
  return findOperatorTracks(context).length;
}

/** Picks a value for one parameter. In SNIFF_MODE uses SNIFF_RULES and defaults unruled
 *  params to their neutral default; otherwise uses SAMPLING_RULES and full-random. */
function chooseValue(param: DeviceParameter<ApiVersion>): number {
  const rules = SNIFF_MODE ? SNIFF_RULES : SAMPLING_RULES;
  const rule = rules[param.name];
  const span = param.max - param.min;
  const nOpts = param.valueItems.length || 1;

  if (rule) {
    if (rule.kind === "forceMax") return param.max;
    if (rule.kind === "forceIndex") return Math.min(nOpts - 1, Math.max(0, rule.index));
    if (rule.kind === "forceValue") return Math.min(param.max, Math.max(param.min, rule.value));
    if (rule.kind === "randomChoice") {
      const pick = rule.indices[Math.floor(Math.random() * rule.indices.length)]!;
      return Math.min(nOpts - 1, Math.max(0, pick));
    }
    // fracRange: sample within a fraction of the full range
    const frac = rule.lo + Math.random() * (rule.hi - rule.lo);
    const value = param.min + frac * span;
    return param.isQuantized ? Math.round(value) : value;
  }

  // SNIFF_MODE: every unruled param stays at its neutral default (clean single-osc patch).
  if (SNIFF_MODE) return param.defaultValue;

  // Normal mode, no rule -> fully random (timbral params).
  if (param.isQuantized) return Math.floor(Math.random() * nOpts);
  return param.min + Math.random() * span;
}

/** Per-track record of the raw parameter values that were applied. */
export type TrackParams = Record<string, number>;

/** Per-track played-note record (pitch is a MIDI note number; velocity 0..127). */
export type NoteInfo = { pitch: number; velocity: number };

/** Inclusive random integer in [lo, hi]. */
function randInt(lo: number, hi: number): number {
  return lo + Math.floor(Math.random() * (hi - lo + 1));
}

/** Replaces a track's clip with a fresh one-note clip (the buildRack primitive). */
async function setClipNote(
  track: MidiTrack<ApiVersion>,
  pitch: number,
  velocity: number,
): Promise<void> {
  await track.clearClipsInRange(0, NOTE_BEATS);
  const clip = await track.createMidiClip(0, NOTE_BEATS);
  clip.notes = [{ pitch, startTime: 0, duration: NOTE_BEATS, velocity }];
}

/**
 * Re-randomizes the played note + velocity on every rack track for this batch by
 * REPLACING each track's clip (clear the [0, NOTE_BEATS) range, then create a fresh
 * one-note clip). Uses only the same primitives buildRack already relies on, so it's
 * robust for long unattended runs. Returns track name -> { pitch, velocity } for the
 * manifest. With RANDOMIZE_NOTE = false it pins the old fixed C3/100 note.
 */
export async function randomizeNotes(
  context: Ctx,
  tracks: MidiTrack<ApiVersion>[],
): Promise<Record<string, NoteInfo>> {
  const recorded: Record<string, NoteInfo> = {};
  // SNIFF_MODE pins to a fixed C3 so the test isolates waveform + envelope (no pitch variance).
  const randomize = RANDOMIZE_NOTE && !SNIFF_MODE;
  for (const track of tracks) {
    const pitch = randomize ? randInt(NOTE_PITCH_MIN, NOTE_PITCH_MAX) : NOTE_PITCH;
    const velocity = randomize ? randInt(VELOCITY_MIN, VELOCITY_MAX) : NOTE_VELOCITY;
    await setClipNote(track, pitch, velocity);
    recorded[track.name] = { pitch, velocity };
  }
  return recorded;
}

/**
 * Pins EVERY rack track's clip to one known note + velocity. Measurement sweeps
 * (freq_map.py) need this: after Auto Collect the rack's clips hold RANDOM pitches,
 * which would make measured frequencies meaningless — each freq-map manifest declares
 * the exact note it must be rendered at.
 */
export async function pinNotes(
  context: Ctx,
  tracks: MidiTrack<ApiVersion>[],
  pitch: number,
  velocity: number,
): Promise<void> {
  for (const track of tracks) {
    await setClipNote(track, pitch, velocity);
  }
}

/**
 * Randomizes every parameter on every rack track in a single transaction.
 * Returns a map of track name -> { paramName: rawValue } for the manifest.
 */
export async function randomizeAll(
  context: Ctx,
  tracks: MidiTrack<ApiVersion>[],
): Promise<Record<string, TrackParams>> {
  const recorded: Record<string, TrackParams> = {};

  const promises = context.withinTransaction(() =>
    tracks.flatMap((track) => {
      const device = track.devices[0] as Device<ApiVersion>;
      const params: TrackParams = {};
      recorded[track.name] = params;
      return device.parameters.map((p) => {
        const value = chooseValue(p);
        params[p.name] = value;
        return p.setValue(value);
      });
    }),
  );

  await Promise.all(promises);
  return recorded;
}

/**
 * Applies specific parameter values (e.g. model predictions) to the rack tracks,
 * by track name. Used by the "hear it" eval to render what the model predicted.
 */
export async function applyParams(
  context: Ctx,
  tracks: MidiTrack<ApiVersion>[],
  tracksParams: Record<string, TrackParams>,
): Promise<number> {
  let applied = 0;
  const promises = context.withinTransaction(() =>
    tracks.flatMap((track) => {
      const params = tracksParams[track.name];
      if (!params) return [];
      applied++;
      const device = track.devices[0] as Device<ApiVersion>;
      return device.parameters
        .filter((p) => p.name in params)
        .map((p) => p.setValue(params[p.name]!));
    }),
  );
  await Promise.all(promises);
  return applied;
}
