/**
 * doppelganger — Phase 0: Render Validation Spike
 * =================================================
 *
 * PURPOSE
 * -------
 * Answer the project's make-or-break question:
 *
 *     Can an Ableton Extension render an *instrument* (MIDI) track's audio to a WAV?
 *
 * The SDK only documents `resources.renderPreFxAudio(track: AudioTrack, ...)`, and there is
 * no freeze / bounce / record API. Instruments (Operator) live on MIDI tracks. So we test
 * empirically whether `renderPreFxAudio` accepts a MIDI track at runtime.
 *
 * WHAT THIS RUN REPORTS
 * ---------------------
 *  TEST 1 (MIDI):  MIDI track + Operator + one C3 note  -> try render -> PASS/FAIL + details
 *  TEST 2 (AUDIO): empty AudioTrack                      -> try render -> PASS/FAIL  (control)
 *
 * The control test matters: if AUDIO renders but MIDI doesn't, that is conclusive proof the
 * limitation is specifically "instrument tracks can't be rendered" (not a broken render API).
 *
 * Trigger: right-click any MIDI track header -> "doppelganger: Test Operator Render".
 * It cleans up the tracks it creates.
 */

import {
  initialize,
  type ActivationContext,
  type AudioTrack,
  type MidiTrack,
  type Track,
  type Device,
} from "@ableton-extensions/sdk";

import * as fs from "fs/promises";
import * as path from "path";
import { decodeWav } from "./wav.js";

// The SDK's data-model classes are generic over the API version they target.
type ApiVersion = "1.0.0";

// ---- Tunables --------------------------------------------------------------
const INSTRUMENT_NAME = "Operator";
const NOTE_PITCH = 60; // C3 in Live's convention
const NOTE_VELOCITY = 100;
const NOTE_BEATS = 2; // note length
const RENDER_BEATS = 4; // render window (note + release tail)

// Where the dumped parameter schema is written (absolute path on this machine).
const SCHEMA_OUT = "D:/AbletonExtensions/doppelganger/schemas/operator.json";
// ---------------------------------------------------------------------------

type Ctx = ReturnType<typeof initialize>;

export function activate(activation: ActivationContext) {
  const context = initialize(activation, "1.0.0");

  context.commands.registerCommand("doppelganger.testRender", () =>
    void runRenderTest(context).catch((e) =>
      console.error("[doppelganger]", formatError(e)),
    ),
  );

  context.ui.registerContextMenuAction(
    "MidiTrack",
    "doppelganger: Test Operator Render",
    "doppelganger.testRender",
  );

  context.commands.registerCommand("doppelganger.dumpSchema", () =>
    void dumpOperatorSchema(context).catch((e) =>
      console.error("[doppelganger]", formatError(e)),
    ),
  );

  context.ui.registerContextMenuAction(
    "MidiTrack",
    "doppelganger: Dump Operator Schema",
    "doppelganger.dumpSchema",
  );
}

async function runRenderTest(context: Ctx): Promise<void> {
  const song = context.application.song;
  if (!song) {
    await report(context, ["FAIL: No active song."]);
    return;
  }

  const lines: string[] = [];
  const cleanup: Track<ApiVersion>[] = [];

  try {
    await context.ui.withinProgressDialog(
      "doppelganger render test",
      { progress: 0 },
      async (update) => {
        // ---- TEST 1: MIDI track + Operator -------------------------------
        update("TEST 1: MIDI + Operator…", 20);
        lines.push("=== TEST 1: MIDI track + Operator ===");
        const midi = await buildTestTrack(context);
        cleanup.push(midi.track);
        lines.push(`Instrument inserted: "${midi.deviceName}"`);
        lines.push(`Parameter count: ${midi.paramCount}`);

        update("TEST 1: rendering MIDI track…", 45);
        await renderInto(lines, context, midi.track, "MIDI");

        // ---- TEST 2: empty AudioTrack (control) --------------------------
        update("TEST 2: empty AudioTrack…", 70);
        lines.push("");
        lines.push("=== TEST 2: empty AudioTrack (control) ===");
        const audio = await song.createAudioTrack();
        cleanup.push(audio);
        await renderInto(lines, context, audio, "AUDIO");

        update("Done", 100);
      },
    );
  } catch (e) {
    lines.push("");
    lines.push("UNEXPECTED ERROR (outside the isolated render calls):");
    lines.push(formatError(e));
  } finally {
    for (const t of cleanup) {
      await song.deleteTrack(t).catch(() => undefined);
    }
  }

  // Interpretation footer
  lines.push("");
  lines.push("--- How to read this ---");
  lines.push("MIDI PASS + non-silent  -> Tool 1 can render Operator directly. 🎉");
  lines.push("MIDI FAIL + AUDIO PASS  -> instrument tracks can't be rendered -> AbletonOSC.");
  lines.push("Both FAIL               -> render API issue; send me the errors.");

  await report(context, lines);
}

/** Runs renderPreFxAudio on a track with isolated error capture and appends results. */
async function renderInto(
  lines: string[],
  context: Ctx,
  track: Track<ApiVersion>,
  label: string,
): Promise<void> {
  // renderPreFxAudio is typed AudioTrack-only; cast so we can test any track at runtime.
  let wavPath: string;
  try {
    wavPath = await context.resources.renderPreFxAudio(
      track as unknown as AudioTrack<ApiVersion>,
      0,
      RENDER_BEATS,
    );
  } catch (e) {
    lines.push(`${label} RENDER: ❌ FAIL at renderPreFxAudio()`);
    lines.push(`  error: ${formatError(e)}`);
    return;
  }

  lines.push(`${label} RENDER: ✅ renderPreFxAudio returned a path`);
  lines.push(`  path: ${wavPath}`);

  try {
    const a = analyze(decodeWav(await fs.readFile(wavPath)));
    lines.push(
      `  ${a.sampleRate} Hz, ${a.channels} ch, ${a.duration.toFixed(3)} s, ` +
        `peak ${a.peak.toFixed(5)}, rms ${a.rms.toFixed(5)}`,
    );
    lines.push(
      a.peak > 1e-4
        ? "  signal: ✅ non-silent"
        : "  signal: ⚠️ silent (no audio in the rendered window)",
    );
  } catch (e) {
    lines.push(`  decode: ❌ FAIL — ${formatError(e)}`);
  }
}

/** Creates a fresh MIDI track and inserts the instrument (default preset). */
async function createOperatorTrack(context: Ctx): Promise<{
  track: MidiTrack<ApiVersion>;
  device: Device<ApiVersion>;
}> {
  const song = context.application.song!;
  const track = await song.createMidiTrack();
  const device = await track.insertDevice(INSTRUMENT_NAME, 0);
  return { track, device };
}

/** Creates a MIDI track, inserts Operator, and adds one sustained note. */
async function buildTestTrack(context: Ctx): Promise<{
  track: MidiTrack<ApiVersion>;
  deviceName: string;
  paramCount: number;
}> {
  const { track, device } = await createOperatorTrack(context);

  const clip = await track.createMidiClip(0, NOTE_BEATS);
  clip.notes = [
    { pitch: NOTE_PITCH, startTime: 0, duration: NOTE_BEATS, velocity: NOTE_VELOCITY },
  ];

  return { track, deviceName: device.name, paramCount: device.parameters.length };
}

/**
 * Reads every parameter of a fresh Operator and writes a JSON schema to disk.
 * This schema is the shared contract used by training (model output heads) and
 * inference (denormalizing predictions back to real parameter values).
 */
async function dumpOperatorSchema(context: Ctx): Promise<void> {
  const song = context.application.song;
  if (!song) {
    await report(context, ["FAIL: No active song."]);
    return;
  }

  const lines: string[] = [];
  let track: MidiTrack<ApiVersion> | null = null;

  try {
    const created = await createOperatorTrack(context);
    track = created.track;
    const params = created.device.parameters;

    // Read each parameter's metadata. getValue() is async, so resolve in parallel.
    const parameters = await Promise.all(
      params.map(async (p, index) => ({
        index,
        name: p.name,
        min: p.min,
        max: p.max,
        defaultValue: p.defaultValue,
        value: await p.getValue(),
        isQuantized: p.isQuantized,
        valueItems: p.isQuantized
          ? p.valueItems.map((v) => ({ name: v.name, shortName: v.shortName }))
          : null,
      })),
    );

    const quantized = parameters.filter((p) => p.isQuantized);
    const schema = {
      device: created.device.name,
      apiVersion: "1.0.0" as const,
      parameterCount: parameters.length,
      quantizedCount: quantized.length,
      continuousCount: parameters.length - quantized.length,
      parameters,
    };

    await fs.mkdir(path.dirname(SCHEMA_OUT), { recursive: true });
    await fs.writeFile(SCHEMA_OUT, JSON.stringify(schema, null, 2), "utf8");

    lines.push("✅ Wrote Operator parameter schema");
    lines.push(`Device: ${schema.device}`);
    lines.push(
      `Parameters: ${schema.parameterCount} ` +
        `(${schema.quantizedCount} quantized, ${schema.continuousCount} continuous)`,
    );
    lines.push(`File: ${SCHEMA_OUT}`);
    lines.push("");
    lines.push("First quantized params (categorical model heads):");
    for (const q of quantized.slice(0, 10)) {
      const items = (q.valueItems ?? []).map((v) => v.name).join(", ");
      lines.push(`  ${q.name}: [${items}]`);
    }
  } catch (e) {
    lines.push("❌ FAIL: " + formatError(e));
  } finally {
    if (track) await song.deleteTrack(track).catch(() => undefined);
  }

  await report(context, lines);
}

function analyze(decoded: ReturnType<typeof decodeWav>): {
  sampleRate: number;
  channels: number;
  duration: number;
  peak: number;
  rms: number;
} {
  const channels = decoded.numberOfChannels;
  let peak = 0;
  let sumSq = 0;
  let count = 0;
  for (let c = 0; c < channels; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < data.length; i++) {
      const v = Math.abs(data[i]!);
      if (v > peak) peak = v;
      sumSq += data[i]! * data[i]!;
      count++;
    }
  }
  return {
    sampleRate: decoded.sampleRate,
    channels,
    duration: decoded.duration,
    peak,
    rms: count > 0 ? Math.sqrt(sumSq / count) : 0,
  };
}

/** Robustly stringifies anything that might be thrown (Error, string, object, null/undefined). */
function formatError(e: unknown): string {
  if (e === undefined) return "undefined (a non-Error value was thrown with no detail)";
  if (e === null) return "null";
  if (e instanceof Error) return e.stack ?? `${e.name}: ${e.message}`;
  if (typeof e === "string") return e;
  try {
    const keys = Object.keys(e as object);
    return `[${typeof e}] ${JSON.stringify(e)}${keys.length ? ` keys=[${keys.join(",")}]` : ""}`;
  } catch {
    return `[${typeof e}] ${String(e)}`;
  }
}

/** Shows the report in a modal dialog (and mirrors it to the host log). */
async function report(context: Ctx, lines: string[]): Promise<void> {
  const text = lines.join("\n");
  console.log("[doppelganger render test]\n" + text);

  const html = `<!doctype html><html><head><meta charset="utf-8">
<style>
  body { font: 13px/1.5 -apple-system, Segoe UI, sans-serif; margin: 16px; background:#1e1e1e; color:#eee; }
  pre  { white-space: pre-wrap; word-break: break-word; }
  button { margin-top: 12px; padding: 6px 18px; font-size: 13px; }
</style></head>
<body>
  <h3>doppelganger — Render Test</h3>
  <pre>${escapeHtml(text)}</pre>
  <button onclick="done()">OK</button>
  <script>
    function done() {
      const msg = "ok";
      if (window.chrome && window.chrome.webview) window.chrome.webview.postMessage(msg);
      else if (window.webkit) window.webkit.messageHandlers.live.postMessage(msg);
    }
  </script>
</body></html>`;

  await context.ui.showModalDialog(
    `data:text/html,${encodeURIComponent(html)}`,
    560,
    480,
  );
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
