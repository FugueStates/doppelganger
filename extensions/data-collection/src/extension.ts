/**
 * doppelganger — Data Collection extension
 * =========================================
 *
 * Generates training data for the Operator sound-matching model. Because Ableton
 * exposes no offline instrument render to any API, we exploit the one offline path
 * that exists — the GUI "Export Audio" (faster-than-realtime) — and batch it hard
 * with many tracks at once.
 *
 * Workflow (right-click any MIDI track):
 *   1. "doppelganger: Build Operator Rack"
 *        Creates NUM_TRACKS MIDI tracks (op_0000…) each with Operator + a C3 note.
 *        Run once per Live set.
 *   2. "doppelganger: Randomize Batch"
 *        Randomizes every parameter on every rack track and writes a manifest under
 *        dataset/pending/<batchId>/. It then tells you the export prefix to use.
 *   3. You (for now) Export Audio → "All Individual Tracks" into that batch folder
 *        using the given prefix and a fixed length (RENDER_SECONDS).
 *   4. Python pairs the WAVs with the manifest:
 *        uv run python -m doppelganger.datagen.split_export ... (process step)
 *   Repeat 2–4. (Steps 2–4 will be automated next via pywinauto.)
 */

import { initialize, type ActivationContext } from "@ableton-extensions/sdk";

import * as fs from "fs/promises";
import * as path from "path";

import {
  NUM_TRACKS,
  RENDER_SECONDS,
  MAX_BATCHES,
  BATCH_TIMEOUT_MS,
  STOP_FILE,
  PREDICT_MANIFEST,
} from "./config.js";
import { buildRack, findOperatorTracks, randomizeAll, applyParams } from "./operator.js";
import { nextBatchId, writeManifest } from "./batch.js";
import { report, formatError } from "./report.js";

type Ctx = ReturnType<typeof initialize>;

export function activate(activation: ActivationContext) {
  const context = initialize(activation, "1.0.0");

  register(context, "doppelganger.buildRack", "doppelganger: Build Operator Rack", runBuildRack);
  register(context, "doppelganger.randomizeBatch", "doppelganger: Randomize Batch", runRandomizeBatch);
  register(context, "doppelganger.autoCollect", "doppelganger: Auto Collect", runAutoCollect);
  register(context, "doppelganger.applyPredicted", "doppelganger: Apply Predicted Batch", runApplyPredicted);
}

/** "Hear it" eval: apply model-predicted params from the predict manifest to the rack. */
async function runApplyPredicted(context: Ctx): Promise<void> {
  const tracks = findOperatorTracks(context);
  if (tracks.length === 0) {
    await report(context, ['No rack found. Run "Build Operator Rack" first.']);
    return;
  }
  try {
    const manifest = JSON.parse(await fs.readFile(PREDICT_MANIFEST, "utf8"));
    const applied = await applyParams(context, tracks, manifest.tracks);
    await report(context, [
      `✅ Applied predicted params to ${applied} track(s).`,
      "Now export (All Individual Tracks) into dataset/predict/render/,",
      "then run: uv run python -m doppelganger.training.hear_it compare",
    ]);
  } catch (e) {
    await report(context, ["❌ Apply failed: " + formatError(e)]);
  }
}

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

/** Polls for a file to appear, up to a timeout. Returns true if it appeared. */
async function waitForFile(p: string, timeoutMs: number): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await fileExists(p)) return true;
    await delay(500);
  }
  return false;
}

/**
 * Unattended loop: randomize -> write manifest + READY, then wait for the Python
 * watcher to export+process and write DONE, then repeat. No modal dialog is shown
 * during the loop so Live stays free for the watcher's Export keystrokes.
 *
 * Run the watcher first: `uv run python -m doppelganger.datagen.run_loop`
 * Stop early by creating the STOP file (the watcher's Ctrl+C also stalls the loop,
 * which then times out and exits).
 */
async function runAutoCollect(context: Ctx): Promise<void> {
  const tracks = findOperatorTracks(context);
  if (tracks.length === 0) {
    await report(context, ["No rack found. Run \"Build Operator Rack\" first."]);
    return;
  }

  let completed = 0;
  let stoppedReason = `reached MAX_BATCHES (${MAX_BATCHES})`;

  for (let i = 0; i < MAX_BATCHES; i++) {
    if (await fileExists(STOP_FILE)) {
      stoppedReason = "STOP file found";
      break;
    }

    const recorded = await randomizeAll(context, tracks);
    const batchId = await nextBatchId();
    const info = await writeManifest(batchId, recorded);
    await fs.writeFile(path.join(info.batchDir, "READY"), "", "utf8");
    console.log(`[doppelganger] ${batchId} READY — waiting for export…`);

    const done = await waitForFile(path.join(info.batchDir, "DONE"), BATCH_TIMEOUT_MS);
    if (!done) {
      stoppedReason = `timed out waiting for ${batchId} (is the Python watcher running?)`;
      break;
    }
    completed++;
    console.log(`[doppelganger] ${batchId} done (${completed} batches, ~${completed * tracks.length} samples)`);
  }

  await report(context, [
    `Auto Collect finished: ${completed} batch(es) (~${completed * tracks.length} samples).`,
    `Stopped because: ${stoppedReason}.`,
  ]);
}

/** Registers a command and a MIDI-track context-menu entry that triggers it. */
function register(
  context: Ctx,
  commandId: string,
  label: string,
  handler: (context: Ctx) => Promise<void>,
): void {
  context.commands.registerCommand(commandId, () =>
    void handler(context).catch((e) => console.error("[doppelganger]", formatError(e))),
  );
  context.ui.registerContextMenuAction("MidiTrack", label, commandId);
}

async function runBuildRack(context: Ctx): Promise<void> {
  const lines: string[] = [];
  try {
    let total = 0;
    await context.ui.withinProgressDialog(
      "Building Operator rack",
      { progress: 0 },
      async (update) => {
        update(`Creating up to ${NUM_TRACKS} Operator tracks…`, 20);
        total = await buildRack(context, NUM_TRACKS);
        update("Done", 100);
      },
    );
    lines.push(`✅ Rack ready: ${total} Operator tracks (op_0000…).`);
    lines.push("Next: right-click → \"doppelganger: Randomize Batch\".");
  } catch (e) {
    lines.push("❌ Build failed: " + formatError(e));
  }
  await report(context, lines);
}

async function runRandomizeBatch(context: Ctx): Promise<void> {
  const lines: string[] = [];
  try {
    const tracks = findOperatorTracks(context);
    if (tracks.length === 0) {
      await report(context, [
        "No Operator rack found.",
        'Run "doppelganger: Build Operator Rack" first.',
      ]);
      return;
    }

    let info;
    await context.ui.withinProgressDialog(
      "Randomizing batch",
      { progress: 0 },
      async (update) => {
        update(`Randomizing ${tracks.length} tracks…`, 40);
        const recorded = await randomizeAll(context, tracks);
        update("Writing manifest…", 80);
        const batchId = await nextBatchId();
        info = await writeManifest(batchId, recorded);
        update("Done", 100);
      },
    );

    if (!info) throw new Error("Batch info missing.");
    lines.push(`✅ Batch ${info.batchId} ready (${info.trackCount} tracks).`);
    lines.push("");
    lines.push("Now Export Audio in Live:");
    lines.push("  • Rendered Track: All Individual Tracks");
    lines.push(`  • Length: ${RENDER_SECONDS}s (keep this consistent!)`);
    lines.push(`  • File name prefix: ${info.batchId}`);
    lines.push(`  • Save into: ${info.batchDir}`);
    lines.push("");
    lines.push("Then process it with:");
    lines.push(`  uv run python -m doppelganger.datagen.split_export process "${info.batchDir}"`);
  } catch (e) {
    lines.push("❌ Randomize failed: " + formatError(e));
  }
  await report(context, lines);
}
