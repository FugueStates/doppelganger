/**
 * Batch bookkeeping: a persistent counter and the per-batch manifest.
 *
 * Each batch gets a unique id like "b0000001". You export the rack in Live using
 * that id as the file name prefix, so files land as "b0000001 op_0000.wav" etc.
 * The manifest records which parameter values produced each track, for pairing.
 */

import * as fs from "fs/promises";
import * as path from "path";

import { PENDING_DIR, STATE_FILE, INSTRUMENT_NAME, RENDER_SECONDS } from "./config.js";
import type { TrackParams, NoteInfo } from "./operator.js";

interface State {
  next: number;
}

async function readState(): Promise<State> {
  try {
    return JSON.parse(await fs.readFile(STATE_FILE, "utf8")) as State;
  } catch {
    return { next: 1 };
  }
}

async function writeState(state: State): Promise<void> {
  await fs.mkdir(path.dirname(STATE_FILE), { recursive: true });
  await fs.writeFile(STATE_FILE, JSON.stringify(state, null, 2), "utf8");
}

/** Reserves and returns the next batch id (e.g. "b0000001"), persisting the counter. */
export async function nextBatchId(): Promise<string> {
  const state = await readState();
  const id = `b${String(state.next).padStart(7, "0")}`;
  await writeState({ next: state.next + 1 });
  return id;
}

export interface BatchInfo {
  batchId: string;
  batchDir: string;
  manifestPath: string;
  trackCount: number;
}

/** Writes the manifest for a batch and returns where everything lives.
 *  `notes` (optional) maps track name -> { pitch, velocity } so the played note is
 *  recorded per sample alongside the params (consumed by split_export). */
export async function writeManifest(
  batchId: string,
  tracks: Record<string, TrackParams>,
  notes?: Record<string, NoteInfo>,
): Promise<BatchInfo> {
  const batchDir = path.join(PENDING_DIR, batchId);
  await fs.mkdir(batchDir, { recursive: true });

  const manifest = {
    batch: batchId,
    device: INSTRUMENT_NAME,
    prefix: batchId, // export name prefix to use in Live
    renderSeconds: RENDER_SECONDS,
    trackCount: Object.keys(tracks).length,
    tracks,
    notes: notes ?? {},
  };

  const manifestPath = path.join(batchDir, "manifest.json");
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

  return {
    batchId,
    batchDir,
    manifestPath,
    trackCount: manifest.trackCount,
  };
}
