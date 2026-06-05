---
name: ableton-extensions-sdk
description: Reference for building Ableton Live "Extensions" (the JS/TS extension SDK, 1.0.0-beta). Use when writing, debugging, or designing any Ableton extension — covers project structure, build, the API surface (Song/Track/Device/DeviceParameter/MidiClip/Resources/Ui), audio rendering, MIDI, device parameters, and known limitations.
---

# Ableton Live Extensions SDK — Reference

> The new Ableton "Extensions" system lets you automate Live with **JavaScript/TypeScript** running in a **Node.js** runtime embedded in Live. This is the cheat-sheet for building one. SDK version covered: **1.0.0-beta.0**.
>
> Source of truth (unzipped SDK): `D:\AbletonExtensions\abletonExtensionsSDK`
> - `api/` — HTML API reference (one file per class/interface/enum). Read as text; strip the markup.
> - `docs/` — HTML guide pages (astro site).
> - `examples/` — **7 working TS example extensions** — the best learning resource. Read these first.
>
> Markers below: ✅ = verified from API HTML or example source · ⚠️ = inferred/uncertain, verify before relying on it.

---

## 1. Mental model (for someone coming from Python/C++)

An extension is an **npm package** written in TS, **bundled with esbuild** into a single CommonJS file, that Live loads into an embedded Node runtime. You don't write a `main()`; you export an `activate()` function. Inside it you **register commands** (named callbacks) and **attach them to UI entry points** (context-menu items). When the user clicks your menu item, Live calls your command with a *handle* to whatever was clicked (a track, clip slot, arrangement selection…). You resolve that handle into a typed object and call async methods on it.

Key idioms that differ from typical Python/C++:
- **Everything that touches Live is `async`** (returns `Promise`). `await` constantly.
- You interact with Live's data model through **proxy objects** (`Song`, `Track`, `Device`, `DeviceParameter`, …). Reading a value is often an `await someParam.getValue()`, not a plain field.
- The runtime is **Node**, so you have `fs/promises`, `Buffer`, npm packages, etc.
- UI is **HTML in a webview** (modal dialog), communicating back via `postMessage`. There is no native widget toolkit.

---

## 2. Project structure & build ✅

A minimal extension (mirrors `examples/strip-silence`):

```
my-extension/
  manifest.json        # extension metadata Live reads
  package.json         # npm deps + build scripts
  build.ts             # esbuild config
  tsconfig.json
  src/
    extension.ts       # entry point: exports activate()
  dist/
    extension.js       # build output (manifest.entry points here)
```

**manifest.json** ✅
```json
{
  "name": "Warp Mode",
  "author": "Ableton",
  "entry": "dist/extension.js",
  "version": "0.0.1",
  "minimumApiVersion": "1.0.0"
}
```

**package.json** ✅ (note `"type": "module"`, SDK consumed as a local tarball)
```json
{
  "type": "module",
  "main": "dist/extension.js",
  "scripts": {
    "build": "tsc --noEmit && tsx build.ts",
    "start": "tsx build.ts && extensions-cli run"
  },
  "dependencies": {
    "@ableton-extensions/sdk": "file:../../ableton-extensions-sdk-1.0.0-beta.0.tgz",
    "audio-decode": "^2.2.3"
  },
  "devDependencies": {
    "@ableton-extensions/cli": "file:../../ableton-extensions-cli-1.0.0-beta.0.tgz",
    "@types/node": "^25.0.10",
    "esbuild": "0.28.0",
    "tsx": "^4.19.0",
    "typescript": "^5.9.3"
  }
}
```

**build.ts** ✅ — esbuild bundles to a single CJS file targeting Node:
```ts
import * as esbuild from "esbuild";
import * as fs from "node:fs";
const manifest = JSON.parse(fs.readFileSync("manifest.json", "utf8"));
const production = process.argv.includes("--production");
await esbuild.build({
  entryPoints: ["src/extension.ts"],
  outfile: manifest.entry,
  bundle: true,
  format: "cjs",
  platform: "node",
  sourcesContent: false,
  logLevel: "info",
  minify: production,
  sourcemap: !production,
});
```

**Two ways to load an extension (`@ableton-extensions/cli`):** ✅
- **`extensions-cli package [dir] -o out.ablx`** → builds a **`.ablx`** archive (zip of `manifest.json` + the `dist/` bundle). The user **drags the `.ablx` into Live → Settings → Extensions** to install. This is the distribution/install path. (Requires a built `dist/` first.)
- **`extensions-cli run [dir]`** → hot-loads into Live's **Extension Host** for development. Needs `EXTENSION_HOST_PATH` (path to `ExtensionHostNodeModule.node`) set in the env or a `.env` file in the extension dir, or `--live <path>`. Without it you get *"No Extension Host path provided."* `--inspect` attaches the VS Code debugger.

Scaffold new extensions with `@ableton-extensions/create-extension` (in the SDK zip).

**Run/develop:** `npm run start` typically = build + `extensions-cli run`; add a `package` script = build + `extensions-cli package`. ✅ Importing HTML works: esbuild can **inline an `.html` file** as a string import (see `examples/modal-dialog`, which has a `src/html.d.ts` declaring the module). ✅

---

## 3. Entry point & context ✅

```ts
import { initialize, type ActivationContext, ... } from "@ableton-extensions/sdk";

export function activate(activation: ActivationContext) {
  const context = initialize(activation, "1.0.0"); // ExtensionContext

  // register a command (named async callback)
  context.commands.registerCommand("example.doThing", (arg: unknown) =>
    void (async (selection /* or handle */) => {
      // ... your logic ...
    })(arg as /* ArrangementSelection | Handle | ... */).catch((e) => console.error(e)),
  );

  // attach it to a UI entry point
  context.ui.registerContextMenuAction(
    "AudioTrack.ArrangementSelection", // scope
    "Do Thing",                        // menu label
    "example.doThing",                 // commandId registered above
  );
}
```

**`ExtensionContext` members** (the hub for everything): ✅
- `context.application` — `Application`; `context.application.song` → the `Song`.
- `context.commands` — `registerCommand(id, callback)`.
- `context.ui` — dialogs, context menus (see §8).
- `context.resources` — file import + audio render (see §7).
- `context.getObjectFromHandle(handle, ExpectedClass)` — resolve a UI handle into a typed object. ✅
- `context.withinTransaction(callback)` — batch many model mutations into one undoable transaction; returns whatever the callback returns (commonly an array of promises you then `await Promise.all` on). ✅
- `context.withinProgressDialog(...)` exists via `context.ui` (see §8). 

`console.log` / `console.error` work for debugging. ✅

---

## 4. The data model — navigation ✅

```
Application
 └─ song: Song
     ├─ tracks: Track[]            (MidiTrack | AudioTrack)
     ├─ returnTracks, mainTrack
     ├─ scenes: Scene[]
     ├─ tempo: number   (get/set)  ← BPM; needed for beats↔seconds
     ├─ createMidiTrack(...), createAudioTrack(...), deleteTrack(...)
     ├─ createScene(...), deleteScene(...), duplicateTrack/Scene(...)
     └─ cuePoints, gridQuantization, rootNote, scale*, ...
```

`Track` (base) / `MidiTrack` / `AudioTrack` share: ✅
- `devices: Device[]`, `insertDevice(name, index)`, `deleteDevice(...)`, `duplicateDevice(...)`
- `clipSlots`, `arrangementClips`, `clearClipsInRange(startBeat, endBeat)`
- `mixer`, `name` (get/set), `arm`, `mute`, `solo`, `groupTrack`, `takeLanes`, `createTakeLane`, `deleteClip`
- `MidiTrack.createMidiClip(startTime, duration)` → `MidiClip`
- `AudioTrack.createAudioClip({ filePath, startTime, duration, isWarped, loopSettings })`

Resolve a handle, then narrow by class:
```ts
const track = context.getObjectFromHandle(handle, AudioTrack);
// or test: device instanceof Simpler / DrumRack
```

**Device classes:** `Device` (base) → `RackDevice` → `DrumRack`; `Simpler`. Identify a generic device by `device.name` (e.g. `"Operator"`). ✅

---

## 5. Devices & parameters ✅ (core of synth automation)

```ts
const device = track.devices[0];           // Device
device.name;                               // string, e.g. "Operator"
const params = device.parameters;          // DeviceParameter[]
```

**`DeviceParameter`** ✅
| Member | Kind | Notes |
|---|---|---|
| `name` | getter `string` | parameter display name |
| `min` | getter `number` | **real units**, not 0–1 |
| `max` | getter `number` | real units |
| `defaultValue` | getter `number` | |
| `isQuantized` | getter `boolean` | discrete/enum vs continuous |
| `valueItems` | getter `DeviceParameterValueItem[]` | the enum options (when quantized) |
| `getValue()` | `Promise<number>` | current value (real units) |
| `setValue(value)` | `Promise<void>` | set value (real units) |

`DeviceParameterValueItem = { name: string; shortName: string }` ✅ — for a quantized param, the **index** into `valueItems` is the parameter's value (e.g. value `2` ⇒ `valueItems[2]`). ⚠️ confirm the index/value mapping empirically.

Read/write pattern:
```ts
for (const p of device.parameters) {
  const v = await p.getValue();
  if (p.isQuantized) { /* v indexes p.valueItems */ }
  else { /* v in [p.min, p.max] */ }
}
await someParam.setValue(0.5);
```

⚠️ **Caveat:** values are in **real units** with per-parameter ranges; for ML you'll normalize each param to [0,1] using `min`/`max` (continuous) or one-hot over `valueItems.length` (quantized). Some Operator params are quantized enums (waveform, algorithm, on/off toggles); most are continuous.

**Adding/loading devices:** `track.insertDevice(deviceName, index)` ✅ — **built-in Live devices with default preset only**; ⚠️ no third-party plugins and (apparently) no loading custom presets via API. So you program a device by setting its parameters, not by loading `.adv` presets.

---

## 6. MIDI & making sound ✅

```ts
const clip = await midiTrack.createMidiClip(startTimeBeats, durationBeats);
clip.notes = [
  { pitch: 60, startTime: 0, duration: 4, velocity: 100 },
];
```

**`NoteDescription`** ✅ (type alias):
```ts
type NoteDescription = {
  pitch: number;        // MIDI note (60 = C3 in Live's convention)
  startTime: number;    // in beats, relative to clip start
  duration: number;     // in beats
  velocity?: number;
  muted?: boolean; probability?: number; releaseVelocity?: number;
  selected?: boolean; velocityDeviation?: number;
};
```

⚠️ **There is NO transport API** (`play`/`stop`/`record`/seek not found on `Song`). You don't trigger sound live; instead you **place a note in the arrangement and render that beat range offline** (see §7). `Song.tempo` (get/set) is the only transport-ish control. To convert: `beatsPerSecond = tempo / 60`.

---

## 7. Audio rendering & file I/O ✅ (and the big limitation ⚠️)

**Render to WAV** ✅:
```ts
const wavPath: string = await context.resources.renderPreFxAudio(
  track,            // ⚠️ typed AudioTrack ONLY
  startTimeBeats,
  endTimeBeats,
);
```
Renders the **pre-effects** audio of an **arrangement** region (times **in beats**) to a WAV in the extension's temp dir, returns its path.

> ⚠️⚠️ **CRITICAL LIMITATION — read before designing any render pipeline.**
> `renderPreFxAudio` is **strictly typed to `AudioTrack`**. Instruments (Operator, etc.) live on **MIDI tracks**, and there is **no `freeze` / `flatten` / `bounce` / `resample` / `record` / output-routing API** anywhere (`Song`, `Track`, `MidiTrack` all checked). So **rendering an instrument's output to audio is not a documented capability**. Before building a data-generation pipeline that depends on it, **empirically test whether `renderPreFxAudio` accepts a `MidiTrack` at runtime** despite the type. If it does not, you must render the instrument outside this SDK (e.g. drive Live via AbletonOSC/Max-for-Live freeze+export, or reimplement the instrument's DSP). There is also **no post-FX render**.

**Read the rendered PCM** ✅ (via npm `audio-decode`, Node `fs`):
```ts
import * as fs from "fs/promises";
import decodeAudio from "audio-decode";
const decoded = await decodeAudio(await fs.readFile(wavPath));
decoded.numberOfChannels; decoded.sampleRate; decoded.duration;
const ch0: Float32Array = decoded.getChannelData(0);
```

**Other file I/O** ✅:
- `context.resources.importIntoProject(filePath)` → copies a file into the Live project, returns the managed path (use the returned path in subsequent calls).
- General disk access via Node `fs/promises` (read/write arbitrary files — e.g. save JSON params, write datasets). ✅
- `Sample.filePath` (read-only) gives a sample's path. ⚠️ No raw-PCM accessor on `Sample`/`AudioClip`; go through render+decode.

---

## 8. UI ✅

**Modal dialog (HTML webview):**
```ts
const html = `<html>... <script> /* call back to extension */ </script></html>`;
const result: string = await context.ui.showModalDialog(
  `data:text/html,${encodeURIComponent(html)}`, // also supports file:, https:, http://localhost
  width, height,
);
```
The page posts a result back via `window.webkit.messageHandlers.live.postMessage(...)` (macOS) / `window.chrome.webview.postMessage(...)` (Windows); `showModalDialog` resolves with that string. ✅ The webview runs **arbitrary JS** and can load `https://` resources — so charts/visualizations and even client-side ML are feasible in the panel. ⚠️ Only **modal** dialogs are documented; no persistent docked panel API.

**Progress dialog (with cancel):** ✅
```ts
await context.ui.withinProgressDialog("Title", { progress: 0 }, async (update, abortSignal) => {
  update("Working…", 50);
  if (abortSignal.aborted) return;
});
```

**Context menu actions:** `context.ui.registerContextMenuAction(scope, label, commandId)` → returns an unregister fn. ✅ Scopes seen: `"ClipSlot"`, `"AudioTrack"`, `"MidiTrack"`, `"AudioClip"`, `"MidiClip"`, `"AudioTrack.ArrangementSelection"`, `"MidiTrack.ArrangementSelection"`. ⚠️ exact full list — see `api/types/ContextMenuScope.html`.

---

## 9. The 7 example extensions (study these) ✅

| Example | Demonstrates |
|---|---|
| `strip-silence` | **`renderPreFxAudio` + `audio-decode` + `withinProgressDialog` + `withinTransaction`** — the rendering reference |
| `audio-clips` | `importIntoProject`, `createAudioClip`, clip slots vs arrangement |
| `arrangementselection` | resolving `ArrangementSelection`, `createMidiClip`, setting `clip.notes` |
| `modal-dialog` | HTML modal, inlined `.html` import, `postMessage` round-trip |
| `progress-dialog` | progress + abort signal |
| `context-menu` | registering menu actions across scopes |
| `warpMode` | `AudioClip.warpMode` / warp markers; minimal manifest |

---

## 9b. Environment & troubleshooting ✅ (learned the hard way)

- **Requires the Live *Beta* build** (from Centercode) with **Preferences → Extensions**. Plain/stable Live won't have it. Match the Live beta to the SDK release it shipped with.
- **Developer Mode** (Preferences → Extensions) must be **ON** for `extensions-cli run` / `npm start` to connect. Without it, `npm start` cannot reach Live.
- **Extension Host module** (the Node host Live spawns): on Windows at
  `C:\ProgramData\Ableton\Live 12 Beta\Program\ExtensionHost\ExtensionHostNodeModule.node`.
  Put `EXTENSION_HOST_PATH=<that path>` in a `.env` in the extension dir (or pass `--live`).
- **Logs:**
  - Extension Host log (your `console.*` + uncaught stack traces): `%APPDATA%\Ableton\Live x.x.x\Preferences\ExtensionHost.txt`. **If this file does not exist, the host never successfully ran an extension** — look upstream.
  - Live's main log: `%APPDATA%\Ableton\Live x.x.x\Preferences\Log.txt`. Grep it for `Extension Host` — e.g. `warning: Extension Host: check for installed extensions timed out` means the host's discovery phase failed (environment/startup issue, not your extension's code).
- **`.ablx` install registers in `Preferences.cfg`**; if grepping it shows no trace of your extension after install, the install didn't take.
- **If the host times out / nothing loads:** fully quit all Live + stray `node`/ExtensionHost processes; check antivirus/firewall isn't blocking the host's local socket; confirm the beta is licensed (not demo); prefer the **dev loop (`npm start`)** over `.ablx` for diagnostics since it streams errors to your terminal.
- **Keep the bundle small.** Heavy deps (e.g. `audio-decode` ≈ 4 MB) bloat the `.ablx` and add load-time risk; for reading your *own* rendered WAVs a ~100-line WAV parser is enough (PCM 16/24/32 + IEEE float, handle `WAVE_FORMAT_EXTENSIBLE`).

## 10. Quick gotchas

- **Beats, not seconds**, everywhere in clips/render. Convert with `song.tempo`.
- **Everything is async** — forget an `await` and you'll set params before the device exists.
- **Param values are real units** with per-param ranges; normalize for ML.
- **`renderPreFxAudio` = AudioTrack + pre-FX only** (see §7 warning).
- **No transport / no preset loading / built-in devices only.**
- Wrap multi-mutation work in `withinTransaction` for a single undo step and atomicity.
- When unsure of an exact signature, grep the relevant `api/classes/*.html` (strip tags) — that's authoritative for this beta.
