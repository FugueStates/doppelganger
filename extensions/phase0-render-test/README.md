# Phase 0 — Render Validation Spike

A throwaway extension that answers the project's #1 question: **can an Ableton Extension
render an instrument (MIDI) track to a WAV?** See `../../docs/PLAN.md` §2.

## What it does
Adds a right-click item **"doppelganger: Test Operator Render"** on any MIDI track. It
creates a temporary MIDI track + Operator + one note, tries to render it, measures the
audio, shows a PASS/FAIL report, and deletes the temporary track.

## Install it (drag-and-drop `.ablx`)
This is the simple path — no Extension Host path needed.

```powershell
cd D:\AbletonExtensions\doppelganger\extensions\phase0-render-test
npm run package      # builds, then writes dist\phase0-render-test.ablx
```

1. In Live: **Settings → Extensions**, then **drag `dist\phase0-render-test.ablx`** onto that
   panel (or use its add/install button). Enable it.
2. **Right-click any MIDI track header** → **"doppelganger: Test Operator Render"**.
3. Read the dialog.

## Alternative: hot-reload dev loop (`npm run start`)
`extensions-cli run` loads the extension live without packaging, but it needs the path to
Live's Extension Host module. Set it once via a `.env` file in this folder:

```
EXTENSION_HOST_PATH=<path to ExtensionHostNodeModule.node>
```

(See the SDK's bundled `docs/` for where to find `ExtensionHostNodeModule.node`.) Then
`npm run start`. For a one-off render test, the `.ablx` install above is easier.

## What to report back
Copy the dialog text. The key lines:
- **RENDER: ✅ PASS** + a non-zero **Peak amplitude** → MIDI-track rendering works; we build
  Tool 1 (data collection) as a pure extension. 🎉
- **RENDER: ✅ PASS** but **Peak ≈ 0 (SILENT)** → render works but the note isn't sounding;
  small fix (note/preset/window).
- **RENDER: ❌ FAIL** (exception at the render step) → MIDI tracks can't be rendered → we
  switch to the AbletonOSC fallback (PLAN.md §2 / §8 Q1).

Also note the **Parameter count** — that's how many parameters Operator exposes, which sizes
the model's output and the dataset for Tool 1/2.
